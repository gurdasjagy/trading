//! Directive 5: Dual Token-Bucket Rate Limiter for Gate.io API.
//!
//! Gate.io limits:
//!   - Private endpoints (orders): 400 req/s per User ID
//!   - Public endpoints (market data): 300 req/s per IP
//!   - WebSocket connections: max 300 per IP
//!
//! This module provides two independent token buckets with safe headroom:
//!   - Public: 250 req/s (83% of 300 limit)
//!   - Private: 350 req/s (87.5% of 400 limit)
//!
//! If a request threatens to breach the limit, it is queued with a
//! microsecond-precision delay rather than dropped or rejected.
//!
//! # Testnet vs Live
//!
//! Testnet API may have lower limits. The rate limiter auto-adjusts
//! based on a `testnet` flag.
//!
//! # Thread Safety
//!
//! The `RateLimiterPool` is designed to be shared across threads via `Arc`.
//! Each bucket uses atomic operations for lock-free token acquisition.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tracing::{debug, warn, info};

// ═══════════════════════════════════════════════════════════════════════════
// Token Bucket
// ═══════════════════════════════════════════════════════════════════════════

/// A single token bucket rate limiter.
///
/// Uses atomic operations for lock-free token acquisition. Tokens are
/// refilled continuously based on wall-clock time.
pub struct TokenBucket {
    /// Maximum tokens in the bucket.
    max_tokens: u64,
    /// Tokens added per second.
    refill_rate: u64,
    /// Current token count (atomic, scaled by 1000 for sub-token precision).
    tokens_x1000: AtomicU64,
    /// Timestamp of last refill (nanoseconds since epoch).
    last_refill_ns: AtomicU64,
    /// Total requests passed through.
    total_passed: AtomicU64,
    /// Total requests that had to wait.
    total_waited: AtomicU64,
    /// Name for logging.
    name: &'static str,
}

impl TokenBucket {
    pub fn new(name: &'static str, max_tokens: u64, refill_rate: u64) -> Self {
        Self {
            max_tokens,
            refill_rate,
            tokens_x1000: AtomicU64::new(max_tokens * 1000),
            last_refill_ns: AtomicU64::new(now_ns()),
            total_passed: AtomicU64::new(0),
            total_waited: AtomicU64::new(0),
            name,
        }
    }

    /// Try to acquire one token. Returns the wait time in microseconds if
    /// the bucket is empty (the caller should sleep this long before retrying).
    /// Returns 0 if the token was acquired immediately.
    pub fn acquire(&self) -> u64 {
        self.refill();

        // Try to acquire a token
        loop {
            let current = self.tokens_x1000.load(Ordering::Relaxed);
            if current >= 1000 {
                // Try to consume one token
                if self
                    .tokens_x1000
                    .compare_exchange_weak(current, current - 1000, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    self.total_passed.fetch_add(1, Ordering::Relaxed);
                    return 0; // Immediate acquisition
                }
                // CAS failed — retry
                continue;
            } else {
                // No tokens available — calculate wait time
                self.total_waited.fetch_add(1, Ordering::Relaxed);
                let wait_us = 1_000_000 / self.refill_rate.max(1);
                return wait_us;
            }
        }
    }

    /// Acquire a token, blocking if necessary with async sleep.
    /// This is the primary interface for the execution pipeline.
    pub async fn acquire_async(&self) {
        loop {
            let wait_us = self.acquire();
            if wait_us == 0 {
                return;
            }
            debug!(
                "[rate-limiter/{}] Throttled: waiting {}μs",
                self.name, wait_us
            );
            tokio::time::sleep(Duration::from_micros(wait_us)).await;
        }
    }

    /// Refill tokens based on elapsed time.
    fn refill(&self) {
        let now = now_ns();
        let prev = self.last_refill_ns.load(Ordering::Relaxed);
        let elapsed_ns = now.saturating_sub(prev);

        if elapsed_ns < 1_000_000 {
            // Less than 1ms — not worth refilling
            return;
        }

        // Calculate tokens to add
        let tokens_to_add = (elapsed_ns as u128 * self.refill_rate as u128 / 1_000_000_000) as u64;
        if tokens_to_add == 0 {
            return;
        }

        // Update last refill time
        let _ = self.last_refill_ns.compare_exchange(
            prev,
            now,
            Ordering::AcqRel,
            Ordering::Relaxed,
        );

        // Add tokens (capped at max)
        let tokens_to_add_x1000 = tokens_to_add * 1000;
        let max_x1000 = self.max_tokens * 1000;
        loop {
            let current = self.tokens_x1000.load(Ordering::Relaxed);
            let new = (current + tokens_to_add_x1000).min(max_x1000);
            if new == current {
                break;
            }
            if self
                .tokens_x1000
                .compare_exchange_weak(current, new, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Get current available tokens (for monitoring).
    pub fn available_tokens(&self) -> u64 {
        self.refill();
        self.tokens_x1000.load(Ordering::Relaxed) / 1000
    }

    /// Get total requests passed through.
    pub fn total_passed(&self) -> u64 {
        self.total_passed.load(Ordering::Relaxed)
    }

    /// Get total requests that had to wait.
    pub fn total_waited(&self) -> u64 {
        self.total_waited.load(Ordering::Relaxed)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Rate Limiter Pool — dual bucket
// ═══════════════════════════════════════════════════════════════════════════

/// Which type of API endpoint is being called.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointType {
    /// Public endpoints: market data, contract info, etc.
    Public,
    /// Private endpoints: orders, positions, balance, etc.
    Private,
}

/// Dual token-bucket rate limiter for Gate.io API.
///
/// Shared across all threads via `Arc<RateLimiterPool>`.
pub struct RateLimiterPool {
    /// Rate limiter for public endpoints.
    pub public: TokenBucket,
    /// Rate limiter for private/order endpoints.
    pub private: TokenBucket,
    /// Whether this is a testnet instance (may have lower limits).
    pub testnet: bool,
    /// WebSocket connection count tracker.
    ws_connections: AtomicU64,
    /// Maximum WS connections (Gate.io: 300 per IP).
    max_ws_connections: u64,
}

impl RateLimiterPool {
    /// Create a new rate limiter pool for live trading.
    pub fn new_live() -> Self {
        Self {
            public: TokenBucket::new("public", 250, 250),
            private: TokenBucket::new("private", 350, 350),
            testnet: false,
            ws_connections: AtomicU64::new(0),
            max_ws_connections: 280, // 300 limit with 20 headroom
        }
    }

    /// Create a new rate limiter pool for testnet (more conservative).
    pub fn new_testnet() -> Self {
        Self {
            public: TokenBucket::new("public-test", 100, 100),
            private: TokenBucket::new("private-test", 150, 150),
            testnet: true,
            ws_connections: AtomicU64::new(0),
            max_ws_connections: 100,
        }
    }

    /// Create the appropriate rate limiter based on whether we're using testnet.
    pub fn new(testnet: bool) -> Self {
        if testnet {
            info!("[rate-limiter] Initialized TESTNET rate limits (public=100/s, private=150/s)");
            Self::new_testnet()
        } else {
            info!("[rate-limiter] Initialized LIVE rate limits (public=250/s, private=350/s)");
            Self::new_live()
        }
    }

    /// Acquire a rate limit token for the given endpoint type.
    /// Blocks asynchronously if the bucket is empty.
    pub async fn acquire(&self, endpoint_type: EndpointType) {
        match endpoint_type {
            EndpointType::Public => self.public.acquire_async().await,
            EndpointType::Private => self.private.acquire_async().await,
        }
    }

    /// Try to acquire a rate limit token synchronously.
    /// Returns the wait time in microseconds (0 = immediate).
    pub fn try_acquire(&self, endpoint_type: EndpointType) -> u64 {
        match endpoint_type {
            EndpointType::Public => self.public.acquire(),
            EndpointType::Private => self.private.acquire(),
        }
    }

    /// Track a new WebSocket connection.
    /// Returns false if the connection limit would be exceeded.
    pub fn ws_connect(&self) -> bool {
        let current = self.ws_connections.fetch_add(1, Ordering::AcqRel);
        if current >= self.max_ws_connections {
            self.ws_connections.fetch_sub(1, Ordering::AcqRel);
            warn!(
                "[rate-limiter] WS connection limit reached ({}/{})",
                current, self.max_ws_connections
            );
            false
        } else {
            true
        }
    }

    /// Track a WebSocket disconnection.
    pub fn ws_disconnect(&self) {
        self.ws_connections.fetch_sub(1, Ordering::AcqRel);
    }

    /// Get the current number of active WebSocket connections.
    pub fn ws_connection_count(&self) -> u64 {
        self.ws_connections.load(Ordering::Relaxed)
    }

    /// Log a health report of the rate limiter state.
    pub fn log_health(&self) {
        info!(
            "[rate-limiter] Public: {}/{} tokens, passed={}, waited={} | Private: {}/{} tokens, passed={}, waited={} | WS: {}/{}",
            self.public.available_tokens(), self.public.max_tokens,
            self.public.total_passed(), self.public.total_waited(),
            self.private.available_tokens(), self.private.max_tokens,
            self.private.total_passed(), self.private.total_waited(),
            self.ws_connection_count(), self.max_ws_connections,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_basic() {
        let bucket = TokenBucket::new("test", 10, 10);
        // Should be able to acquire 10 tokens immediately
        for _ in 0..10 {
            assert_eq!(bucket.acquire(), 0);
        }
        // 11th should require waiting
        let wait = bucket.acquire();
        assert!(wait > 0);
    }

    #[test]
    fn test_pool_creation() {
        let pool = RateLimiterPool::new(false);
        assert!(!pool.testnet);
        assert_eq!(pool.public.max_tokens, 250);
        assert_eq!(pool.private.max_tokens, 350);

        let pool = RateLimiterPool::new(true);
        assert!(pool.testnet);
        assert_eq!(pool.public.max_tokens, 100);
        assert_eq!(pool.private.max_tokens, 150);
    }

    #[test]
    fn test_ws_connection_tracking() {
        let pool = RateLimiterPool::new(false);
        assert_eq!(pool.ws_connection_count(), 0);
        assert!(pool.ws_connect());
        assert_eq!(pool.ws_connection_count(), 1);
        pool.ws_disconnect();
        assert_eq!(pool.ws_connection_count(), 0);
    }
}
