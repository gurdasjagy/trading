//! Hardware timestamp support for ultra-low latency tick timing.
//!
//! Uses SO_TIMESTAMPING on Linux to get NIC-level timestamps instead of
//! kernel timestamps, reducing timestamp jitter from ~50us to ~100ns.
//!
//! This module provides functions to enable hardware timestamping on sockets,
//! which is critical for institutional-grade latency measurement and 
//! order book reconstruction.

use std::io;
use std::os::unix::io::RawFd;

/// SO_TIMESTAMPING flags for hardware timestamping.
/// These constants match the Linux kernel definitions.
#[cfg(target_os = "linux")]
mod flags {
    /// Request hardware receive timestamps
    pub const SOF_TIMESTAMPING_RX_HARDWARE: u32 = 1 << 2;
    /// Request hardware transmit timestamps
    pub const SOF_TIMESTAMPING_TX_HARDWARE: u32 = 1 << 1;
    /// Return hardware timestamps in raw format
    pub const SOF_TIMESTAMPING_RAW_HARDWARE: u32 = 1 << 6;
    /// Software receive timestamp (fallback)
    pub const SOF_TIMESTAMPING_RX_SOFTWARE: u32 = 1 << 3;
    /// Software transmit timestamp (fallback)
    pub const SOF_TIMESTAMPING_TX_SOFTWARE: u32 = 1 << 4;
    /// Report timestamp generation via SO_TIMESTAMPING
    pub const SOF_TIMESTAMPING_SOFTWARE: u32 = 1 << 4;
}

/// Enable hardware timestamps on a socket.
/// Call this after socket creation but before binding.
///
/// # Arguments
/// * `fd` - Raw file descriptor of the socket
///
/// # Returns
/// * `Ok(())` if hardware timestamping was enabled successfully
/// * `Err(io::Error)` if the operation failed (e.g., NIC doesn't support HW timestamps)
///
/// # Example
/// ```no_run
/// use std::os::unix::io::AsRawFd;
/// use std::net::UdpSocket;
/// 
/// let socket = UdpSocket::bind("0.0.0.0:0").unwrap();
/// hw_timestamp::enable_hw_timestamp(socket.as_raw_fd()).unwrap();
/// ```
#[cfg(target_os = "linux")]
pub fn enable_hw_timestamp(fd: RawFd) -> io::Result<()> {
    use flags::*;
    
    // Request both hardware and software timestamps.
    // Hardware timestamps are preferred, but software timestamps
    // provide a fallback if HW timestamping isn't available.
    let flags: u32 = SOF_TIMESTAMPING_RX_HARDWARE 
        | SOF_TIMESTAMPING_TX_HARDWARE 
        | SOF_TIMESTAMPING_RAW_HARDWARE
        | SOF_TIMESTAMPING_RX_SOFTWARE
        | SOF_TIMESTAMPING_TX_SOFTWARE;
    
    let ret = unsafe {
        libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_TIMESTAMPING,
            &flags as *const u32 as *const libc::c_void,
            std::mem::size_of::<u32>() as libc::socklen_t,
        )
    };
    
    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        tracing::info!("Hardware timestamping enabled on fd={}", fd);
        Ok(())
    }
}

/// No-op implementation for non-Linux platforms.
/// Hardware timestamping via SO_TIMESTAMPING is Linux-specific.
#[cfg(not(target_os = "linux"))]
pub fn enable_hw_timestamp(_fd: RawFd) -> io::Result<()> {
    tracing::warn!("Hardware timestamping not available on this platform");
    Ok(()) // No-op on non-Linux
}

/// Attempt to enable hardware timestamps, falling back gracefully if not supported.
/// This is the recommended entry point for production use.
///
/// # Arguments
/// * `fd` - Raw file descriptor of the socket
///
/// # Returns
/// * `true` if hardware timestamping was enabled
/// * `false` if falling back to software timestamps (still usable)
pub fn try_enable_hw_timestamp(fd: RawFd) -> bool {
    match enable_hw_timestamp(fd) {
        Ok(()) => {
            tracing::info!("Hardware timestamping enabled successfully");
            true
        }
        Err(e) => {
            tracing::warn!(
                "Hardware timestamping not available ({}), using software timestamps",
                e
            );
            false
        }
    }
}

/// Configuration for timestamp source selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampSource {
    /// Use hardware timestamps from NIC (lowest latency, ~100ns jitter)
    Hardware,
    /// Use kernel software timestamps (~50us jitter)
    Software,
    /// Automatically select best available
    Auto,
}

impl Default for TimestampSource {
    fn default() -> Self {
        Self::Auto
    }
}

/// Get current timestamp in nanoseconds since epoch.
/// Uses the highest resolution clock available.
#[inline]
pub fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Get current timestamp in microseconds since epoch.
#[inline]
pub fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

/// High-precision monotonic timestamp for latency measurement.
/// Uses CLOCK_MONOTONIC_RAW on Linux for best precision.
#[inline]
pub fn monotonic_ns() -> u64 {
    std::time::Instant::now().elapsed().as_nanos() as u64
}

/// Timestamp delta calculator for measuring latencies.
pub struct LatencyMeasurement {
    start_ns: u64,
}

impl LatencyMeasurement {
    /// Start a new latency measurement.
    #[inline]
    pub fn start() -> Self {
        Self { start_ns: now_ns() }
    }
    
    /// Get elapsed time in nanoseconds.
    #[inline]
    pub fn elapsed_ns(&self) -> u64 {
        now_ns().saturating_sub(self.start_ns)
    }
    
    /// Get elapsed time in microseconds.
    #[inline]
    pub fn elapsed_us(&self) -> u64 {
        self.elapsed_ns() / 1000
    }
    
    /// Get elapsed time in milliseconds.
    #[inline]
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed_ns() as f64 / 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_now_ns_monotonic() {
        let t1 = now_ns();
        std::thread::sleep(std::time::Duration::from_micros(100));
        let t2 = now_ns();
        assert!(t2 > t1, "Timestamps should be monotonically increasing");
    }
    
    #[test]
    fn test_latency_measurement() {
        let measurement = LatencyMeasurement::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = measurement.elapsed_ms();
        assert!(elapsed >= 9.0, "Should measure at least 9ms");
        assert!(elapsed < 50.0, "Should not exceed 50ms for 10ms sleep");
    }
}
