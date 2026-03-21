# Docker Deployment Guide

## Overview

This trading bot application is fully containerized using Docker Compose with three main services:
- **bot**: The trading bot application (Python/FastAPI)
- **redis**: Cache and task queue
- **timescaledb**: Time-series PostgreSQL database

## Recent Fixes Applied

### 1. TimescaleDB Background Worker Errors - FIXED ✅

**Problem:**
```
WARNING: failed to launch job 1 "Telemetry Reporter [1]": failed to start a background worker
WARNING: failed to launch job 3 "Job History Log Retention Policy [3]": failed to start a background worker
```

**Root Cause:**
TimescaleDB requires additional PostgreSQL background worker processes to run its internal jobs (telemetry reporting and job history management). The default `max_worker_processes=4` was insufficient.

**Solution Applied:**
Updated `docker-compose.yml` with proper worker configuration:
- Increased `max_worker_processes` from 4 to 8
- Added `timescaledb.max_background_workers=4`

This provides sufficient worker slots for:
- PostgreSQL core processes
- Parallel query workers
- TimescaleDB background jobs (telemetry, retention policies, continuous aggregates, etc.)

### 2. Dashboard Features Not Showing Up - EXPLANATION

**Problem:**
Features from PR #30 (Settings tab, advanced dashboard features) not visible in running container.

**Root Cause:**
Docker images are **immutable snapshots**. When you pull code changes (like PR #30), they exist in your local repository but NOT in the already-built Docker image.

**Why Rebuild is Required:**
The Dockerfile uses `COPY . .` (line 60) which copies **all application code** at build time:
- Python source files
- HTML templates (`templates/*.html`)
- Static assets (`static/css/*.css`, `static/js/*.js`)
- Configuration files

Changes made after the image was built are NOT automatically reflected in running containers.

## How to Deploy Updates

### Step 1: Stop Running Containers
```bash
cd crypto_trading_bot
docker-compose down
```

### Step 2: Rebuild the Bot Image

**Option A: Quick rebuild (no cache, clean build)**
```bash
docker-compose build --no-cache bot
```

**Option B: Full rebuild (recommended for major changes)**
```bash
# Remove old images to ensure clean state
docker rmi crypto_trading_bot:latest 2>/dev/null || true

# Rebuild with no cache
docker-compose build --no-cache
```

### Step 3: Start Services
```bash
docker-compose up -d
```

### Step 4: Verify Deployment
```bash
# Check all services are healthy
docker-compose ps

# Check bot logs for errors
docker-compose logs -f bot

# Check TimescaleDB logs (should see no background worker warnings)
docker-compose logs timescaledb | tail -20

# Access dashboard
curl http://localhost:8080/health
```

## When to Rebuild

You **MUST** rebuild the Docker image when:
- ✅ Code changes (Python files, HTML templates, CSS/JS)
- ✅ New dependencies added to `requirements.txt`
- ✅ Configuration file changes (not environment variables)
- ✅ After pulling changes from Git (merging PRs, updating branches)

You **DON'T** need to rebuild for:
- ❌ Environment variable changes in `.env` (just restart: `docker-compose restart bot`)
- ❌ Volume-mounted files (models directory)
- ❌ Database schema changes (handled by migrations at runtime)

## Troubleshooting

### Issue: "Container keeps restarting"
```bash
# Check logs for specific error
docker-compose logs bot --tail 50

# Common causes:
# - Missing .env file → Copy .env.example to .env
# - Invalid API keys → Check .env configuration
# - Database connection failed → Verify timescaledb is healthy
```

### Issue: "Port already in use"
```bash
# Check what's using port 8080
sudo lsof -i :8080

# Kill existing process or change port in docker-compose.yml
```

### Issue: "Database connection errors"
```bash
# Verify TimescaleDB is healthy
docker-compose ps timescaledb

# Check database logs
docker-compose logs timescaledb

# Restart database if needed
docker-compose restart timescaledb
```

### Issue: "Changes not reflected after rebuild"
```bash
# Force complete rebuild with no cache
docker-compose down
docker system prune -f  # Warning: removes unused Docker objects
docker-compose build --no-cache
docker-compose up -d
```

## Performance Optimization

### Database Persistence
TimescaleDB data persists in named volume `timescale_data`. To reset:
```bash
docker-compose down -v  # WARNING: Deletes all data!
docker-compose up -d
```

### Log Management
Logs are automatically rotated:
- Bot logs: 10MB max size, 5 files
- Redis logs: 5MB max size, 3 files
- TimescaleDB logs: 10MB max size, 3 files

### Resource Limits
Default configuration allocates:
- Redis: 512MB memory with LRU eviction
- TimescaleDB: 512MB shared buffers, 1.5GB cache
- Bot: No explicit limit (adjust in docker-compose.yml if needed)

## Quick Reference

| Command | Purpose |
|---------|---------|
| `docker-compose up -d` | Start all services in background |
| `docker-compose down` | Stop all services |
| `docker-compose build --no-cache bot` | Rebuild bot image |
| `docker-compose logs -f bot` | Follow bot logs |
| `docker-compose restart bot` | Restart bot only |
| `docker-compose ps` | Check service status |
| `docker-compose exec bot bash` | Enter bot container shell |
| `docker-compose exec timescaledb psql -U trading_bot -d trading_bot` | Access database |

## Verification Checklist After Deployment

- [ ] All services show "healthy" status: `docker-compose ps`
- [ ] No background worker warnings in TimescaleDB logs
- [ ] Dashboard accessible at http://localhost:8080
- [ ] New features from PR #30 visible (Settings tab, advanced features)
- [ ] Health check returns 200: `curl http://localhost:8080/health`
- [ ] WebSocket connection working (check browser console)
- [ ] Trading bot initializing correctly (check logs)

## Next Steps

1. **Stop your current containers:**
   ```bash
   docker-compose down
   ```

2. **Pull latest changes (if not done):**
   ```bash
   git pull origin main
   ```

3. **Rebuild and restart:**
   ```bash
   docker-compose build --no-cache
   docker-compose up -d
   ```

4. **Monitor startup:**
   ```bash
   docker-compose logs -f
   ```

Your TimescaleDB warnings should be gone, and all features from PR #30 should now be visible!
