#!/bin/bash
# Quick deployment script for Trading Bot
# This script rebuilds and redeploys the Docker containers with the latest fixes

set -e  # Exit on any error

echo "=========================================="
echo "Trading Bot - Quick Deployment Script"
echo "=========================================="
echo ""

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed or not in PATH"
    exit 1
fi

echo "✅ Docker is available"
echo ""

# Navigate to the correct directory
cd "$(dirname "$0")"

# Step 1: Stop existing containers
echo "🛑 Step 1: Stopping existing containers..."
docker compose down
echo "✅ Containers stopped"
echo ""

# Step 2: Rebuild the bot image
echo "🔨 Step 2: Rebuilding bot image (this may take a few minutes)..."
docker compose build --no-cache bot
echo "✅ Bot image rebuilt"
echo ""

# Step 3: Start all services
echo "🚀 Step 3: Starting all services..."
docker compose up -d
echo "✅ Services started"
echo ""

# Step 4: Wait for services to be healthy
echo "⏳ Step 4: Waiting for services to become healthy..."
sleep 5

# Check service status
echo ""
echo "📊 Service Status:"
docker compose ps
echo ""

# Step 5: Check for TimescaleDB warnings
echo "🔍 Step 5: Checking TimescaleDB logs for warnings..."
sleep 3
if docker compose logs timescaledb 2>&1 | grep -i "failed to launch job" > /dev/null; then
    echo "⚠️  Warning: Still seeing background worker errors. You may need to restart:"
    echo "   docker compose restart timescaledb"
else
    echo "✅ No background worker errors detected!"
fi
echo ""

# Step 6: Verify dashboard
echo "🌐 Step 6: Verifying dashboard availability..."
sleep 2
if curl -s -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Dashboard is accessible at http://localhost:8080"
else
    echo "⚠️  Dashboard not yet ready. Check logs with: docker compose logs -f bot"
fi
echo ""

# Show recent logs
echo "📋 Recent bot logs:"
echo "----------------------------------------"
docker compose logs --tail 20 bot
echo "----------------------------------------"
echo ""

echo "=========================================="
echo "🎉 Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  • View logs: docker compose logs -f bot"
echo "  • Access dashboard: http://localhost:8080"
echo "  • Check status: docker compose ps"
echo "  • View all logs: docker compose logs -f"
echo ""
echo "If you see any issues, check DOCKER_DEPLOYMENT.md for troubleshooting."
