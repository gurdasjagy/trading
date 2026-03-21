#!/bin/bash
set -euo pipefail

echo "=== VPS Kernel Tuning for Low-Latency Trading ==="

# TCP optimizations
sysctl -w net.ipv4.tcp_nodelay=1
sysctl -w net.ipv4.tcp_low_latency=1
sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"

# Disable swap (latency killer)
swapoff -a || true

# Set CPU governor to performance
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$cpu" 2>/dev/null || true
done

# Create shared memory mount
mkdir -p /dev/shm
mount -t tmpfs -o size=64m tmpfs /dev/shm 2>/dev/null || true

echo "=== Starting Trading Stack ==="
cd "$(dirname "$0")/.."
docker compose up -d --build

echo "=== Deployment Complete ==="
docker compose ps
