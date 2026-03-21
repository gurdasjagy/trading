#!/usr/bin/env bash
# =============================================================================
# setup_droplet.sh — DigitalOcean Droplet Setup for Crypto Trading Bot
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/JashanxJagy0/trading-bot/main/crypto_trading_bot/scripts/setup_droplet.sh | bash
#   # — or — clone the repo first then run:
#   bash crypto_trading_bot/scripts/setup_droplet.sh
#
# Tested on: Ubuntu 22.04 LTS / 24.04 LTS (DigitalOcean)
# Recommended droplet: Basic, 2 vCPU, 2 GB RAM, 50 GB SSD ($18/mo)
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
CYAN="\033[36m"
RESET="\033[0m"

info()  { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }
step()  { echo -e "\n${BOLD}${CYAN}▶ $*${RESET}"; }

# ── Root check ────────────────────────────────────────────────────────────────
if [[ "$EUID" -ne 0 ]]; then
    error "Please run this script as root: sudo bash $0"
fi

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/JashanxJagy0/trading-bot.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt/trading-bot}"
BOT_PORT="${BOT_PORT:-8080}"

# =============================================================================
# Step 1 — System update
# =============================================================================
step "Updating system packages"
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    curl \
    git \
    ufw \
    ca-certificates \
    gnupg \
    lsb-release
info "System packages updated ✓"

# =============================================================================
# Step 2 — Install Docker
# =============================================================================
step "Installing Docker Engine"
if command -v docker &>/dev/null; then
    info "Docker is already installed ($(docker --version)) — skipping."
else
    # Add Docker's official GPG key and repository
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list

    apt-get update -qq
    apt-get install -y -qq \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin

    systemctl enable --now docker
    info "Docker installed ✓"
fi

# =============================================================================
# Step 3 — Install Docker Compose (standalone v2 CLI plugin already included;
#           also provide the classic `docker-compose` alias for convenience)
# =============================================================================
step "Verifying Docker Compose"
if docker compose version &>/dev/null; then
    info "Docker Compose plugin available: $(docker compose version)"
else
    error "Docker Compose plugin not found. Please install docker-compose-plugin."
fi

# =============================================================================
# Step 4 — Clone the repository
# =============================================================================
step "Cloning trading-bot repository to ${INSTALL_DIR}"
if [[ -d "${INSTALL_DIR}/.git" ]]; then
    info "Repository already cloned — pulling latest changes."
    git -C "${INSTALL_DIR}" pull --ff-only
else
    git clone "${REPO_URL}" "${INSTALL_DIR}"
    info "Repository cloned ✓"
fi

# =============================================================================
# Step 5 — Configure environment variables
# =============================================================================
step "Configuring environment"
BOT_DIR="${INSTALL_DIR}/crypto_trading_bot"

if [[ ! -f "${BOT_DIR}/.env" ]]; then
    cp "${BOT_DIR}/.env.example" "${BOT_DIR}/.env"
    info ".env created from .env.example ✓"
    warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    warn "ACTION REQUIRED: Edit ${BOT_DIR}/.env with your"
    warn "exchange API keys before starting the bot."
    warn "  nano ${BOT_DIR}/.env"
    warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    info ".env already exists — skipping copy."
fi

# =============================================================================
# Step 6 — Configure firewall (UFW)
# =============================================================================
step "Configuring firewall (UFW)"
# Always allow SSH so we don't lock ourselves out
ufw allow OpenSSH
# Allow the bot dashboard port
ufw allow "${BOT_PORT}/tcp" comment "Crypto Trading Bot Dashboard"
# Enable UFW (non-interactively)
ufw --force enable
info "Firewall configured: SSH + port ${BOT_PORT} allowed ✓"
ufw status verbose

# =============================================================================
# Step 7 — Start the bot with Docker Compose
# =============================================================================
step "Starting the trading bot"
cd "${BOT_DIR}"

if [[ ! -f ".env" ]]; then
    error ".env file not found. Please create it before starting the bot."
fi

docker compose pull
docker compose up -d --build
info "Trading bot started ✓"

# =============================================================================
# Post-install summary
# =============================================================================
DROPLET_IP=$(curl -s --max-time 5 http://169.254.169.254/metadata/v1/interfaces/public/0/ipv4/address 2>/dev/null || echo "")
if [[ -z "$DROPLET_IP" ]]; then
    warn "Could not auto-detect droplet IP from metadata service."
    DROPLET_IP="YOUR_DROPLET_IP"
fi

echo ""
echo -e "${BOLD}${GREEN}✅  Setup complete!${RESET}"
echo ""
echo -e "  📊 Dashboard:   ${BOLD}http://${DROPLET_IP}:${BOT_PORT}${RESET}"
echo -e "  📱 Mobile:      Open the URL above in your phone browser"
echo -e "  📋 View logs:   ${BOLD}docker compose logs -f bot${RESET}"
echo -e "  🛑 Stop bot:    ${BOLD}docker compose down${RESET}"
echo -e "  🔄 Restart bot: ${BOLD}docker compose restart bot${RESET}"
echo ""

# =============================================================================
# HTTPS / Domain setup (optional — requires a domain name)
# =============================================================================
# To serve the dashboard over HTTPS with a custom domain, install Caddy or
# nginx as a reverse proxy. The steps below use Caddy (simplest option):
#
#   1. Point your domain's A record to this droplet's IP address.
#
#   2. Install Caddy:
#        apt-get install -y debian-keyring debian-archive-keyring apt-transport-https
#        curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
#            | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
#        curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
#            | tee /etc/apt/sources.list.d/caddy-stable.list
#        apt-get update && apt-get install -y caddy
#
#   3. Create /etc/caddy/Caddyfile:
#        your-domain.com {
#            reverse_proxy localhost:8080
#        }
#
#   4. Reload Caddy:
#        systemctl reload caddy
#
#   Caddy automatically provisions and renews a Let's Encrypt TLS certificate.
#   Your dashboard will then be accessible at https://your-domain.com
#
# ── nginx alternative ─────────────────────────────────────────────────────────
#
#   1. Install nginx + certbot:
#        apt-get install -y nginx certbot python3-certbot-nginx
#
#   2. Create /etc/nginx/sites-available/trading-bot:
#        server {
#            listen 80;
#            server_name your-domain.com;
#            location / {
#                proxy_pass http://localhost:8080;
#                proxy_set_header Host $host;
#                proxy_set_header X-Real-IP $remote_addr;
#                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#                proxy_set_header X-Forwarded-Proto $scheme;
#                # WebSocket support
#                proxy_http_version 1.1;
#                proxy_set_header Upgrade $http_upgrade;
#                proxy_set_header Connection "upgrade";
#            }
#        }
#
#   3. Enable the site and get a certificate:
#        ln -s /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/
#        certbot --nginx -d your-domain.com
#        systemctl reload nginx
# =============================================================================
