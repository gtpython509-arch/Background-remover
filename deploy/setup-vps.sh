#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot provisioning for a fresh Hostinger VPS (Ubuntu 22.04/24.04).
# Run as root:   bash deploy/setup-vps.sh api.yourdomain.com
#
# Review it before running. It installs nginx + Docker, creates a swap file,
# opens the firewall and requests a certificate.
# ---------------------------------------------------------------------------
set -euo pipefail

DOMAIN="${1:-}"
if [[ -z "$DOMAIN" ]]; then
  echo "usage: bash deploy/setup-vps.sh <domain>" >&2
  exit 1
fi

if [[ $EUID -ne 0 ]]; then
  echo "run as root (sudo -i)" >&2
  exit 1
fi

echo "==> Updating packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg ufw nginx certbot python3-certbot-nginx

# Torch's peak allocation during model load can briefly exceed free RAM on a
# 4 GB box. Swap turns an OOM kill into a slow start.
if [[ ! -f /swapfile ]]; then
  echo "==> Creating a 2G swap file"
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  sysctl -w vm.swappiness=10
  grep -q '^vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' >> /etc/sysctl.conf
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "==> Installing Docker"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
  systemctl enable --now docker
fi

echo "==> Configuring the firewall"
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw --force enable

echo "==> Installing the nginx site"
SITE=/etc/nginx/sites-available/rmbg
sed "s/api\.yourdomain\.com/${DOMAIN}/g" "$(dirname "$0")/nginx.conf" > "$SITE"

# certbot cannot start nginx if the config references a certificate that does
# not exist yet, so serve plain HTTP for the challenge first.
TMP=/etc/nginx/sites-available/rmbg-bootstrap
cat > "$TMP" <<NGINX
server {
    listen 80;
    server_name ${DOMAIN};
    location /.well-known/acme-challenge/ { root /var/www/html; }
    location / { return 200 'bootstrap'; add_header Content-Type text/plain; }
}
NGINX
rm -f /etc/nginx/sites-enabled/default /etc/nginx/sites-enabled/rmbg
ln -sf "$TMP" /etc/nginx/sites-enabled/rmbg-bootstrap
nginx -t && systemctl reload nginx

echo "==> Requesting a certificate for ${DOMAIN}"
certbot certonly --nginx -d "$DOMAIN" --non-interactive --agree-tos \
  --register-unsafely-without-email || {
    echo "certbot failed - check that ${DOMAIN}'s A record points at this server" >&2
    exit 1
  }

rm -f /etc/nginx/sites-enabled/rmbg-bootstrap
ln -sf "$SITE" /etc/nginx/sites-enabled/rmbg
nginx -t && systemctl reload nginx

cat <<DONE

==> Host is ready.

Next:
  1. Put the project in /srv/rmbg-service
  2. cp .env.example .env  and set RMBG_API_KEYS + RMBG_CORS_ORIGINS
  3. docker compose -f deploy/docker-compose.yml up -d --build
  4. curl https://${DOMAIN}/healthz

The first build downloads torch and the weights; expect 10-20 minutes.
DONE
