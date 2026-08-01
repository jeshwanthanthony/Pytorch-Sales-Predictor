# Free production deployment

The app needs persistent disk and enough RAM to train PyTorch. Oracle Cloud's
Always Free Ampere A1 VM is the supported free target. Use one VM with 2 OCPUs,
12 GB RAM, and a 50 GB boot volume.

## 1. Create the VM

- Image: Ubuntu 24.04 aarch64
- Shape: `VM.Standard.A1.Flex`, 2 OCPUs, 12 GB RAM
- Assign a public IPv4 address.
- In the subnet security list, allow inbound TCP 22, 80, and 443.
- Keep the VM and boot volume marked Always Free eligible.

## 2. Install and configure

```bash
ssh ubuntu@YOUR_SERVER_IP
sudo mkdir -p /opt/restaurant-forecast
sudo chown ubuntu:ubuntu /opt/restaurant-forecast
cd /opt/restaurant-forecast
git clone https://github.com/jeshwanthanthony/restaurant-forecast-ai.git app
cd app
sudo bash deploy/bootstrap-ubuntu.sh
cp deploy/.env.production.example deploy/.env.production
```

Replace every placeholder in `deploy/.env.production`. Generate `APP_SECRET`
on the VM with:

```bash
openssl rand -base64 48
```

Use `forecast.YOUR_SERVER_IP.sslip.io` for `APP_DOMAIN`. Caddy obtains and
renews HTTPS automatically. Start the service:

```bash
cd /opt/restaurant-forecast/app/deploy
sudo docker compose up -d --build
sudo docker compose logs -f app caddy
```

## 3. Square callback

In the Square Developer Dashboard, set the production OAuth Redirect URL to:

```text
https://forecast.YOUR_SERVER_IP.sslip.io/api/square/callback
```

It must exactly match `SQUARE_REDIRECT_URL` in `deploy/.env.production`.

## Operations

```bash
sudo docker compose ps
sudo docker compose logs --tail=200 app
sudo docker compose up -d --build       # deploy an updated checkout
sudo docker compose restart app
```

Seller tokens, raw Square data, SQLite databases, model files, and terminal
history live in the `forecast-data` Docker volume and survive app redeploys.
Back up that volume before deleting the VM.
