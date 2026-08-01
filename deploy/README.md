# Free Deployment

The app needs permanent storage for Square tokens, SQLite data, and PyTorch
models. The included setup uses an Oracle Cloud Always Free ARM VM with Docker.

## Create the server

Use Ubuntu 24.04 ARM, shape `VM.Standard.A1.Flex`, 2 OCPUs, 12 GB RAM, and a
50 GB boot disk. Give it a public IP and allow TCP ports 22, 80, and 443.

## Install

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

Fill in every value in `deploy/.env.production`. Create `APP_SECRET` with:

```bash
openssl rand -base64 48
```

Set `APP_DOMAIN` to `forecast.YOUR_SERVER_IP.sslip.io`, then start the app:

```bash
cd /opt/restaurant-forecast/app/deploy
sudo docker compose up -d --build
sudo docker compose logs -f app caddy
```

Set the Square production callback to this exact URL:

```text
https://forecast.YOUR_SERVER_IP.sslip.io/api/square/callback
```

It must match `SQUARE_REDIRECT_URL`. Caddy handles HTTPS, and the Docker volume
keeps restaurant data during updates. Back up that volume before deleting the
server.

```bash
sudo docker compose ps
sudo docker compose logs --tail=200 app
sudo docker compose up -d --build
```
