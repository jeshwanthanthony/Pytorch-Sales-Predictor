#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "run with sudo: sudo bash deploy/bootstrap-ubuntu.sh"
  exit 1
fi

apt-get update
apt-get install -y ca-certificates curl docker.io docker-compose-v2
systemctl enable --now docker

mkdir -p /opt/restaurant-forecast
echo "Docker is ready. Put the repository in /opt/restaurant-forecast/app."
