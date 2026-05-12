#!/usr/bin/env bash
# Idempotent installer for the SED edge device.
# Tested on Ubuntu Server 24.04 LTS. Adapt as needed for Debian 12.
#
# Usage (as root):
#   PROJECT_ID=the_hawthorn ./deploy/install.sh

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID to the venue id, e.g. 'the_hawthorn'}"
SED_HOME="${SED_HOME:-/opt/sed}"
SED_USER="${SED_USER:-sed}"

echo "==> 1) system packages"
apt-get update
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    libsndfile1 \
    libasound2 alsa-utils \
    libusb-1.0-0 \
    git rsync

echo "==> 2) sed user"
if ! id -u "$SED_USER" >/dev/null 2>&1; then
    useradd --system --create-home --home-dir "$SED_HOME" --shell /usr/sbin/nologin "$SED_USER"
    usermod -aG audio "$SED_USER"
fi

echo "==> 3) repo at $SED_HOME"
mkdir -p "$SED_HOME"
chown -R "$SED_USER":"$SED_USER" "$SED_HOME"

# Sync the current repo into $SED_HOME (assumes we're running from inside the repo)
THIS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
sudo -u "$SED_USER" rsync -a --delete \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='projects/*/recordings' --exclude='projects/*/dataset' \
    "$THIS_DIR/" "$SED_HOME/"

echo "==> 4) python venv (inference deps only — no torch on edge)"
sudo -u "$SED_USER" python3 -m venv "$SED_HOME/.venv"
sudo -u "$SED_USER" "$SED_HOME/.venv/bin/pip" install --upgrade pip wheel
# Inference-only: numpy, scipy, sounddevice, soundfile, librosa, openvino,
# onnxruntime, stupidArtnet, pyzmq, rich, typer, pydantic, jinja2.
# Do NOT install torch / lightning on the edge device.
sudo -u "$SED_USER" "$SED_HOME/.venv/bin/pip" install \
    numpy scipy pydantic pydantic-settings \
    "typer[all]" rich \
    sounddevice soundfile librosa \
    openvino onnxruntime \
    stupidArtnet pyzmq jinja2 \
    matplotlib

# Install the local package
sudo -u "$SED_USER" "$SED_HOME/.venv/bin/pip" install --no-deps -e "$SED_HOME"

echo "==> 5) /etc/sed/env"
mkdir -p /etc/sed
cat > /etc/sed/env <<EOF
PROJECT_ID=${PROJECT_ID}
PYTHONUNBUFFERED=1
EOF
chmod 0644 /etc/sed/env

echo "==> 6) systemd units"
cp "$SED_HOME/deploy/sed.service" /etc/systemd/system/sed.service
# instantiate path unit for this project
sed "s|%i|${PROJECT_ID}|g" "$SED_HOME/deploy/sed-model.path" \
    > /etc/systemd/system/sed-model.path
cp "$SED_HOME/deploy/sed-model-reload.service" /etc/systemd/system/sed-model-reload.service
systemctl daemon-reload
systemctl enable --now sed.service
systemctl enable --now sed-model.path

echo "==> 7) logging"
mkdir -p /var/log/sed
chown "$SED_USER":"$SED_USER" /var/log/sed

echo "==> done"
echo "  status:    systemctl status sed"
echo "  follow:    journalctl -u sed -f"
echo "  restart:   systemctl restart sed"
echo "  swap mod:  rsync new model into $SED_HOME/projects/${PROJECT_ID}/models/"
