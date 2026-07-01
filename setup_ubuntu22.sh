#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

echo "[KGMCF] Ubuntu 22.04 LTS setup"
echo "[KGMCF] Project directory: ${PROJECT_DIR}"

if command -v apt-get >/dev/null 2>&1; then
  echo "[KGMCF] Installing Ubuntu system dependencies..."
  sudo apt-get update
  sudo apt-get install -y \
    python3 python3-pip python3-venv python3-dev build-essential \
    libjpeg-dev zlib1g-dev libfreetype6-dev libopenjp2-7-dev \
    fonts-wqy-microhei fonts-wqy-zenhei fonts-noto-cjk \
    curl git nodejs npm
else
  echo "[KGMCF] apt-get not found; skipping system package installation."
fi

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
pip install -r "${PROJECT_DIR}/requirements.txt"
pip install -e "${PROJECT_DIR}"

echo "[KGMCF] Setup completed. Activate with:"
echo "source ${VENV_DIR}/bin/activate"
