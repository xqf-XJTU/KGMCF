#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
export FLASK_ENV="production"
export KGMCF_HOST="${KGMCF_HOST:-127.0.0.1}"
export KGMCF_PORT="${KGMCF_PORT:-5000}"
python apps/prototype/app.py
