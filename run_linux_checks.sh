#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
python scripts/run_pipeline.py
python scripts/run_all_experiments.py
python -m pytest -q
python -m compileall -q src scripts apps experiments tests
if command -v node >/dev/null 2>&1; then
  node --check apps/prototype/static/js/prototype.js
else
  echo "[KGMCF] node not found; skipped JavaScript syntax check."
fi
python - <<'PY'
from apps.prototype.app import app
c = app.test_client()
c.post('/api/process_card', json={'feature_id':'F05','method':'KGMCF'})
r = c.get('/api/process_card_pdf?feature_id=F05&method=KGMCF')
assert r.status_code == 200, r.status_code
assert r.data[:4] == b'%PDF', r.data[:16]
print('[KGMCF] PDF endpoint check: PASS')
PY
