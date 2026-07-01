from pathlib import Path
import sys
import pytest
pytest.importorskip('flask')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'apps' / 'prototype'))
from app import app


def test_prototype_api_smoke():
    client = app.test_client()
    r = client.get('/api/features')
    assert r.status_code == 200
    assert len(r.get_json()) == 20
    r = client.get('/api/retrieve/F12')
    assert r.status_code == 200
    assert r.get_json()['feature_id'] == 'F12'
    r = client.post('/api/plan', json={'feature_id':'F12','method':'KGMCF'})
    assert r.status_code == 200
    assert 'plan' in r.get_json()
    r = client.post('/api/verify_loop', json={'feature_id':'F12','method':'Vanilla MLLM','max_retries':2})
    assert r.status_code == 200
    assert r.get_json()['retry_count'] <= 2
