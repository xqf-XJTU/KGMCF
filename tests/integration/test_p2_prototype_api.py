from pathlib import Path
import sys
import pytest
pytest.importorskip('flask')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'apps' / 'prototype'))
from app import app


def test_p2_api_workflow_smoke():
    client = app.test_client()
    assert client.get('/api/health').status_code == 200
    spec = client.get('/api/workflow').get_json()
    assert len(spec['stages']) == 6
    detail = client.get('/api/features/F05')
    assert detail.status_code == 200
    img = detail.get_json()['image_examples'][0]
    assert client.get('/api/image?path=' + img).status_code == 200
    run = client.post('/api/run_workflow', json={'feature_id':'F12','method':'KGMCF','max_retries':2})
    assert run.status_code == 200
    data = run.get_json()
    assert data['feature']['feature_id'] == 'F12'
    assert 'verification_loop' in data
