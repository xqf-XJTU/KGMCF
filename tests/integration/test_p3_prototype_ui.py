from pathlib import Path
import sys
import pytest
pytest.importorskip('flask')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'apps' / 'prototype'))
from app import app


def test_p3_prototype_page_and_new_endpoints():
    client = app.test_client()
    page = client.get('/')
    assert page.status_code == 200
    assert b'Aero-Instruct-5K Dataset Management' in page.data
    assert b'Aero-Instruct-5K' in page.data
    state = client.get('/api/app_state').get_json()
    assert state['health']['sample_count'] == 5000
    rows = client.get('/api/dataset/samples?feature_id=F05&page_size=3').get_json()
    assert rows['records']
    run = client.post('/api/run_workflow', json={'feature_id':'F05','method':'KGMCF','intent':'Generate process plan'})
    assert run.status_code == 200
    data = run.get_json()
    assert data['process_card']['route_steps']
    assert 'route_steps' in data['process_card']
