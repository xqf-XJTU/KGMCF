from pathlib import Path
import sys
import pytest
pytest.importorskip('flask')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'apps' / 'prototype'))
from app import app


def test_p4_layout_matches_prototype_pages_and_uses_local_data():
    client = app.test_client()
    page = client.get('/')
    assert page.status_code == 200
    html = page.data.decode('utf-8')
    for text in [
        'KGCF System',
        'Aero-Instruct-5K Dataset Management',
        'Aero-MMKG Visualization & Visual anchoring',
        'MLLM Cognitive Reasoning Cockpit',
        'Aerospace Process Report Center',
        'Aero-LMM Engine (LoRA)',
        'Qwen2.5-VL-72B Runtime',
    ]:
        assert text in html
    assert 'Aero-Instruct-5K' in html
    graph = client.get('/api/graph/F05').get_json()
    assert len(graph['nodes']) > 20
    assert len(graph['edges']) > 20
    rows = client.get('/api/dataset/samples?feature_id=F05&page_size=5').get_json()['records']
    assert rows and all(r['feature_id'] == 'F05' for r in rows)
    card = client.post('/api/process_card', json={'feature_id':'F05','method':'KGMCF'}).get_json()
    assert card['route_steps']
    assert card['route_steps']
    engine = client.get('/api/engine_status').get_json()
    assert engine['status'] == 'ready'
