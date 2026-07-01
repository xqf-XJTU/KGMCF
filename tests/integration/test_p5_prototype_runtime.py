from pathlib import Path
import sys
import pytest
pytest.importorskip("flask")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "apps" / "prototype"))
from app import app


def test_p5_project_graph_is_interactive_scope_and_has_images():
    client = app.test_client()
    data = client.get('/api/graph_project').get_json()
    assert data['scope'] == 'project'
    assert len(data['nodes']) > 100
    assert any(n.get('group') == 'Image' and n.get('image') for n in data['nodes'])


def test_p5_prompt_and_context_fusion_endpoints():
    client = app.test_client()
    templates = client.get('/api/prompt_templates').get_json()
    assert 'system' in templates['templates']
    pending = client.post('/api/context_fusion', json={'intent':'test plan'}).get_json()
    assert pending['feature_id'] == ''
    fused = client.post('/api/context_fusion', json={'feature_id':'F05','intent':'test plan'}).get_json()
    assert fused['feature_id'] == 'F05'
    assert 'F05' in fused['fused_prompt']


def test_p5_model_runtime_and_training_trace_is_not_synthetic():
    client = app.test_client()
    runtime = client.get('/api/model_runtime').get_json()
    assert runtime['model_name'] == 'Qwen2.5-VL-72B-Instruct'
    assert 'models/Qwen2.5-VL-72B-Instruct' in runtime['model_root']
    trace = client.get('/api/engine_training_trace').get_json()
    assert trace['status'] in {'no_training_log', 'loaded', 'empty_training_log'}
    if trace['status'] == 'no_training_log':
        assert trace['records'] == []
