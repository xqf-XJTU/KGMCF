from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'apps' / 'prototype'))

from services import PrototypeService


def test_p2_service_workflow_contract():
    svc = PrototypeService(ROOT)
    health = svc.health()
    assert health['status'] == 'ready'
    assert health['feature_count'] == 20
    spec = svc.workflow_spec()
    assert len(spec['stages']) == 6
    assert any(m['name'] == 'KGMCF' for m in spec['methods'])


def test_p2_service_feature_images_and_workflow():
    svc = PrototypeService(ROOT)
    detail = svc.feature_detail('F05')
    assert detail['feature_id'] == 'F05'
    assert detail['image_examples']
    assert (ROOT / detail['image_examples'][0]).exists()
    result = svc.run_workflow('F12', method='KGMCF', max_retries=2)
    assert result['feature']['feature_id'] == 'F12'
    assert result['retrieval']['node_count'] > 0
    assert result['plan']['method'] == 'KGMCF'
    assert result['verification_loop']['retry_count'] <= 2
    assert 'process_route' in result['plan']


def test_p2_planning_records_and_traceback_linkage():
    svc = PrototypeService(ROOT)
    cases = svc.planning_record_cases(feature_id='F12', limit=5)
    assert cases
    evidence = svc.planning_records(cases[0]['case_id'])
    assert 'generated_process_plans_raw_400.csv' in evidence
    assert 'process_generation_metrics_raw_400.csv' in evidence
    assert 'semantic_traceback_records.csv' in evidence
    assert set(evidence).issubset({'generated_process_plans_raw_400.csv', 'process_generation_metrics_raw_400.csv', 'semantic_traceback_records.csv'})


def test_p3_dataset_browser_and_process_card_are_data_driven():
    svc = PrototypeService(ROOT)
    page = svc.dataset_samples(feature_id='F05', search='radius', page=1, page_size=5)
    assert page['total'] > 0
    assert len(page['records']) <= 5
    card = svc.process_card('F05', 'KGMCF')
    assert card['feature']['feature_id'] == 'F05'
    assert card['route_steps']
    assert card['route_steps']
