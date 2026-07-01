from pathlib import Path
import csv, sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kgmcf.data.audit import audit_project
from kgmcf.data.dataset_index import validate_dataset_split_index
from kgmcf.data.instructions import validate_instruction_corpus


def test_dataset_audit_counts():
    audit = audit_project(ROOT)
    assert audit["status"] == "PASS"
    assert audit["totals"]["total_png_images"] == 1400
    assert audit["totals"]["total_xt_models"] == 20
    assert audit["totals"]["total_kg_json"] == 20


def test_instruction_corpus_schema():
    report = validate_instruction_corpus(ROOT / "data/processed/aero_instruct_5k.jsonl")
    assert report["status"] == "PASS"


def test_dataset_split_index_semantics():
    report = validate_dataset_split_index(ROOT / "data/processed/dataset_split_index.csv")
    assert report["status"] == "PASS"
    assert report["validation_case_count"] == 100
    assert report["validation_representative_sample_count"] == 100
