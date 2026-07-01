from pathlib import Path
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kgmcf.kg.loader import AeroMPKG
from kgmcf.metrics.process_metrics import summarize_generation_metrics
from kgmcf.planning.planner import build_process_plan
from kgmcf.verification.symbolic import verify_plan


def test_kg_load_and_validate():
    kg = AeroMPKG.load_dir(ROOT / "data/raw/aero_mpkg/json")
    report = kg.validate()
    assert report["feature_count"] == 20
    assert report["node_count"] > 800
    assert report["edge_count"] > 800
    assert report["missing_source_count"] == 0
    assert report["missing_target_count"] == 0


def test_process_generation_metrics_are_available():
    rows = list(csv.DictReader((ROOT / "data/supplementary/planning_records/process_generation_metrics_raw_400.csv").open("r", encoding="utf-8-sig")))
    metrics = summarize_generation_metrics(rows)
    assert metrics["methods"]["KGMCF"]["MSRV"] >= metrics["methods"]["Vanilla MLLM"]["MSRV"]
    assert set(metrics["methods"]["KGMCF"]).issuperset({"MPSA", "MSRV", "TSA"})


def test_planner_and_verifier():
    kg = AeroMPKG.load_dir(ROOT / "data/raw/aero_mpkg/json")
    plan = build_process_plan(kg.feature_summary("F12"), "KGMCF")
    report = verify_plan(plan)
    assert "process_route" in plan
    assert "checked_items" in report
