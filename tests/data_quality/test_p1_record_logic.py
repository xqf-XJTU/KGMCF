from pathlib import Path
import csv
ROOT = Path(__file__).resolve().parents[2]


def test_generated_process_plan_records_are_linkable():
    plans = list(csv.DictReader((ROOT / "data/supplementary/planning_records/generated_process_plans_raw_400.csv").open("r", encoding="utf-8-sig")))
    metrics = list(csv.DictReader((ROOT / "data/supplementary/planning_records/process_generation_metrics_raw_400.csv").open("r", encoding="utf-8-sig")))
    plan_ids = {r["plan_id"] for r in plans}
    metric_ids = {r["plan_id"] for r in metrics}
    assert len(plans) == 400
    assert plan_ids == metric_ids


def test_semantic_traceback_records_are_linkable():
    plans = list(csv.DictReader((ROOT / "data/supplementary/planning_records/generated_process_plans_raw_400.csv").open("r", encoding="utf-8-sig")))
    trace = list(csv.DictReader((ROOT / "data/supplementary/planning_records/semantic_traceback_records.csv").open("r", encoding="utf-8-sig")))
    assert {r["plan_id"] for r in trace}.issubset({r["plan_id"] for r in plans})


def test_dataset_split_image_paths_exist():
    rows = list(csv.DictReader((ROOT / "data/processed/dataset_split_index.csv").open("r", encoding="utf-8-sig")))
    for r in rows[:200]:
        assert (ROOT / r["image_file"]).exists()
