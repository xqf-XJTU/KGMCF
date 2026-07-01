from pathlib import Path
import csv
ROOT = Path(__file__).resolve().parents[2]


def test_planning_records_link_to_split_index():
    split_rows = list(csv.DictReader((ROOT / "data/processed/dataset_split_index.csv").open("r", encoding="utf-8-sig")))
    split_cases = {r.get("validation_record_key", "") for r in split_rows if r.get("validation_record_key")}
    plan_rows = list(csv.DictReader((ROOT / "data/supplementary/planning_records/generated_process_plans_raw_400.csv").open("r", encoding="utf-8-sig")))
    plan_cases = {r.get("case_id", "") for r in plan_rows if r.get("case_id")}
    assert len(split_cases) == 100
    assert len(plan_cases) == 100
    assert plan_cases.issubset(split_cases)


def test_planning_records_folder_contains_only_planning_files():
    folder = ROOT / "data/supplementary/planning_records"
    names = {p.name for p in folder.glob("*.csv")}
    assert names == {"generated_process_plans_raw_400.csv", "process_generation_metrics_raw_400.csv", "semantic_traceback_records.csv"}
