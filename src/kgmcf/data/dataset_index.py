"""Dataset split-index validation and controlled rebuild utilities."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter, defaultdict
import csv

REQUIRED_FIELDS = [
    "sample_id", "case_id", "feature_id", "source_geometry_id", "source_instance_id",
    "view_id", "augmentation_id", "instruction_id", "split_id", "image_file",
    "cad_model_file", "kg_file", "prompt_template", "retrieval_exclusion_key",
    "is_from_heldout_validation_case", "is_validation_representative_sample", "validation_record_key",
]


def read_dataset_split_index(path: Path) -> List[Dict[str, str]]:
    return list(csv.DictReader(path.open("r", encoding="utf-8-sig")))


def validate_dataset_split_index(path: Path) -> Dict[str, Any]:
    rows = read_dataset_split_index(path)
    fields = list(rows[0].keys()) if rows else []
    by_source = defaultdict(set)
    for r in rows:
        by_source[r["source_instance_id"]].add(r["split_id"])
    split_counts = Counter(r["split_id"] for r in rows)
    rep_count = sum(1 for r in rows if r.get("is_validation_representative_sample") == "1")
    heldout_keys = {r.get("validation_record_key", "") for r in rows if r.get("validation_record_key")}
    leakage_sources = [s for s, splits in by_source.items() if len(splits) > 1]
    return {
        "records": len(rows),
        "fields": len(fields),
        "required_fields_present": all(f in fields for f in REQUIRED_FIELDS),
        "split_counts": dict(split_counts),
        "source_instance_count": len(by_source),
        "split_leakage_source_count": len(leakage_sources),
        "validation_case_count": len(heldout_keys),
        "validation_representative_sample_count": rep_count,
        "status": "PASS" if len(rows) == 5000 and split_counts == {"train": 3500, "validation": 750, "test": 750} and len(leakage_sources) == 0 and len(heldout_keys) == 100 and rep_count == 100 else "CHECK",
    }
