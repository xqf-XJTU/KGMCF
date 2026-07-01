"""Instruction-corpus checks."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json

REQUIRED_FIELDS = {"sample_id", "feature_id", "image_file", "instruction", "input_context", "reasoning_text", "process_plan_reference", "record_source"}


def validate_instruction_corpus(path: Path) -> Dict[str, Any]:
    count = 0
    missing = 0
    feature_ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            rec = json.loads(line)
            feature_ids.add(rec.get("feature_id"))
            if not REQUIRED_FIELDS.issubset(rec):
                missing += 1
    return {"records": count, "feature_count": len(feature_ids), "missing_required_field_rows": missing, "status": "PASS" if count == 5000 and len(feature_ids) == 20 and missing == 0 else "CHECK"}
