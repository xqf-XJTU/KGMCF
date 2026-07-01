"""Dataset and file-integrity audit utilities."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict
import hashlib
import json


def audit_project(root: Path) -> Dict[str, Any]:
    image_root = root / "data" / "raw" / "cv20" / "images"
    model_root = root / "data" / "raw" / "cv20" / "cad_xt"
    kg_dir = root / "data" / "raw" / "aero_mpkg" / "json"
    aero5k = root / "data" / "processed" / "aero_instruct_5k.jsonl"
    split_index = root / "data" / "processed" / "dataset_split_index.csv"
    features: List[Dict[str, Any]] = []
    for i in range(20):
        fid = f"F{i:02d}"
        img_dir = image_root / str(i)
        images = sorted(img_dir.rglob("*.png")) if img_dir.exists() else []
        model_file = model_root / f"{i}.x_t"
        kg_file = kg_dir / f"{fid}.json"
        node_count = edge_count = None
        if kg_file.exists():
            data = json.loads(kg_file.read_text(encoding="utf-8"))
            node_count = len(data.get("nodes", []))
            edge_count = len(data.get("relationships", data.get("edges", [])))
        features.append({
            "feature_id": fid,
            "image_count": len(images),
            "has_cad_xt": model_file.exists(),
            "has_kg_json": kg_file.exists(),
            "kg_nodes": node_count,
            "kg_edges": edge_count,
            "image_examples": [str(p.relative_to(root)).replace("\\", "/") for p in images[:3]],
        })
    instruction_count = sum(1 for _ in aero5k.open("r", encoding="utf-8")) if aero5k.exists() else 0
    split_count = 0
    if split_index.exists():
        import csv
        split_count = sum(1 for _ in csv.DictReader(split_index.open("r", encoding="utf-8-sig")))
    duplicate_report = duplicate_image_audit(image_root)
    totals = {
        "feature_count": len(features),
        "total_png_images": sum(f["image_count"] for f in features),
        "total_xt_models": sum(1 for f in features if f["has_cad_xt"]),
        "total_kg_json": sum(1 for f in features if f["has_kg_json"]),
        "features_with_70_images": sum(1 for f in features if f["image_count"] == 70),
        "aero_instruct_samples": instruction_count,
        "dataset_split_records": split_count,
        "duplicate_image_groups": duplicate_report["duplicate_group_count"],
    }
    status = "PASS" if totals["total_png_images"] == 1400 and totals["total_xt_models"] == 20 and totals["total_kg_json"] == 20 and totals["aero_instruct_samples"] == 5000 and totals["dataset_split_records"] == 5000 else "CHECK"
    return {"status": status, "totals": totals, "features": features, "duplicate_image_audit": duplicate_report}


def duplicate_image_audit(image_root: Path) -> Dict[str, Any]:
    hashes = defaultdict(list)
    if image_root.exists():
        for p in sorted(image_root.rglob("*.png")):
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            hashes[h].append(str(p.relative_to(image_root)).replace("\\", "/"))
    groups = [{"hash": h, "count": len(v), "files": v} for h, v in hashes.items() if len(v) > 1]
    return {"duplicate_group_count": len(groups), "duplicate_file_count": sum(g["count"] for g in groups), "groups": groups}
