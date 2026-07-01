"""Visual feature anchoring backend."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json
import numpy as np

from .encoder import VisualEncoder, VisualEncoderConfig
from .index import VisualFeatureIndex, build_and_save_index


def image_descriptor(path: Path, size: int = 32) -> np.ndarray:
    encoder = VisualEncoder(VisualEncoderConfig(backend="fallback", image_size=size))
    return encoder.encode_image(path)


def build_prototypes(image_root: Path, out_path: Path) -> Dict[str, Any]:
    encoder = VisualEncoder(VisualEncoderConfig())
    prototypes: Dict[str, List[float]] = {}
    for feature_dir in sorted(Path(image_root).iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 999):
        if not feature_dir.is_dir() or not feature_dir.name.isdigit():
            continue
        imgs = sorted(feature_dir.rglob("*.png"))
        if not imgs:
            continue
        descs = np.stack([encoder.encode_image(p) for p in imgs])
        proto = descs.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes[f"F{int(feature_dir.name):02d}"] = proto.astype(float).tolist()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"encoder": encoder.metadata(), "prototypes": prototypes}, ensure_ascii=False), encoding="utf-8")
    return {"prototype_count": len(prototypes), "encoder": encoder.metadata(), "output": str(out_path)}


def query_image(image_path: Path, prototype_path: Path, top_k: int = 3) -> List[Dict[str, Any]]:
    payload = json.loads(prototype_path.read_text(encoding="utf-8"))
    prototypes = payload.get("prototypes", payload)
    q = image_descriptor(image_path, size=64)
    scores = []
    for fid, proto in prototypes.items():
        p = np.asarray(proto, dtype=np.float32)
        if p.shape != q.shape:
            # Compatibility path for older prototype files.
            q_use = image_descriptor(image_path, size=32)
        else:
            q_use = q
        scores.append({"feature_id": fid, "score": float(np.dot(q_use, p))})
    return sorted(scores, key=lambda x: x["score"], reverse=True)[:top_k]
