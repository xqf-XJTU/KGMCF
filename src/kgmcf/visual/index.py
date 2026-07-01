"""Feature-level visual retrieval index.

The index stores image embeddings and feature-category prototypes. It uses a
portable NumPy search backend by default, and can use HNSW or FAISS when those
libraries are installed in the execution environment.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import json
import re
import numpy as np

from .encoder import VisualEncoder, VisualEncoderConfig


@dataclass
class VisualIndexRecord:
    image_file: str
    feature_id: str
    view_id: str


class VisualFeatureIndex:
    def __init__(self, vectors: np.ndarray, records: List[VisualIndexRecord], encoder_metadata: Dict[str, Any], backend: str = "numpy") -> None:
        self.vectors = vectors.astype(np.float32)
        self.records = records
        self.encoder_metadata = encoder_metadata
        self.backend = backend
        self._backend_index = None
        self._build_optional_backend()

    @classmethod
    def build(cls, image_root: Path, encoder: VisualEncoder | None = None, backend: str = "auto") -> "VisualFeatureIndex":
        image_root = Path(image_root)
        encoder = encoder or VisualEncoder(VisualEncoderConfig())
        images = sorted(image_root.rglob("*.png"), key=lambda p: str(p))
        records: List[VisualIndexRecord] = []
        vectors: List[np.ndarray] = []
        for img in images:
            fid = _feature_id_from_path(img, image_root)
            records.append(VisualIndexRecord(str(img), fid, img.stem))
            vectors.append(encoder.encode_image(img))
        matrix = np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, encoder.dimension), dtype=np.float32)
        matrix = _row_normalize(matrix)
        index = cls(matrix, records, encoder.metadata(), backend="numpy" if backend == "auto" else backend)
        return index

    @classmethod
    def load(cls, path: Path) -> "VisualFeatureIndex":
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        vectors = np.load(path / "vectors.npy")
        records = [VisualIndexRecord(**r) for r in json.loads((path / "records.json").read_text(encoding="utf-8"))]
        return cls(vectors, records, meta.get("encoder", {}), backend=meta.get("backend", "numpy"))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        (path / "records.json").write_text(json.dumps([asdict(r) for r in self.records], ensure_ascii=False, indent=2), encoding="utf-8")
        meta = {"backend": self.backend, "encoder": self.encoder_metadata, "record_count": len(self.records)}
        (path / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_optional_backend(self) -> None:
        if self.backend == "hnsw":
            try:
                import hnswlib  # type: ignore
                index = hnswlib.Index(space="cosine", dim=self.vectors.shape[1])
                index.init_index(max_elements=len(self.records), ef_construction=100, M=16)
                index.add_items(self.vectors, np.arange(len(self.records)))
                index.set_ef(min(50, max(10, len(self.records))))
                self._backend_index = index
                return
            except Exception:
                self.backend = "numpy"
        elif self.backend == "faiss":
            try:
                import faiss  # type: ignore
                index = faiss.IndexFlatIP(self.vectors.shape[1])
                index.add(self.vectors.astype(np.float32))
                self._backend_index = index
                return
            except Exception:
                self.backend = "numpy"
        else:
            self.backend = "numpy"

    def query_vector(self, vector: np.ndarray, top_k: int = 10) -> Dict[str, Any]:
        if self.vectors.size == 0:
            return {"matches": [], "category_scores": []}
        q = vector.astype(np.float32).reshape(1, -1)
        q = _row_normalize(q)[0]
        if self.backend == "hnsw" and self._backend_index is not None:
            labels, distances = self._backend_index.knn_query(q, k=min(top_k, len(self.records)))
            idx_scores = [(int(i), float(1.0 - d)) for i, d in zip(labels[0], distances[0])]
        elif self.backend == "faiss" and self._backend_index is not None:
            scores, labels = self._backend_index.search(q.reshape(1, -1), min(top_k, len(self.records)))
            idx_scores = [(int(i), float(s)) for i, s in zip(labels[0], scores[0])]
        else:
            scores = self.vectors @ q
            top = np.argsort(-scores)[: min(top_k, len(scores))]
            idx_scores = [(int(i), float(scores[i])) for i in top]
        matches = [{"rank": rank + 1, "score": round(score, 6), **asdict(self.records[i])} for rank, (i, score) in enumerate(idx_scores)]
        by_feature: Dict[str, List[float]] = {}
        for i, score in idx_scores:
            by_feature.setdefault(self.records[i].feature_id, []).append(score)
        category_scores = sorted(
            [{"feature_id": fid, "score": round(float(sum(vals) / len(vals)), 6), "match_count": len(vals)} for fid, vals in by_feature.items()],
            key=lambda r: r["score"],
            reverse=True,
        )
        return {"backend": self.backend, "matches": matches, "category_scores": category_scores}

    def query_image(self, image_path: Path, encoder: VisualEncoder | None = None, top_k: int = 10) -> Dict[str, Any]:
        encoder = encoder or VisualEncoder(VisualEncoderConfig())
        return self.query_vector(encoder.encode_image(Path(image_path)), top_k=top_k)


def build_and_save_index(image_root: Path, output_dir: Path, backend: str = "auto") -> Dict[str, Any]:
    encoder = VisualEncoder(VisualEncoderConfig())
    index = VisualFeatureIndex.build(image_root, encoder=encoder, backend=backend)
    index.save(output_dir)
    return {"record_count": len(index.records), "backend": index.backend, "encoder": index.encoder_metadata, "output_dir": str(output_dir)}


def _feature_id_from_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    first = rel.parts[0]
    if first.isdigit():
        return f"F{int(first):02d}"
    m = re.search(r"F(\d{2})", str(rel), flags=re.IGNORECASE)
    if m:
        return f"F{int(m.group(1)):02d}"
    return "F00"


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return (matrix / norms).astype(np.float32)
