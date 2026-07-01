"""Visual encoding backends used by the feature-anchoring pipeline.

The default backend is dependency-light and deterministic so that the project can
be checked in a clean environment. The same interface can be configured to use a
pretrained SigLIP encoder when the required model files and libraries are
available locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import numpy as np
from PIL import Image


@dataclass
class VisualEncoderConfig:
    backend: str = "auto"
    model_name: str = "google/siglip-base-patch16-224"
    image_size: int = 64
    normalize: bool = True


class VisualEncoder:
    """Image encoder with an optional SigLIP backend and a stable local backend."""

    def __init__(self, config: VisualEncoderConfig | None = None) -> None:
        self.config = config or VisualEncoderConfig()
        self.actual_backend = "fallback"
        self._processor = None
        self._model = None
        if self.config.backend in {"auto", "siglip"}:
            self._try_load_siglip()
        if self.config.backend == "siglip" and self.actual_backend != "siglip":
            raise RuntimeError("SigLIP backend is requested but its local dependencies or model files are unavailable.")

    def _try_load_siglip(self) -> None:
        try:
            from transformers import AutoProcessor, AutoModel  # type: ignore
            import torch  # type: ignore
            self._torch = torch
            self._processor = AutoProcessor.from_pretrained(self.config.model_name, local_files_only=True)
            self._model = AutoModel.from_pretrained(self.config.model_name, local_files_only=True)
            self._model.eval()
            self.actual_backend = "siglip"
        except Exception:
            self._processor = None
            self._model = None
            self.actual_backend = "fallback"

    def encode_image(self, image_path: Path) -> np.ndarray:
        image_path = Path(image_path)
        if self.actual_backend == "siglip" and self._processor is not None and self._model is not None:
            return self._encode_siglip_image(image_path)
        return self._encode_local_image(image_path)

    def encode_batch(self, image_paths: Iterable[Path]) -> np.ndarray:
        vectors = [self.encode_image(Path(p)) for p in image_paths]
        if not vectors:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)

    @property
    def dimension(self) -> int:
        if self.actual_backend == "siglip":
            return int(getattr(getattr(self._model, "config", None), "projection_dim", 768) or 768)
        s = self.config.image_size
        return s * s + 2 * s + 16

    def _encode_siglip_image(self, image_path: Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        with self._torch.no_grad():
            inputs = self._processor(images=image, return_tensors="pt")
            if hasattr(self._model, "get_image_features"):
                output = self._model.get_image_features(**inputs)
            else:
                output = self._model(**inputs).pooler_output
        vec = output.detach().cpu().numpy()[0].astype(np.float32)
        return _normalize(vec) if self.config.normalize else vec

    def _encode_local_image(self, image_path: Path) -> np.ndarray:
        size = self.config.image_size
        img = Image.open(image_path).convert("L").resize((size, size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        hist, _ = np.histogram(arr, bins=16, range=(0.0, 1.0), density=True)
        gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1])).mean(axis=0)
        gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :])).mean(axis=1)
        vec = np.concatenate([arr.flatten(), gx, gy, hist.astype(np.float32)])
        return _normalize(vec) if self.config.normalize else vec.astype(np.float32)

    def metadata(self) -> Dict[str, Any]:
        return {
            "requested_backend": self.config.backend,
            "actual_backend": self.actual_backend,
            "model_name": self.config.model_name if self.actual_backend == "siglip" else None,
            "dimension": self.dimension,
        }


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm
