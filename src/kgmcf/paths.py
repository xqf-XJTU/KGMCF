from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SUPPLEMENTARY_DIR = DATA_DIR / "supplementary"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
