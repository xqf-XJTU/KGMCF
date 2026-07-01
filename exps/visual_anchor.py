from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from kgmcf.visual.anchor import build_prototypes
print(json.dumps(build_prototypes(ROOT/"data/raw/cv20/images", ROOT/"data/processed/visual_anchor_prototypes.json"),ensure_ascii=False,indent=2))
