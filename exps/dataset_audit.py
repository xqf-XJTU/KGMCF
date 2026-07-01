from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from kgmcf.data.audit import audit_project
print(json.dumps(audit_project(ROOT),ensure_ascii=False,indent=2))
