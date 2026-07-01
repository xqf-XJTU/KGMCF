from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from kgmcf.kg.loader import AeroMPKG
from kgmcf.kg.lgse import extract_logic_guided_subgraph
from kgmcf.planning.adapters import PlanningInput, generate_process_plan
from kgmcf.verification.loop import run_symbolic_verification_loop

kg=AeroMPKG.load_dir(ROOT/"data/raw/aero_mpkg/json")
examples=[]
for fid in ["F00","F05","F11","F12"]:
    retrieved=extract_logic_guided_subgraph(kg,fid)
    plan=generate_process_plan("Vanilla MLLM", PlanningInput(fid, kg.feature_summary(fid), retrieved))
    result=run_symbolic_verification_loop(plan, max_retries=3)
    examples.append({"feature_id":fid,"result":result})
out=ROOT/"artifacts/reports/symbolic_traceback_examples.json"
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"records":len(examples),"output":str(out)},ensure_ascii=False,indent=2))
