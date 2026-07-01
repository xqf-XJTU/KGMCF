from pathlib import Path
import csv, json, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from kgmcf.kg.loader import AeroMPKG
from kgmcf.kg.lgse import extract_logic_guided_subgraph
from kgmcf.planning.adapters import PlanningInput, generate_process_plan

kg=AeroMPKG.load_dir(ROOT/"data/raw/aero_mpkg/json")
rows=[]
for fid in sorted(kg.by_feature):
    summary=kg.feature_summary(fid)
    retrieved=extract_logic_guided_subgraph(kg,fid)
    for method in ["Vanilla MLLM","RAG-based Planner","Standard ReAct Agent","KGMCF"]:
        rows.append(generate_process_plan(method, PlanningInput(fid, summary, retrieved)))
out=ROOT/"artifacts/reports/process_planning_examples.json"
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"records":len(rows),"output":str(out)},ensure_ascii=False,indent=2))
