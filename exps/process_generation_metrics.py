from pathlib import Path
import csv, json, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from kgmcf.kg.loader import AeroMPKG
from kgmcf.kg.lgse import extract_logic_guided_subgraph
from kgmcf.planning.adapters import PlanningInput, generate_process_plan
from kgmcf.metrics.process_metrics import evaluate_process_pair, summarize_generation_metrics

kg=AeroMPKG.load_dir(ROOT/"data/raw/aero_mpkg/json")
plans_path=ROOT/"data/supplementary/planning_records/generated_process_plans_raw_400.csv"
out_path=ROOT/"data/supplementary/planning_records/process_generation_metrics_raw_400.csv"
rows=list(csv.DictReader(plans_path.open("r",encoding="utf-8-sig")))
metric_rows=[]
for r in rows:
    summary=kg.feature_summary(r["feature_id"])
    reference=generate_process_plan("KGMCF", PlanningInput(r["feature_id"], summary, extract_logic_guided_subgraph(kg,r["feature_id"])))
    reference["reference_tools"]=summary.get("tools",[])+summary.get("equipment",[])
    reference["required_terms"]=[]
    generated={"feature_id":r["feature_id"],"method":r["method"],"operation_sequence":r.get("operation_sequence",""),"selected_tool_or_holder":r.get("selected_tool_or_holder",""),"cutting_parameters":r.get("cutting_parameters",""),"manufacturing_reasoning_notes":r.get("manufacturing_reasoning_notes","")}
    metric_rows.append({"case_id":r["case_id"],"feature_id":r["feature_id"],"method":r["method"],"plan_id":r["plan_id"],**evaluate_process_pair(generated,reference)})
out_path.parent.mkdir(parents=True,exist_ok=True)
with out_path.open("w",encoding="utf-8-sig",newline="") as f:
    fieldnames=["case_id","feature_id","method","plan_id","MPSA","MSRV","TSA","G-PvR","ROUGE-L"]
    writer=csv.DictWriter(f,fieldnames=fieldnames); writer.writeheader(); writer.writerows(metric_rows)
summary=summarize_generation_metrics(metric_rows)
summary_path=ROOT/"artifacts/reports/process_generation_metrics_summary.json"
summary_path.parent.mkdir(parents=True,exist_ok=True)
summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"records":len(metric_rows),"methods":list(summary.get("methods",{}).keys()),"output":str(out_path)},ensure_ascii=False,indent=2))
