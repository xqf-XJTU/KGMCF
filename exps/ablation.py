from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from kgmcf.metrics.process_metrics import summarize_generation_metrics
import csv
rows=list(csv.DictReader((ROOT/"data/supplementary/planning_records/process_generation_metrics_raw_400.csv").open("r",encoding="utf-8-sig")))
summary=summarize_generation_metrics(rows)
base=summary.get("methods",{}).get("Vanilla MLLM",{})
out_rows=[]
for method, vals in summary.get("methods",{}).items():
    out_rows.append({
        "method":method,
        "MPSA":vals.get("MPSA"),
        "MSRV":vals.get("MSRV"),
        "TSA":vals.get("TSA"),
        "G-PvR":vals.get("G-PvR"),
        "ROUGE-L":vals.get("ROUGE-L"),
        "MPSA_gain_vs_vanilla":None if not base else round(vals.get("MPSA",0)-base.get("MPSA",0),4),
        "MSRV_gain_vs_vanilla":None if not base else round(vals.get("MSRV",0)-base.get("MSRV",0),4),
    })
out=ROOT/"artifacts/reports/ablation_summary.json"
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(out_rows,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"records":len(out_rows),"output":str(out)},ensure_ascii=False,indent=2))
