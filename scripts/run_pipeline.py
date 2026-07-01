#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, json, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kgmcf.data.audit import audit_project
from kgmcf.data.dataset_index import validate_dataset_split_index
from kgmcf.data.instructions import validate_instruction_corpus
from kgmcf.kg.loader import AeroMPKG, write_json
from kgmcf.planning.process_mapping import diagnose_process
from kgmcf.visual.anchor import build_prototypes
from kgmcf.visual.index import build_and_save_index
from kgmcf.kg.lgse import extract_logic_guided_subgraph
from kgmcf.planning.adapters import PlanningInput, generate_process_plan
from kgmcf.verification.loop import run_symbolic_verification_loop
from kgmcf.metrics.process_metrics import evaluate_process_pair, summarize_generation_metrics
import csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KGMCF data, graph, process-generation, and symbolic-verification checks.")
    parser.add_argument("--root", default=str(ROOT), help="Project root path")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    reports = root / "artifacts" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    graph_out = root / "artifacts" / "graph"
    graph_out.mkdir(parents=True, exist_ok=True)

    audit = audit_project(root)
    write_json(reports / "dataset_audit.json", audit)

    kg = AeroMPKG.load_dir(root / "data" / "raw" / "aero_mpkg" / "json")
    kg_validation = kg.validate()
    write_json(reports / "kg_validation_report.json", kg_validation)
    write_json(graph_out / "aero_mpkg_vis.json", kg.to_vis_network())
    summaries = kg.all_feature_summaries()
    write_json(reports / "feature_summaries.json", summaries)
    write_json(reports / "process_mapping_diagnosis.json", [diagnose_process(s).__dict__ for s in summaries])

    instr_report = validate_instruction_corpus(root / "data" / "processed" / "aero_instruct_5k.jsonl")
    split_report = validate_dataset_split_index(root / "data" / "processed" / "dataset_split_index.csv")
    proto = build_prototypes(root / "data" / "raw" / "cv20" / "images", root / "data" / "processed" / "visual_anchor_prototypes.json")
    visual_index = build_and_save_index(root / "data" / "raw" / "cv20" / "images", root / "artifacts" / "visual_index", backend="auto")
    lgse_examples = {fid: extract_logic_guided_subgraph(kg, fid) for fid in sorted(kg.by_feature)}
    write_json(reports / "lgse_retrieval_examples.json", lgse_examples)
    planning_examples = []
    traceback_examples = []
    for fid in sorted(kg.by_feature):
        summary_i = kg.feature_summary(fid)
        retrieved_i = lgse_examples[fid]
        for method in ["Vanilla MLLM", "RAG-based Planner", "Standard ReAct Agent", "KGMCF"]:
            plan_i = generate_process_plan(method, PlanningInput(fid, summary_i, retrieved_i))
            planning_examples.append(plan_i)
        if fid in {"F00", "F05", "F11", "F12"}:
            base_plan = generate_process_plan("Vanilla MLLM", PlanningInput(fid, summary_i, retrieved_i))
            traceback_examples.append({"feature_id": fid, "result": run_symbolic_verification_loop(base_plan, max_retries=3)})
    write_json(reports / "process_planning_examples.json", planning_examples)
    write_json(reports / "symbolic_traceback_examples.json", traceback_examples)
    generation_metric_rows = _build_generation_metrics(root, kg)
    write_json(reports / "process_generation_metrics_summary.json", summarize_generation_metrics(generation_metric_rows))
    generation_summary = summarize_generation_metrics(generation_metric_rows)
    write_json(reports / "process_generation_metrics_summary.json", generation_summary)
    write_json(root / "data" / "quality" / "duplicate_image_audit.json", audit["duplicate_image_audit"])

    summary = {
        "dataset_audit_status": audit["status"],
        "dataset_totals": audit["totals"],
        "instruction_corpus": instr_report,
        "dataset_split_index": split_report,
        "kg_validation": {k: kg_validation[k] for k in ["node_count", "edge_count", "feature_count", "missing_source_count", "missing_target_count"]},
        "visual_anchor": proto,
        "visual_index": visual_index,
        "lgse_feature_count": len(lgse_examples),
        "process_planning_example_count": len(planning_examples),
        "symbolic_traceback_example_count": len(traceback_examples),
        "generation_metric_methods": list(generation_summary.get("methods", {}).keys()),
        "process_generation_metrics": generation_summary,
        "external_validation_records": "not_included",
    }
    write_json(reports / "pipeline_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _build_generation_metrics(root: Path, kg: AeroMPKG) -> list[dict]:
    plans_path = root / "data" / "supplementary" / "planning_records" / "generated_process_plans_raw_400.csv"
    out_path = root / "data" / "supplementary" / "planning_records" / "process_generation_metrics_raw_400.csv"
    rows = list(csv.DictReader(plans_path.open("r", encoding="utf-8-sig")))
    metric_rows = []
    for r in rows:
        summary = kg.feature_summary(r["feature_id"])
        reference = generate_process_plan("KGMCF", PlanningInput(r["feature_id"], summary, extract_logic_guided_subgraph(kg, r["feature_id"])))
        reference["reference_tools"] = summary.get("tools", []) + summary.get("equipment", [])
        reference["required_terms"] = []
        generated = {
            "feature_id": r["feature_id"],
            "method": r["method"],
            "operation_sequence": r.get("operation_sequence", ""),
            "selected_tool_or_holder": r.get("selected_tool_or_holder", ""),
            "cutting_parameters": r.get("cutting_parameters", ""),
            "manufacturing_reasoning_notes": r.get("manufacturing_reasoning_notes", ""),
        }
        metrics = evaluate_process_pair(generated, reference)
        metric_rows.append({"case_id": r["case_id"], "feature_id": r["feature_id"], "method": r["method"], "plan_id": r["plan_id"], **metrics})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["case_id", "feature_id", "method", "plan_id", "MPSA", "MSRV", "TSA", "G-PvR", "ROUGE-L"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(metric_rows)
    return metric_rows


if __name__ == "__main__":
    main()
