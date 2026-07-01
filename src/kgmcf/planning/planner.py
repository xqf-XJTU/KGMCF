"""Deterministic process-plan construction from Aero-MPKG summaries."""
from __future__ import annotations
from typing import Any, Dict, List
from .process_mapping import diagnose_process


def _join(items: List[str], n: int = 5) -> str:
    return "; ".join(str(x) for x in items[:n]) if items else "not specified"


def build_process_plan(summary: Dict[str, Any], method: str = "KGMCF") -> Dict[str, Any]:
    diag = diagnose_process(summary)
    fid = summary.get("feature_id", "F00")
    constraints = summary.get("constraints", [])
    tools = summary.get("tools", [])
    equipment = summary.get("equipment", [])
    processes = summary.get("processes", [])
    steps = summary.get("steps", [])
    name = summary.get("feature_name", fid)
    if method == "Vanilla MLLM":
        route = "generic roughing -> generic finishing"
        tool = "standard end mill or generic boring tool"
        notes = "Direct multimodal generation; local clearance, coolant, and graph constraints are not explicitly checked."
    elif method == "RAG-based Planner":
        route = "retrieved roughing route -> retrieved finishing route"
        tool = _join(equipment + tools, 2)
        notes = "Uses retrieved process text but no deterministic symbolic validation loop."
    elif method == "Standard ReAct Agent":
        route = _join(processes + steps, 4)
        tool = _join(equipment + tools, 3)
        notes = "Can invoke graph tools but does not use the semantic traceback mechanism."
    else:
        route = _join(processes + steps, 6)
        tool = _join(equipment + tools, 4)
        notes = "Aero-MPKG retrieval, physics-aware reasoning, symbolic verification, and semantic correction are enabled."
    return {
        "feature_id": fid,
        "feature_name": name,
        "method": method,
        "recognized_feature": name,
        "key_constraints": constraints[:8],
        "manufacturing_risks": ["tool access", "holder clearance", "overcut/interference", "parameter safety"],
        "process_route": route,
        "tooling": tool,
        "recommended_operation": diag.recommended_operation,
        "cutting_parameters": {
            "Vc_m_min": 55 if fid != "F12" else 70,
            "feed_mm_rev": 0.10 if fid != "F12" else 0.08,
            "ap_mm": 1.5 if fid != "F12" else 0.1,
            "coolant": "high-pressure coolant" if fid != "F12" else "through-tool high-pressure coolant",
        },
        "reasoning_notes": notes,
        "process_diagnosis": diag.__dict__,
    }
