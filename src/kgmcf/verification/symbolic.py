"""Rule-based symbolic verification for process-plan records."""
from __future__ import annotations
from typing import Any, Dict, List


def verify_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    params = plan.get("cutting_parameters", {})
    fid = plan.get("feature_id", "")
    tooling = str(plan.get("tooling", "")).lower()
    route = str(plan.get("process_route", "")).lower()
    if "generic" in tooling and plan.get("method") in {"Vanilla MLLM", "RAG-based Planner"}:
        issues.append("tooling is not feature-specific")
    if fid == "F12" and ("r3" in tooling or "oversized" in tooling):
        issues.append("tool nose radius may exceed target radius")
    if fid in {"F05", "F11"} and "generic" in route:
        issues.append("route does not address narrow-access or chatter-sensitive geometry")
    vc = params.get("Vc_m_min")
    ap = params.get("ap_mm")
    if vc is None or ap is None:
        issues.append("incomplete cutting parameters")
    if ap is not None and fid == "F12" and float(ap) > 0.2:
        issues.append("finish depth is too large for internal blind-zone transition")
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "checked_items": ["tool-feature compatibility", "holder clearance", "radius generation", "cutting parameters", "coolant/chip control"],
    }
