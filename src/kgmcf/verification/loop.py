"""Symbolic verification loop with semantic feedback."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List
import copy

from .symbolic import verify_plan

PlanGenerator = Callable[[Dict[str, Any], List[str], int], Dict[str, Any]]


@dataclass
class VerificationStep:
    retry: int
    passed: bool
    issues: List[str]
    feedback: List[str]


@dataclass
class VerificationLoopResult:
    passed: bool
    final_status: str
    retry_count: int
    final_unresolved_failure: bool
    steps: List[Dict[str, Any]]
    final_plan: Dict[str, Any]


def run_symbolic_verification_loop(
    initial_plan: Dict[str, Any],
    generator: PlanGenerator | None = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    plan = copy.deepcopy(initial_plan)
    steps: List[Dict[str, Any]] = []
    retry = 0
    while True:
        report = verify_plan(plan)
        feedback = semantic_feedback(report.get("issues", []))
        steps.append(asdict(VerificationStep(retry=retry, passed=bool(report.get("passed")), issues=report.get("issues", []), feedback=feedback)))
        if report.get("passed"):
            return asdict(VerificationLoopResult(True, "verified", retry, False, steps, plan))
        if retry >= max_retries:
            return asdict(VerificationLoopResult(False, "unresolved", retry, True, steps, plan))
        if generator is not None:
            plan = generator(plan, feedback, retry + 1)
        else:
            plan = apply_semantic_correction(plan, feedback)
        retry += 1


def semantic_feedback(issues: List[str]) -> List[str]:
    feedback: List[str] = []
    for issue in issues:
        if "tooling" in issue:
            feedback.append("Replace generic tooling with feature-specific holder and insert resources from the Aero-MPKG context.")
        elif "tool nose radius" in issue:
            feedback.append("Constrain tool nose radius below the target transition radius and use a small-radius internal profiling insert.")
        elif "generic" in issue or "route" in issue:
            feedback.append("Replace the generic route with feature-level roughing, semi-finishing, and profile finishing steps.")
        elif "cutting parameters" in issue:
            feedback.append("Complete cutting speed, feed, depth of cut, and coolant parameters before acceptance.")
        elif "finish depth" in issue:
            feedback.append("Reduce finishing depth for internal blind-zone transition to protect radius generation and clearance.")
        else:
            feedback.append(f"Correct symbolic issue: {issue}")
    return feedback


def apply_semantic_correction(plan: Dict[str, Any], feedback: List[str]) -> Dict[str, Any]:
    updated = copy.deepcopy(plan)
    fid = updated.get("feature_id", "")
    joined = " ".join(feedback).lower()
    if "generic tooling" in joined or "feature-specific" in joined:
        if fid == "F12":
            updated["tooling"] = "CER2525 holder; U5R-R1.5 insert"
        elif fid == "F05":
            updated["tooling"] = "CGGL2525P1604-WK014 holder; QC04-R1.6R0.9-WK014 insert"
        elif fid == "F00":
            updated["tooling"] = "CFGL2525 holder; QC04-3.7R2 insert"
        else:
            tools = updated.get("key_constraints", [])
            updated["tooling"] = updated.get("tooling", "feature-specific tool") if "generic" not in str(updated.get("tooling", "")).lower() else "feature-specific form tool"
    if "feature-level" in joined or "generic route" in joined:
        updated["process_route"] = "feature roughing -> allowance-controlled semi-finishing -> profile finishing -> clearance verification"
    if "finish depth" in joined or fid == "F12":
        params = dict(updated.get("cutting_parameters", {}))
        params["ap_mm"] = min(float(params.get("ap_mm", 0.1)), 0.1)
        params.setdefault("Vc_m_min", 70)
        params.setdefault("feed_mm_rev", 0.08)
        params.setdefault("coolant", "through-tool high-pressure coolant")
        updated["cutting_parameters"] = params
    elif "complete cutting" in joined:
        updated.setdefault("cutting_parameters", {"Vc_m_min": 55, "feed_mm_rev": 0.10, "ap_mm": 0.5, "coolant": "high-pressure coolant"})
    updated["semantic_correction_applied"] = True
    return updated
