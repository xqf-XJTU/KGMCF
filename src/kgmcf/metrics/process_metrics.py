"""Manufacturing-oriented process-plan metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from collections import defaultdict
import csv
import re


def normalize_tokens(text: str) -> List[str]:
    text = str(text or "").lower()
    pieces = re.split(r"\s*(?:->|→|,|;|/|\||\n)\s*", text)
    return [p.strip() for p in pieces if p.strip()]


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def macro_process_sequence_accuracy(generated_sequence: str, reference_sequence: str) -> float:
    g = normalize_tokens(generated_sequence)
    r = normalize_tokens(reference_sequence)
    if not g and not r:
        return 1.0
    return round(1.0 - levenshtein_distance(g, r) / max(len(g), len(r), 1), 4)


def micro_step_resource_validity(plan: Dict[str, Any], required_terms: Iterable[str] | None = None) -> float:
    checks = []
    tooling = str(plan.get("tooling") or plan.get("selected_tool_or_holder") or "").lower()
    route = str(plan.get("process_route") or plan.get("operation_sequence") or "").lower()
    params = plan.get("cutting_parameters", {})
    if isinstance(params, str):
        params_text = params.lower()
        has_params = all(k in params_text for k in ["vc", "ap"]) and ("f=" in params_text or "feed" in params_text)
    else:
        has_params = all(k in params for k in ["Vc_m_min", "feed_mm_rev", "ap_mm"])
    checks.append(bool(tooling and "generic" not in tooling and "standard end mill" not in tooling))
    checks.append(bool(route and "generic" not in route))
    checks.append(bool(has_params))
    for term in required_terms or []:
        term_l = str(term).lower()
        checks.append(term_l in tooling or term_l in route or term_l in str(plan).lower())
    return round(sum(1 for c in checks if c) / max(1, len(checks)), 4)


def graph_process_violation_rate(plan: Dict[str, Any], required_terms: Iterable[str] | None = None) -> float:
    return round(1.0 - micro_step_resource_validity(plan, required_terms), 4)


def tool_selection_accuracy(selected_tool: str, reference_tools: Iterable[str]) -> float:
    selected = _tool_terms(selected_tool)
    refs = [_tool_terms(t) for t in reference_tools if str(t).strip()]
    if not selected and not refs:
        return 1.0
    if not selected or not refs:
        return 0.0
    best = 0.0
    for ref in refs:
        inter = len(selected & ref)
        union = len(selected | ref) or 1
        best = max(best, inter / union)
    return round(best, 4)


def rouge_l_f1(generated: str, reference: str) -> float:
    g = _word_tokens(generated)
    r = _word_tokens(reference)
    if not g and not r:
        return 1.0
    if not g or not r:
        return 0.0
    lcs = _lcs_length(g, r)
    precision = lcs / len(g)
    recall = lcs / len(r)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def evaluate_process_pair(generated_plan: Dict[str, Any], reference_plan: Dict[str, Any]) -> Dict[str, float]:
    reference_tools = reference_plan.get("reference_tools") or [reference_plan.get("tooling", "")]
    required_terms = reference_plan.get("required_terms") or []
    g_seq = generated_plan.get("process_route") or generated_plan.get("operation_sequence") or ""
    r_seq = reference_plan.get("process_route") or reference_plan.get("operation_sequence") or ""
    selected_tool = generated_plan.get("tooling") or generated_plan.get("selected_tool_or_holder") or ""
    g_text = _plan_text(generated_plan)
    r_text = _plan_text(reference_plan)
    msrv = micro_step_resource_validity(generated_plan, required_terms)
    return {
        "MPSA": macro_process_sequence_accuracy(g_seq, r_seq),
        "MSRV": msrv,
        "TSA": tool_selection_accuracy(str(selected_tool), reference_tools),
        "G-PvR": round(1.0 - msrv, 4),
        "ROUGE-L": rouge_l_f1(g_text, r_text),
    }


def summarize_generation_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    out: Dict[str, Any] = {"methods": {}}
    for method, rs in by.items():
        metrics = ["MPSA", "MSRV", "TSA", "G-PvR", "ROUGE-L"]
        out["methods"][method] = {m: round(sum(float(r[m]) for r in rs) / len(rs), 4) for m in metrics}
        out["methods"][method]["records"] = len(rs)
    return out


def _tool_terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+\d*[a-z]*|r\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}", str(text).lower()))


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9\.]+|[\u4e00-\u9fff]", str(text).lower())


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    prev = [0] * (len(b) + 1)
    for ca in a:
        curr = [0]
        for j, cb in enumerate(b, 1):
            curr.append(prev[j - 1] + 1 if ca == cb else max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def _plan_text(plan: Dict[str, Any]) -> str:
    keys = ["recognized_feature", "recognized_geometry", "key_constraints", "process_route", "operation_sequence", "tooling", "selected_tool_or_holder", "cutting_parameters", "manufacturing_reasoning_notes", "reasoning_notes"]
    return "\n".join(str(plan.get(k, "")) for k in keys)
