"""Feature-to-process mapping and validation hints.

This module adds the missing process-type diagnosis layer It distinguishes generic milling operations from form-groove or
turning-style resources by inspecting tool-holder names, insert names and feature
semantics.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import re

TURNING_HINTS = ["车", "立车", "卧车", "turn", "lathe", "grooving", "切槽刀杆", "刀杆"]
FORM_HINTS = ["成型", "forming", "profile", "仿形", "R2", "R4", "QC", "CFGL", "CFEL", "CER", "DVXNN"]
MILLING_HINTS = ["mill", "铣", "end mill", "cavity", "z-level", "fixed contour", "adaptive"]


@dataclass
class ProcessDiagnosis:
    feature_id: str
    feature_name: str
    part_level_process: str
    feature_level_strategy: str
    recommended_operation: str
    is_generic_cavity: bool
    confidence: float
    rationale: List[str]


def diagnose_process(summary: Dict[str, Any]) -> ProcessDiagnosis:
    text = " ".join(map(str, [summary.get("feature_name", ""), summary.get("raw_desc", "")] + summary.get("constraints", []) + summary.get("tools", []) + summary.get("equipment", []) + summary.get("processes", []) + summary.get("steps", []))).lower()
    fid = summary.get("feature_id", "UNKNOWN")
    name = summary.get("feature_name", fid)
    rationale: List[str] = []
    turning_score = sum(1 for h in TURNING_HINTS if h.lower() in text)
    form_score = sum(1 for h in FORM_HINTS if h.lower() in text)
    milling_score = sum(1 for h in MILLING_HINTS if h.lower() in text)

    if turning_score:
        rationale.append(f"检测到车削/槽刀资源关键词 {turning_score} 个。")
    if form_score:
        rationale.append(f"检测到成型/仿形刀具或R角匹配关键词 {form_score} 个。")
    if milling_score:
        rationale.append(f"检测到铣削关键词 {milling_score} 个。")

    # Aerospace casing/ring segment models are usually validated as milling-part
    # contexts, but feature-level strategy may still require form-tool machining.
    part_level = "MILLING_PART_OR_RING_SEGMENT"
    if turning_score >= 2 and "环" in text:
        rationale.append("零件语义包含机匣/环件，整体可按铣削环件段验证；局部特征可采用成型槽策略。")
    if form_score >= 2:
        feature_strategy = "FORM_GROOVE_OR_FORM_PROFILE_MACHINING"
        op = "FORM_TOOL_CONTOUR / FIXED_CONTOUR_FINISHING; avoid generic cavity-style mapping as final strategy"
        is_generic = False
        confidence = min(0.95, 0.55 + 0.08 * form_score + 0.04 * turning_score)
        rationale.append("该特征更适合成型刀具/轮廓精加工，不宜仅映射为普通Cavity Mill。")
    elif milling_score >= turning_score:
        feature_strategy = "GENERAL_MILLING"
        op = "CAVITY_MILL or ZLEVEL_PROFILE depending on floor/side-wall geometry"
        is_generic = True
        confidence = 0.65
        rationale.append("当前证据支持普通铣削映射。")
    else:
        feature_strategy = "TURNING_OR_GROOVING_RESOURCE_REQUIRED"
        op = "TURNING_GROOVE / PROFILE_TURNING if full rotational geometry is available"
        is_generic = False
        confidence = 0.70
        rationale.append("刀具资源显示为车削槽刀；若几何为完整回转体，应使用车削模块。")
    return ProcessDiagnosis(fid, name, part_level, feature_strategy, op, is_generic, confidence, rationale)
