"""Planner adapters for experimental methods."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Type

from .planner import build_process_plan


@dataclass
class PlanningInput:
    feature_id: str
    feature_summary: Dict[str, Any]
    retrieved_context: Dict[str, Any] | None = None
    anchored_matches: List[Dict[str, Any]] | None = None
    prompt: str | None = None


class PlannerAdapter:
    method_name = "Base"
    uses_retrieval = False
    uses_symbolic_verification = False
    uses_semantic_traceback = False

    def generate(self, planning_input: PlanningInput) -> Dict[str, Any]:
        plan = build_process_plan(planning_input.feature_summary, self.method_name)
        plan["method_settings"] = self.settings()
        plan["retrieved_evidence"] = self._evidence(planning_input)
        return plan

    def settings(self) -> Dict[str, Any]:
        return {
            "uses_retrieval": self.uses_retrieval,
            "uses_symbolic_verification": self.uses_symbolic_verification,
            "uses_semantic_traceback": self.uses_semantic_traceback,
        }

    def _evidence(self, planning_input: PlanningInput) -> Dict[str, Any]:
        return {"nodes": 0, "edges": 0, "evidence_paths": []}


class VanillaMLLMAdapter(PlannerAdapter):
    method_name = "Vanilla MLLM"


class RAGPlannerAdapter(PlannerAdapter):
    method_name = "RAG-based Planner"
    uses_retrieval = True

    def _evidence(self, planning_input: PlanningInput) -> Dict[str, Any]:
        ctx = planning_input.retrieved_context or {}
        return {
            "nodes": len(ctx.get("nodes", [])),
            "edges": len(ctx.get("edges", [])),
            "evidence_paths": ctx.get("evidence_paths", [])[:8],
        }


class ReActPlannerAdapter(RAGPlannerAdapter):
    method_name = "Standard ReAct Agent"
    uses_symbolic_verification = True


class KGMCFPlannerAdapter(RAGPlannerAdapter):
    method_name = "KGMCF"
    uses_symbolic_verification = True
    uses_semantic_traceback = True

    def generate(self, planning_input: PlanningInput) -> Dict[str, Any]:
        plan = super().generate(planning_input)
        ctx = planning_input.retrieved_context or {}
        context = ctx.get("context", {})
        if context:
            constraints = context.get("constraints", [])[:8]
            tools = context.get("tools", [])[:4]
            processes = context.get("processes", []) + context.get("steps", [])
            if constraints:
                plan["key_constraints"] = constraints
            if tools:
                plan["tooling"] = "; ".join(tools)
            if processes:
                plan["process_route"] = " -> ".join(processes[:6])
            plan["physics_aware_context"] = {
                "physics": context.get("physics", [])[:6],
                "parameters": context.get("parameters", [])[:4],
                "coolant": context.get("coolant", [])[:3],
            }
        plan["reasoning_notes"] = "Aero-MPKG evidence, physics constraints, symbolic checks, and feedback correction are used in sequence."
        return plan


ADAPTERS: Mapping[str, Type[PlannerAdapter]] = {
    "Vanilla MLLM": VanillaMLLMAdapter,
    "RAG-based Planner": RAGPlannerAdapter,
    "Standard ReAct Agent": ReActPlannerAdapter,
    "KGMCF": KGMCFPlannerAdapter,
}


def get_planner(method: str) -> PlannerAdapter:
    if method not in ADAPTERS:
        raise KeyError(f"Unknown method: {method}")
    return ADAPTERS[method]()


def generate_process_plan(method: str, planning_input: PlanningInput) -> Dict[str, Any]:
    return get_planner(method).generate(planning_input)
