"""Logic-guided subgraph extraction for Aero-MPKG retrieval."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Set, Tuple

from .loader import AeroMPKG


PRIORITY_LABELS = {"特征", "几何约束", "物理约束", "工序", "工步", "Tool", "Equipment", "刀具", "刀片", "刀杆", "材料", "切削参数", "冷却要求", "力学逻辑", "分析"}
PRIORITY_RELATIONS = {"HAS_CONSTRAINT", "USES_TOOL", "USES_EQUIPMENT", "HAS_PROCESS", "INCLUDES_STEP", "HAS_STEP", "HAS_PARAMETER", "HAS_RISK", "PHYSICS_CAUSE", "REQUIRES_COOLANT", "RECOMMENDS"}


@dataclass
class EvidencePath:
    source: str
    relation: str
    target: str
    source_label: str
    target_label: str
    source_name: str
    target_name: str


@dataclass
class RetrievedSubgraph:
    feature_id: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    evidence_paths: List[Dict[str, Any]]
    context: Dict[str, List[str]]


def extract_logic_guided_subgraph(
    kg: AeroMPKG,
    feature_id: str,
    anchored_constraints: Iterable[str] | None = None,
    max_nodes: int = 60,
) -> Dict[str, Any]:
    feature_id = feature_id.upper()
    data = kg.by_feature.get(feature_id)
    if not data:
        raise KeyError(f"Unknown feature_id: {feature_id}")
    nodes = data.get("nodes", [])
    raw_edges = data.get("relationships", data.get("edges", []))
    node_by_id = {str(n.get("id")): n for n in nodes}
    feature_nodes = [n for n in nodes if n.get("label") == "特征"]
    seed_ids: Set[str] = {str(n.get("id")) for n in feature_nodes}
    constraint_terms = [str(x).lower() for x in (anchored_constraints or []) if str(x).strip()]
    for n in nodes:
        name = str(n.get("properties", {}).get("name", "")).lower()
        label = str(n.get("label", ""))
        if label in {"几何约束", "物理约束"} and any(term in name for term in constraint_terms):
            seed_ids.add(str(n.get("id")))

    selected_ids: Set[str] = set(seed_ids)
    selected_edges: List[Dict[str, Any]] = []
    # First pass: keep feature-neighborhood and priority relations.
    for e in raw_edges:
        s = str(e.get("startNode") or e.get("source") or e.get("from") or "")
        t = str(e.get("endNode") or e.get("target") or e.get("to") or "")
        r = str(e.get("type") or e.get("relation") or "RELATED_TO")
        if not s or not t:
            continue
        sl = str(node_by_id.get(s, {}).get("label", ""))
        tl = str(node_by_id.get(t, {}).get("label", ""))
        if s in seed_ids or t in seed_ids or r in PRIORITY_RELATIONS or sl in PRIORITY_LABELS or tl in PRIORITY_LABELS:
            selected_edges.append({"source": s, "target": t, "relation": r, "properties": e.get("properties", {})})
            selected_ids.add(s)
            selected_ids.add(t)
    selected_nodes = [n for n in nodes if str(n.get("id")) in selected_ids]
    if len(selected_nodes) > max_nodes:
        priority = {"特征": 0, "几何约束": 1, "物理约束": 2, "工序": 3, "工步": 4, "Tool": 5, "Equipment": 5, "刀具": 5, "刀片": 5, "刀杆": 5}
        selected_nodes = sorted(selected_nodes, key=lambda n: (priority.get(str(n.get("label")), 9), str(n.get("id"))))[:max_nodes]
        kept = {str(n.get("id")) for n in selected_nodes}
        selected_edges = [e for e in selected_edges if e["source"] in kept and e["target"] in kept]
    evidence = [_edge_to_evidence(e, node_by_id) for e in selected_edges]
    context = _context_from_nodes(selected_nodes)
    return asdict(RetrievedSubgraph(feature_id, selected_nodes, selected_edges, [asdict(x) for x in evidence], context))


def _edge_to_evidence(edge: Dict[str, Any], node_by_id: Dict[str, Any]) -> EvidencePath:
    s = node_by_id.get(edge["source"], {})
    t = node_by_id.get(edge["target"], {})
    return EvidencePath(
        source=edge["source"],
        relation=edge["relation"],
        target=edge["target"],
        source_label=str(s.get("label", "")),
        target_label=str(t.get("label", "")),
        source_name=str(s.get("properties", {}).get("name", edge["source"])),
        target_name=str(t.get("properties", {}).get("name", edge["target"])),
    )


def _context_from_nodes(nodes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    buckets = {
        "constraints": [],
        "physics": [],
        "tools": [],
        "processes": [],
        "steps": [],
        "parameters": [],
        "coolant": [],
    }
    for n in nodes:
        label = str(n.get("label", ""))
        name = str(n.get("properties", {}).get("name", n.get("id", "")))
        if "约束" in label:
            buckets["physics" if "物理" in label else "constraints"].append(name)
        elif label in {"Tool", "Equipment", "刀具", "刀片", "刀杆", "工具"}:
            buckets["tools"].append(name)
        elif label == "工序":
            buckets["processes"].append(name)
        elif label == "工步":
            buckets["steps"].append(name)
        elif label == "切削参数":
            buckets["parameters"].append(name)
        elif label == "冷却要求":
            buckets["coolant"].append(name)
    return {k: _dedupe(v) for k, v in buckets.items()}


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out
