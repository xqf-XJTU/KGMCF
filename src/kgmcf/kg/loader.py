"""Aero-MPKG local graph loading utilities.

This module provides a Neo4j-free loader for the JSON knowledge graph files
included in the project. It normalizes node/edge records and can export a
visualization-ready graph for the local prototype system.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import re

FEATURE_RE = re.compile(r"F(\d{2})", re.IGNORECASE)


@dataclass
class KGNode:
    id: str
    label: str
    properties: Dict[str, Any]
    source_file: str

    @property
    def name(self) -> str:
        return str(self.properties.get("name") or self.id)


@dataclass
class KGEdge:
    source: str
    target: str
    relation: str
    properties: Dict[str, Any]
    source_file: str


def feature_id_from_filename(path: Path) -> str:
    m = FEATURE_RE.search(path.stem)
    if not m:
        raise ValueError(f"Cannot infer feature id from {path}")
    return f"F{int(m.group(1)):02d}"


class AeroMPKG:
    """In-memory representation of the project KG JSON files."""

    def __init__(self) -> None:
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[KGEdge] = []
        self.by_feature: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def load_dir(cls, kg_dir: Path) -> "AeroMPKG":
        """Load feature KG files from either MMKG/json or MMKG."""
        kg = cls()
        candidates = [kg_dir, kg_dir / "json"]
        chosen = None
        for candidate in candidates:
            if candidate.exists() and any(candidate.glob("F*.json")):
                chosen = candidate
                break
        if chosen is None:
            chosen = kg_dir
        for path in sorted(chosen.glob("F*.json")):
            kg.load_file(path)
        return kg

    def load_file(self, path: Path) -> None:
        feature_id = feature_id_from_filename(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_nodes = data.get("nodes", [])
        raw_edges = data.get("relationships", data.get("edges", []))
        self.by_feature[feature_id] = data
        for n in raw_nodes:
            node_id = str(n.get("id"))
            if not node_id or node_id == "None":
                raise ValueError(f"Missing node id in {path}")
            self.nodes[node_id] = KGNode(
                id=node_id,
                label=str(n.get("label", "Node")),
                properties=dict(n.get("properties", {})),
                source_file=path.name,
            )
        for e in raw_edges:
            # Support several common relationship schemas.
            source = e.get("source") or e.get("from") or e.get("start") or e.get("start_node") or e.get("startNode")
            target = e.get("target") or e.get("to") or e.get("end") or e.get("end_node") or e.get("endNode")
            relation = e.get("type") or e.get("label") or e.get("relation") or e.get("name") or "RELATED_TO"
            if source is None or target is None:
                raise ValueError(f"Missing relationship endpoints in {path}: {e}")
            self.edges.append(KGEdge(
                source=str(source),
                target=str(target),
                relation=str(relation),
                properties=dict(e.get("properties", {})),
                source_file=path.name,
            ))

    def validate(self) -> Dict[str, Any]:
        missing_sources = []
        missing_targets = []
        for e in self.edges:
            if e.source not in self.nodes:
                missing_sources.append(asdict(e))
            if e.target not in self.nodes:
                missing_targets.append(asdict(e))
        labels: Dict[str, int] = {}
        for n in self.nodes.values():
            labels[n.label] = labels.get(n.label, 0) + 1
        relations: Dict[str, int] = {}
        for e in self.edges:
            relations[e.relation] = relations.get(e.relation, 0) + 1
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "feature_count": len(self.by_feature),
            "labels": dict(sorted(labels.items(), key=lambda kv: kv[0])),
            "relations": dict(sorted(relations.items(), key=lambda kv: kv[0])),
            "missing_source_count": len(missing_sources),
            "missing_target_count": len(missing_targets),
            "missing_sources": missing_sources[:20],
            "missing_targets": missing_targets[:20],
        }

    def feature_summary(self, feature_id: str) -> Dict[str, Any]:
        data = self.by_feature[feature_id]
        nodes = data.get("nodes", [])
        feat_nodes = [n for n in nodes if n.get("label") == "特征"]
        constraints = [n for n in nodes if "约束" in str(n.get("label", ""))]
        tools = [n for n in nodes if str(n.get("label")) in {"Tool", "工具", "刀具", "刀片"}]
        equipment = [n for n in nodes if str(n.get("label")) in {"Equipment", "设备", "刀杆"}]
        processes = [n for n in nodes if str(n.get("label")) == "工序"]
        steps = [n for n in nodes if str(n.get("label")) == "工步"]
        return {
            "feature_id": feature_id,
            "feature_name": feat_nodes[0].get("properties", {}).get("name", feature_id) if feat_nodes else feature_id,
            "raw_desc": feat_nodes[0].get("properties", {}).get("raw_desc", "") if feat_nodes else "",
            "constraints": [n.get("properties", {}).get("name", n.get("id")) for n in constraints],
            "tools": [n.get("properties", {}).get("name", n.get("id")) for n in tools],
            "equipment": [n.get("properties", {}).get("name", n.get("id")) for n in equipment],
            "processes": [n.get("properties", {}).get("name", n.get("id")) for n in processes],
            "steps": [n.get("properties", {}).get("name", n.get("id")) for n in steps],
        }

    def all_feature_summaries(self) -> List[Dict[str, Any]]:
        return [self.feature_summary(fid) for fid in sorted(self.by_feature)]

    def to_vis_network(self) -> Dict[str, Any]:
        label_colors = {
            "零件": "#f7c948", "特征": "#4dabf7", "几何约束": "#74c69d",
            "Tool": "#e64980", "Equipment": "#9b5de5", "工具": "#e64980", "刀具": "#e64980", "刀片": "#e64980", "刀杆": "#9b5de5",
            "工序": "#ff922b", "工步": "#63e6be", "材料": "#adb5bd", "物理约束": "#fa5252", "复杂度": "#868e96",
            "分析": "#6c757d", "切削参数": "#5c7cfa", "冷却要求": "#15aabf", "力学逻辑": "#fa8c16",
        }
        nodes = []
        for idx, node in enumerate(self.nodes.values()):
            nodes.append({
                "id": node.id,
                "label": node.name,
                "group": node.label,
                "title": json.dumps(node.properties, ensure_ascii=False),
                "color": label_colors.get(node.label, "#dee2e6"),
            })
        edges = []
        for i, edge in enumerate(self.edges):
            if edge.source in self.nodes and edge.target in self.nodes:
                edges.append({"id": f"e{i}", "from": edge.source, "to": edge.target, "label": edge.relation})
        return {"nodes": nodes, "edges": edges}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
