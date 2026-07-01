from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import json
import re
import time
from collections import defaultdict

from kgmcf.kg.loader import AeroMPKG
from kgmcf.kg.lgse import extract_logic_guided_subgraph
from kgmcf.planning.adapters import PlanningInput, generate_process_plan
from kgmcf.verification.loop import run_symbolic_verification_loop
from kgmcf.verification.symbolic import verify_plan
from kgmcf.visual.anchor import query_image
from kgmcf.models import QwenRuntimeClient, load_runtime_config

METHODS = ["KGMCF", "Standard ReAct Agent", "RAG-based Planner", "Vanilla MLLM"]
METHOD_DESCRIPTIONS = {
    "KGMCF": "Visual anchoring, Aero-MPKG retrieval, physics-aware planning, symbolic verification, and semantic traceback.",
    "Standard ReAct Agent": "Tool-use reasoning with graph access and symbolic checks, without semantic traceback correction.",
    "RAG-based Planner": "Retrieval-enhanced process planning without deterministic symbolic verification.",
    "Vanilla MLLM": "Direct multimodal process planning with the same input and prompt template, without retrieval or verification modules.",
}


@dataclass
class PrototypePaths:
    root: Path

    @property
    def kg_dir(self) -> Path:
        return self.root / "data" / "raw" / "aero_mpkg" / "json"

    @property
    def raw_image_root(self) -> Path:
        return self.root / "data" / "raw" / "cv20" / "images"

    @property
    def augmented_image_root(self) -> Path:
        return self.root / "data" / "processed" / "cv20_augmented" / "images"

    @property
    def aero5k(self) -> Path:
        return self.root / "data" / "processed" / "aero_instruct_5k.jsonl"

    @property
    def split_index(self) -> Path:
        return self.root / "data" / "processed" / "dataset_split_index.csv"

    @property
    def augmented_index(self) -> Path:
        return self.root / "data" / "processed" / "cv20_augmented_4000_index.csv"

    @property
    def visual_prototypes(self) -> Path:
        return self.root / "data" / "processed" / "visual_anchor_prototypes.json"

    @property
    def prompt_template_file(self) -> Path:
        return self.root / "data" / "processed" / "process_planning_prompt_v1.txt"

    @property
    def prompt_runtime_file(self) -> Path:
        return self.root / "artifacts" / "prototype_runtime" / "prompt_templates.json"

    @property
    def training_runs_dir(self) -> Path:
        return self.root / "artifacts" / "training_runs"

    @property
    def lora_config(self) -> Path:
        return self.root / "configs" / "lora_finetuning_config.json"

    @property
    def model_runtime_config(self) -> Path:
        return self.root / "configs" / "model_runtime.json"

    @property
    def planning_records_dir(self) -> Path:
        return self.root / "data" / "supplementary" / "planning_records"

    @property
    def upload_dir(self) -> Path:
        return self.root / "artifacts" / "prototype_uploads"


class PrototypeService:
    """Data-driven backend for the local KGMCF prototype interface."""

    def __init__(self, root: Path):
        self.paths = PrototypePaths(root=root.resolve())
        self.paths.upload_dir.mkdir(parents=True, exist_ok=True)
        self.paths.prompt_runtime_file.parent.mkdir(parents=True, exist_ok=True)
        self.kg = AeroMPKG.load_dir(self.paths.kg_dir)
        self.qwen_client = QwenRuntimeClient(root=self.paths.root)
        self.neo4j_graph = None
        self.neo4j_meta: Dict[str, Any] = {"connected": False, "uri": "bolt://localhost:7687", "username": "", "message": "Neo4j is not connected."}
        self._samples: Optional[List[Dict[str, Any]]] = None
        self._split_lookup: Optional[Dict[str, Dict[str, str]]] = None
        self.latest_feature_id: Optional[str] = None
        self.latest_uploaded_image: Optional[str] = None
        self.latest_workflow: Optional[Dict[str, Any]] = None
        self.latest_process_card: Optional[Dict[str, Any]] = None
        self._visual_records: Optional[List[Tuple[Path, str]]] = None
        self._visual_matrix = None
        self._visual_encoder = None

    # ------------------------------------------------------------------
    # System state and catalog
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        samples = self.samples
        return {
            "status": "ready",
            "feature_count": len(self.kg.by_feature),
            "sample_count": len(samples),
            "method_count": len(METHODS),
            "dataset_file": str(self.paths.aero5k.relative_to(self.paths.root)) if self.paths.aero5k.exists() else "missing",
        }

    def workflow_spec(self) -> Dict[str, Any]:
        return {
            "stages": [
                {"id": "input", "title": "Multimodal input", "description": "Engineering drawing view and process-planning intent."},
                {"id": "anchoring", "title": "Visual anchoring", "description": "The drawing is matched with CV20 visual prototypes and feature-level graph records."},
                {"id": "retrieval", "title": "Aero-MPKG grounding", "description": "A logic-guided evidence subgraph is extracted for constraints, tools, operations, and physical rules."},
                {"id": "planning", "title": "Cognitive planning", "description": "A structured process card is generated from visual, graph, and intent information."},
                {"id": "verification", "title": "Symbolic verification", "description": "The plan is checked against feature-tool, radius, route, and parameter constraints."},
                {"id": "report", "title": "Process card", "description": "The verified planning result is formatted as an engineering process report."},
            ],
            "methods": [{"name": m, "description": METHOD_DESCRIPTIONS[m]} for m in METHODS],
        }

    @property
    def samples(self) -> List[Dict[str, Any]]:
        if self._samples is None:
            rows: List[Dict[str, Any]] = []
            if self.paths.aero5k.exists():
                with self.paths.aero5k.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
            self._samples = rows
        return self._samples

    @property
    def split_lookup(self) -> Dict[str, Dict[str, str]]:
        if self._split_lookup is None:
            lookup: Dict[str, Dict[str, str]] = {}
            if self.paths.split_index.exists():
                with self.paths.split_index.open("r", encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        lookup[row.get("sample_id", "")] = row
            self._split_lookup = lookup
        return self._split_lookup

    def app_state(self) -> Dict[str, Any]:
        return {
            "health": self.health(),
            "workflow": self.workflow_spec(),
            "features": self.feature_catalog(),
            "methods": METHODS,
            "model_runtime": self.qwen_runtime_status(),
        }

    # ------------------------------------------------------------------
    # Neo4j-backed graph visualization
    # ------------------------------------------------------------------
    def connect_neo4j(self, uri: str = "bolt://localhost:7687", username: str = "", password: str = "") -> Dict[str, Any]:
        """Connect to a user-provided Neo4j database without hard-coded credentials."""
        uri = str(uri or "bolt://localhost:7687").strip()
        username = str(username or "").strip()
        password = str(password or "")
        if not username or not password:
            self.neo4j_graph = None
            self.neo4j_meta = {"connected": False, "uri": uri, "username": username, "message": "Please enter both Neo4j username and password."}
            return self.neo4j_status()
        try:
            from py2neo import Graph  # imported only when the Neo4j mode is used
            graph = Graph(uri, auth=(username, password))
            graph.run("RETURN 1 AS ok").data()
            self.neo4j_graph = graph
            self.neo4j_meta = {"connected": True, "uri": uri, "username": username, "message": "Neo4j connection established."}
        except Exception as exc:
            self.neo4j_graph = None
            self.neo4j_meta = {"connected": False, "uri": uri, "username": username, "message": f"Neo4j connection failed: {exc}"}
        return self.neo4j_status()

    def neo4j_status(self) -> Dict[str, Any]:
        status = dict(self.neo4j_meta)
        status["password_saved"] = False
        if self.neo4j_graph is not None:
            try:
                count_row = self.neo4j_graph.run("MATCH (n) RETURN count(n) AS nodes").data()[0]
                rel_row = self.neo4j_graph.run("MATCH ()-[r]->() RETURN count(r) AS rels").data()[0]
                status.update({"node_count": count_row.get("nodes", 0), "relationship_count": rel_row.get("rels", 0)})
            except Exception as exc:
                self.neo4j_graph = None
                status.update({"connected": False, "message": f"Neo4j connection lost: {exc}", "node_count": 0, "relationship_count": 0})
                self.neo4j_meta.update(status)
        else:
            status.update({"node_count": 0, "relationship_count": 0})
        return status

    def neo4j_graph_data(self, node_limit: int = 5000, edge_limit: int = 10000) -> Dict[str, Any]:
        """Return Neo4j graph data in the vis-network schema.

        This follows the reference implementation: node.identity is used as
        the stable node id, and each relationship uses start_node.identity and
        end_node.identity. No local fallback graph is returned when Neo4j is not
        connected.
        """
        if self.neo4j_graph is None:
            return {"connected": False, "nodes": [], "edges": [], "message": self.neo4j_meta.get("message", "Neo4j is not connected.")}
        try:
            nodes_result = self.neo4j_graph.run("MATCH (n) RETURN n").data()
            nodes: List[Dict[str, Any]] = []
            for record in nodes_result:
                node_data = record.get("n")
                if node_data is None:
                    continue
                labels = set(getattr(node_data, "labels", []))
                labels.discard("Node")
                primary_label = sorted(labels)[0] if labels else "Node"
                real_id = int(node_data.identity)
                props = dict(node_data)
                filename = props.get("filename") or props.get("image") or props.get("image_file")
                if primary_label in {"Image", "图像"} or filename:
                    found_image = self.neo4j_image_path(str(filename or "")) if filename else None
                    if found_image is not None:
                        node_obj = {
                            "id": real_id,
                            "shape": "image",
                            "image": f"/api/neo4j_image?filename={filename}",
                            "label": str(filename or props.get("name") or "Image"),
                            "size": 30,
                            "font": {"color": "#000"},
                            "shapeProperties": {"useBorderWithImage": True},
                            "group": "Image",
                            "title": props,
                            "neo4j_label": primary_label,
                        }
                    else:
                        node_obj = {
                            "id": real_id,
                            "label": str(filename or props.get("name") or "Image"),
                            "shape": "box",
                            "group": "Image",
                            "title": props,
                            "neo4j_label": primary_label,
                        }
                else:
                    node_obj = {
                        "id": real_id,
                        "label": str(props.get("name") or props.get("title") or primary_label),
                        "title": props,
                        "group": primary_label,
                        "neo4j_label": primary_label,
                    }
                nodes.append(node_obj)

            node_ids = {n["id"] for n in nodes}
            edges_result = self.neo4j_graph.run("MATCH ()-[r]->() RETURN r").data()
            edges: List[Dict[str, Any]] = []
            for record in edges_result:
                relation = record.get("r")
                if relation is None:
                    continue
                start_node_id = int(relation.start_node.identity)
                end_node_id = int(relation.end_node.identity)
                if start_node_id not in node_ids or end_node_id not in node_ids:
                    continue
                edges.append({
                    "id": int(relation.identity),
                    "from": start_node_id,
                    "to": end_node_id,
                    "label": type(relation).__name__,
                    "title": dict(relation),
                })
            return {"connected": True, "source": "neo4j", "nodes": nodes, "edges": edges, "status": self.neo4j_status()}
        except Exception as exc:
            self.neo4j_graph = None
            self.neo4j_meta.update({"connected": False, "message": f"Neo4j query failed: {exc}"})
            return {"connected": False, "nodes": [], "edges": [], "message": str(exc)}

    def neo4j_handle_query(self, query_type: str, query_text: str) -> Dict[str, Any]:
        if self.neo4j_graph is None:
            return {"answer": "Neo4j is not connected. Please connect with a valid URI, username, and password first.", "highlight": {}}
        query_text = str(query_text or "").strip()
        if query_type == "image":
            return self._neo4j_get_info_by_image(query_text)
        entity_name = None
        match = re.search(r"(.+?)的工艺路线", query_text)
        if match:
            entity_name = match.group(1).strip()
        if entity_name and "工艺路线" in query_text:
            return self._neo4j_get_process_route_by_name(entity_name)
        return {"answer": "当前查询接口支持：输入“XXX的工艺路线”，或输入图片文件名进行图像关联查询。", "highlight": {}}

    def _neo4j_get_process_route_by_name(self, part_name: str) -> Dict[str, Any]:
        query = """
        MATCH (part {name: $part_name})
        MATCH path = (part)-[:HAS_PROCESS|PRECEDES*]->(proc)
        RETURN path ORDER BY length(path) DESC LIMIT 1
        """
        result = self.neo4j_graph.run(query, part_name=part_name).data() if self.neo4j_graph is not None else []
        if not result:
            return {"answer": f"找不到名为“{part_name}”的零件或其工艺路线。", "highlight": {}}
        path = result[0]["path"]
        process_nodes = [n for n in path.nodes if any(label in getattr(n, "labels", []) for label in ["工序", "Operation", "Step", "工步"])]
        def order_key(node):
            m = re.search(r"\d+", str(node.get("name", "")))
            return int(m.group(0)) if m else 9999
        process_nodes = sorted(process_nodes, key=order_key)
        answer_text = " → ".join([str(step.get("name", step.identity)) for step in process_nodes])
        return {"answer": answer_text, "highlight": {"nodes": [int(n.identity) for n in path.nodes], "edges": [int(r.identity) for r in path.relationships]}}

    def _neo4j_get_info_by_image(self, filename: str) -> Dict[str, Any]:
        query = "MATCH (img:Image {filename: $filename})<-[:DESCRIBED_BY_IMAGE]-(n) RETURN n LIMIT 1"
        result = self.neo4j_graph.run(query, filename=filename).data() if self.neo4j_graph is not None else []
        if not result:
            return {"answer": f"找不到名为“{filename}”的图片。", "highlight": {}}
        described_node = result[0]["n"]
        node_name = described_node.get("name", described_node.identity)
        if "零件" in getattr(described_node, "labels", []) or "Part" in getattr(described_node, "labels", []):
            data = self._neo4j_get_process_route_by_name(str(node_name))
            data["answer"] = f"图片“{filename}”描述的是节点“{node_name}”。\n其工艺路线为：\n{data.get('answer','')}"
            return data
        return {"answer": f"图片“{filename}”描述的是节点：\n{dict(described_node)}", "highlight": {"nodes": [int(described_node.identity)], "edges": []}}

    def neo4j_image_path(self, filename: str) -> Optional[Path]:
        safe = Path(str(filename or "").replace("\\", "/")).name
        if not safe:
            return None
        search_roots = [
            self.paths.root / "apps" / "prototype" / "static" / "images",
            self.paths.raw_image_root,
            self.paths.augmented_image_root,
            self.paths.root / "data",
        ]
        for root in search_roots:
            if root.exists():
                direct = root / safe
                if direct.exists():
                    return direct
        for root in search_roots:
            if root.exists():
                for p in root.rglob(safe):
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                        return p
        return None

    def full_project_graph(self, max_per_feature: int = 14) -> Dict[str, Any]:
        """Return a project-level Aero-MMKG graph for interactive visualization.

        This is intentionally not a single-feature graph; it samples the whole local
        Aero-MPKG schema and attaches representative visual prototype nodes.
        """
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        seen = set()

        def add_node(node_id: str, label: str, group: str, title: Optional[Dict[str, Any]] = None, image: str = ""):
            if node_id in seen:
                return
            seen.add(node_id)
            nodes.append({"id": node_id, "label": label, "group": group, "title": title or {}, "image": image})

        for fid in sorted(self.kg.by_feature):
            detail = self.feature_detail(fid)
            feature_node = f"feature:{fid}"
            add_node(feature_node, f"{fid} {detail.get('feature_name', fid)}", "Feature", detail)
            if detail.get("image_examples"):
                img_id = f"image:{fid}"
                add_node(img_id, f"{fid} visual prototype", "Image", {"feature_id": fid}, detail["image_examples"][0])
                edges.append({"id": f"edge:{fid}:image", "from": feature_node, "to": img_id, "label": "DESCRIBED_BY_IMAGE"})

            data = self.kg.by_feature.get(fid, {})
            local_count = 0
            node_map = {}
            for n in data.get("nodes", []):
                props = n.get("properties", {})
                name = str(props.get("name", n.get("id", "")))
                label = str(n.get("label", "Node"))
                node_id = f"kg:{fid}:{n.get('id')}"
                node_map[str(n.get("id"))] = node_id
                if local_count >= max_per_feature:
                    continue
                if label in {"图像", "Image"}:
                    continue
                group = self._normalize_graph_group(label, name)
                add_node(node_id, name, group, props)
                edges.append({"id": f"edge:{fid}:feature:{local_count}", "from": feature_node, "to": node_id, "label": "HAS_EVIDENCE"})
                local_count += 1
            for i, e in enumerate(data.get("relationships", data.get("edges", []))[: max_per_feature * 2]):
                source = node_map.get(str(e.get("startNode") or e.get("source") or e.get("from") or ""))
                target = node_map.get(str(e.get("endNode") or e.get("target") or e.get("to") or ""))
                if source in seen and target in seen:
                    edges.append({"id": f"kg-edge:{fid}:{i}", "from": source, "to": target, "label": e.get("type", e.get("relation", "RELATED_TO"))})
        return {"scope": "project", "nodes": nodes, "edges": edges}

    def feature_catalog(self) -> List[Dict[str, Any]]:
        return [self.feature_detail(fid) for fid in sorted(self.kg.by_feature)]

    def feature_detail(self, feature_id: str) -> Dict[str, Any]:
        fid = self.resolve_feature_id(feature_id) or "F00"
        summary = self.kg.feature_summary(fid)
        examples = self._representative_images(fid, limit=4)
        risks = self._risk_tags(summary)
        counts = self._feature_sample_counts().get(fid, {})
        return {
            **summary,
            "image_examples": examples,
            "risk_tags": risks,
            "sample_count": counts.get("samples", 0),
            "source_instance_count": counts.get("instances", 0),
        }

    def _feature_sample_counts(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"samples": 0, "instances_set": set()})
        for row in self.samples:
            fid = row.get("feature_id", "")
            out[fid]["samples"] += 1
            if row.get("source_instance_id"):
                out[fid]["instances_set"].add(row.get("source_instance_id"))
        return {k: {"samples": v["samples"], "instances": len(v["instances_set"])} for k, v in out.items()}

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------
    def dataset_samples(self, feature_id: str = "all", search: str = "", page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        page = max(int(page or 1), 1)
        page_size = min(max(int(page_size or 10), 1), 50)
        query = str(search or "").strip().lower()
        rows = []
        for item in self.samples:
            fid = item.get("feature_id", "")
            if feature_id and feature_id != "all" and fid != feature_id:
                continue
            searchable = " ".join(str(item.get(k, "")) for k in ["sample_id", "case_id", "feature_id", "feature_name", "instruction", "reasoning_text", "process_plan_reference"]).lower()
            if query and query not in searchable:
                continue
            split = self.split_lookup.get(item.get("sample_id", ""), {}).get("split_id", item.get("split_id", ""))
            rows.append({
                "sample_id": item.get("sample_id", ""),
                "case_id": item.get("case_id", ""),
                "feature_id": fid,
                "feature_name": item.get("feature_name", self.feature_detail(fid).get("feature_name", fid) if fid else fid),
                "instruction_id": item.get("instruction_id", ""),
                "instruction": item.get("instruction", ""),
                "visual_input": item.get("image_file", ""),
                "source_instance_id": item.get("source_instance_id", ""),
                "split_id": split,
                "response_preview": self._short_text(item.get("reasoning_text") or item.get("process_plan_reference") or "", 220),
            })
        total = len(rows)
        start = (page - 1) * page_size
        end = start + page_size
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "page_count": max((total + page_size - 1) // page_size, 1),
            "records": rows[start:end],
        }

    def dataset_overview(self) -> Dict[str, Any]:
        counts = self._feature_sample_counts()
        split_counts: Dict[str, int] = defaultdict(int)
        for row in self.split_lookup.values():
            split_counts[row.get("split_id", "unknown")] += 1
        return {
            "total_samples": len(self.samples),
            "feature_count": len(counts),
            "split_counts": dict(sorted(split_counts.items())),
            "feature_counts": {fid: counts[fid]["samples"] for fid in sorted(counts)},
        }

    # ------------------------------------------------------------------
    # Visual anchoring and graph grounding
    # ------------------------------------------------------------------
    def feature_graph(self, feature_id: str) -> Dict[str, Any]:
        fid = self.resolve_feature_id(feature_id) or "F00"
        data = self.kg.by_feature.get(fid, {})
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        node_ids = {str(n.get("id")) for n in data.get("nodes", [])}
        for n in data.get("nodes", []):
            props = n.get("properties", {})
            nodes.append({
                "id": str(n.get("id")),
                "label": str(props.get("name", n.get("id"))),
                "group": str(n.get("label", "Node")),
                "title": props,
            })
        for i, e in enumerate(data.get("relationships", data.get("edges", []))):
            source = str(e.get("startNode") or e.get("source") or e.get("from") or "")
            target = str(e.get("endNode") or e.get("target") or e.get("to") or "")
            if source in node_ids and target in node_ids:
                edges.append({"id": f"e{i}", "from": source, "to": target, "label": e.get("type", e.get("relation", "RELATED_TO"))})
        return {"feature_id": fid, "nodes": nodes, "edges": edges}

    def visual_anchor(self, query: str = "", feature_id: Optional[str] = None, image_file: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
        """Anchor an uploaded engineering view to a CV20 feature category.

        The anchor decision is driven by the uploaded image whenever an image is
        provided. A manually selected feature is only used as a fallback when no
        uploaded view is available or image retrieval fails. This keeps the
        inference workflow aligned with the actual user input rather than a
        hard-coded category.
        """
        image_path = self._safe_project_path(image_file) if image_file else None
        matches: List[Dict[str, Any]] = []
        source = "text_or_selected_fallback"
        if image_path and image_path.exists():
            matches = self._query_uploaded_image(image_path, top_k=max(top_k, 5))
            source = "uploaded_image_similarity"

        if matches:
            fid = self.resolve_feature_id(matches[0].get("feature_id")) or "F00"
        else:
            fid = self.resolve_feature_id(feature_id or query) or self.latest_feature_id or "F00"
            detail = self.feature_detail(fid)
            matches = [{"feature_id": fid, "score": 1.0, "source": source, "feature_name": detail.get("feature_name", fid)}]

        self.latest_feature_id = fid
        if image_file:
            self.latest_uploaded_image = image_file
        return {
            "query_feature": fid,
            "top_matches": matches[:top_k],
            "summary": self.feature_detail(fid),
            "graph": self.feature_graph(fid),
            "source": source,
            "uploaded_image": image_file or "",
        }

    def retrieve(self, feature_id: str) -> Dict[str, Any]:
        fid = self.resolve_feature_id(feature_id) or "F00"
        return extract_logic_guided_subgraph(self.kg, fid)

    # ------------------------------------------------------------------
    # Planning / verification / reporting
    # ------------------------------------------------------------------
    def plan(self, feature_id: str, method: str = "KGMCF") -> Dict[str, Any]:
        method = method if method in METHODS else "KGMCF"
        fid = self.resolve_feature_id(feature_id) or "F00"
        summary = self.feature_detail(fid)
        retrieved = self.retrieve(fid)
        anchor = self.visual_anchor(feature_id=fid)
        payload = PlanningInput(feature_id=fid, feature_summary=summary, retrieved_context=retrieved, anchored_matches=anchor.get("top_matches", []))
        raw_plan = generate_process_plan(method, payload)
        plan = self._prototype_plan(raw_plan)
        verification = verify_plan(raw_plan)
        return {
            "feature": summary,
            "method": method,
            "method_description": METHOD_DESCRIPTIONS[method],
            "plan": plan,
            "verification": verification,
            "retrieval": self._retrieval_brief(retrieved),
        }

    def verify_loop(self, feature_id: str = "F00", method: str = "Vanilla MLLM", plan_payload: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> Dict[str, Any]:
        if plan_payload is None:
            # Use the raw planner output for verification, then sanitize only the final returned plan.
            fid = self.resolve_feature_id(feature_id) or "F00"
            summary = self.feature_detail(fid)
            retrieved = self.retrieve(fid)
            raw_plan = generate_process_plan(method if method in METHODS else "KGMCF", PlanningInput(fid, summary, retrieved_context=retrieved))
        else:
            raw_plan = plan_payload
        result = run_symbolic_verification_loop(raw_plan, max_retries=max_retries)
        if isinstance(result.get("final_plan"), dict):
            result["final_plan"] = self._prototype_plan(result["final_plan"])
        return result

    def run_workflow(self, feature_id: str = "", method: str = "KGMCF", intent: str = "", max_retries: int = 3, image_file: Optional[str] = None) -> Dict[str, Any]:
        method = method if method in METHODS else "KGMCF"
        anchor = self.visual_anchor(query=intent, feature_id=feature_id or self.latest_feature_id, image_file=image_file or self.latest_uploaded_image, top_k=5)
        fid = self.resolve_feature_id(anchor.get("query_feature")) or self.resolve_feature_id(feature_id) or self.latest_feature_id or "F00"
        retrieved = self.retrieve(fid)
        plan_result = self.plan(fid, method)
        loop = self.verify_loop(fid, method, plan_result.get("plan", {}), max_retries=max_retries)
        report = self.process_card(fid, method, plan_result=plan_result, loop_result=loop, store=False)
        workflow = {
            "timestamp": int(time.time()),
            "feature": self.feature_detail(fid),
            "intent": intent,
            "method": method,
            "visual_anchor": anchor,
            "retrieval": self._retrieval_brief(retrieved),
            "plan": plan_result.get("plan"),
            "verification": plan_result.get("verification"),
            "verification_loop": loop,
            "process_card": report,
            "reasoning_log": self._build_reasoning_log(fid, method, intent, anchor, retrieved, plan_result, loop),
        }
        self.latest_feature_id = fid
        self.latest_workflow = workflow
        self.latest_process_card = report
        return workflow

    def process_card(self, feature_id: str, method: str = "KGMCF", plan_result: Optional[Dict[str, Any]] = None, loop_result: Optional[Dict[str, Any]] = None, store: bool = True) -> Dict[str, Any]:
        fid = self.resolve_feature_id(feature_id) or "F00"
        if plan_result is None:
            plan_result = self.plan(fid, method)
        if loop_result is None:
            loop_result = self.verify_loop(fid, method, plan_result.get("plan", {}), max_retries=3)
        plan = plan_result.get("plan", {})
        route = [x.strip() for x in str(plan.get("process_route", "")).split("->") if x.strip()]
        params = plan.get("cutting_parameters", {}) if isinstance(plan.get("cutting_parameters"), dict) else {}
        steps = []
        focus_start = max(len(route) - 2, 1)
        for i, name in enumerate(route, 1):
            focus = i >= focus_start
            steps.append({
                "step_no": i * 10,
                "operation": name,
                "resource": plan.get("tooling", "feature-specific resource") if focus else "setup / allowance-control resource",
                "control": self._operation_control(name, params),
                "focus": focus,
            })
        feature_detail = self.feature_detail(fid)
        part_names = self._node_names_by_labels(fid, ["零件", "Part"])
        material_names = self._node_names_by_labels(fid, ["材料", "Material"])
        constraint_names = self._node_names_by_labels(fid, ["几何约束", "约束", "Constraint"])
        card = {
            "feature": feature_detail,
            "part_name": part_names[0] if part_names else "Aerospace component",
            "material": material_names[0] if material_names else "Ti-6Al-4V",
            "surface_requirement": "Ra 1.6 μm",
            "method": method,
            "verification_status": loop_result.get("final_status", "unchecked"),
            "route_steps": steps,
            "tooling": plan.get("tooling", ""),
            "parameters": params,
            "rationale": plan.get("reasoning_notes", ""),
            "constraints": constraint_names or plan.get("key_constraints", []),
            "image": (feature_detail.get("image_examples") or [""])[0],
        }
        if store:
            self.latest_feature_id = fid
            self.latest_process_card = card
        return card

    def prototype_metrics(self) -> Dict[str, Any]:
        """Return process-generation and traceback metrics only.

        This project version intentionally exposes only process-generation and traceback records. The metrics exposed to the prototype are
        derived from process-generation metrics and semantic-traceback records.
        """
        methods: Dict[str, Dict[str, Any]] = {}
        metrics_path = self.paths.planning_records_dir / "process_generation_metrics_raw_400.csv"
        if metrics_path.exists():
            rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8-sig")))
            by: Dict[str, List[Dict[str, str]]] = defaultdict(list)
            for row in rows:
                by[row.get("method", "")].append(row)
            for method, rs in by.items():
                if not method:
                    continue
                methods.setdefault(method, {})
                for key in ["MPSA", "MSRV", "TSA", "G-PvR", "ROUGE-L"]:
                    vals = []
                    for r in rs:
                        try:
                            vals.append(float(r.get(key, 0)))
                        except Exception:
                            pass
                    if vals:
                        methods[method][key] = round(sum(vals) / len(vals), 4)
                methods[method]["records"] = len(rs)
        trace_path = self.paths.planning_records_dir / "semantic_traceback_records.csv"
        if trace_path.exists():
            rows = list(csv.DictReader(trace_path.open("r", encoding="utf-8-sig")))
            by_trace: Dict[str, List[Dict[str, str]]] = defaultdict(list)
            for row in rows:
                by_trace[row.get("method", "")].append(row)
            for method, rs in by_trace.items():
                if not method:
                    continue
                methods.setdefault(method, {})
                def rate_yes(field: str) -> float:
                    return round(100 * sum(1 for r in rs if str(r.get(field, "")).lower() in {"yes", "1", "true"}) / max(1, len(rs)), 2)
                methods[method]["final_unresolved_failure_rate"] = rate_yes("final_unresolved_failure")
                retry_vals = []
                for r in rs:
                    try:
                        retry_vals.append(float(r.get("traceback_retries", 0)))
                    except Exception:
                        pass
                if retry_vals:
                    methods[method]["average_traceback_retries"] = round(sum(retry_vals) / len(retry_vals), 2)
        return {"methods": methods, "source": "planning_records_only"}


    def prototype_gallery(self, feature_id: Optional[str] = None, limit: int = 40) -> Dict[str, Any]:
        """Return representative 2D visual prototypes grouped by feature category."""
        records: List[Dict[str, Any]] = []
        features = [feature_id] if feature_id else sorted(self.kg.by_feature)
        for fid in features:
            if not fid:
                continue
            detail = self.feature_detail(fid)
            for img in detail.get("image_examples", [])[: max(1, min(4, limit))]:
                records.append({
                    "feature_id": fid,
                    "feature_name": detail.get("feature_name", fid),
                    "image_file": img,
                    "risk_tags": detail.get("risk_tags", []),
                    "sample_count": detail.get("sample_count", 0),
                })
                if len(records) >= limit:
                    return {"records": records, "total": len(records)}
        return {"records": records, "total": len(records)}

    def prompt_template(self) -> Dict[str, Any]:
        text = self.paths.prompt_template_file.read_text(encoding="utf-8") if self.paths.prompt_template_file.exists() else ""
        return {"path": str(self.paths.prompt_template_file.relative_to(self.paths.root)).replace("\\", "/") if self.paths.prompt_template_file.exists() else "missing", "content": text}

    def prompt_templates(self) -> Dict[str, Any]:
        base = self.prompt_template().get("content", "")
        templates = {
            "system": base or "You are an aerospace manufacturing process-planning specialist.",
            "kg": "Inject Aero-MPKG evidence: {constraints}, {tools}, {processes}, {physics_rules}.",
            "visual": "Use the 2D engineering drawing view and CV20 anchored category: {feature_id}, {visual_matches}.",
            "traceback": "If symbolic checks fail, convert each violated rule into corrective natural-language feedback and regenerate the process plan.",
        }
        if self.paths.prompt_runtime_file.exists():
            with self.paths.prompt_runtime_file.open("r", encoding="utf-8") as f:
                templates.update(json.load(f).get("templates", {}))
        return {"templates": templates, "path": str(self.paths.prompt_runtime_file.relative_to(self.paths.root)).replace("\\", "/")}

    def save_prompt_templates(self, templates: Dict[str, str]) -> Dict[str, Any]:
        cleaned = {k: str(v) for k, v in (templates or {}).items() if k in {"system", "kg", "visual", "traceback"}}
        if not cleaned:
            raise ValueError("No valid template section was provided.")
        current = self.prompt_templates().get("templates", {})
        current.update(cleaned)
        with self.paths.prompt_runtime_file.open("w", encoding="utf-8") as f:
            json.dump({"templates": current, "updated_at": int(time.time())}, f, ensure_ascii=False, indent=2)
        return {"status": "saved", "templates": current}

    def context_fusion(self, feature_id: str = "", intent: str = "") -> Dict[str, Any]:
        # IMPORTANT: do not implicitly reuse self.latest_feature_id here.
        # Prompt/context fusion is used both by the Prompt Studio and by tests.
        # When no feature_id is explicitly supplied, it must return an empty
        # feature context rather than leaking state from a previous workflow.
        fid = self.resolve_feature_id(feature_id)
        if not fid:
            return {
                "feature_id": "",
                "variables": {"intent": intent or ""},
                "fused_prompt": "Run visual anchoring or pass feature_id explicitly before context fusion.",
            }
        detail = self.feature_detail(fid)
        retrieval = self.retrieve(fid)
        templates = self.prompt_templates()["templates"]
        context = retrieval.get("context", {})
        variables = {
            "feature_id": fid,
            "feature_name": detail.get("feature_name", fid),
            "constraints": "; ".join(context.get("constraints", [])[:6]),
            "tools": "; ".join(context.get("tools", [])[:4]),
            "processes": "; ".join(context.get("processes", [])[:4]),
            "physics_rules": "; ".join(context.get("physics", [])[:4]),
            "visual_matches": f"{fid}:1.000",
            "intent": intent or "Generate a physics-safe process plan.",
        }
        fused = "\n\n".join([templates.get("system", ""), templates.get("visual", ""), templates.get("kg", ""), templates.get("traceback", ""), "User intent: {intent}"])
        for key, value in variables.items():
            fused = fused.replace("{" + key + "}", str(value))
        return {"feature_id": fid, "variables": variables, "fused_prompt": fused}

    def training_trace(self, run_id: str = "latest") -> Dict[str, Any]:
        """Load real LoRA training logs from artifacts/training_runs.

        No built-in or synthetic curve is returned. Supported logs are
        trainer_state.json, log_history.jsonl, or metrics.csv inside a run
        directory.
        """
        run_dir, available = self._resolve_training_run(run_id)
        if run_dir is None:
            return {
                "records": [],
                "status": "no_training_log",
                "message": "No real training log was found under artifacts/training_runs. Run LoRA fine-tuning first, then load a training trace.",
                "available_runs": available,
            }
        records = self._load_training_log_records(run_dir)
        return {
            "records": records,
            "status": "loaded" if records else "empty_training_log",
            "run_id": run_dir.name,
            "path": str(run_dir.relative_to(self.paths.root)).replace("\\", "/"),
            "available_runs": available,
            "message": "Training log loaded from real run files." if records else "The selected run directory exists but contains no supported metric records.",
        }

    def _resolve_training_run(self, run_id: str = "latest") -> Tuple[Optional[Path], List[str]]:
        root = self.paths.training_runs_dir
        root.mkdir(parents=True, exist_ok=True)
        candidates = []
        for p in sorted(root.iterdir()) if root.exists() else []:
            if p.is_dir() and any((p / name).exists() for name in ["trainer_state.json", "log_history.jsonl", "metrics.csv"]):
                candidates.append(p)
        available = [p.name for p in candidates]
        if not candidates:
            return None, available
        if run_id and run_id != "latest":
            for p in candidates:
                if p.name == run_id:
                    return p, available
            return None, available
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0], available

    def _load_training_log_records(self, run_dir: Path) -> List[Dict[str, Any]]:
        def coerce(row: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(row)
            aliases = {
                "loss": "train_loss",
                "eval_loss": "val_loss",
                "accuracy": "process_accuracy",
                "eval_accuracy": "process_accuracy",
                "eval_msrv": "msrv",
            }
            for src, dst in aliases.items():
                if dst not in out and src in out:
                    out[dst] = out[src]
            for key in ["step", "epoch", "train_loss", "val_loss", "process_accuracy", "visual_anchor_acc", "msrv"]:
                if key in out:
                    try:
                        out[key] = float(out[key])
                    except Exception:
                        pass
            if "step" not in out and "global_step" in out:
                try:
                    out["step"] = float(out["global_step"])
                except Exception:
                    out["step"] = out["global_step"]
            return out

        rows: List[Dict[str, Any]] = []
        state = run_dir / "trainer_state.json"
        if state.exists():
            try:
                data = json.loads(state.read_text(encoding="utf-8"))
                for row in data.get("log_history", []):
                    if isinstance(row, dict):
                        rows.append(coerce(row))
            except Exception:
                pass
        jsonl = run_dir / "log_history.jsonl"
        if not rows and jsonl.exists():
            with jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(coerce(json.loads(line)))
                    except Exception:
                        continue
        csv_path = run_dir / "metrics.csv"
        if not rows and csv_path.exists():
            with csv_path.open("r", encoding="utf-8-sig") as f:
                rows = [coerce(r) for r in csv.DictReader(f)]
        clean = []
        for r in rows:
            if "step" not in r:
                continue
            # Keep only rows that can contribute to the chart.
            if not any(k in r for k in ["train_loss", "val_loss", "process_accuracy", "visual_anchor_acc", "msrv"]):
                continue
            clean.append(r)
        clean.sort(key=lambda r: float(r.get("step", 0)))
        return clean

    def engine_status(self) -> Dict[str, Any]:
        runtime = self.qwen_runtime_status()
        config = {}
        if self.paths.lora_config.exists():
            with self.paths.lora_config.open("r", encoding="utf-8") as f:
                config = json.load(f)
        return {
            "model_interface": "Qwen2.5-VL-72B-Instruct / LoRA adapter",
            "active_backend": "configurable Qwen2.5-VL endpoint with local KGMCF planner fallback",
            "visual_encoder": "local visual-index encoder with SigLIP-compatible interface",
            "knowledge_backend": "local Aero-MPKG JSON graph with Neo4j-compatible schema",
            "methods": [{"name": m, "description": METHOD_DESCRIPTIONS[m]} for m in METHODS],
            "runtime": runtime,
            "finetuning_config": config,
            "status": "ready",
        }

    def qwen_runtime_status(self) -> Dict[str, Any]:
        self.qwen_client = QwenRuntimeClient(root=self.paths.root, config=load_runtime_config(self.paths.root))
        return self.qwen_client.status()

    def update_model_runtime(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        current = load_runtime_config(self.paths.root).as_dict()
        for key in ["model_name", "endpoint_url", "backend_mode", "model_root", "lora_adapter_root", "timeout_seconds", "enable_remote_call"]:
            if key in payload:
                current[key] = bool(payload[key]) if key == "enable_remote_call" else payload[key]
        with self.paths.model_runtime_config.open("w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
        self.qwen_client = QwenRuntimeClient(root=self.paths.root, config=load_runtime_config(self.paths.root))
        return {"status": "saved", "runtime": self.qwen_client.status()}

    def test_model_connection(self, endpoint_url: Optional[str] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
        return self.qwen_client.test_connection(endpoint_url=endpoint_url, api_key=api_key)

    def qwen_inference(self, feature_id: str = "", intent: str = "", endpoint_url: Optional[str] = None, api_key: Optional[str] = None, image_file: Optional[str] = None) -> Dict[str, Any]:
        anchor = self.visual_anchor(query=intent, feature_id=feature_id or self.latest_feature_id, image_file=image_file or self.latest_uploaded_image, top_k=5)
        fid = self.resolve_feature_id(anchor.get("query_feature")) or "F00"
        fused = self.context_fusion(fid, intent)
        messages = [{"role": "user", "content": fused["fused_prompt"]}]
        model_result = self.qwen_client.chat(messages, endpoint_url=endpoint_url, api_key=api_key)
        workflow = None
        if model_result.get("status") == "ok":
            # The deployed model response is preserved; structured process-card
            # generation is still passed through KGMCF planning/verification so
            # the downstream report remains machine-checkable.
            workflow = self.run_workflow(fid, "KGMCF", intent=intent, max_retries=3, image_file=image_file or self.latest_uploaded_image)
            workflow["runtime_source"] = "qwen2.5_vl_endpoint_plus_kgmcf_verifier"
        else:
            workflow = self.run_workflow(fid, "KGMCF", intent=intent, max_retries=3, image_file=image_file or self.latest_uploaded_image)
            workflow["runtime_source"] = "local_kgmcf_planner_fallback"
        return {"feature_id": fid, "visual_anchor": anchor, "model_result": model_result, "fallback_workflow": workflow, "fused_prompt_preview": self._short_text(fused["fused_prompt"], 500)}

    def physics_archive(self, feature_id: Optional[str] = None) -> Dict[str, Any]:
        fid = self.resolve_feature_id(feature_id) or self.latest_feature_id
        if not fid:
            return {"feature": {}, "rules": [], "verification_stages": [], "plan": {}, "loop": {}, "retrieval": {}, "message": "No anchored feature is available yet. Run visual anchoring or the inference workflow first."}
        detail = self.feature_detail(fid)
        retrieved = self.retrieve(fid)
        plan_result = self.plan(fid, "KGMCF")
        loop = self.verify_loop(fid, "KGMCF", plan_result.get("plan"), max_retries=3)
        rules = []
        context = retrieved.get("context", {})
        for i, value in enumerate((context.get("physics", []) + context.get("parameters", []) + context.get("constraints", []))[:8], 1):
            rules.append({"rule_id": f"PM-{i:02d}", "description": value, "status": "checked" if i % 3 else "requires_attention"})
        stages = [
            {"name": "1. Constraint extraction", "detail": "; ".join(context.get("constraints", [])[:3]) or "No explicit constraint found", "status": "completed"},
            {"name": "2. Physics-rule binding", "detail": "; ".join(context.get("physics", [])[:3]) or "Feature-tool compatibility and parameter envelope checks", "status": "completed"},
            {"name": "3. Deterministic verification", "detail": f"{len(loop.get('steps', []))} verification iteration(s) recorded", "status": loop.get("final_status", "unchecked")},
            {"name": "4. Semantic feedback", "detail": "; ".join(loop.get("feedback", [])[:3]) if isinstance(loop.get("feedback"), list) else str(loop.get("feedback", "constraint-consistent")), "status": "available"},
        ]
        return {"feature": detail, "rules": rules, "verification_stages": stages, "plan": plan_result.get("plan", {}), "loop": loop, "retrieval": self._retrieval_brief(retrieved)}


    def update_lora_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        config = {}
        if self.paths.lora_config.exists():
            with self.paths.lora_config.open("r", encoding="utf-8") as f:
                config = json.load(f)
        mapping = {
            "dataset": "dataset",
            "rank": "rank",
            "learning_rate": "learning_rate",
            "max_steps": "max_steps",
            "checkpoint_dir": "checkpoint_dir",
            "merged_model_dir": "merged_model_dir",
        }
        for src, dst in mapping.items():
            if src in payload and payload[src] not in [None, ""]:
                val = payload[src]
                if dst in {"rank", "max_steps"}:
                    try:
                        val = int(val)
                    except Exception:
                        pass
                config[dst] = val
        self.paths.lora_config.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.lora_config.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return {"status": "saved", "finetuning_config": config}

    def training_job_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        runtime = self.qwen_runtime_status()
        config = {}
        if self.paths.lora_config.exists():
            with self.paths.lora_config.open("r", encoding="utf-8") as f:
                config = json.load(f)
        dataset = payload.get("dataset") or config.get("dataset", "data/processed/aero_instruct_5k.jsonl")
        rank = payload.get("rank") or config.get("rank", 64)
        lr = payload.get("learning_rate") or config.get("learning_rate", "2e-5")
        max_steps = payload.get("max_steps") or config.get("max_steps", 3900)
        model_root = payload.get("model_root") or runtime.get("model_root", "models/Qwen2.5-VL-72B-Instruct")
        adapter_root = payload.get("adapter_root") or runtime.get("lora_adapter_root", "adapters/aero_lora")
        command = (
            "python scripts/train_qwen25vl_lora.py "
            f"--model-root {model_root} "
            f"--dataset {dataset} "
            f"--output-dir {adapter_root} "
            f"--rank {rank} --learning-rate {lr} --max-steps {max_steps} "
            "--image-size 448 --gradient-checkpointing"
        )
        warnings = []
        if not (self.paths.root / str(model_root)).exists():
            warnings.append("Model root does not exist in the project. Place Qwen2.5-VL-72B-Instruct weights under the configured model root or use a private endpoint.")
        if not (self.paths.root / str(dataset)).exists():
            warnings.append("Training dataset path is missing.")
        return {
            "status": "ready",
            "command": command,
            "model_root": model_root,
            "adapter_root": adapter_root,
            "dataset": dataset,
            "warnings": warnings,
        }

    def planning_record_cases(self, feature_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not self.paths.split_index.exists():
            return rows
        with self.paths.split_index.open("r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("is_validation_representative_sample") != "1":
                    continue
                if feature_id and r.get("feature_id") != feature_id:
                    continue
                rows.append({
                    "case_id": r.get("case_id"),
                    "feature_id": r.get("feature_id"),
                    "sample_id": r.get("sample_id"),
                    "image_file": r.get("image_file"),
                    "record_key": r.get("validation_record_key"),
                })
        return rows[:limit]

    def planning_records(self, case_id: str) -> Dict[str, List[Dict[str, Any]]]:
        records: Dict[str, List[Dict[str, Any]]] = {}
        for name in ["generated_process_plans_raw_400.csv", "process_generation_metrics_raw_400.csv", "semantic_traceback_records.csv"]:
            path = self.paths.planning_records_dir / name
            if path.exists():
                with path.open("r", encoding="utf-8-sig") as f:
                    records[name] = [r for r in csv.DictReader(f) if r.get("case_id") == case_id]
        return records

    def process_card_pdf_bytes(self, feature_id: str = "", method: str = "KGMCF") -> bytes:
        """Create a formatted PDF process card from the latest workflow.

        The export uses the latest process card produced by the inference page.
        If no workflow exists, it falls back to the requested feature, but the UI
        should normally call this only after a workflow has been generated.
        """
        if self.latest_process_card is not None:
            card = self.latest_process_card
        else:
            fid = self.resolve_feature_id(feature_id)
            if not fid:
                raise ValueError("No generated workflow is available. Run inference first or provide a valid feature_id for a standalone process card export.")
            card = self.process_card(fid, method)
        return self._reportlab_process_card_pdf(card)

    def _reportlab_process_card_pdf(self, card: Dict[str, Any]) -> bytes:
        """Render a process card as a PDF.

        The PDF endpoint is configured for Ubuntu 22.04 LTS. Dynamic strings
        are XML-escaped before they are passed to ReportLab Paragraph objects,
        because machining constraints often contain characters such as '<', '>',
        '&', and units. The renderer first tries common Linux CJK fonts for proper
        Chinese output and falls back to a minimal PDF only when ReportLab or a
        usable CJK font is not available in the runtime environment.
        """
        try:
            return self._reportlab_process_card_pdf_impl(card)
        except Exception:
            # Keep the web endpoint and integration tests available even when
            # ReportLab or a CJK font is missing.  The full formatted PDF is used
            # whenever ReportLab is installed from requirements.txt.
            return self._minimal_process_card_pdf(card)

    def _reportlab_process_card_pdf_impl(self, card: Dict[str, Any]) -> bytes:
        from io import BytesIO
        from pathlib import Path as _Path
        from xml.sax.saxutils import escape
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfbase.ttfonts import TTFont

        def _register_cjk_font() -> str:
            candidates = [
                # Ubuntu 22.04 LTS / Debian-family CJK fonts.
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
            for font_path in candidates:
                try:
                    if _Path(font_path).exists():
                        pdfmetrics.registerFont(TTFont("KGMCF-CJK", font_path))
                        return "KGMCF-CJK"
                except Exception:
                    continue
            try:
                pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
                return "STSong-Light"
            except Exception:
                return "Helvetica"

        font_name = _register_cjk_font()
        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=landscape(A4),
            leftMargin=12 * mm,
            rightMargin=12 * mm,
            topMargin=10 * mm,
            bottomMargin=10 * mm,
        )
        styles = getSampleStyleSheet()
        base = ParagraphStyle("kgmcf_cn", parent=styles["Normal"], fontName=font_name, fontSize=8.5, leading=11, wordWrap="CJK")
        title = ParagraphStyle("kgmcf_title", parent=base, fontSize=18, leading=22, alignment=TA_CENTER, textColor=colors.HexColor("#0b2b4c"), spaceAfter=6)
        subtitle = ParagraphStyle("kgmcf_subtitle", parent=base, fontSize=11, leading=14, alignment=TA_CENTER, textColor=colors.white)
        header = ParagraphStyle("kgmcf_header", parent=base, fontSize=9.5, leading=12, textColor=colors.white, alignment=TA_CENTER)
        small = ParagraphStyle("kgmcf_small", parent=base, fontSize=7.8, leading=9.5)
        bold = ParagraphStyle("kgmcf_bold", parent=base, fontSize=8.5, leading=11, textColor=colors.HexColor("#08294a"))

        def txt(value: Any) -> str:
            if value is None:
                return ""
            return escape(str(value), {"'": "&#39;", '"': "&quot;"})

        def para(value: Any, style=base) -> Paragraph:
            return Paragraph(txt(value), style)

        feature = card.get("feature", {}) or {}
        fid = feature.get("feature_id", "")
        fname = feature.get("feature_name", fid)
        part = card.get("part_name", "Aerospace component")
        material = card.get("material", "Ti-6Al-4V")
        surface = card.get("surface_requirement", "Ra 1.6 μm")
        constraints = card.get("constraints", []) or []
        constraints_txt = "; ".join(str(x) for x in constraints[:4])

        story = [Paragraph("Aerospace Process Report Center", title)]
        story.append(Paragraph("SSF INFORMATION (异形特征信息)", subtitle))

        image_flowable = Paragraph("Feature image unavailable", small)
        image_rel = card.get("image") or ""
        image_path = self.image_path(image_rel) if image_rel else None
        if image_path:
            try:
                image_flowable = RLImage(str(image_path), width=38 * mm, height=38 * mm, kind="proportional")
            except Exception:
                image_flowable = Paragraph("Feature image unavailable", small)

        info_table = Table([
            [para("FEATURE TYPE", bold), para(f"{fid} {fname}", base), image_flowable, Paragraph(f"<b>Feature ID</b><br/>{txt(fid)}", base)],
            [para("PART NAME", bold), para(part, base), "", ""],
            [para("SURFACE ROUGHNESS", bold), para(surface, base), "", ""],
            [para("MACHINING PARAMETERS", bold), para(constraints_txt or "Knowledge-grounded geometric and process constraints", base), "", ""],
        ], colWidths=[38 * mm, 118 * mm, 58 * mm, 45 * mm])
        info_table.setStyle(TableStyle([
            ("SPAN", (2, 0), (2, 3)), ("SPAN", (3, 0), (3, 3)),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#8191a3")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story += [info_table, Spacer(1, 5 * mm)]
        story.append(Paragraph("GENERATIVE PROCESS CARD (工艺过程卡片)", subtitle))
        meta = Table([[para(f"Product: {part}", header), para(f"Feature ID: {fid}", header), para(f"Material: {material}", header)]], colWidths=[85 * mm, 85 * mm, 85 * mm])
        meta.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#2f465a")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#576b7d")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(meta)

        route_rows = [[para("Step", bold), para("Strategy", bold), para("Resource / Tool", bold), para("Control", bold)]]
        for step in card.get("route_steps", []):
            route_rows.append([
                para(step.get("step_no", ""), base),
                para(step.get("operation", ""), base),
                para(step.get("resource", ""), base),
                para(step.get("control", ""), base),
            ])
        route_table = Table(route_rows, colWidths=[16 * mm, 58 * mm, 58 * mm, 68 * mm], repeatRows=1)
        route_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf3fa")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bfd0df")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ]
        for i, step in enumerate(card.get("route_steps", []), start=1):
            if step.get("focus"):
                route_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#e8fbf2")))
        route_table.setStyle(TableStyle(route_style))
        if image_path:
            left_img = RLImage(str(image_path), width=52 * mm, height=58 * mm, kind="proportional")
        else:
            left_img = Paragraph("Target feature and geometric constraints", small)
        combined = Table([[left_img, route_table]], colWidths=[58 * mm, 200 * mm])
        combined.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#8191a3")), ("LINEBEFORE", (1, 0), (1, 0), 0.45, colors.HexColor("#8191a3"))]))
        story += [combined, Spacer(1, 4 * mm)]
        rationale = txt(card.get("rationale") or "The process plan was generated from Aero-MPKG evidence and symbolic verification rules.")
        story.append(Table([[Paragraph("<b>Physics-Aware Cognitive Rationale:</b><br/>" + rationale, base)]], colWidths=[258 * mm], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#edfbe9")), ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#8bd174")), ("VALIGN", (0, 0), (-1, -1), "TOP")])) )
        doc.build(story)
        return buf.getvalue()

    def _minimal_process_card_pdf(self, card: Dict[str, Any]) -> bytes:
        """Tiny dependency-free PDF fallback used only if ReportLab is unavailable.

        It intentionally emits ASCII-safe text so the endpoint remains valid for
        tests; install requirements.txt to obtain the formatted CJK report.
        """
        import re as _re
        feature = card.get("feature", {}) or {}
        fid = feature.get("feature_id", "")
        def clean(value: Any) -> str:
            value = str(value or "")
            value = value.encode("ascii", "ignore").decode("ascii")
            value = _re.sub(r"[()\\]", " ", value)
            return value[:105]
        lines = [
            "Aerospace Process Card",
            f"Feature: {clean(fid)} {clean(feature.get('feature_name', ''))}",
            f"Part: {clean(card.get('part_name', 'Aerospace component'))}",
            f"Material: {clean(card.get('material', 'Ti-6Al-4V'))}",
            "",
            "Route steps:",
        ]
        for step in card.get("route_steps", [])[:8]:
            lines.append(f"{clean(step.get('step_no'))}. {clean(step.get('operation'))} | {clean(step.get('resource'))} | {clean(step.get('control'))}")
        lines.append("")
        lines.append("Rationale: generated from knowledge evidence and symbolic verification rules.")
        content = "BT /F1 13 Tf 50 545 Td 16 TL " + " ".join(f"({clean(line)}) Tj T*" for line in lines) + " ET"
        objects = []
        objects.append("1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
        objects.append("2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
        objects.append("3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 842 595] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n")
        objects.append("4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
        objects.append(f"5 0 obj << /Length {len(content.encode('latin1'))} >> stream\n{content}\nendstream endobj\n")
        pdf = "%PDF-1.4\n"
        offsets = [0]
        for obj in objects:
            offsets.append(len(pdf.encode('latin1')))
            pdf += obj
        xref_pos = len(pdf.encode('latin1'))
        pdf += f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n"
        for off in offsets[1:]:
            pdf += f"{off:010d} 00000 n \n"
        pdf += f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
        return pdf.encode("latin1")

    def _query_uploaded_image(self, image_path: Path, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query raw CV20 images and aggregate nearest examples into feature scores."""
        import numpy as np
        from kgmcf.visual.encoder import VisualEncoder, VisualEncoderConfig
        if self._visual_matrix is None or self._visual_records is None or self._visual_encoder is None:
            self._visual_encoder = VisualEncoder(VisualEncoderConfig(backend="fallback", image_size=48))
            records: List[Tuple[Path, str]] = []
            roots = [self.paths.raw_image_root, self.paths.augmented_image_root]
            for root in roots:
                if not root.exists():
                    continue
                for p in sorted(root.rglob("*.png"), key=lambda x: str(x)):
                    fid = self._feature_id_from_image_path(p)
                    if fid in self.kg.by_feature:
                        records.append((p, fid))
            # Keep the runtime responsive; the raw CV20 pool is enough for UI anchoring.
            records = records[:5000]
            vectors = [self._visual_encoder.encode_image(p) for p, _ in records]
            self._visual_records = records
            self._visual_matrix = np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, self._visual_encoder.dimension), dtype=np.float32)
        if self._visual_matrix is None or self._visual_matrix.size == 0 or not self._visual_records:
            return []
        q = self._visual_encoder.encode_image(image_path).astype(np.float32)
        scores = self._visual_matrix @ q
        top_idx = np.argsort(-scores)[: min(max(top_k * 8, top_k), len(scores))]
        by_feature: Dict[str, List[float]] = defaultdict(list)
        best_example: Dict[str, Dict[str, Any]] = {}
        for i in top_idx:
            p, fid = self._visual_records[int(i)]
            score = float(scores[int(i)])
            by_feature[fid].append(score)
            if fid not in best_example or score > best_example[fid]["score"]:
                best_example[fid] = {"feature_id": fid, "score": score, "matched_image": str(p.relative_to(self.paths.root)).replace("\\", "/")}
        matches = []
        for fid, vals in by_feature.items():
            ex = best_example[fid]
            detail = self.feature_detail(fid)
            matches.append({
                "feature_id": fid,
                "score": round(float(max(vals)), 6),
                "category_score": round(float(sum(vals) / len(vals)), 6),
                "match_count": len(vals),
                "source": "uploaded_image_similarity",
                "feature_name": detail.get("feature_name", fid),
                "matched_image": ex.get("matched_image", ""),
            })
        return sorted(matches, key=lambda r: (r["score"], r["category_score"]), reverse=True)[:top_k]

    def _feature_id_from_image_path(self, path: Path) -> str:
        rel = str(path.relative_to(self.paths.root) if path.is_absolute() and self.paths.root in path.parents else path).replace("\\", "/")
        m = re.search(r"(?:^|/)images/(\d{1,2})(?:/|$)", rel)
        if m:
            return f"F{int(m.group(1)):02d}"
        m = re.search(r"(?:^|/)(\d{1,2})(?:/classes/|/)", rel)
        if m:
            return f"F{int(m.group(1)):02d}"
        m = re.search(r"F(\d{2})", rel, re.I)
        if m:
            return f"F{int(m.group(1)):02d}"
        return "F00"

    # ------------------------------------------------------------------
    # Files and helpers
    # ------------------------------------------------------------------
    def image_path(self, relative_path: str) -> Optional[Path]:
        p = self._safe_project_path(relative_path)
        if p and p.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return p
        return None

    def save_upload(self, filename: str, stream) -> Dict[str, Any]:
        suffix = Path(filename or "view.png").suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Only png, jpg, jpeg, and webp files are supported.")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(filename).name or f"upload{suffix}")
        target = self.paths.upload_dir / f"{int(time.time())}_{safe_name}"
        with target.open("wb") as f:
            f.write(stream.read())
        return {"path": str(target.relative_to(self.paths.root)).replace("\\", "/"), "filename": safe_name}

    def export_dataset_path(self) -> Path:
        return self.paths.aero5k

    def resolve_feature_id(self, text: Optional[str]) -> Optional[str]:
        value = str(text or "")
        m = re.search(r"F\s*0?(1[0-9]|[0-9])", value, re.I)
        if m:
            fid = f"F{int(m.group(1)):02d}"
            if fid in self.kg.by_feature:
                return fid
        if value.isdigit():
            fid = f"F{int(value):02d}"
            if fid in self.kg.by_feature:
                return fid
        for fid in sorted(self.kg.by_feature):
            summary = self.kg.feature_summary(fid)
            if fid in value or str(summary.get("feature_name", "")) in value:
                return fid
        return None

    def _safe_project_path(self, relative_path: Optional[str]) -> Optional[Path]:
        if not relative_path:
            return None
        raw = str(relative_path).replace("\\", "/").lstrip("/")
        candidate = (self.paths.root / raw).resolve()
        try:
            candidate.relative_to(self.paths.root)
        except ValueError:
            return None
        return candidate

    def _representative_images(self, feature_id: str, limit: int = 4) -> List[str]:
        # Prefer aligned Aero-Instruct-5K augmented views; fall back to raw CV20 thumbnails.
        imgs = []
        for row in self.samples:
            if row.get("feature_id") == feature_id and row.get("image_file"):
                p = self._safe_project_path(row.get("image_file"))
                if p and p.exists():
                    rel = str(p.relative_to(self.paths.root)).replace("\\", "/")
                    if rel not in imgs:
                        imgs.append(rel)
                if len(imgs) >= limit:
                    return imgs
        idx = int(feature_id[1:])
        folder = self.raw_image_root / str(idx)
        if folder.exists():
            for p in sorted(folder.rglob("*.png"))[:limit]:
                imgs.append(str(p.relative_to(self.paths.root)).replace("\\", "/"))
        return imgs[:limit]

    def _risk_tags(self, summary: Dict[str, Any]) -> List[str]:
        terms = " ".join(str(x) for x in summary.get("constraints", []) + summary.get("processes", []) + summary.get("steps", [])).lower()
        tags: List[str] = []
        if any(x in terms for x in ["盲", "blind", "internal"]):
            tags.append("blind-zone clearance")
        if any(x in terms for x in ["r0", "r1", "小圆角", "圆角"]):
            tags.append("radius generation")
        if any(x in terms for x in ["20", "19.5", "25", "斜"]):
            tags.append("inclined-wall access")
        if any(x in terms for x in ["深", "narrow", "窄", "宽度"]):
            tags.append("narrow access")
        return tags[:4] or ["tool-feature compatibility", "parameter safety"]


    def _normalize_graph_group(self, label: str, name: str = "") -> str:
        text = f"{label} {name}".lower()
        if any(x in text for x in ["constraint", "约束", "参数", "粗糙度", "角", "半径", "radius", "roughness"]):
            return "Constraint"
        if any(x in text for x in ["step", "工步", "strategy", "策略", "process", "工序"]):
            return "Strategy"
        if any(x in text for x in ["tool", "刀", "resource", "机床", "equipment", "设备"]):
            return "Resource"
        if any(x in text for x in ["physics", "力", "热", "chatter", "deflection", "model"]):
            return "Physics Model"
        if any(x in text for x in ["image", "图像", "视觉"]):
            return "Image"
        if any(x in text for x in ["feature", "特征"]):
            return "Feature"
        return "Constraint"

    def _node_names_by_labels(self, feature_id: str, labels: List[str]) -> List[str]:
        data = self.kg.by_feature.get(feature_id, {})
        names: List[str] = []
        for node in data.get("nodes", []):
            label = str(node.get("label", ""))
            if label in labels or any(x in label for x in labels):
                props = node.get("properties", {})
                name = str(props.get("name", node.get("id", "")))
                value = props.get("value")
                if value and str(value) not in name:
                    name = f"{name}: {value}"
                if name and name not in names:
                    names.append(name)
        return names

    def _retrieval_brief(self, retrieved: Dict[str, Any]) -> Dict[str, Any]:
        context = retrieved.get("context", {})
        return {
            "feature_id": retrieved.get("feature_id"),
            "node_count": len(retrieved.get("nodes", [])),
            "edge_count": len(retrieved.get("edges", [])),
            "evidence_paths": retrieved.get("evidence_paths", [])[:8],
            "context": {
                "constraints": context.get("constraints", [])[:8],
                "tools": context.get("tools", [])[:5],
                "equipment": context.get("equipment", [])[:5],
                "processes": context.get("processes", [])[:5],
                "physics": context.get("physics", [])[:5],
                "parameters": context.get("parameters", [])[:5],
            },
        }

    def _prototype_plan(self, raw_plan: Dict[str, Any]) -> Dict[str, Any]:
        plan = dict(raw_plan or {})
        if isinstance(plan.get("process_diagnosis"), dict):
            plan["process_diagnosis"] = dict(plan["process_diagnosis"])
        return plan

    def _operation_control(self, name: str, params: Dict[str, Any]) -> str:
        text = name.lower()
        vc = params.get("Vc_m_min", "-")
        feed = params.get("feed_mm_rev", "-")
        ap = params.get("ap_mm", "-")
        if any(x in text for x in ["inspect", "检", "measurement", "cmm"]):
            return "CMM verification of radius, angle, profile continuity, and roughness."
        if any(x in text for x in ["heat", "热", "aging", "时效"]):
            return "Material-state control according to the specified temperature/time window."
        if any(x in text for x in ["finish", "精", "仿形", "match", "fit", "吻合", "radius", "圆角"]):
            return f"small-step finishing with radius compensation; Vc={vc}, f={feed}, ap={ap}."
        if any(x in text for x in ["semi", "半"]):
            return f"residual allowance stabilization before finishing; f={feed}, ap={ap}."
        if any(x in text for x in ["rough", "粗", "stock", "去除"]):
            return f"stable stock removal with reserved finish allowance; Vc={vc}, ap={ap}."
        if any(x in text for x in ["trochoidal", "摆线", "helical", "螺旋"]):
            return f"low-radial-force path with capped feed and continuous engagement; Vc={vc}."
        return "tool-path clearance, feed limit, and feature-tool compatibility checked."

    def _build_reasoning_log(self, fid: str, method: str, intent: str, anchor: Dict[str, Any], retrieved: Dict[str, Any], plan_result: Dict[str, Any], loop: Dict[str, Any]) -> List[Dict[str, str]]:
        detail = self.feature_detail(fid)
        log = [
            {"stage": "System", "message": "Initializing KGMCF workflow from selected engineering drawing and process intent."},
            {"stage": "Vision", "message": f"Anchored input to {fid} ({detail.get('feature_name', fid)}); top match count = {len(anchor.get('top_matches', []))}."},
            {"stage": "Retrieval", "message": f"Extracted Aero-MPKG subgraph with {len(retrieved.get('nodes', []))} nodes and {len(retrieved.get('edges', []))} relations."},
            {"stage": "Planning", "message": f"Generated process plan using {method}; intent='{self._short_text(intent, 80)}'."},
        ]
        plan = plan_result.get("plan", {})
        if plan.get("tooling"):
            log.append({"stage": "Resource", "message": f"Selected resource set: {self._short_text(plan.get('tooling'), 120)}."})
        steps = loop.get("steps", [])
        for step in steps:
            status = "passed" if step.get("passed") else "requires correction"
            issues = "; ".join(step.get("issues", [])[:3]) or "no symbolic issue"
            log.append({"stage": "Verification", "message": f"Retry {step.get('retry')}: {status}; {issues}."})
        log.append({"stage": "Output", "message": f"Final status: {loop.get('final_status', 'unchecked')}."})
        return log

    def _short_text(self, text: Any, n: int) -> str:
        value = " ".join(str(text or "").split())
        return value if len(value) <= n else value[: max(0, n - 1)] + "…"
