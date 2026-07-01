from __future__ import annotations

from pathlib import Path
import os
import sys

try:
    from flask import Flask, jsonify, render_template, request, send_file
except Exception as exc:  # pragma: no cover
    raise SystemExit("Flask is required. Install project requirements first.") from exc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from services import METHODS, PrototypeService  # noqa: E402

service = PrototypeService(ROOT)
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)


@app.errorhandler(KeyError)
def handle_key_error(exc):
    return jsonify({"error": "not_found", "message": str(exc)}), 404


@app.errorhandler(ValueError)
def handle_value_error(exc):
    return jsonify({"error": "bad_request", "message": str(exc)}), 400


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify(service.health())


@app.route("/api/app_state")
def app_state():
    return jsonify(service.app_state())


@app.route("/api/workflow")
def workflow_spec():
    return jsonify(service.workflow_spec())


@app.route("/api/features")
def features():
    return jsonify(service.feature_catalog())


@app.route("/api/features/<feature_id>")
def feature_detail(feature_id: str):
    return jsonify(service.feature_detail(feature_id))


@app.route("/api/dataset/overview")
def dataset_overview():
    return jsonify(service.dataset_overview())


@app.route("/api/dataset/samples")
def dataset_samples():
    return jsonify(service.dataset_samples(
        feature_id=request.args.get("feature_id", "all"),
        search=request.args.get("search", ""),
        page=int(request.args.get("page", 1)),
        page_size=int(request.args.get("page_size", 10)),
    ))


@app.route("/api/dataset/export")
def dataset_export():
    return send_file(service.export_dataset_path(), as_attachment=True, download_name="aero_instruct_5k.jsonl")




@app.route("/api/neo4j/status")
def neo4j_status():
    return jsonify(service.neo4j_status())


@app.route("/api/neo4j/connect", methods=["POST"])
def neo4j_connect():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(service.connect_neo4j(
        uri=data.get("uri", "bolt://localhost:7687"),
        username=data.get("username", ""),
        password=data.get("password", ""),
    ))


@app.route("/api/graph_data")
def graph_data_neo4j_alias():
    return jsonify(service.neo4j_graph_data())


@app.route("/api/neo4j/graph_data")
def neo4j_graph_data():
    return jsonify(service.neo4j_graph_data())


@app.route("/api/neo4j/query", methods=["POST"])
def neo4j_query():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(service.neo4j_handle_query(data.get("type", "text"), data.get("query", "")))


@app.route("/api/query", methods=["POST"])
def neo4j_query_alias():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(service.neo4j_handle_query(data.get("type", "text"), data.get("query", "")))


@app.route("/api/neo4j_image")
def neo4j_image():
    image_path = service.neo4j_image_path(request.args.get("filename", ""))
    if image_path is None:
        return jsonify({"error": "not_found", "message": "Neo4j image file was not found in project image directories."}), 404
    return send_file(image_path)


@app.route("/api/graph/<feature_id>")
def graph(feature_id: str):
    return jsonify(service.feature_graph(feature_id))


@app.route("/api/graph_project")
def graph_project():
    return jsonify(service.full_project_graph())


@app.route("/api/visual_anchor", methods=["POST"])
def visual_anchor():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(service.visual_anchor(
        query=data.get("query", ""),
        feature_id=data.get("feature_id"),
        image_file=data.get("image_file"),
        top_k=int(data.get("top_k", 3)),
    ))


@app.route("/api/upload_view", methods=["POST"])
def upload_view():
    if "file" not in request.files:
        raise ValueError("No file field was provided.")
    file = request.files["file"]
    return jsonify(service.save_upload(file.filename, file.stream))


@app.route("/api/retrieve/<feature_id>")
def retrieve(feature_id: str):
    return jsonify(service.retrieve(feature_id))


@app.route("/api/plan", methods=["POST"])
def plan():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(service.plan(data.get("feature_id") or service.latest_feature_id or "F00", data.get("method", "KGMCF")))


@app.route("/api/verify_loop", methods=["POST"])
def verify_loop():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(service.verify_loop(
        feature_id=data.get("feature_id") or service.latest_feature_id or "F00",
        method=data.get("method", "Vanilla MLLM"),
        plan_payload=data.get("plan"),
        max_retries=int(data.get("max_retries", 3)),
    ))


@app.route("/api/run_workflow", methods=["POST"])
def run_workflow():
    data = request.get_json(force=True, silent=True) or {}
    method = data.get("method", "KGMCF")
    if method not in METHODS:
        method = "KGMCF"
    return jsonify(service.run_workflow(
        feature_id=data.get("feature_id", ""),
        method=method,
        intent=data.get("intent", ""),
        max_retries=int(data.get("max_retries", 3)),
        image_file=data.get("image_file"),
    ))


@app.route("/api/process_card", methods=["POST"])
def process_card():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(service.process_card(data.get("feature_id") or service.latest_feature_id or "F00", data.get("method", "KGMCF")))




@app.route("/api/process_card_pdf")
def process_card_pdf():
    feature_id = request.args.get("feature_id", "F00")
    method = request.args.get("method", "KGMCF")
    pdf_bytes = service.process_card_pdf_bytes(feature_id, method)
    from flask import Response
    filename = f"process_card_{feature_id}.pdf"
    return Response(pdf_bytes, mimetype="application/pdf", headers={"Content-Disposition": f"attachment; filename={filename}"})

@app.route("/api/prototypes")
def prototypes():
    return jsonify(service.prototype_gallery(
        feature_id=request.args.get("feature_id"),
        limit=int(request.args.get("limit", 40)),
    ))


@app.route("/api/physics_archive")
def physics_archive():
    return jsonify(service.physics_archive(request.args.get("feature_id")))


@app.route("/api/prompt_template")
def prompt_template():
    return jsonify(service.prompt_template())


@app.route("/api/prompt_templates")
def prompt_templates():
    return jsonify(service.prompt_templates())


@app.route("/api/prompt_templates", methods=["POST"])
def save_prompt_templates():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(service.save_prompt_templates(data.get("templates", {})))


@app.route("/api/context_fusion", methods=["POST"])
def context_fusion():
    data = request.get_json(force=True, silent=True) or {}
    # Do not silently fall back to the latest workflow feature here.
    # The prompt studio and tests expect an empty feature context unless the
    # caller explicitly supplies feature_id. Runtime pages that need the
    # current anchored feature pass it from the frontend state.
    return jsonify(service.context_fusion(data.get("feature_id") or "", data.get("intent", "")))


@app.route("/api/engine_status")
def engine_status():
    return jsonify(service.engine_status())


@app.route("/api/engine_training_trace")
def engine_training_trace():
    return jsonify(service.training_trace(request.args.get("run_id", "latest")))


@app.route("/api/model_runtime")
def model_runtime():
    return jsonify(service.qwen_runtime_status())


@app.route("/api/model_runtime", methods=["POST"])
def update_model_runtime():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(service.update_model_runtime(data))


@app.route("/api/model_connection_test", methods=["POST"])
def model_connection_test():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(service.test_model_connection(endpoint_url=data.get("endpoint_url"), api_key=data.get("api_key")))


@app.route("/api/model_inference", methods=["POST"])
def model_inference():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(service.qwen_inference(
        feature_id=data.get("feature_id", ""),
        intent=data.get("intent", ""),
        endpoint_url=data.get("endpoint_url"),
        api_key=data.get("api_key"),
        image_file=data.get("image_file"),
    ))



@app.route("/api/lora_config", methods=["POST"])
def save_lora_config():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(service.update_lora_config(data))


@app.route("/api/training_job", methods=["POST"])
def training_job():
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(service.training_job_command(data))

@app.route("/api/prototype_metrics")
def prototype_metrics():
    return jsonify(service.prototype_metrics())


@app.route("/api/planning_record_cases")
def planning_record_cases():
    return jsonify(service.planning_record_cases(
        feature_id=request.args.get("feature_id"),
        limit=int(request.args.get("limit", 100)),
    ))


@app.route("/api/planning_records/<case_id>")
def planning_records(case_id: str):
    return jsonify(service.planning_records(case_id))


@app.route("/api/image")
def image():
    image_path = service.image_path(request.args.get("path", ""))
    if image_path is None:
        return jsonify({"error": "not_found", "message": "Image does not exist or is outside project data."}), 404
    return send_file(image_path)


if __name__ == "__main__":
    host = os.environ.get("KGMCF_HOST", "127.0.0.1")
    port = int(os.environ.get("KGMCF_PORT", "5000"))
    debug = os.environ.get("KGMCF_DEBUG", "0") in {"1", "true", "True", "yes"}
    app.run(host=host, port=port, debug=debug)
