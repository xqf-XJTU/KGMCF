from pathlib import Path
import csv, sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from kgmcf.kg.loader import AeroMPKG
from kgmcf.kg.lgse import extract_logic_guided_subgraph
from kgmcf.planning.adapters import PlanningInput, generate_process_plan
from kgmcf.verification.loop import run_symbolic_verification_loop
from kgmcf.metrics.process_metrics import macro_process_sequence_accuracy, evaluate_process_pair
from kgmcf.visual.encoder import VisualEncoder, VisualEncoderConfig
from kgmcf.visual.index import VisualFeatureIndex


def test_lgse_returns_evidence_context():
    kg = AeroMPKG.load_dir(ROOT / "data/raw/aero_mpkg/json")
    sub = extract_logic_guided_subgraph(kg, "F05")
    assert sub["feature_id"] == "F05"
    assert len(sub["nodes"]) > 0
    assert len(sub["evidence_paths"]) > 0
    assert "constraints" in sub["context"]


def test_planner_adapters_and_traceback_loop():
    kg = AeroMPKG.load_dir(ROOT / "data/raw/aero_mpkg/json")
    retrieved = extract_logic_guided_subgraph(kg, "F12")
    plan = generate_process_plan("KGMCF", PlanningInput("F12", kg.feature_summary("F12"), retrieved_context=retrieved))
    assert plan["method_settings"]["uses_semantic_traceback"] is True
    base = generate_process_plan("Vanilla MLLM", PlanningInput("F12", kg.feature_summary("F12"), retrieved_context=retrieved))
    result = run_symbolic_verification_loop(base, max_retries=3)
    assert result["final_status"] in {"verified", "unresolved"}
    assert result["retry_count"] <= 3


def test_process_plan_metrics():
    assert macro_process_sequence_accuracy("roughing -> semi-finishing -> finishing", "roughing -> finishing") < 1.0
    generated = {"process_route":"roughing -> finishing", "tooling":"CER2525 holder; U5R-R1.5 insert", "cutting_parameters":{"Vc_m_min":70,"feed_mm_rev":0.08,"ap_mm":0.1}}
    reference = {"process_route":"roughing -> semi-finishing -> finishing", "tooling":"CER2525 holder; U5R-R1.5 insert", "reference_tools":["CER2525 holder", "U5R-R1.5 insert"]}
    metrics = evaluate_process_pair(generated, reference)
    assert set(metrics) == {"MPSA", "MSRV", "TSA", "G-PvR", "ROUGE-L"}
    assert metrics["MSRV"] >= 0.9


def test_visual_encoder_and_index_query():
    image = next((ROOT / "data/raw/cv20/images/0/classes/0").glob("*.png"))
    enc = VisualEncoder(VisualEncoderConfig(backend="fallback", image_size=16))
    vec = enc.encode_image(image)
    assert vec.shape[0] == enc.dimension
    index = VisualFeatureIndex.build(ROOT / "data/raw/cv20/images", encoder=enc, backend="numpy")
    result = index.query_image(image, encoder=enc, top_k=5)
    assert result["matches"]
    assert result["matches"][0]["feature_id"] == "F00"
