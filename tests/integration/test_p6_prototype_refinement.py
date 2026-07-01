from pathlib import Path
import sys
import pytest
pytest.importorskip("flask")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "apps" / "prototype"))
from app import app


def test_p6_sidebar_and_inference_inputs_are_refined():
    client = app.test_client()
    html = client.get("/").data.decode("utf-8")
    assert "data-group=\"kg-group\"" in html
    assert "Detected feature: waiting for input" in html
    assert "reasonFeatureSelect" not in html
    assert "Download Process Card PDF" in html
    assert "Qwen2.5-VL-72B Runtime" in html


def test_p6_process_pdf_and_controls_are_available():
    client = app.test_client()
    card = client.post("/api/process_card", json={"feature_id":"F05","method":"KGMCF"}).get_json()
    controls = [s["control"] for s in card["route_steps"]]
    assert controls
    assert not any(c == "constraint-consistent execution" for c in controls)
    pdf = client.get("/api/process_card_pdf?feature_id=F05&method=KGMCF")
    assert pdf.status_code == 200
    assert pdf.data.startswith(b"%PDF")
    assert pdf.headers["Content-Type"].startswith("application/pdf")
