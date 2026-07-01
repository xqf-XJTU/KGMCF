from pathlib import Path
import sys
import pytest
pytest.importorskip("flask")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "apps" / "prototype"))
from app import app


def test_neo4j_requires_user_credentials_before_graph_load():
    client = app.test_client()
    status = client.get("/api/neo4j/status").get_json()
    assert status["connected"] is False
    graph = client.get("/api/graph_data").get_json()
    assert graph["connected"] is False
    assert graph["nodes"] == []
    bad = client.post("/api/neo4j/connect", json={"uri":"bolt://localhost:7687", "username":"", "password":""}).get_json()
    assert bad["connected"] is False
    assert bad["password_saved"] is False
