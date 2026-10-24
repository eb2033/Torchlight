"""API tests against the real trained artifacts (run train.py first)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if not (ROOT / "artifacts" / "meta.json").exists():
    pytest.skip("artifacts/ missing: run src/train.py first", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
from preprocess import FEATURES, load_kdd, readable_name  # noqa: E402

app_module.ARTIFACTS = ROOT / "artifacts"


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture(scope="module")
def test_rows():
    df = load_kdd(ROOT / "data" / "KDDTest+.txt")
    return df


def records(df):
    return {"records": df[FEATURES].to_dict(orient="records")}


def test_readable_name():
    assert readable_name("num__src_bytes") == "src_bytes"
    assert readable_name("cat__flag_REJ") == "flag=REJ"
    assert readable_name("cat__protocol_type_tcp") == "protocol_type=tcp"


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"]


def test_attack_batch_is_flagged_with_context(client, test_rows):
    # neptune = SYN flood; the model should flag most of these.
    batch = test_rows[test_rows["label"] == "neptune"].head(50)
    body = client.post("/score", json=records(batch)).json()
    assert body["batch_size"] == 50
    assert body["anomaly_count"] > 25
    assert body["anomaly_rate"] == round(body["anomaly_count"] / 50, 4)
    flagged = [r for r in body["results"] if r["is_anomaly"]]
    for r in flagged:
        assert len(r["top_features"]) == 3
        assert {"protocol_type", "service", "flag"} <= r.keys()


def test_normal_results_are_minimal(client, test_rows):
    batch = test_rows[~test_rows["is_attack"]].head(50)
    body = client.post("/score", json=records(batch)).json()
    for r in body["results"]:
        if not r["is_anomaly"]:
            assert set(r) == {"index", "score", "is_anomaly"}


def test_invalid_input_rejected(client):
    assert client.post("/score", json={"records": []}).status_code == 422
    assert client.post("/score", json={"records": [{"duration": "x"}]}).status_code == 422


def test_metrics_exposed(client, test_rows):
    client.post("/score", json=records(test_rows.head(5)))
    text = client.get("/metrics").text
    for name in ("requests_total", "records_scored_total", "anomalies_total",
                 "score_latency_seconds_bucket"):
        assert name in text
