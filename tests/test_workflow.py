"""Static checks on the exported n8n workflow."""
import json
from pathlib import Path

WF = json.loads((Path(__file__).resolve().parents[1] / "workflows" / "anomaly_alert.json").read_text())
NODES = {n["name"]: n for n in WF["nodes"]}


def test_connections_point_at_real_nodes():
    for src, outputs in WF["connections"].items():
        assert src in NODES
        for branch in outputs["main"]:
            for link in branch:
                assert link["node"] in NODES


def test_no_secrets_or_credentials_embedded():
    text = json.dumps(WF)
    assert "sk-ant-" not in text
    assert "discord.com/api/webhooks" not in text
    assert all("credentials" not in n for n in WF["nodes"])


def test_llm_call_has_timeout_retry_and_fallback():
    llm = NODES["LLM triage (Claude)"]
    assert llm["parameters"]["options"]["timeout"] == 20000
    assert llm["retryOnFail"] and llm["maxTries"] == 2
    assert llm["onError"] == "continueErrorOutput"
    # Both the success and the error output go to validation, so a failure
    # still reaches the fallback alert.
    outputs = WF["connections"]["LLM triage (Claude)"]["main"]
    assert [b[0]["node"] for b in outputs] == ["Validate triage JSON"] * 2


def test_scorer_reached_by_service_name():
    assert NODES["Score batch"]["parameters"]["url"] == "http://scorer:8000/score"
