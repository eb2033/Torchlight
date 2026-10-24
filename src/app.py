"""FastAPI scorer: /health, /score, /metrics.

Loads artifacts/ at startup (train first, or startup fails).
"""
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, create_model

from model import Autoencoder, per_feature_error
from preprocess import CATEGORICAL, FEATURES, readable_name

ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "5000"))
TOP_K = 3

# --- structured logging: one JSON object per line on stdout ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger("scorer")

# --- Prometheus metrics ---
REQUESTS = Counter("requests_total", "Scoring requests", ["status"])
RECORDS = Counter("records_scored_total", "Records scored")
ANOMALIES = Counter("anomalies_total", "Records flagged as anomalous")
LATENCY = Histogram("score_latency_seconds", "Time to score one batch",
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))

# --- request schema: every one of the 41 features is required and typed ---
Record = create_model(
    "Record",
    **{f: (str if f in CATEGORICAL else float, ...) for f in FEATURES},
)


class ScoreRequest(BaseModel):
    records: list[Record] = Field(min_length=1, max_length=MAX_BATCH)


state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    model = Autoencoder(meta["input_dim"])
    model.load_state_dict(torch.load(ARTIFACTS / "model.pt", map_location="cpu"))
    model.eval()
    state.update(
        model=model,
        preprocessor=joblib.load(ARTIFACTS / "preprocessor.joblib"),
        threshold=meta["threshold"],
        feature_names=[readable_name(n) for n in meta["feature_names"]],
    )
    log.info(json.dumps({"event": "startup", "threshold": meta["threshold"],
                         "input_dim": meta["input_dim"]}))
    yield
    state.clear()


app = FastAPI(title="Torchlight anomaly scorer", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "threshold": state.get("threshold")}


@app.post("/score")
def score(req: ScoreRequest):
    start = time.perf_counter()
    try:
        df = pd.DataFrame([r.model_dump() for r in req.records], columns=FEATURES)
        x = torch.tensor(state["preprocessor"].transform(df), dtype=torch.float32)
        errors = per_feature_error(state["model"], x).numpy()  # (records, features)
        scores = errors.mean(axis=1)
        threshold = state["threshold"]
        flagged = scores > threshold

        results = []
        for i, s in enumerate(scores):
            item = {"index": i, "score": round(float(s), 6), "is_anomaly": bool(flagged[i])}
            if flagged[i]:
                # Which encoded features the model failed to reconstruct the most.
                top = np.argsort(errors[i])[::-1][:TOP_K]
                item["top_features"] = [
                    {"feature": state["feature_names"][j], "error": round(float(errors[i, j]), 6)}
                    for j in top
                ]
                for col in CATEGORICAL:
                    item[col] = df.at[i, col]
            results.append(item)

        anomaly_scores = scores[flagged]
        body = {
            "batch_size": len(scores),
            "anomaly_count": int(flagged.sum()),
            "anomaly_rate": round(float(flagged.mean()), 4),
            "threshold": round(threshold, 6),
            "max_score": round(float(scores.max()), 6),
            "mean_score": round(float(scores.mean()), 6),
            "mean_anomaly_score": round(float(anomaly_scores.mean()), 6) if len(anomaly_scores) else None,
            "results": results,
        }
    except Exception:
        REQUESTS.labels(status="error").inc()
        log.exception(json.dumps({"event": "score_error", "batch_size": len(req.records)}))
        raise

    latency = time.perf_counter() - start
    REQUESTS.labels(status="ok").inc()
    RECORDS.inc(body["batch_size"])
    ANOMALIES.inc(body["anomaly_count"])
    LATENCY.observe(latency)
    log.info(json.dumps({
        "event": "score", "batch_size": body["batch_size"],
        "anomaly_count": body["anomaly_count"], "latency_ms": round(latency * 1000, 2),
    }))
    return body


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
