"""Replay sampled KDDTest+ records to the n8n webhook (or straight to /score).

Labels are stripped before sending: the pipeline only sees raw features.
The ground-truth mix is printed locally so you can compare with the alert.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from preprocess import FEATURES, load_kdd


def sample(data_dir: str, n: int, attack_ratio: float, seed: int):
    df = load_kdd(Path(data_dir) / "KDDTest+.txt")
    n_attack = round(n * attack_ratio)
    attacks = df[df["is_attack"]].sample(n_attack, random_state=seed)
    normal = df[~df["is_attack"]].sample(n - n_attack, random_state=seed)
    return pd.concat([attacks, normal]).sample(frac=1, random_state=seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--webhook", default="http://localhost:5678/webhook-test/traffic",
                   help="n8n webhook URL, or http://localhost:8000/score to bypass n8n")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--attack-ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=60)
    args = p.parse_args()

    batch = sample(args.data_dir, args.n, args.attack_ratio, args.seed)
    print(f"sending {len(batch)} records ({int(batch['is_attack'].sum())} attacks, "
          f"labels: {batch['label'].value_counts().head(5).to_dict()})")

    payload = {"records": batch[FEATURES].to_dict(orient="records")}
    resp = requests.post(args.webhook, json=payload, timeout=args.timeout)
    print(f"HTTP {resp.status_code}")
    try:
        body = resp.json()
        body.pop("results", None)  # keep console output short
        print(json.dumps(body, indent=2))
    except ValueError:
        print(resp.text[:500])


if __name__ == "__main__":
    main()
