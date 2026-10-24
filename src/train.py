"""Train the autoencoder on normal traffic, pick a threshold, evaluate.

Steps:
1. Split normal rows of KDDTrain+ into train (80%) / validation (20%).
2. Fit the preprocessor on the normal training rows only.
3. Train the autoencoder to reconstruct normal rows.
4. Threshold = chosen percentile of validation error (held-out normal data),
   i.e. at p99 we accept ~1% false positives on normal traffic.
5. Report precision/recall on KDDTest+ (attacks = positives).
6. Save model.pt, preprocessor.joblib and meta.json to artifacts/.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import Autoencoder, reconstruction_error
from preprocess import FEATURES, build_preprocessor, load_kdd


def to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float32)


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = scores > threshold
    return {
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "false_positive_rate": round(float(y_pred[~y_true].mean()), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--percentile", type=float, default=99)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_kdd(data_dir / "KDDTrain+.txt")
    test_df = load_kdd(data_dir / "KDDTest+.txt")

    # 1. Normal rows only, shuffled, 80/20 split.
    normal = train_df[~train_df["is_attack"]].iloc[rng.permutation((~train_df["is_attack"]).sum())]
    n_val = int(len(normal) * 0.2)
    val_df, fit_df = normal.iloc[:n_val], normal.iloc[n_val:]
    print(f"normal train rows: {len(fit_df)}, validation rows: {len(val_df)}")

    # 2. Preprocessor sees normal training rows only (no leakage from attacks/val).
    pre = build_preprocessor()
    x_fit = to_tensor(pre.fit_transform(fit_df[FEATURES]))
    x_val = to_tensor(pre.transform(val_df[FEATURES]))
    x_test = to_tensor(pre.transform(test_df[FEATURES]))

    # 3. Train: minimise mean squared reconstruction error.
    model = Autoencoder(x_fit.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_fit), batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for (batch,) in loader:
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        val_loss = reconstruction_error(model, x_val).mean().item()
        print(f"epoch {epoch:3d}  train_loss {total / len(x_fit):.5f}  val_loss {val_loss:.5f}")

    # 4. Threshold from held-out normal data.
    val_scores = reconstruction_error(model, x_val).numpy()
    threshold = float(np.percentile(val_scores, args.percentile))

    # 5. Evaluate on KDDTest+ (includes attack types unseen in training).
    test_scores = reconstruction_error(model, x_test).numpy()
    y_test = test_df["is_attack"].to_numpy()
    metrics = evaluate(y_test, test_scores, threshold)
    print(f"\nthreshold (p{args.percentile:g} of validation error): {threshold:.5f}")
    print(f"KDDTest+ at p{args.percentile:g}: {metrics}")

    # Show what other percentiles would trade off (for the README / interview).
    comparison = {}
    for pct in (95, 99, 99.9):
        t = float(np.percentile(val_scores, pct))
        comparison[f"p{pct:g}"] = {"threshold": round(t, 5), **evaluate(y_test, test_scores, t)}
        print(f"  p{pct:g}: {comparison[f'p{pct:g}']}")

    # 6. Save artifacts.
    torch.save(model.state_dict(), out_dir / "model.pt")
    joblib.dump(pre, out_dir / "preprocessor.joblib")
    meta = {
        "threshold": threshold,
        "percentile": args.percentile,
        "input_dim": int(x_fit.shape[1]),
        "feature_names": [str(n) for n in pre.get_feature_names_out()],
        "raw_features": FEATURES,
        "epochs": args.epochs,
        "test_metrics": metrics,
        "threshold_comparison": comparison,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved artifacts to {out_dir}/")


if __name__ == "__main__":
    main()
