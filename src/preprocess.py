"""Feature list, data loader and sklearn preprocessor for NSL-KDD.

Shared by train.py and app.py so the API transforms records exactly the way
the model saw them during training.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

# The 41 NSL-KDD features, in file order.
FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]

# Each file row is the 41 features, then the label, then a difficulty score.
COLUMNS = FEATURES + ["label", "difficulty"]


def load_kdd(path: str | Path) -> pd.DataFrame:
    """Load one NSL-KDD file. Adds is_attack (label != 'normal')."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["is_attack"] = df["label"] != "normal"
    return df


def build_preprocessor() -> ColumnTransformer:
    """Numeric: log1p (byte counts are heavy-tailed) then scale to [0, 1]
    (bounded, so a rare-but-normal value cannot dominate the error).
    Categorical: one-hot, ignoring categories never seen in training."""
    numeric = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", MinMaxScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def readable_name(encoded: str) -> str:
    """Turn sklearn's encoded names into something a human (or LLM) can read.

    'num__src_bytes'  -> 'src_bytes'
    'cat__flag_REJ'   -> 'flag=REJ'
    """
    kind, _, name = encoded.partition("__")
    if kind == "cat":
        for col in CATEGORICAL:
            if name.startswith(col + "_"):
                return f"{col}={name[len(col) + 1:]}"
    return name
