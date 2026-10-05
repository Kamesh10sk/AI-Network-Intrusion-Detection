"""Turn cleaned flows into a numeric model matrix."""
import numpy as np
import pandas as pd
from preprocessing import LABEL_COL, ID_COLS


def add_features(df):
    """Add behavioural ratios that help separate attack types. Only uses columns that exist."""
    df = df.copy()
    c = df.columns
    if {"Total Fwd Packets", "Total Backward Packets"} <= set(c):
        df["Total Packets"] = df["Total Fwd Packets"] + df["Total Backward Packets"]
        df["Fwd/Bwd Packet Ratio"] = df["Total Fwd Packets"] / (df["Total Backward Packets"] + 1)
    if {"SYN Flag Count", "ACK Flag Count"} <= set(c):
        df["SYN/ACK Ratio"] = df["SYN Flag Count"] / (df["ACK Flag Count"] + 1)
    if "Flow Duration" in c:
        df["Log Flow Duration"] = np.log1p(df["Flow Duration"].clip(lower=0))
    return df


def to_model_matrix(df, feature_columns=None):
    """Drop labels/IDs, one-hot encode categoricals (e.g. Protocol), align to training columns."""
    X = df.drop(columns=[c for c in [LABEL_COL] + ID_COLS if c in df.columns])
    cat = X.select_dtypes(exclude=[np.number]).columns.tolist()
    X = pd.get_dummies(X, columns=cat, dtype=int)
    X = X.loc[:, X.nunique() > 1] if feature_columns is None else X   # drop constant columns
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)          # same columns as training
    return X.astype("float32")
