"""Loading and cleaning network-flow data (works for CICIDS2017 CSVs too)."""
from pathlib import Path
import numpy as np
import pandas as pd

LABEL_COL = "Label"
# Columns that identify a flow / host rather than describe behaviour. Models must not learn these.
ID_COLS = ["Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
           "Source Port", "Src Port", "Destination Port", "Dst Port", "Timestamp"]


def load_data(path, exclude=("sample_flows.csv",)):
    """Load one CSV, or every CSV in a folder (CICIDS2017 ships as one file per day).
    The synthetic sample file is skipped whenever real files are present."""
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    real = [f for f in files if f.name not in exclude]
    files = real or files
    if not files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    return pd.concat([_read(f) for f in files], ignore_index=True)


def _read(f):
    """CICIDS labels contain a non-UTF-8 dash ('Web Attack \x96 ...'), so fall back to latin-1."""
    try:
        return pd.read_csv(f, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(f, low_memory=False, encoding="latin-1")


def clean(df):
    """Strip names, drop duplicates, fix infinite/missing values."""
    df = df.copy()
    df.columns = df.columns.str.strip()                       # CICIDS has leading spaces
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)                # rate features can divide by 0
    df = df.dropna().reset_index(drop=True)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.replace(r"[^\x20-\x7e]", "-", regex=True).str.replace(r"\s*-\s*", " - ", regex=True).str.strip()
    return df


def cap_rows(df, max_rows):
    """Downsample huge datasets but never wipe out rare attack classes (keeps >=2000 rows/class)."""
    if not max_rows or len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    parts = [g.sample(n=min(len(g), max(int(len(g) * frac), 2000)), random_state=42)
             for _, g in df.groupby(LABEL_COL)]
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)


def encode_labels(df):
    """BENIGN = 0, any attack = 1."""
    return (df[LABEL_COL].astype(str).str.strip().str.upper() != "BENIGN").astype(int)


def class_balance(y):
    counts = y.value_counts().sort_index()
    return {"benign": int(counts.get(0, 0)), "attack": int(counts.get(1, 0)),
            "attack_rate_pct": round(100 * counts.get(1, 0) / counts.sum(), 2)}
