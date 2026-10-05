"""Use the saved model to score network flows and build security alerts."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from feature_engineering import add_features, to_model_matrix

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "intrusion_detection_model.pkl"


def risk_level(p):
    """Map attack probability to an analyst-friendly risk label."""
    return np.select([p >= 0.90, p >= 0.70, p >= 0.50], ["High", "Medium", "Low"], "None")


def load_bundle(path=MODEL_PATH):
    return joblib.load(path)


def predict(df, bundle=None):
    """Return df with Prediction, Attack Probability, Attack Type and Risk Level columns."""
    bundle = bundle or load_bundle()
    d = df.copy()
    d.columns = d.columns.str.strip()
    d = d.replace([np.inf, -np.inf], np.nan)
    num = d.select_dtypes(include=[np.number]).columns
    d[num] = d[num].fillna(0)                       # at inference we must keep every row
    X = to_model_matrix(add_features(d), bundle["features"])
    p_attack = bundle["model"].predict_proba(X)[:, 1]
    is_attack = p_attack >= bundle["threshold"]
    types = bundle["attack_model"].predict(X)
    out = df.copy()
    out["Prediction"] = np.where(is_attack, "Attack", "Normal")
    out["Attack Probability"] = p_attack.round(4)
    out["Attack Type"] = np.where(is_attack, types, "-")
    out["Risk Level"] = np.where(is_attack, risk_level(p_attack), "None")
    return out
