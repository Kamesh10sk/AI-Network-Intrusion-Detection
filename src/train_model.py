"""Train, compare and save the intrusion-detection models.

Run from the project root:   python src/train_model.py
Put real CICIDS2017 CSVs in data/raw/ first (otherwise sample data is generated).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import LABEL_COL, load_data, clean, cap_rows, encode_labels, class_balance
from feature_engineering import add_features, to_model_matrix
from prediction import risk_level, MODEL_PATH
from generate_sample_data import generate

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
THRESHOLD = 0.5  # lower it (e.g. 0.3) to trade more false positives for fewer missed attacks


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def main():
    # 1. Load -------------------------------------------------------------
    max_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000   # usage: python src/train_model.py 200000
    csvs = [p for p in RAW.glob("*.csv") if p.name != "sample_flows.csv"]
    if not csvs and not (RAW / "sample_flows.csv").exists():
        print("No CSVs in data/raw/ -> generating CICIDS-style SAMPLE data (synthetic).")
        generate().to_csv(RAW / "sample_flows.csv", index=False)
    using_sample = not csvs
    print("Using:", [p.name for p in csvs] if csvs else "synthetic sample_flows.csv")
    df = load_data(RAW)
    print(f"Loaded {len(df):,} rows")

    # 2. Clean + features ------------------------------------------------
    df = cap_rows(clean(df), max_rows)
    print("Label counts:\n" + df[LABEL_COL].value_counts().to_string())
    y = encode_labels(df)
    balance = class_balance(y)
    print(f"After cleaning: {len(df):,} rows | {balance}")
    X = to_model_matrix(add_features(df))
    features = X.columns.tolist()

    # 3. Split (stratified keeps the attack ratio the same in train and test)
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_te, y_tr, y_te = X.loc[idx_tr], X.loc[idx_te], y.loc[idx_tr], y.loc[idx_te]

    # 4. Train and compare three models ---------------------------------
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=3,
                                                class_weight="balanced", n_jobs=-1, random_state=42),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        prob = m.predict_proba(X_te)[:, 1]
        results[name] = metrics(y_te, (prob >= THRESHOLD).astype(int))
        r = results[name]
        print(f"{name:20s} acc={r['accuracy']:.4f} prec={r['precision']:.4f} "
              f"recall={r['recall']:.4f} f1={r['f1']:.4f} fpr={r['false_positive_rate']:.4f}")

    best = max(results, key=lambda k: results[k]["f1"])
    print(f"Primary model (best F1): {best}")
    model = models[best]

    # 5. Feature importance (Random Forest) -----------------------------
    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    # 6. Attack-type classifier (tells the analyst WHAT kind of attack) --
    attack_model = RandomForestClassifier(n_estimators=80, max_depth=20, min_samples_leaf=3,
                                          class_weight="balanced", n_jobs=-1, random_state=42)
    attack_model.fit(X_tr, df.loc[idx_tr, LABEL_COL])

    # 7. Save model bundle, metrics, and scored test set for the dashboard
    joblib.dump({"model": model, "attack_model": attack_model, "features": features,
                 "threshold": THRESHOLD, "model_name": best}, MODEL_PATH, compress=3)

    p = model.predict_proba(X_te)[:, 1]
    is_attack = p >= THRESHOLD
    test = df.loc[idx_te].copy()
    test["Actual"] = np.where(y_te == 1, "Attack", "Normal")
    test["Prediction"] = np.where(is_attack, "Attack", "Normal")
    test["Attack Probability"] = p.round(4)
    test["Attack Type"] = np.where(is_attack, attack_model.predict(X_te), "-")
    test["Risk Level"] = np.where(is_attack, risk_level(p), "None")
    keep = ["Timestamp", "Protocol", "Destination Port", "Flow Duration", "Total Fwd Packets",
            "SYN Flag Count", LABEL_COL, "Actual", "Prediction", "Attack Probability",
            "Attack Type", "Risk Level"]
    test[[c for c in keep if c in test.columns]].to_csv(
        ROOT / "data" / "processed" / "predictions.csv", index=False)

    report = {"using_sample_data": using_sample, "rows": int(len(df)), "class_balance": balance,
              "primary_model": best, "threshold": THRESHOLD, "models": results,
              "feature_importance": imp.head(15).round(4).to_dict()}
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(report, indent=2))
    print("Saved model, metrics.json and predictions.csv")


if __name__ == "__main__":
    main()
