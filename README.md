# AI-Powered Network Intrusion Detection System

A machine-learning system that classifies network flows as **normal or malicious**, names the
likely attack type, assigns a risk level, and presents everything in a Streamlit security dashboard.

> **Scope (honest version):** this is an **offline** flow-classification prototype evaluated on a
> held-out test set. It is *not* a real-time IDS. Live capture, SIEM integration and automated
> response are listed under Future Work.

## Pipeline
Dataset → cleaning → feature engineering → stratified train/test split → 3 models compared →
evaluation → attack-type classifier → saved model → dashboard alerts

## Quick start
```bash
pip install -r requirements.txt
# Option A: real data  -> put CICIDS2017 "MachineLearningCSV" files in data/raw/
# Option B: no data yet -> training auto-generates synthetic CICIDS-style sample data
python src/train_model.py
streamlit run dashboard/app.py
```
Results shown on synthetic data are only a demo of the pipeline. **Re-train on CICIDS2017 and
report those numbers** on your resume and in interviews.

## Project structure
| Path | Purpose |
|---|---|
| `src/preprocessing.py` | load CSVs, strip column names, drop duplicates, handle inf/NaN, encode labels |
| `src/feature_engineering.py` | derived features (packet ratios, SYN/ACK ratio), one-hot encoding |
| `src/train_model.py` | trains Logistic Regression, Decision Tree, Random Forest; saves model + metrics |
| `src/prediction.py` | loads the model, scores new flows, computes risk level |
| `src/generate_sample_data.py` | synthetic CICIDS-style data for trying the pipeline |
| `dashboard/app.py` | Streamlit dashboard (6 pages) |
| `models/` | saved model bundle (`.pkl`) |
| `reports/metrics.json` | evaluation results used by the dashboard |

## Design decisions
- **Stratified split + `class_weight="balanced"`** because attacks are the minority class.
- **Recall and false-positive rate reported next to accuracy**: accuracy alone is misleading on imbalanced data.
- **ID columns (IPs, ports, timestamps) excluded from features** so the model learns behaviour, not addresses.
- **Alert threshold** (`THRESHOLD` in `train_model.py`, default 0.5): lowering it catches more attacks at the cost of more false alarms.

## Future work
Live packet capture → feature extraction → prediction → SIEM/SOC integration → SOAR response;
deep-learning and anomaly-detection models for unseen attacks; SHAP explanations; role-based dashboard access.

## Ethical scope
For defensive research and authorised monitoring only. Use public datasets or traffic you are permitted to analyse.
