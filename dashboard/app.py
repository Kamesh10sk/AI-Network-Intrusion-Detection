"""Signal Deck — network intrusion console.   Run:  streamlit run dashboard/app.py

A thermal/signal-analyzer themed dashboard: cool = calm, hot = dangerous.
No sidebar — a top segmented control switches sections on one scrolling page.
"""
import json
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prediction import predict, MODEL_PATH  # noqa: E402

st.set_page_config(page_title="Signal Deck — Intrusion Analyzer", page_icon="◈", layout="wide")

# ---------------------------------------------------------------- palette --
INK = "#0b0710"        # page background — near-black violet
PANEL = "#15101f"
PANEL_2 = "#1d1630"
LINE = "#2f2545"
TEXT = "#f0ecf7"
MUTED = "#9388ab"
CYAN = "#4cc9f0"       # calm / benign
VIOLET = "#9d4edd"     # low risk
MAGENTA = "#f72585"    # medium risk
ORANGE = "#ff5400"     # high risk — the hottest reading on the dial

HEAT_SCALE = [[0.0, CYAN], [0.33, VIOLET], [0.66, MAGENTA], [1.0, ORANGE]]

TEMPLATE = go.layout.Template(layout=go.Layout(
    paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="JetBrains Mono, monospace", size=12),
    colorway=[CYAN, MAGENTA, ORANGE, VIOLET, "#4ea8de", "#f4a261"],
    xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
    yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=40, l=10, r=10, b=10)))

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{ --ink:{INK}; --panel:{PANEL}; --panel2:{PANEL_2}; --line:{LINE}; --text:{TEXT}; --muted:{MUTED};
  --cyan:{CYAN}; --violet:{VIOLET}; --magenta:{MAGENTA}; --orange:{ORANGE}; }}
html, body, [class*="css"] {{ font-family:'Sora', sans-serif; }}
.stApp {{ background: var(--ink); color: var(--text); }}
#MainMenu, footer, header[data-testid="stHeader"], section[data-testid="stSidebar"] {{ display:none; }}
.block-container {{ padding-top: 0 !important; max-width: 1180px; }}
/* ---- ticker ---- */
.ticker-wrap {{ background:var(--panel2); border-bottom:1px solid var(--line); overflow:hidden;
  white-space:nowrap; padding:.5rem 0; margin: 0 -5rem 1.4rem -5rem; }}
.ticker {{ display:inline-block; animation: scroll 38s linear infinite; font-family:'JetBrains Mono',monospace; font-size:.78rem; }}
.ticker span {{ margin-right:2.5rem; color:var(--muted); }}
.ticker span b {{ color:var(--orange); }}
@keyframes scroll {{ 0% {{ transform:translateX(0); }} 100% {{ transform:translateX(-50%); }} }}
/* ---- top bar ---- */
.topbar {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:1.6rem; }}
.wordmark {{ font-family:'Sora'; font-weight:700; font-size:1.15rem; letter-spacing:-.01em; }}
.wordmark span {{ color:var(--cyan); }}
.subline {{ color:var(--muted); font-size:.76rem; font-family:'JetBrains Mono',monospace; }}
/* ---- segmented nav (st.tabs restyle) ---- */
.stTabs [data-baseweb="tab-list"] {{ gap:.35rem; background:var(--panel2); border:1px solid var(--line); border-radius:8px; padding:4px; flex-wrap:wrap; }}
.stTabs [data-baseweb="tab"] {{ height:2.3rem; padding:0 1.1rem; border-radius:6px; color:var(--muted); font-size:.82rem; font-family:'JetBrains Mono',monospace; white-space:nowrap; display:flex; align-items:center; justify-content:center; }}
.stTabs [data-baseweb="tab"] p {{ margin:0; }}
.stTabs [aria-selected="true"] {{ background:var(--panel); color:var(--text) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ display:none; }}
.stTabs [data-baseweb="tab-border"] {{ display:none; }}
/* ---- panel / spec-sheet (st.container(border=True)) ---- */
[data-testid="stVerticalBlockBorderWrapper"] {{ background:var(--panel); border:1px solid var(--line) !important; border-radius:10px; }}
[data-testid="stVerticalBlockBorderWrapper"] > div {{ background:transparent; }}
.eyebrow {{ color:var(--muted); font-size:.74rem; font-family:'JetBrains Mono',monospace; letter-spacing:.03em; margin-bottom:.6rem; }}
.spec-row {{ display:flex; justify-content:space-between; align-items:baseline; padding:.55rem 0; border-bottom:1px solid var(--line); }}
.spec-row:last-child {{ border-bottom:none; }}
.spec-label {{ color:var(--muted); font-size:.82rem; }}
.spec-value {{ font-family:'JetBrains Mono',monospace; font-weight:600; font-size:1.05rem; }}
/* ---- tags ---- */
.tag {{ display:inline-block; font-family:'JetBrains Mono',monospace; font-size:.72rem; padding:.15rem .5rem;
  border-radius:4px; border:1px solid var(--line); }}
.tag-high {{ color:var(--orange); border-color:var(--orange); }}
.tag-med {{ color:var(--magenta); border-color:var(--magenta); }}
.tag-low {{ color:var(--violet); border-color:var(--violet); }}
[data-testid="stDataFrame"] {{ border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
hr {{ border-color:var(--line); }}
</style>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python src/train_model.py` first.")
    st.stop()


@st.cache_data
def load_report():
    return json.loads((ROOT / "reports" / "metrics.json").read_text())


@st.cache_data
def load_predictions():
    return pd.read_csv(ROOT / "data" / "processed" / "predictions.csv")


report, df = load_report(), load_predictions()
alerts_all = df[df["Prediction"] == "Attack"].copy()

# ------------------------------------------------------------------ ticker --
top_alerts = alerts_all.sort_values("Attack Probability", ascending=False).head(14)
items = "".join(
    f'<span>⚠ <b>{r["Attack Type"]}</b> · {r["Attack Probability"]*100:.1f}% confidence · {r["Risk Level"]} risk</span>'
    for _, r in top_alerts.iterrows())
st.markdown(f'<div class="ticker-wrap"><div class="ticker">{items}{items}</div></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ topbar --
st.markdown(f"""<div class="topbar">
  <div><div class="wordmark">◈ SIGNAL<span>DECK</span></div>
       <div class="subline">network intrusion analyzer · offline flow classification</div></div>
  <div class="subline" style="text-align:right;">MODEL&nbsp; {report['primary_model'].upper()}<br>
       SOURCE&nbsp; {'SYNTHETIC SAMPLE' if report.get('using_sample_data') else 'CICIDS2017'}</div>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["OVERVIEW", "ALERTS", "ATTACK MIX", "MODELS", "TRAFFIC", "SCAN CSV"])

# ----------------------------------------------------------------- helpers --
def risk_tag(level):
    cls = {"High": "tag-high", "Medium": "tag-med", "Low": "tag-low"}.get(level, "tag-low")
    return f'<span class="tag {cls}">{level}</span>'


def spec_row(label, value):
    st.markdown(f'<div class="spec-row"><span class="spec-label">{label}</span>'
               f'<span class="spec-value">{value}</span></div>', unsafe_allow_html=True)


# =========================================================== OVERVIEW =====
with tabs[0]:
    total = len(df)
    attacks = int((df["Prediction"] == "Attack").sum())
    rate = 100 * attacks / total
    posture = "ELEVATED" if rate > 15 else ("WATCH" if rate > 5 else "CALM")
    posture_color = ORANGE if posture == "ELEVATED" else (MAGENTA if posture == "WATCH" else CYAN)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=rate, number={"suffix": "%", "font": {"size": 42, "color": posture_color}},
            gauge={"axis": {"range": [0, 50], "tickcolor": MUTED, "tickfont": {"size": 10}},
                  "bar": {"color": posture_color, "thickness": 0.28},
                  "bgcolor": PANEL, "borderwidth": 0,
                  "steps": [{"range": [0, 5], "color": "rgba(76,201,240,.12)"},
                            {"range": [5, 15], "color": "rgba(247,37,133,.10)"},
                            {"range": [15, 50], "color": "rgba(255,84,0,.10)"}]}))
        gauge.update_layout(template=TEMPLATE, height=280, margin=dict(t=10, b=10, l=30, r=30))
        with st.container(border=True):
            st.markdown(f'<div class="eyebrow">THREAT TEMPERATURE · POSTURE: '
                        f'<span style="color:{posture_color};">{posture}</span></div>', unsafe_allow_html=True)
            st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})
    with c2:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">READOUT</div>', unsafe_allow_html=True)
            spec_row("Total connections", f"{total:,}")
            spec_row("Normal traffic", f"{total - attacks:,}")
            spec_row("Flagged suspicious", f"{attacks:,}")
            high = int((alerts_all["Risk Level"] == "High").sum())
            spec_row("High-risk alerts", f"{high:,}")
            spec_row("Primary model", report["primary_model"])
            spec_row("Model F1 score", f"{report['models'][report['primary_model']]['f1']:.3f}")

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="eyebrow">ALERT INTENSITY BY RISK LEVEL</div>', unsafe_allow_html=True)
        risk = alerts_all["Risk Level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
        risk.columns = ["Risk Level", "Alerts"]
        fig = px.bar(risk, x="Alerts", y="Risk Level", orientation="h", color="Risk Level",
                    color_discrete_map={"High": ORANGE, "Medium": MAGENTA, "Low": VIOLET})
        fig.update_layout(template=TEMPLATE, showlegend=False, height=220)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ============================================================= ALERTS =====
with tabs[1]:
    c1, c2 = st.columns(2)
    levels = c1.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    types = c2.multiselect("Attack type", sorted(alerts_all["Attack Type"].unique()),
                           default=sorted(alerts_all["Attack Type"].unique()))
    filtered = alerts_all[alerts_all["Risk Level"].isin(levels) & alerts_all["Attack Type"].isin(types)]
    st.markdown(f'<div class="eyebrow">{len(filtered):,} ALERTS MATCHED</div>', unsafe_allow_html=True)
    cols = [c for c in ["Timestamp", "Attack Type", "Attack Probability", "Risk Level", "Protocol",
                        "Destination Port", "Flow Duration", "Total Fwd Packets", "SYN Flag Count"]
            if c in filtered.columns]
    st.dataframe(filtered.sort_values("Attack Probability", ascending=False)[cols].head(500),
                 width="stretch", height=440)

# ========================================================== ATTACK MIX ====
with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">TRAFFIC CATEGORIES (GROUND TRUTH)</div>', unsafe_allow_html=True)
            actual = df["Label"].value_counts().reset_index(); actual.columns = ["Category", "Flows"]
            fig = px.pie(actual, names="Category", values="Flows", hole=0.6,
                        color_discrete_sequence=[CYAN, VIOLET, MAGENTA, ORANGE, "#4ea8de", "#f4a261", MUTED])
            fig.update_layout(template=TEMPLATE, height=340)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">PREDICTED ATTACK TYPES</div>', unsafe_allow_html=True)
            pred = alerts_all["Attack Type"].value_counts().reset_index(); pred.columns = ["Type", "Alerts"]
            fig = px.bar(pred, x="Alerts", y="Type", orientation="h")
            fig.update_traces(marker_color=MAGENTA)
            fig.update_layout(template=TEMPLATE, height=340, yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ============================================================== MODELS ====
with tabs[3]:
    m = report["models"]
    with st.container(border=True):
        st.markdown('<div class="eyebrow">MODEL COMPARISON</div>', unsafe_allow_html=True)
        table = pd.DataFrame(m).T.drop(columns="confusion_matrix")
        st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#2a1f45"), width="stretch")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            choice = st.selectbox("Confusion matrix for", list(m), index=list(m).index(report["primary_model"]))
            cm = m[choice]["confusion_matrix"]
            fig = px.imshow(cm, text_auto=True, x=["Pred Normal", "Pred Attack"],
                            y=["Actual Normal", "Actual Attack"], color_continuous_scale=[[0, PANEL], [1, MAGENTA]])
            fig.update_layout(template=TEMPLATE, height=300, coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.markdown(f'<div class="subline">Missed attacks: {cm[1][0]:,} &nbsp;·&nbsp; False alarms: {cm[0][1]:,}</div>',
                        unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">TOP FEATURES (RANDOM FOREST)</div>', unsafe_allow_html=True)
            fi = pd.Series(report["feature_importance"]).sort_values().reset_index()
            fi.columns = ["Feature", "Importance"]
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h")
            fig.update_traces(marker_color=CYAN)
            fig.update_layout(template=TEMPLATE, height=300)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ============================================================= TRAFFIC ====
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            cmap = {"Normal": CYAN, "Attack": ORANGE}
            if "Protocol" in df.columns:
                st.markdown('<div class="eyebrow">PROTOCOL VS PREDICTION</div>', unsafe_allow_html=True)
                fig = px.histogram(df, x="Protocol", color="Prediction", barmode="group", color_discrete_map=cmap)
            elif "Destination Port" in df.columns:
                top = df["Destination Port"].value_counts().head(10).index
                st.markdown('<div class="eyebrow">TOP DESTINATION PORTS VS PREDICTION</div>', unsafe_allow_html=True)
                fig = px.histogram(df[df["Destination Port"].isin(top)].astype({"Destination Port": str}),
                                   x="Destination Port", color="Prediction", barmode="group", color_discrete_map=cmap)
            fig.update_layout(template=TEMPLATE, height=300)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">FORWARD PACKETS PER FLOW</div>', unsafe_allow_html=True)
            fig = px.box(df, x="Prediction", y="Total Fwd Packets", log_y=True, color="Prediction",
                        color_discrete_map={"Normal": CYAN, "Attack": ORANGE})
            fig.update_layout(template=TEMPLATE, height=300, showlegend=False)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.write("")
    with st.container(border=True):
        st.markdown('<div class="eyebrow">MODEL CONFIDENCE BY TRUE CLASS</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="Attack Probability", color="Actual", nbins=40, barmode="overlay",
                           color_discrete_map={"Normal": CYAN, "Attack": ORANGE})
        fig.update_traces(opacity=0.75)
        fig.update_layout(template=TEMPLATE, height=280)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ============================================================ SCAN CSV ====
with tabs[5]:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">SCAN A CSV</div>', unsafe_allow_html=True)
        st.write("Upload a CSV of network flows (same columns as CICIDS2017) to score it. "
                "Only analyse traffic you are authorised to inspect.")
        up = st.file_uploader("Network-flow CSV", type="csv", label_visibility="collapsed")
        if up:
            scored = predict(pd.read_csv(up))
            n_att = int((scored["Prediction"] == "Attack").sum())
            spec_row("Flagged as attack", f"{n_att:,} of {len(scored):,}")
            spec_row("Classified normal", f"{len(scored) - n_att:,}")
            st.write("")
            st.dataframe(scored[scored["Prediction"] == "Attack"].head(500), width="stretch")
            st.download_button("Download scored CSV", scored.to_csv(index=False), "scored_flows.csv")
