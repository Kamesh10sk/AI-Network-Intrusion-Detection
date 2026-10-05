"""Generate CICIDS2017-style sample network-flow data (synthetic).

Use this ONLY to try the pipeline when you don't have the real dataset yet.
Column names match CICIDS2017 so real CSVs drop in with no code changes.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _benign(rng, n):
    fwd = rng.poisson(10, n) + 1
    bwd = rng.poisson(8, n) + 1
    return dict(
        dur=rng.lognormal(11, 2, n), fwd=fwd, bwd=bwd,
        avg=rng.lognormal(5.5, 0.8, n),
        proto=rng.choice(["TCP", "UDP", "ICMP"], n, p=[0.80, 0.18, 0.02]),
        syn=np.minimum(rng.poisson(1, n), 3), ack=rng.poisson((fwd + bwd) * 0.7))


def _dos(rng, n):
    fwd = rng.poisson(200, n) + 1
    bwd = rng.poisson(2, n)
    return dict(
        dur=rng.lognormal(9.5, 1.5, n), fwd=fwd, bwd=bwd,
        avg=rng.lognormal(4.2, 0.6, n),
        proto=rng.choice(["TCP", "UDP"], n, p=[0.85, 0.15]),
        syn=rng.poisson(30, n), ack=rng.poisson(3, n))


def _portscan(rng, n):
    return dict(
        dur=rng.lognormal(6, 1, n), fwd=rng.integers(1, 4, n), bwd=rng.integers(0, 2, n),
        avg=rng.lognormal(3.8, 0.4, n), proto=np.array(["TCP"] * n),
        syn=np.ones(n, dtype=int), ack=rng.integers(0, 2, n))


def _bruteforce(rng, n):
    return dict(
        dur=rng.lognormal(12, 1, n), fwd=rng.poisson(25, n) + 1, bwd=rng.poisson(20, n) + 1,
        avg=rng.lognormal(4.6, 0.5, n), proto=np.array(["TCP"] * n),
        syn=np.full(n, 2), ack=rng.poisson(40, n))


GENERATORS = {"BENIGN": _benign, "DoS": _dos, "PortScan": _portscan, "BruteForce": _bruteforce}
MIX = {"BENIGN": 0.75, "DoS": 0.12, "PortScan": 0.08, "BruteForce": 0.05}


def generate(n=30000, seed=42, overlap=0.06):
    rng = np.random.default_rng(seed)
    frames = []
    for label, frac in MIX.items():
        m = int(n * frac)
        d = GENERATORS[label](rng, m)
        # 'overlap' fraction of attack rows look like normal traffic (stealthy attacks)
        if label != "BENIGN":
            k = int(m * overlap)
            b = _benign(rng, k)
            for key in d:
                d[key][:k] = b[key]
        total_pkts = d["fwd"] + d["bwd"]
        total_bytes = total_pkts * d["avg"]
        dur = np.maximum(d["dur"], 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            bps = total_bytes / (dur / 1e6)
            pps = total_pkts / (dur / 1e6)
        frames.append(pd.DataFrame({
            "Flow Duration": dur.round(0), "Total Fwd Packets": d["fwd"],
            "Total Backward Packets": d["bwd"], "Flow Bytes/s": bps, "Flow Packets/s": pps,
            "Protocol": d["proto"], "SYN Flag Count": d["syn"], "ACK Flag Count": d["ack"],
            "Average Packet Size": d["avg"].round(2), "Label": label}))
    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Mimic real-world messiness: timestamps, infinities, NaNs, duplicates
    df.insert(0, "Timestamp", pd.Timestamp("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 86400, len(df)), unit="s"))
    df.loc[rng.choice(len(df), 15, replace=False), "Flow Duration"] = 0   # -> inf rates
    with np.errstate(divide="ignore", invalid="ignore"):
        z = df["Flow Duration"] == 0
        df.loc[z, "Flow Bytes/s"] = np.inf
        df.loc[z, "Flow Packets/s"] = np.inf
    df.loc[rng.choice(len(df), 40, replace=False), "Flow Bytes/s"] = np.nan
    df = pd.concat([df, df.sample(150, random_state=seed)], ignore_index=True)
    return df


if __name__ == "__main__":
    out = ROOT / "data" / "raw" / "sample_flows.csv"
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    generate(n).to_csv(out, index=False)
    print(f"Wrote {out}")
