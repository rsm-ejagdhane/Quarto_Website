"""Generate projection-friendly chart PNGs for the VC deal flow deck."""

import json
import os
import re
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT_DIR, exist_ok=True)

BG = "#0b1020"
PANEL = "#121a2e"
PRIMARY = "#6366f1"
SECONDARY = "#22c55e"
ACCENT = "#a78bfa"
TEXT = "#f4f6fa"
MUTED = "#a8b4c8"
GRID = "#3d4a63"

# Larger type for classroom projection
plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.labelsize": 15,
        "axes.titlesize": 18,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "text.color": TEXT,
        "grid.color": GRID,
        "font.family": "sans-serif",
        "font.size": 13,
        "legend.fontsize": 12,
    }
)

DPI = 220
BASE = "/Users/eeshajagdhane/Desktop/100x/Data/Simulated"
THESIS = "AI infra for data-heavy workflows; Seed-A; US/EU"
TARGET_GEOS = {"SF", "LA", "NY", "Austin", "Boston", "Seattle"}
weights = {
    "semantic_sim": 0.40,
    "stage_fit": 0.20,
    "geo_fit": 0.10,
    "traction": 0.15,
    "investor_signal": 0.10,
    "hygiene": 0.05,
}


def extract_domain(url):
    if not isinstance(url, str) or not url.strip():
        return None
    u = url.strip()
    if not re.match(r"^https?://", u, re.I):
        u = "http://" + u
    try:
        return urlparse(u).netloc.lower().replace("www.", "") or None
    except Exception:
        return None


def p_at_k(df, k):
    top = df.head(k)
    lab = top[top.crm_label.notna()]
    return 0 if len(lab) == 0 else (lab.crm_label == 1).sum() / len(lab)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", path)


crm = pd.read_csv(f"{BASE}/Simulated_CRM_DealFlow.csv")
harm = pd.read_csv(f"{BASE}/Simulated_Harmonic_DataPull.csv")
pb = pd.read_csv(f"{BASE}/Simulated_Pitchbook_DataPull.csv")

records = []
for src, df, name_col, dom_col in [
    ("crm", crm, "Name", "Website"),
    ("harm", harm, "Company Name", "Website URL"),
    ("pb", pb, "Companies", "Website"),
]:
    tmp = df.copy()
    tmp["domain"] = tmp[dom_col].apply(extract_domain)
    tmp["src"] = src
    tmp["canon_name"] = tmp[name_col]
    records.append(tmp)
all_df = pd.concat(records, ignore_index=True)

groups = []
for dom, g in all_df.groupby("domain"):
    if dom is None:
        continue
    row = {"domain": dom, "name": g["canon_name"].dropna().iloc[0]}
    descs = []
    if "Description" in g.columns:
        descs.extend(g["Description"].dropna().astype(str).tolist())
    row["description"] = max(descs, key=len) if descs else ""
    stage = None
    if "Last Funding Type" in g.columns:
        s = g["Last Funding Type"].dropna()
        if len(s):
            stage = s.iloc[0]
    row["stage"] = stage or "UNKNOWN"
    geo = (
        g["Geo"].dropna().iloc[0]
        if "Geo" in g.columns and g["Geo"].notna().any()
        else None
    )
    row["geo"] = geo
    fund = np.nan
    if "Funding Total" in g.columns:
        s = pd.to_numeric(g["Funding Total"], errors="coerce").dropna()
        if len(s):
            fund = s.max()
    row["funding_total_usd"] = fund
    if "Headcount % (90d)" in g.columns:
        s = pd.to_numeric(g["Headcount % (90d)"], errors="coerce").dropna()
        row["headcount_growth_90d"] = s.iloc[0] if len(s) else np.nan
    else:
        row["headcount_growth_90d"] = np.nan
    invs = []
    for c in ["Investors", "Active Investors"]:
        if c in g.columns:
            invs.extend(g[c].dropna().astype(str).tolist())
    row["investors"] = invs[0] if invs else ""
    row["crm_status"] = (
        g["Status"].dropna().iloc[0]
        if "Status" in g.columns and g["Status"].notna().any()
        else None
    )
    row["tags"] = (
        g["Tags"].dropna().iloc[0]
        if "Tags" in g.columns and g["Tags"].notna().any()
        else ""
    )
    groups.append(row)

companies = pd.DataFrame(groups)
raw_total = len(crm) + len(harm) + len(pb)
dedupe_rate = 1 - len(companies) / raw_total

pos = {"Active", "Portfolio", "Due Diligence", "Term Sheet"}
neg = {"Dispose", "Pass"}
companies["crm_label"] = companies["crm_status"].apply(
    lambda s: 1 if s in pos else (0 if s in neg else None)
)

texts = (
    companies["description"].fillna("")
    + " | "
    + companies["tags"].fillna("")
    + " | "
    + companies["investors"].fillna("")
).tolist()
vec = TfidfVectorizer(stop_words="english", max_features=5000)
X = vec.fit_transform(texts + [THESIS])
companies["semantic_sim"] = np.clip(cosine_similarity(X[:-1], X[-1:]).ravel(), 0, 1)


def stage_fit(s):
    su = str(s).upper().replace(" ", "_")
    if su in {"SEED", "PRE_SEED", "SERIES_A", "SERIES A"}:
        return 1.0
    if su in {"SERIES_B"}:
        return 0.3
    return 0.0


companies["stage_fit"] = companies["stage"].apply(stage_fit)
companies["geo_fit"] = companies["geo"].apply(
    lambda g: 1.0 if g in TARGET_GEOS else 0.0
)
g = companies["headcount_growth_90d"].fillna(0)
gn = (
    (g - g.min()) / (g.max() - g.min())
    if g.max() > g.min()
    else pd.Series(0.0, index=companies.index)
)
companies["traction"] = np.clip(gn, 0, 1) * 0.6
companies["investor_signal"] = companies["investors"].apply(
    lambda x: min(1.0, len(str(x).split(";")) / 5) if x else 0
)


def hygiene(r):
    vals = [
        r.get("domain"),
        r.get("description"),
        r.get("geo"),
        r.get("funding_total_usd"),
        r.get("investors"),
    ]
    return sum(1 for v in vals if pd.notna(v) and str(v).strip()) / 5


companies["hygiene"] = companies.apply(hygiene, axis=1)
companies["score"] = sum(
    100 * w * companies[k].clip(0, 1) for k, w in weights.items()
)
companies = companies.sort_values("score", ascending=False).reset_index(drop=True)

metrics = {
    "raw_total": raw_total,
    "unique": len(companies),
    "dedupe_pct": dedupe_rate * 100,
    "mean_score": float(companies.score.mean()),
    "max_score": float(companies.score.max()),
    "p20": p_at_k(companies, 20) * 100,
    "p50": p_at_k(companies, 50) * 100,
    "positives": int((companies.crm_label == 1).sum()),
    "negatives": int((companies.crm_label == 0).sum()),
}
with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# 1. Data sources
fig, ax = plt.subplots(figsize=(9, 5.5))
sources = ["CRM", "Harmonic", "PitchBook"]
vals = [20000, 20000, 20000]
colors = [PRIMARY, ACCENT, SECONDARY]
bars = ax.bar(sources, vals, color=colors, width=0.55, edgecolor=GRID, linewidth=1.5)
ax.set_ylabel("Records", fontweight="600")
ax.set_title("Simulated Multi-Source Dataset", fontweight="bold", pad=14)
ax.yaxis.grid(True, alpha=0.4, linewidth=1)
ax.set_axisbelow(True)
for b, v in zip(bars, vals):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v + 400,
        f"{v:,}",
        ha="center",
        va="bottom",
        color=TEXT,
        fontsize=14,
        fontweight="bold",
    )
ax.text(
    0.5,
    -0.16,
    f"Total raw: {raw_total:,}  →  Unique: {len(companies):,}  ({dedupe_rate:.1%} deduped)",
    transform=ax.transAxes,
    ha="center",
    color=MUTED,
    fontsize=12,
)
plt.tight_layout()
save(fig, "data_sources.png")

# 2. Dedupe funnel
fig, ax = plt.subplots(figsize=(9, 5))
stages = ["Raw Records\n(3 sources)", "After Dedupe\n(unique domains)"]
vals = [raw_total, len(companies)]
colors = [PRIMARY, SECONDARY]
y = np.arange(len(stages))
ax.barh(y, vals, color=colors, height=0.48, edgecolor=GRID, linewidth=1.5)
ax.set_yticks(y)
ax.set_yticklabels(stages, fontweight="600")
ax.invert_yaxis()
ax.set_xlabel("Company count", fontweight="600")
ax.set_title("Entity Resolution: Raw → Canonical", fontweight="bold", pad=14)
ax.xaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
for i, v in enumerate(vals):
    ax.text(v + 600, i, f"{v:,}", va="center", color=TEXT, fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "dedupe_funnel.png")

# 3. Driver weights
fig, ax = plt.subplots(figsize=(9, 5.5))
labels = [
    "Semantic\nSimilarity",
    "Stage Fit",
    "Geo Fit",
    "Traction",
    "Investor\nSignal",
    "Hygiene",
]
keys = [
    "semantic_sim",
    "stage_fit",
    "geo_fit",
    "traction",
    "investor_signal",
    "hygiene",
]
vals = [weights[k] * 100 for k in keys]
colors = [PRIMARY, ACCENT, SECONDARY, PRIMARY, ACCENT, MUTED]
y = np.arange(len(labels))
ax.barh(y, vals, color=colors, height=0.55, edgecolor=GRID, linewidth=1.5)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontweight="600")
ax.invert_yaxis()
ax.set_xlabel("Partner weight (%)", fontweight="600")
ax.set_title("Key Drivers · Weighted Scoring Model", fontweight="bold", pad=14)
ax.set_xlim(0, 48)
ax.xaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
for i, v in enumerate(vals):
    ax.text(v + 0.9, i, f"{v:.0f}%", va="center", color=TEXT, fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "driver_weights.png")

# 4. Score distribution
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.hist(
    companies["score"],
    bins=40,
    color=PRIMARY,
    edgecolor=GRID,
    alpha=0.9,
    linewidth=1.2,
)
ax.axvline(
    companies.score.mean(),
    color=SECONDARY,
    linestyle="--",
    linewidth=2.5,
    label=f"Mean: {companies.score.mean():.1f}",
)
ax.axvline(
    companies.score.max(),
    color=ACCENT,
    linestyle="--",
    linewidth=2.5,
    label=f"Max: {companies.score.max():.1f}",
)
ax.set_xlabel("Relevance Score (0–100)", fontweight="600")
ax.set_ylabel("Companies", fontweight="600")
ax.set_title("Score Distribution (Simulated)", fontweight="bold", pad=14)
ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
save(fig, "score_distribution.png")

# 5. Precision metrics
fig, ax = plt.subplots(figsize=(8, 5))
ks = ["Precision@20", "Precision@50"]
vals = [p_at_k(companies, 20) * 100, p_at_k(companies, 50) * 100]
bars = ax.bar(ks, vals, color=[PRIMARY, ACCENT], width=0.48, edgecolor=GRID, linewidth=1.5)
ax.set_ylim(0, 55)
ax.set_ylabel("Rate (%)", fontweight="600")
ax.set_title("Ranking Quality vs. CRM Labels", fontweight="bold", pad=14)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
for b, v in zip(bars, vals):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v + 1.5,
        f"{v:.1f}%",
        ha="center",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
    )
plt.tight_layout()
save(fig, "precision_metrics.png")

# 6. Top company contribution breakdown
row = companies.iloc[0]
contrib = {k: 100 * weights[k] * min(1, max(0, row[k])) for k in weights}
labels = [
    "Stage Fit",
    "Geo Fit",
    "Investor Signal",
    "Semantic Sim",
    "Traction",
    "Hygiene",
]
keys2 = [
    "stage_fit",
    "geo_fit",
    "investor_signal",
    "semantic_sim",
    "traction",
    "hygiene",
]
vals = [contrib[k] for k in keys2]
fig, ax = plt.subplots(figsize=(9, 5))
colors = [SECONDARY, PRIMARY, ACCENT, PRIMARY, SECONDARY, MUTED]
y = np.arange(len(labels))
ax.barh(y, vals, color=colors, height=0.55, edgecolor=GRID, linewidth=1.5)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontweight="600")
ax.invert_yaxis()
ax.set_xlabel("Points (of 100)", fontweight="600")
ax.set_title(
    f"Top-Ranked Company · Score = {row.score:.1f} / 100",
    fontweight="bold",
    pad=14,
)
ax.xaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
for i, v in enumerate(vals):
    ax.text(v + 0.5, i, f"{v:.1f}", va="center", color=TEXT, fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "score_breakdown.png")

print(json.dumps(metrics, indent=2))
