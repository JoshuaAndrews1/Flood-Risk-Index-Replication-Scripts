"""
Scatter plot comparing the baseline full-period (2006-2024) index against the
arithmetic-combine alternative:

  x: 2006-2024_geometric_MinMaxRiskIndex.csv  combined_risk_score  (baseline)
  y: 2006-2024_arithmetic_MinMaxRiskIndex.csv combined_risk_score  (arithmetic combine)

Both are on the same min-max [0, 1] scale, so a 1:1 reference line is drawn
alongside the fitted trend line. 

Output:
  TimeBasedFloodIndex/2006-2024/2006-2024_baseline_vs_arithmetic_scatter.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True


# Package layout: <package_root>/scripts/03_robustness/plot_baseline_vs_arithmetic_scatter.py
# Reads and writes results (computed index CSVs + figure) in data/results/.
INDEX_DIR   = Path(__file__).resolve().parents[2] / "data" / "results" / "TimeBasedFloodIndex" / "2006-2024"
BASELINE_CSV = INDEX_DIR / "2006-2024_geometric_MinMaxRiskIndex.csv"
ARITH_CSV    = INDEX_DIR / "2006-2024_arithmetic_MinMaxRiskIndex.csv"
OUTPUT_PATH  = INDEX_DIR / "2006-2024_baseline_vs_arithmetic_scatter.png"

BASELINE_LABEL   = r"Baseline: Geometric Aggregation (Min$-$Max)"
ARITHMETIC_LABEL = r"Arithmetic Aggregation (Min$-$Max)"


def main() -> None:
    baseline = pd.read_csv(BASELINE_CSV, dtype={"GEOID": str})[["GEOID", "combined_risk_score"]]
    arith    = pd.read_csv(ARITH_CSV,    dtype={"GEOID": str})[["GEOID", "combined_risk_score"]]

    merged = baseline.merge(arith, on="GEOID", suffixes=("_baseline", "_arithmetic"))
    merged = merged.dropna(subset=["combined_risk_score_baseline", "combined_risk_score_arithmetic"])
    print(f"Tracts with complete scores: {len(merged):,}")

    x = merged["combined_risk_score_baseline"].values
    y = merged["combined_risk_score_arithmetic"].values

    r, r_p = stats.pearsonr(x, y)
    rho, rho_p = stats.spearmanr(x, y)
    print(f"Pearson r  = {r:.4f}  (p = {r_p:.4e})")
    print(f"Spearman rho = {rho:.4f}  (p = {rho_p:.4e})")

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(x, y, alpha=0.3, s=6, color="steelblue", rasterized=True)

    lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], color="k", linewidth=0.8, linestyle="--", label="1:1 line")

    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(lo, hi, 100)
    ax.plot(x_line, m * x_line + b, color="red", linewidth=1.5, label="Fitted trend")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(BASELINE_LABEL, fontsize=13)
    ax.set_ylabel(ARITHMETIC_LABEL, fontsize=13)
    ax.tick_params(labelsize=11)

    ax.text(
        0.03, 0.97,
        f"Pearson $r$ = {r:.3f}\nSpearman $\\rho$ = {rho:.3f}",
        transform=ax.transAxes, fontsize=12, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"),
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
