"""
plot_rucc_score_comparison.py

Grouped bar chart of mean hazard, vulnerability, exposure, and combined risk
scores across all nine USDA Rural-Urban Continuum Codes (RUCC_2023), for the
full-period (2006-2024) risk index. A dashed line separates Metro (RUCC 1-3)
from Nonmetro (RUCC 4-9) counties.

Tracts with no RUCC_2023 value (unclassified Connecticut tracts) are excluded.

Output:
  Maps/TimeBasedFloodIndex_Maps/2006-2024/RUCC_ScoreComparison_MinMax.png
  Maps/TimeBasedFloodIndex_Maps/2006-2024/RUCC_ScoreComparison_Rank.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True


# Package layout: <package_root>/scripts/05_rucc_analysis/plot_rucc_score_comparison.py
# Reads and writes results (risk index CSVs + figure) in data/results/.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR   = PACKAGE_ROOT / "data" / "results" / "TimeBasedFloodIndex" / "2006-2024"
OUTPUT_DIR = PACKAGE_ROOT / "data" / "results" / "Maps" / "TimeBasedFloodIndex_Maps" / "2006-2024"


SCORES = [
    ("hazard_score",        "Hazard"),
    ("vulnerability_score", "Vulnerability"),
    ("exposure_score",      "Exposure (combined)"),
    ("combined_risk_score", "Combined Risk"),
]

RUCC_CODES = list(range(1, 10))


DPI     = 150
COLORS  = plt.cm.tab10.colors


def plot_rucc_comparison(df: pd.DataFrame, output_path: Path) -> None:
    means = pd.DataFrame({
        label: df.groupby("RUCC_2023")[col].mean().reindex(RUCC_CODES)
        for col, label in SCORES
    })

    fig, ax = plt.subplots(figsize=(14, 8))

    n_series = len(SCORES)
    bar_width = 0.8 / n_series
    x = np.arange(len(RUCC_CODES))

    for i, (_, label) in enumerate(SCORES):
        offset = (i - (n_series - 1) / 2) * bar_width
        ax.bar(
            x + offset, means[label], width=bar_width,
            label=label, color=COLORS[i], edgecolor="white", linewidth=0.5,
        )

    ax.axvline(2.5, color="#444444", linestyle="--", linewidth=1)
    ax.text(1.0, ax.get_ylim()[1], "Metro", ha="center", va="bottom", fontsize=13, color="#444444")
    ax.text(6.0, ax.get_ylim()[1], "Nonmetro", ha="center", va="bottom", fontsize=13, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(RUCC_CODES)
    ax.set_xlabel("RUCC Code (2023)")
    ax.set_ylabel("Mean Score")
    ax.legend(title="Risk Index Components", fontsize=12, title_fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, out_name in [
        ("2006-2024_geometric_MinMaxRiskIndex.csv", "RUCC_ScoreComparison_MinMax.png"),
        ("2006-2024_geometric_RankRiskIndex.csv",   "RUCC_ScoreComparison_Rank.png"),
    ]:
        print(f"\nPlotting {out_name}...")
        df = pd.read_csv(BASE_DIR / filename)
        plot_rucc_comparison(df, OUTPUT_DIR / out_name)

    print(f"\nAll RUCC comparison plots saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
