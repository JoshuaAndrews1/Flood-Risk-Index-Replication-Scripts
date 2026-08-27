"""

Plots the distribution (KDE) of every risk index component.

Covers the full period (2006-2024).

Note: the original version of this script also plotted per-5-year-period
distributions; that section is omitted here because this replication package
only ships the full-period (2006-2024) index data used by the paper.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Package layout: <package_root>/scripts/07_distributions_summary/plot_index_distributions_fullperiod.py
# Reads and writes results (risk index CSVs + figures) in data/results/.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR   = PACKAGE_ROOT / "data" / "results" / "TimeBasedFloodIndex" / "2006-2024"
OUTPUT_DIR = PACKAGE_ROOT / "data" / "results" / "Maps" / "TimeBasedFloodIndex_Maps" / "2006-2024" / "Distributions"


SCORES = [
    ("hazard_score",         "Hazard"),
    ("vulnerability_score",  "Vulnerability"),
    ("exposure_score",       "Exposure (combined)"),
    ("urban_exposure_score", "Urban Exposure"),
    ("rural_exposure_score", "Rural Exposure"),
    ("combined_risk_score",  "Combined Risk"),
]


DPI     = 150
PALETTE = "tab10"

sns.set_theme(style="whitegrid", font="cmr10")
plt.rcParams["axes.formatter.use_mathtext"] = True



def print_summary_stats(df: pd.DataFrame, period_label: str) -> None:
    print(f"\n  Summary statistics ({period_label}):")
    print(f"  {'Index':<20} {'N':>8} {'Mean':>8} {'SD':>8} {'Min':>8} {'Median':>8} {'Max':>8}")
    print("  " + "-" * 76)
    for col, label in SCORES:
        vals = df[col].dropna()
        print(f"  {label:<20} {len(vals):>8,} {vals.mean():>8.3f} {vals.std():>8.3f} "
              f"{vals.min():>8.3f} {vals.median():>8.3f} {vals.max():>8.3f}")



def plot_distributions(df: pd.DataFrame, period_label: str, output_path: Path) -> None:
    print_summary_stats(df, period_label)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette(PALETTE, n_colors=len(SCORES))

    for (col, label), color in zip(SCORES, colors):
        vals = df[col].dropna()
        sns.kdeplot(
            vals, ax=ax, label=f"{label}  (mean={vals.mean():.2f})",
            color=color, linewidth=2, fill=True, alpha=0.08, clip=(0, 1),
        )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend(title="Risk Index Components", fontsize=13, title_fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {output_path}")



def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, out_name in [
        ("2006-2024_MinMaxRiskIndex.csv", "IndexDistributions_MinMax.png"),
        ("2006-2024_RankRiskIndex.csv",   "IndexDistributions_Rank.png"),
    ]:
        print(f"\nPlotting {out_name}...")
        df = pd.read_csv(BASE_DIR / filename)
        plot_distributions(df, "Full Period", OUTPUT_DIR / out_name)

    print(f"\nFull-period distribution plots saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
