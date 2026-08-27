"""

Compares Metro (RUCC_2023 1-3) vs Nonmetro (RUCC_2023 4-9) tracts across all
risk index components, for both the MinMax- and Rank-normalized full-period
(2006-2024) risk index CSVs.

For each index column, reports per-group count, mean, median, std, and a
Welch's t-test (unequal variance) for the metro/nonmetro mean difference.

Tracts with no RUCC_2023 value (unclassified Connecticut tracts) are excluded.

Requirements:
    pip install pandas scipy
"""

import pandas as pd
from scipy import stats
from pathlib import Path

# Package layout: <package_root>/scripts/05_rucc_analysis/metro_nonmetro_analysis.py
# Reads and writes results (risk index + comparison CSVs) in data/results/.
BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "results" / "TimeBasedFloodIndex" / "2006-2024"

FILES = {
    "MinMax": BASE_DIR / "2006-2024_MinMaxRiskIndex.csv",
    "Rank":   BASE_DIR / "2006-2024_RankRiskIndex.csv",
}

INDEX_COLS = [
    "hazard_score",
    "vulnerability_score",
    "urban_exposure_score",
    "rural_exposure_score",
    "exposure_score",
    "combined_risk_score",
]

METRO_CODES = {1, 2, 3}
NONMETRO_CODES = {4, 5, 6, 7, 8, 9}


def classify(rucc: float) -> str:
    if pd.isna(rucc):
        return None
    rucc = int(rucc)
    if rucc in METRO_CODES:
        return "Metro"
    if rucc in NONMETRO_CODES:
        return "Nonmetro"
    return None


def analyze_file(label: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["group"] = df["RUCC_2023"].apply(classify)

    n_excluded = df["group"].isna().sum()
    print(f"\n{'=' * 78}")
    print(f"  {label}RiskIndex  ({path.name})")
    print(f"{'=' * 78}")
    print(f"  Metro (RUCC 1-3):    {(df['group'] == 'Metro').sum():,} tracts")
    print(f"  Nonmetro (RUCC 4-9): {(df['group'] == 'Nonmetro').sum():,} tracts")
    print(f"  Excluded (no RUCC):  {n_excluded:,} tracts")

    rows = []
    for col in INDEX_COLS:
        metro    = df.loc[df["group"] == "Metro", col].dropna()
        nonmetro = df.loc[df["group"] == "Nonmetro", col].dropna()

        t_stat, p_value = stats.ttest_ind(metro, nonmetro, equal_var=False)

        rows.append({
            "index":            col,
            "metro_n":          len(metro),
            "metro_mean":       metro.mean(),
            "metro_median":     metro.median(),
            "metro_sd":         metro.std(),
            "nonmetro_n":       len(nonmetro),
            "nonmetro_mean":    nonmetro.mean(),
            "nonmetro_median":  nonmetro.median(),
            "nonmetro_sd":      nonmetro.std(),
            "t_stat":           t_stat,
            "p_value":          p_value,
        })

    result = pd.DataFrame(rows)

    print()
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return result


def main():
    for label, path in FILES.items():
        result = analyze_file(label, path)
        out_path = path.with_name(f"{path.stem}_MetroVsNonmetro.csv")
        result.to_csv(out_path, index=False)
        print(f"\n  Saved -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
