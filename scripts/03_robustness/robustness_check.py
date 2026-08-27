"""

Robustness check for the full-period (2006-2024) baseline index:
2006-2024_geometric_MinMaxRiskIndex.csv  (PCA dimension scores, multiplicative
[geometric] combine of hazard/vulnerability/exposure, min-max normalization).

Compares the baseline's combined_risk_score against a set of alternative
construction choices and reports Spearman rho, Pearson r, and top-1% overlap
for each

Alternatives compared:
  1. Already-built axes in TimeBasedFloodIndex/2006-2024/:
       - Rank normalization        (2006-2024_geometric_RankRiskIndex.csv)
       - Arithmetic combine        (2006-2024_arithmetic_MinMaxRiskIndex.csv)
       - Arithmetic + rank         (2006-2024_arithmetic_RankRiskIndex.csv)
  2. Generated into TimeBasedFloodIndex/2006-2024/Robustness Alternatives/:
       - PC1 -> equal-weight-within-dimension 
         same combine/normalize as baseline, but each of the three PCA dimensions -- hazard,
         vulnerability, urban_exposure is built as an equal-weight average
         instead of the PC1 projection. Direction of each indicator is aligned using the sign of its baseline PCA
         loading, so "higher = higher risk" stays consistent with the
         baseline; only the weighting scheme changes.)

       - Leave-one-indicator-out (10 alternatives, one per indicator): drop
         one indicator  (or drop the rural exposure sub-score entirely if pct_farmland_area is
         the dropped indicator), everything else is identical to baseline.

Alternative index CSVs are only (re)generated if missing -- if
Robustness Alternatives/ already contains all expected files, generation is
skipped and the script goes straight to computing the comparison table.

Sensitivity analysis (indicator-level influence on the baseline index):
  1. PCA loadings --  reports each indicator's loading and its share of the
     dimension's total |loading|
  2. Leave-one-out impact -- reuses the leave-one-out alternatives above and
     ranks indicators by how much combined_risk_score shifts when that
     indicator is dropped (1 - Spearman rho vs. baseline, plus the mean
     absolute percentile-rank shift across tracts). 

Requirements:
    pip install pandas numpy scikit-learn matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True

# CONFIG 
# Package layout: <package_root>/scripts/03_robustness/robustness_check.py
# Combined Data (dimension-level indicators) is an input (data/reconstruction/);
# the baseline/alternative index CSVs and summary outputs are results (data/results/).
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR     = PACKAGE_ROOT / "data" / "results" / "TimeBasedFloodIndex" / "2006-2024"
DATA_DIR     = PACKAGE_ROOT / "data" / "reconstruction" / "TimeBasedFloodIndex" / "2006-2024" / "Combined Data"
ALT_DIR      = BASE_DIR / "Robustness Alternatives"
BASELINE_CSV = BASE_DIR / "2006-2024_geometric_MinMaxRiskIndex.csv"
MAPS_DIR     = PACKAGE_ROOT / "data" / "results" / "Maps" / "TimeBasedFloodIndex_Maps" / "2006-2024"

# Pre-existing alternative axes (built by fullperiod_pca.py already; not regenerated here)
EXISTING_AXES = [
    ("Rank normalization",        BASE_DIR / "2006-2024_geometric_RankRiskIndex.csv"),
    ("Arithmetic combine",        BASE_DIR / "2006-2024_arithmetic_MinMaxRiskIndex.csv"),
    ("Arithmetic combine + rank", BASE_DIR / "2006-2024_arithmetic_RankRiskIndex.csv"),
]

SUMMARY_CSV     = BASE_DIR / "2006-2024_RobustnessCheck_Summary.csv"
SENSITIVITY_CSV = BASE_DIR / "2006-2024_IndicatorSensitivity_Summary.csv"
SENSITIVITY_PNG = MAPS_DIR / "IndicatorSensitivity.png"

# Same settings as fullperiod_pca.py, held fixed so only one thing changes at a time.
LOG_HAZARD        = True
HAZARD_LOG_COLS   = ["annual_flood_rate_per_sqkm"]

LOG_EXPOSURE      = True
EXPOSURE_LOG_COLS = ["pop_density_per_sqkm", "housing_density_per_sqkm"]

HAZARD_VARS        = ["annual_flood_rate_per_sqkm", "avg_rainfall_mm", "pct_100yr_flood_area"]
VULNERABILITY_VARS = ["pct_poverty_rate", "median_household_income", "pct_renters"]
URBAN_EXPOSURE_VARS = ["pop_density_per_sqkm", "housing_density_per_sqkm", "pct_houses_built_pre1980"]
RURAL_EXPOSURE_VAR  = "pct_farmland_area"

# Display labels for chart axes -- mathtext (enabled below) renders raw "_"
# column names as subscripts, so plots use these instead.
INDICATOR_LABELS = {
    "annual_flood_rate_per_sqkm": "Annual Flood Rate (per sqkm)",
    "avg_rainfall_mm":            "Avg. Rainfall (mm)",
    "pct_100yr_flood_area":       "% in 100-yr Flood Zone",
    "pct_poverty_rate":           "Poverty Rate (%)",
    "median_household_income":    "Median Household Income",
    "pct_renters":                "Renters (%)",
    "pop_density_per_sqkm":       "Population Density (per sqkm)",
    "housing_density_per_sqkm":   "Housing Density (per sqkm)",
    "pct_houses_built_pre1980":   "Houses Built Pre-1980 (%)",
    "pct_farmland_area":          "Farmland Area (%)",
}

URBAN_EXPOSURE_WEIGHT = 0.5
RURAL_EXPOSURE_WEIGHT = 0.5

TOP_PCT = 0.01 

NON_CONTINENTAL_FIPS = {"02", "15", "60", "66", "69", "72", "74", "78"}



def _state_fips(geoid: str) -> str:
    g = str(geoid)
    if "US" in g:
        g = g[g.index("US") + 2:]
    return g[:2]


def _continental(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("GEOID")
    return df[~df.index.map(_state_fips).isin(NON_CONTINENTAL_FIPS)]


def _apply_log(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = np.log1p(df[col])
    return df


def _minmax_normalize(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return pd.Series(0.0, index=s.index) if rng == 0 else (s - s.min()) / rng


def run_pca(df: pd.DataFrame, variables: list, label: str) -> pd.Series:
    """Fit StandardScaler + PCA(1); return raw PC1 Series. Same method as baseline."""
    subset = df[variables].copy().dropna()
    n_dropped = len(df) - len(subset)
    if n_dropped:
        print(f"  [{label}] Dropping {n_dropped:,} rows with missing values")

    scaler   = StandardScaler()
    pca      = PCA(n_components=1)
    X_scaled = scaler.fit_transform(subset)
    pca.fit(X_scaled)
    pc1 = X_scaled @ pca.components_[0]

    pct = pca.explained_variance_ratio_[0] * 100
    print(f"  [{label}] PC1 explains {pct:.1f}% of variance ({len(variables)} vars)")
    return pd.Series(pc1, index=subset.index)


def compute_dimension_loadings(hazard_df, vuln_df, exposure_df) -> pd.DataFrame:
    """Fit each baseline PCA dimension once more just to expose per-indicator
    loadings. 
    """
    dimensions = [
        ("hazard",         hazard_df,   HAZARD_VARS),
        ("vulnerability",  vuln_df,     VULNERABILITY_VARS),
        ("urban_exposure", exposure_df, URBAN_EXPOSURE_VARS),
    ]

    rows = []
    for dim_label, df, variables in dimensions:
        subset = df[variables].dropna()
        X_scaled = StandardScaler().fit_transform(subset)
        pca = PCA(n_components=1)
        pca.fit(X_scaled)
        loadings = pca.components_[0]
        abs_total = np.abs(loadings).sum()
        for var, loading in zip(variables, loadings):
            rows.append({
                "indicator":         var,
                "dimension":         dim_label,
                "pca_loading":       loading,
                "pca_loading_share": abs(loading) / abs_total,
            })

    # Rural exposure doesnt use PCA 
    rows.append({
        "indicator": RURAL_EXPOSURE_VAR, "dimension": "rural_exposure",
        "pca_loading": np.nan, "pca_loading_share": np.nan,
    })

    return pd.DataFrame(rows)


def run_equal_weight(df: pd.DataFrame, variables: list, label: str) -> pd.Series:
    """
    Equal weight within dimension alternative to PC1. 
    """
    subset = df[variables].copy().dropna()
    n_dropped = len(df) - len(subset)
    if n_dropped:
        print(f"  [{label}] Dropping {n_dropped:,} rows with missing values")

    X_scaled = StandardScaler().fit_transform(subset)

    pca = PCA(n_components=1)
    pca.fit(X_scaled)
    signs = np.sign(pca.components_[0])
    signs[signs == 0] = 1.0

    equal_weights = signs / len(variables)
    score = X_scaled @ equal_weights
    weight_str = ", ".join(f"{v}={w:+.3f}" for v, w in zip(variables, equal_weights))
    print(f"  [{label}] equal weights (sign-aligned): {weight_str}")
    return pd.Series(score, index=subset.index)


def _extract_rural(df: pd.DataFrame, var: str, label: str) -> pd.Series:
    s = df[var].copy()
    n_missing = s.isna().sum()
    if n_missing:
        print(f"  [{label}] Dropping {n_missing:,} rows with missing values")
    return s.dropna()


DIM_BUILDERS = {"pca": run_pca, "eqw": run_equal_weight}


def construct_index(hazard_df, vuln_df, exposure_df, *, hazard_vars, hazard_method,
                     vuln_vars, vuln_method, urban_vars, urban_method,
                     include_rural, label) -> pd.DataFrame:
    """Builds a combined_risk_score index using the given per-dimension choices.
    Combine method (geometric/multiplicative) and normalization (min-max) are
    fixed to match the baseline, only the dimension construction choices vary.
    """
    raw_hazard = DIM_BUILDERS[hazard_method](hazard_df, hazard_vars, f"{label} | hazard")
    raw_vuln   = DIM_BUILDERS[vuln_method](vuln_df, vuln_vars, f"{label} | vulnerability")
    raw_urban  = DIM_BUILDERS[urban_method](exposure_df, urban_vars, f"{label} | urban_exposure")

    scores = pd.DataFrame({
        "hazard_score":         _minmax_normalize(raw_hazard),
        "vulnerability_score":  _minmax_normalize(raw_vuln),
        "urban_exposure_score": _minmax_normalize(raw_urban),
    })

    if include_rural:
        raw_rural = _extract_rural(exposure_df, RURAL_EXPOSURE_VAR, f"{label} | rural_exposure")
        scores["rural_exposure_score"] = _minmax_normalize(raw_rural)

        exposure_cols = ["urban_exposure_score", "rural_exposure_score"]
        complete = scores[exposure_cols].notna().all(axis=1)
        raw_exposure = pd.Series(np.nan, index=scores.index)
        raw_exposure.loc[complete] = (
            URBAN_EXPOSURE_WEIGHT * scores.loc[complete, "urban_exposure_score"]
            + RURAL_EXPOSURE_WEIGHT * scores.loc[complete, "rural_exposure_score"]
        )
        raw_exposure = raw_exposure.dropna()
        scores["exposure_score"] = _minmax_normalize(raw_exposure).reindex(scores.index)
    else:
        scores["exposure_score"] = scores["urban_exposure_score"]

    combined_cols = ["hazard_score", "vulnerability_score", "exposure_score"]
    complete = scores[combined_cols].notna().all(axis=1)
    raw_combined = pd.Series(np.nan, index=scores.index)
    raw_combined.loc[complete] = scores.loc[complete, combined_cols].prod(axis=1) ** (1 / 3)
    raw_combined = raw_combined.dropna()
    scores["combined_risk_score"] = _minmax_normalize(raw_combined).reindex(scores.index)

    return scores.reset_index()



# Alternative specifications

def build_alt_specs() -> list:
    specs = [dict(
        key="equalweight",
        label="PC1 -> equal-weight-within-dimension",
        filename="2006-2024_equalweight_MinMaxRiskIndex.csv",
        hazard_vars=HAZARD_VARS, hazard_method="eqw",
        vuln_vars=VULNERABILITY_VARS, vuln_method="eqw",
        urban_vars=URBAN_EXPOSURE_VARS, urban_method="eqw",
        include_rural=True,
    )]

    all_indicators = (
        [("hazard", v) for v in HAZARD_VARS]
        + [("vulnerability", v) for v in VULNERABILITY_VARS]
        + [("urban_exposure", v) for v in URBAN_EXPOSURE_VARS]
        + [("rural_exposure", RURAL_EXPOSURE_VAR)]
    )

    for dim, var in all_indicators:
        spec = dict(
            key=f"leaveoneout_{var}",
            label=f"Leave-one-out: {var}",
            filename=f"2006-2024_leaveoneout_{var}_MinMaxRiskIndex.csv",
            hazard_vars=HAZARD_VARS, hazard_method="pca",
            vuln_vars=VULNERABILITY_VARS, vuln_method="pca",
            urban_vars=URBAN_EXPOSURE_VARS, urban_method="pca",
            include_rural=True,
        )
        if dim == "hazard":
            spec["hazard_vars"] = [v for v in HAZARD_VARS if v != var]
        elif dim == "vulnerability":
            spec["vuln_vars"] = [v for v in VULNERABILITY_VARS if v != var]
        elif dim == "urban_exposure":
            spec["urban_vars"] = [v for v in URBAN_EXPOSURE_VARS if v != var]
        elif dim == "rural_exposure":
            spec["include_rural"] = False
        specs.append(spec)

    return specs


def _load_dimension_data() -> tuple:
    """Load and log-transform the raw hazard/vulnerability/exposure tables
    exactly as fullperiod_pca.py does, for rebuilding dimension scores."""
    hazard   = _continental(pd.read_csv(DATA_DIR / "hazard.csv",        dtype={"GEOID": str}))
    vuln     = _continental(pd.read_csv(DATA_DIR / "vulnerability.csv", dtype={"GEOID": str}))
    exposure = _continental(pd.read_csv(DATA_DIR / "exposure.csv",      dtype={"GEOID": str}))

    if LOG_HAZARD:
        hazard = _apply_log(hazard, HAZARD_LOG_COLS)
    if LOG_EXPOSURE:
        exposure = _apply_log(exposure, EXPOSURE_LOG_COLS)

    return hazard, vuln, exposure


def generate_missing_alternatives(specs: list, hazard, vuln, exposure) -> None:
    missing = [s for s in specs if not (ALT_DIR / s["filename"]).exists()]
    if not missing:
        print(f"All {len(specs)} alternative index files already exist in "
              f"{ALT_DIR} -- skipping generation.")
        return

    print(f"Generating {len(missing)}/{len(specs)} missing alternative index files...")
    ALT_DIR.mkdir(parents=True, exist_ok=True)

    for spec in missing:
        out_path = ALT_DIR / spec["filename"]
        print(f"\n-- {spec['label']} --")
        result = construct_index(
            hazard, vuln, exposure,
            hazard_vars=spec["hazard_vars"], hazard_method=spec["hazard_method"],
            vuln_vars=spec["vuln_vars"], vuln_method=spec["vuln_method"],
            urban_vars=spec["urban_vars"], urban_method=spec["urban_method"],
            include_rural=spec["include_rural"],
            label=spec["label"],
        )
        result.to_csv(out_path, index=False)
        print(f"  Saved {len(result):,} tracts -> {out_path}")



# Comparison against baseline

def _load_scores(path: Path) -> pd.Series:
    """Return combined_risk_score indexed by GEOID, valid rows only.

    The dimension score CSVs keep the union of GEOIDs across hazard/
    vulnerability/exposure, so combined_risk_score is NaN wherever any
    component was missing for that tract - those rows must be dropped
    before comparing, or n_tracts overstates the real sample size.
    """
    df = pd.read_csv(path, dtype={"GEOID": str})
    return df.set_index("GEOID")["combined_risk_score"].dropna()


def compare_to_baseline(label: str, path: Path, baseline: pd.Series) -> dict:
    alt = _load_scores(path)
    common = baseline.index.intersection(alt.index)
    b = baseline.loc[common]
    a = alt.loc[common]

    n_top = max(1, round(len(common) * TOP_PCT))
    top_b = set(b.sort_values(ascending=False).index[:n_top])
    top_a = set(a.sort_values(ascending=False).index[:n_top])
    overlap = len(top_b & top_a) / n_top

    # Per-tract percentile shift - catches indicators that reshuffle many
    # tracts' relative standing even when the aggregate correlation stays high.
    mean_abs_pct_shift = (b.rank(pct=True) - a.rank(pct=True)).abs().mean()

    return {
        "alternative":               label,
        "n_tracts":                  len(common),
        "pearson_r":                 b.corr(a, method="pearson"),
        "spearman_rho":              b.corr(a, method="spearman"),
        "top1pct_overlap":           overlap,
        "n_top1pct":                 n_top,
        "mean_abs_percentile_shift": mean_abs_pct_shift,
    }


# Indicator sensitivity analysis

def build_sensitivity_summary(results: pd.DataFrame, loadings: pd.DataFrame) -> pd.DataFrame:
    """Isolate the leave-one-out rows and rank indicators by how much their
    removal shifts combined_risk_score 
    """
    loo = results[results["alternative"].str.startswith("Leave-one-out:")].copy()
    loo["indicator"] = loo["alternative"].str.replace("Leave-one-out: ", "", regex=False)
    loo = loo.merge(loadings, on="indicator", how="left")

    # 1 - rho: 0 means removing the indicator didn't change tract ordering at
    # all; higher means the indicator was doing more of the work.
    loo["influence_score"] = 1 - loo["spearman_rho"]
    loo = loo.sort_values("influence_score", ascending=False).reset_index(drop=True)
    loo.insert(0, "influence_rank", loo.index + 1)

    cols = [
        "influence_rank", "indicator", "dimension", "pca_loading", "pca_loading_share",
        "influence_score", "spearman_rho", "pearson_r", "top1pct_overlap",
        "mean_abs_percentile_shift", "n_tracts",
    ]
    return loo[cols]


def plot_sensitivity_tornado(sensitivity: pd.DataFrame, output_path: Path) -> None:
    """Horizontal tornado chart: mean absolute percentile-rank shift in
    combined_risk_score when each indicator is left out, sorted by impact,
    so any indicator with disproportionate influence stands out visually.
    """
    plotted = sensitivity.sort_values("mean_abs_percentile_shift")
    labels = plotted["indicator"].map(INDICATOR_LABELS).fillna(plotted["indicator"])

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(
        labels, plotted["mean_abs_percentile_shift"],
        color="#c0392b", edgecolor="white", linewidth=0.5,
    )
    for bar, rho in zip(bars, plotted["spearman_rho"]):
        ax.text(
            bar.get_width(), bar.get_y() + bar.get_height() / 2,
            f"  rho={rho:.3f}", va="center", fontsize=10, color="#444444",
        )

    ax.set_xlabel("Mean Absolute Percentile-Rank Shift (leave-one-out vs. baseline)")
    ax.set_ylabel("Indicator")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def main():
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(f"Baseline not found: {BASELINE_CSV}")

    hazard, vuln, exposure = _load_dimension_data()

    specs = build_alt_specs()
    generate_missing_alternatives(specs, hazard, vuln, exposure)

    print("\n" + "=" * 60)
    print("Computing robustness comparisons vs. baseline "
          "(2006-2024_geometric_MinMaxRiskIndex.csv)")
    print("=" * 60)

    baseline = _load_scores(BASELINE_CSV)

    comparisons = list(EXISTING_AXES)
    comparisons += [(s["label"], ALT_DIR / s["filename"]) for s in specs]

    rows = []
    for label, path in comparisons:
        if not path.exists():
            print(f"  [skip] Missing file for '{label}': {path}")
            continue
        rows.append(compare_to_baseline(label, path, baseline))

    results = pd.DataFrame(rows)
    results.to_csv(SUMMARY_CSV, index=False)

    with pd.option_context("display.float_format", "{:.4f}".format,
                            "display.width", 120, "display.max_rows", None):
        print()
        print(results.to_string(index=False))

    print(f"\nSaved summary -> {SUMMARY_CSV}")

    print("\n" + "=" * 60)
    print("Indicator sensitivity analysis")
    print("=" * 60)

    loadings = compute_dimension_loadings(hazard, vuln, exposure)
    print("\nPCA loadings by dimension (share of that dimension's total |loading|):")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(loadings.to_string(index=False))

    sensitivity = build_sensitivity_summary(results, loadings)
    sensitivity.to_csv(SENSITIVITY_CSV, index=False)

    print("\nIndicators ranked by leave-one-out influence on combined_risk_score:")
    with pd.option_context("display.float_format", "{:.4f}".format,
                            "display.width", 140, "display.max_rows", None):
        print(sensitivity.to_string(index=False))
    print(f"\nSaved sensitivity summary -> {SENSITIVITY_CSV}")

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    plot_sensitivity_tornado(sensitivity, SENSITIVITY_PNG)


if __name__ == "__main__":
    main()
