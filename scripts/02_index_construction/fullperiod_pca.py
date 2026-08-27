

"""
Creates PCA risk index for the full 2006-2024 composite period.
Exposure is split into urban and rural exposure - combined via weighted average

"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# CONFIG

# Package layout: <package_root>/scripts/02_index_construction/fullperiod_pca.py
# Reads Combined Data (an input) from data/reconstruction/; writes the
# computed risk index CSVs (a result) to data/results/.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR   = PACKAGE_ROOT / "data" / "reconstruction" / "TimeBasedFloodIndex" / "2006-2024" / "Combined Data"
OUTPUT_DIR = PACKAGE_ROOT / "data" / "results" / "TimeBasedFloodIndex" / "2006-2024"

LOG_HAZARD        = True
# "avg_rainfall_mm", "pct_100yr_flood_area"
HAZARD_LOG_COLS   = ["annual_flood_rate_per_sqkm"]

LOG_EXPOSURE      = True
EXPOSURE_LOG_COLS = ["pop_density_per_sqkm", "housing_density_per_sqkm"]

# Variables whose PCA loading is forced positive before computing PC1 scores. For testing purposes
FORCE_POSITIVE_LOADINGS = {
    "hazard":         [],
    "vulnerability":  [],
    "urban_exposure": [],
}

HAZARD_VARS        = ["annual_flood_rate_per_sqkm", "avg_rainfall_mm", "pct_100yr_flood_area"]
VULNERABILITY_VARS = ["pct_poverty_rate", "median_household_income", "pct_renters"]
URBAN_EXPOSURE_VARS = ["pop_density_per_sqkm", "housing_density_per_sqkm", "pct_houses_built_pre1980"]
RURAL_EXPOSURE_VAR = "pct_farmland_area"

# Weights for combining the exposure sub-indices
URBAN_EXPOSURE_WEIGHT = 0.5
RURAL_EXPOSURE_WEIGHT = 0.5


# How hazard_score, vulnerability_score, and exposure_score are combined 
# "equal_weight" - mean (1/3 each)
# "multiplicative" - (hazard_score x vulnerability_score x exposure_score)^(1/3).
COMBINE_METHOD = "multiplicative"

NON_CONTINENTAL_FIPS = {"02", "15", "60", "66", "69", "72", "74", "78"}



# Strip State fip code from geoid
def _state_fips(geoid: str) -> str:
    g = str(geoid)
    if "US" in g:
        g = g[g.index("US") + 2:]
    return g[:2]


# Get rid of non continental tracts
def _continental(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("GEOID")
    return df[~df.index.map(_state_fips).isin(NON_CONTINENTAL_FIPS)]


def _apply_log(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = np.log1p(df[col])
    return df


def run_pca(df: pd.DataFrame, variables: list, label: str,
            force_positive: list = None) -> pd.Series:
    # First fit StandardScaler + PCA; return raw PC1 Series.

   
    subset = df[variables].copy().dropna()
    n_dropped = len(df) - len(subset)
    if n_dropped:
        print(f"  [{label}] Dropping {n_dropped:,} rows with missing values")


    scaler   = StandardScaler()
    pca      = PCA(n_components=1)

    # PCA is sensitive to scale so you must standardize first. 
    X_scaled = scaler.fit_transform(subset)

    # Find PC1 
    pca.fit(X_scaled)

    loadings = pca.components_[0].copy()
    if force_positive:
        for var in force_positive:
            if var in variables:
                idx = variables.index(var)
                loadings[idx] = abs(loadings[idx])
    pc1 = X_scaled @ loadings


    pct = pca.explained_variance_ratio_[0] * 100
    print(f"  [{label}] PC1 explains {pct:.1f}% of variance")
    for var, loading in zip(variables, loadings):
        print(f"    {var:42s} {loading:+.4f}")

    return pd.Series(pc1, index=subset.index)


def _extract_rural(df: pd.DataFrame, var: str, label: str) -> pd.Series:
    # Return the raw (undropped-NaN) rural exposure variable, no PCA needed.
    s = df[var].copy()
    n_missing = s.isna().sum()
    if n_missing:
        print(f"  [{label}] Dropping {n_missing:,} rows with missing values")
    return s.dropna()


def _rank_normalize(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def _minmax_normalize(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return pd.Series(0.0, index=s.index) if rng == 0 else (s - s.min()) / rng


# Take the weighted average of urban_exposure_score and rural_exposure_score.
def _combine_exposure(scores: pd.DataFrame) -> pd.Series:

    exposure_cols = ["urban_exposure_score", "rural_exposure_score"]
    complete = scores[exposure_cols].notna().all(axis=1)
    exposure_score = pd.Series(np.nan, index=scores.index)
    exposure_score.loc[complete] = (
        URBAN_EXPOSURE_WEIGHT * scores.loc[complete, "urban_exposure_score"]
        + RURAL_EXPOSURE_WEIGHT * scores.loc[complete, "rural_exposure_score"]
    )
    return exposure_score



def _combine_risk(scores: pd.DataFrame, combined_cols: list) -> pd.Series:
    
    """Combine hazard/vulnerability/exposure component scores

    equal_weight   - mean of the three components (equal 1/3 weights)
    multiplicative - geometric mean of the three components:
                      (hazard_score x vulnerability_score x exposure_score)^(1/3)
    """
   
    complete = scores[combined_cols].notna().all(axis=1)
    raw = pd.Series(np.nan, index=scores.index)
    if COMBINE_METHOD == "multiplicative":
        raw.loc[complete] = scores.loc[complete, combined_cols].prod(axis=1) ** (1 / 3)
    else:
        raw.loc[complete] = scores.loc[complete, combined_cols].mean(axis=1)
    return raw


def _build_output(raw: dict, normalize_fn) -> pd.DataFrame:
    """Apply normalize_fn to each raw PC1/rural Series, compute combined score."""
    scores = pd.DataFrame({
        "hazard_score":         normalize_fn(raw["hazard"]),
        "vulnerability_score":  normalize_fn(raw["vulnerability"]),
        "urban_exposure_score": normalize_fn(raw["urban_exposure"]),
        "rural_exposure_score": normalize_fn(raw["rural_exposure"]),
    })
    raw_exposure = _combine_exposure(scores).dropna()
    scores["exposure_score"] = normalize_fn(raw_exposure).reindex(scores.index)

    combined_cols = ["hazard_score", "vulnerability_score", "exposure_score"]
    raw_combined = _combine_risk(scores, combined_cols).dropna()
    scores["combined_risk_score"] = normalize_fn(raw_combined).reindex(scores.index)
    return scores.reset_index()


def _save_and_summarize(result: pd.DataFrame, path: Path, label: str) -> None:
    result.to_csv(path, index=False)
    print(f"\n  [{label}] Saved {len(result):,} tracts -> {path}")
    for col in ["hazard_score", "vulnerability_score",
                "urban_exposure_score", "rural_exposure_score",
                "exposure_score", "combined_risk_score"]:
        s = result[col].dropna()
        print(f"    {col:30s}  n={len(s):,}  min={s.min():.3f}  "
              f"max={s.max():.3f}  mean={s.mean():.3f}")


def main():
    print("Full-Period (2006-2024) FloodIndex - PCA Risk Index")
    print(f"Log hazard: {LOG_HAZARD}  |  Log exposure: {LOG_EXPOSURE}")
    print(f"Combine method: {COMBINE_METHOD}")

    # Strip off noncontinental us tracks
    hazard   = _continental(pd.read_csv(DATA_DIR / "hazard.csv",        dtype={"GEOID": str}))
    vuln     = _continental(pd.read_csv(DATA_DIR / "vulnerability.csv", dtype={"GEOID": str}))
    exposure = _continental(pd.read_csv(DATA_DIR / "exposure.csv",      dtype={"GEOID": str}))

    print(f"\n  hazard={len(hazard):,}  vulnerability={len(vuln):,}  exposure={len(exposure):,}")

    # Apply specified logs
    if LOG_HAZARD:
        hazard = _apply_log(hazard, HAZARD_LOG_COLS)
    if LOG_EXPOSURE:
        exposure = _apply_log(exposure, EXPOSURE_LOG_COLS)

    # Run PCA 
    raw = {
        "hazard":         run_pca(hazard,   HAZARD_VARS,          "hazard",         FORCE_POSITIVE_LOADINGS["hazard"]),
        "vulnerability":  run_pca(vuln,     VULNERABILITY_VARS,   "vulnerability",  FORCE_POSITIVE_LOADINGS["vulnerability"]),
        "urban_exposure": run_pca(exposure, URBAN_EXPOSURE_VARS,  "urban_exposure", FORCE_POSITIVE_LOADINGS["urban_exposure"]),
        "rural_exposure": _extract_rural(exposure, RURAL_EXPOSURE_VAR, "rural_exposure"),
    }

    # Normalization setup
    for norm_name, norm_fn, filename in [
        ("rank",   _rank_normalize,   "2006-2024_RankRiskIndex.csv"),
        ("minmax", _minmax_normalize, "2006-2024_MinMaxRiskIndex.csv"),
    ]:
        result = _build_output(raw, norm_fn)
        _save_and_summarize(result, OUTPUT_DIR / filename, norm_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
