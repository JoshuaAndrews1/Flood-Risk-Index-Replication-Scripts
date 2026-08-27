"""

Spatial autocorrelation analysis for the 2006-2024 full-period flood risk index.

For each index file this creates:
  - Pearson & Spearman correlation matrices + heatmap
  - Global Moran's I table
  - Moran scatter plots
  - LISA cluster maps

"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import libpysal
from esda.moran import Moran, Moran_Local
from itertools import combinations
from pathlib import Path

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True


# Package layout: <package_root>/scripts/04_spatial_autocorrelation/spatial_autocorrelation_fullperiod.py
# Risk index CSVs (results) live in data/results/; the tract shapefile
# (an input) lives in data/reconstruction/. Generated figures go to results/.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR  = PACKAGE_ROOT / "data" / "results" / "TimeBasedFloodIndex" / "2006-2024"
OUTPUT_DIR = PACKAGE_ROOT / "data" / "results" / "SpatialAutocorrelation" / "2006-2024"
SHAPEFILE  = PACKAGE_ROOT / "data" / "reconstruction" / "CensusTractShapeFiles" / "tl_2020_us_tract.shp"

INDEX_FILES = [
    ("2006-2024_MinMaxRiskIndex.csv", "Min-Max Normalized (0-1)"),
    ("2006-2024_RankRiskIndex.csv",   "Rank Normalized (percentile)"),
]

SCORE_COLS = [
    "hazard_score",
    "vulnerability_score",
    "exposure_score",
    "combined_risk_score",
]

SCORE_LABELS = {
    "hazard_score":        "Hazard",
    "vulnerability_score": "Vulnerability",
    "exposure_score":      "Exposure",
    "combined_risk_score": "Combined Risk",
}

NON_CONUS = {"02", "15", "60", "66", "69", "72", "74", "78"}
PROJECTION = "ESRI:102003"   # USA Contiguous Albers Equal Area Conic
LISA_P     = 0.01

QUAD_LABELS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
QUAD_COLORS = {
    "HH": "#d7191c",
    "LL": "#2c7bb6",
    "LH": "#abd9e9",
    "HL": "#fdae61",
    "ns": "#eeeeee",
}


# Shapefile 

def load_shapefile() -> gpd.GeoDataFrame:
    print("Loading 2020 census tract shapefile...")
    gdf = gpd.read_file(SHAPEFILE)
    gdf = gdf[gdf["ALAND"] > 0].copy()          # drop water-only coastal tracts
    gdf["GEOID"] = gdf["GEOID"].str.zfill(11)
    gdf = gdf[~gdf["STATEFP"].isin(NON_CONUS)].copy()
    gdf = gdf.to_crs(PROJECTION)
    print(f"  Continental tracts: {len(gdf):,}")
    return gdf


# Correlation analysis

def run_correlations(df: pd.DataFrame, cols: list, out_dir: Path) -> None:
    scores = df[cols]
    tick_labels = [SCORE_LABELS.get(c, c) for c in cols]

    pearson  = scores.corr(method="pearson")
    spearman = scores.corr(method="spearman")

    print(f"\n  Pearson correlation matrix:")
    print(pearson.round(4).to_string())
    print(f"\n  Spearman correlation matrix:")
    print(spearman.round(4).to_string())

    print(f"\n  Pairwise Pearson r (with p-values):")
    print(f"  {'Pair':<55} {'r':>8}  {'p-value':>12}")
    print("  " + "-" * 78)
    from scipy import stats
    for a, b in combinations(cols, 2):
        valid = df[[a, b]].dropna()
        r, p = stats.pearsonr(valid[a], valid[b])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {a} vs {b:<28} {r:>8.4f}  {p:>12.4e}  {sig}")

    print(f"\n  Pairwise Spearman rho (with p-values):")
    print(f"  {'Pair':<55} {'rho':>8}  {'p-value':>12}")
    print("  " + "-" * 78)
    for a, b in combinations(cols, 2):
        valid = df[[a, b]].dropna()
        rho, p = stats.spearmanr(valid[a], valid[b])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {a} vs {b:<28} {rho:>8.4f}  {p:>12.4e}  {sig}")

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(cols) * 2.5) * 2,
                                            max(5, len(cols) * 1.5)))
    for ax, matrix, title in zip(
        axes,
        [pearson, spearman],
        ["Pearson Correlation", "Spearman Correlation"],
    ):
        sns.heatmap(
            matrix, annot=True, fmt=".3f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5,
            xticklabels=tick_labels, yticklabels=tick_labels,
        )
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    plt.tight_layout()
    path = out_dir / "correlation_heatmaps.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved -> {path.name}")


# Global Moran's I 

def run_global_morans(gdf: gpd.GeoDataFrame, w, cols: list) -> None:
    print(f"\n  Global Moran's I:")
    print(f"  {'Variable':<30} {'Moran I':>9}  {'p-value (MC)':>14}  {'z-score':>9}")
    print("  " + "-" * 68)
    for col in cols:
        mi  = Moran(gdf[col], w, permutations=999)
        sig = "***" if mi.p_sim < 0.001 else "**" if mi.p_sim < 0.01 else "*" if mi.p_sim < 0.05 else ""
        print(f"  {col:<30} {mi.I:>9.4f}  {mi.p_sim:>14.4f}  {mi.z_sim:>9.4f}  {sig}")


# Moran scatter plots

def plot_moran_scatter(gdf: gpd.GeoDataFrame, w, cols: list,
                       out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        mi  = Moran(gdf[col], w, permutations=999)
        lag = libpysal.weights.lag_spatial(w, gdf[col].values)
        z     = (gdf[col].values - gdf[col].mean()) / gdf[col].std()
        z_lag = (lag - lag.mean()) / lag.std()

        # Clip display to 1st-99th percentile so outliers don't compress the view.
        # Points outside the window are still included in Moran's I above.
        x_lo, x_hi = np.percentile(z,     [.5, 99.5])
        y_lo, y_hi = np.percentile(z_lag, [.5, 99.5])
        mask = (z >= x_lo) & (z <= x_hi) & (z_lag >= y_lo) & (z_lag <= y_hi)

        ax.scatter(z[mask], z_lag[mask], alpha=0.3, s=6, color="steelblue", rasterized=True)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.axvline(0, color="k", linewidth=0.8)

        m = np.cov(z, z_lag)[0, 1] / np.var(z)
        b = z_lag.mean() - m * z.mean()
        x_line = np.linspace(x_lo, x_hi, 100)
        ax.plot(x_line, m * x_line + b, color="red", linewidth=1.5)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(
            f"{SCORE_LABELS.get(col, col)}\nMoran's I = {mi.I:.4f}  p = {mi.p_sim:.4f}",
            fontsize=20, fontweight="bold",
        )
        ax.set_xlabel("Standardised value", fontsize=13)
        ax.set_ylabel("Spatial lag", fontsize=13)
        ax.tick_params(labelsize=11)

    plt.tight_layout()
    path = out_dir / "morans_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path.name}")


# LISA cluster maps 

def plot_lisa_maps(gdf: gpd.GeoDataFrame, w, cols: list,
                   out_dir: Path) -> None:
    ncols = 2
    nrows = (len(cols) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    print(f"\n  LISA cluster counts (p < {LISA_P}):")
    for ax, col in zip(axes, cols):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in divide")
            lisa = Moran_Local(gdf[col].values, w, permutations=999)

        sig  = lisa.p_sim < LISA_P
        quad = np.where(sig, np.vectorize(QUAD_LABELS.get)(lisa.q), "ns")

        gdf = gdf.copy()
        gdf["_quad"]  = quad
        gdf["_color"] = gdf["_quad"].map(QUAD_COLORS)

        gdf.plot(color=gdf["_color"], ax=ax, linewidth=0)

        title = SCORE_LABELS.get(col, col)
        ax.set_title(f"{title}\n(p $<$ {LISA_P})", fontsize=20, fontweight="bold")
        ax.set_axis_off()

        legend_elements = [
            mpatches.Patch(facecolor=c, label=k) for k, c in QUAD_COLORS.items()
        ]
        ax.legend(handles=legend_elements, loc="lower left", fontsize=11, framealpha=0.8)

        counts = {k: int((quad == k).sum()) for k in ["HH", "HL", "LH", "LL", "ns"]}
        print(f"  {col}: " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    path = out_dir / "lisa_maps.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path.name}")




def main() -> None:
    gdf_base = load_shapefile()

    for filename, label in INDEX_FILES:
        index_name = filename.replace(".csv", "")
        out_dir = OUTPUT_DIR / index_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 64}")
        print(f"  {index_name}  ({label})")
        print(f"{'=' * 64}")

        df = pd.read_csv(INDEX_DIR / filename, dtype={"GEOID": str})
        df["GEOID"] = df["GEOID"].str.zfill(11)

        # Keep only score columns that are actually present
        cols = [c for c in SCORE_COLS if c in df.columns]

        # Merge scores onto shapefile
        gdf = gdf_base.merge(df[["GEOID"] + cols], on="GEOID", how="inner")
        print(f"  Tracts after join: {len(gdf):,}")

        # Drop rows where any score is NaN before building weights
        gdf = gdf.dropna(subset=cols).reset_index(drop=True)
        print(f"  Tracts with complete scores: {len(gdf):,}")

        # Correlation
        print("\n[Correlation]")
        run_correlations(df, cols, out_dir)

        # Spatial weights 
        print("\n[Spatial weights]")
        w = libpysal.weights.Queen.from_dataframe(gdf)
        if w.islands:
            print(f"  Note: {len(w.islands)} island(s) with no neighbours — excluded from LISA.")
        w.transform = "r"

        # Global Moran's I
        print("\n[Global Moran's I]")
        run_global_morans(gdf, w, cols)

        # Moran scatter plots 
        print("\n[Moran scatter plots]")
        plot_moran_scatter(gdf, w, cols, out_dir)

        # LISA maps 
        print("\n[LISA cluster maps]")
        plot_lisa_maps(gdf, w, cols, out_dir)

        print(f"\n  All outputs saved to: {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
