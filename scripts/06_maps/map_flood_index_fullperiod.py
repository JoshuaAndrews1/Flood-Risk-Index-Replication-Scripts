"""

Creates choropleth maps for the full-period (2006-2024) composite flood risk
index. One map per score column for each normalization method.

Output structure:
  Maps/TimeBasedFloodIndex_Maps/2006-2024/
    MinMaxRiskIndex/  -- 4 maps (hazard, vulnerability, exposure, combined)
    RankRiskIndex/    -- 4 maps (same scores, rank-normalized)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True


# Package layout: <package_root>/scripts/06_maps/map_flood_index_fullperiod.py
# Risk index CSVs (results) live in data/results/; the tract shapefile (an input) lives in data/reconstruction/. Generated maps go to results/.
PACKAGE_ROOT  = Path(__file__).resolve().parents[2]
BASE_DIR      = PACKAGE_ROOT / "data" / "results" / "TimeBasedFloodIndex" / "2006-2024"
OUTPUT_DIR    = PACKAGE_ROOT / "data" / "results" / "Maps" / "TimeBasedFloodIndex_Maps" / "2006-2024"
SHAPEFILE     = PACKAGE_ROOT / "data" / "reconstruction" / "CensusTractShapeFiles" / "tl_2020_us_tract.shp"


SCORES = [
    ("hazard_score",        "Hazard Score"),
    ("vulnerability_score", "Vulnerability Score"),
    ("exposure_score",      "Exposure Score"),
    ("combined_risk_score", "Combined Risk Score"),
]

# Combined 2x2 grid: hazard, vulnerability, exposure, combined risk.
COMBINED_GRID_SCORES = [
    ("hazard_score",        "Hazard Score"),
    ("vulnerability_score", "Vulnerability Score"),
    ("exposure_score",      "Exposure Score"),
    ("combined_risk_score", "Combined Risk Score"),
]

# Exposure sub-index row: rural, urban, combined exposure.
EXPOSURE_SUBINDEX_SCORES = [
    ("rural_exposure_score", "Rural / Agricultural Exposure"),
    ("urban_exposure_score", "Urban Exposure"),
    ("exposure_score",       "Combined Exposure"),
]


CMAP       = "RdYlGn_r"
DPI        = 150
PROJECTION = "ESRI:102003"   # USA Contiguous Albers Equal Area Conic

# Color scale mode:
#   "quantile" -- stretch color scale to the 1st-99th percentile of each score
#                 (maximizes visual contrast, but scale differs per map)
#   "fixed"    -- fixed 0-1 scale for every map (comparable across maps, but
#                 low-contrast if a score's values cluster in a narrow range)
SCALE_MODE = "quantile"

NON_CONUS = {"02", "15", "60", "66", "69", "72", "74", "78"}



def load_tract_shapefile() -> gpd.GeoDataFrame:
    print("  Loading 2020 census tract shapefile...")
    gdf = gpd.read_file(SHAPEFILE)
    gdf = gdf[gdf["ALAND"] > 0].copy()          # drop water-only coastal tracts
    gdf = gdf.to_crs(PROJECTION)
    gdf = gdf[~gdf["STATEFP"].isin(NON_CONUS)].copy()
    gdf["GEOID"] = gdf["GEOID"].str.zfill(11)
    print(f"  Contiguous-US tracts: {len(gdf):,}")
    return gdf



def plot_map(
    gdf_conus: gpd.GeoDataFrame,
    risk_df: pd.DataFrame,
    score_col: str,
    score_label: str,
    output_path: Path,
) -> None:
    merged = gdf_conus.merge(
        risk_df[["GEOID", score_col]],
        on="GEOID",
        how="left",
    )

    vals = risk_df[score_col].dropna()
    if SCALE_MODE == "fixed":
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(vals.quantile(0.01))
        vmax = float(vals.quantile(0.99))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))

    merged.plot(
        column=score_col,
        ax=ax,
        cmap=CMAP,
        norm=norm,
        linewidth=0,
        missing_kwds={"color": "#d0d0d0", "label": "No data"},
    )

    stats = f"n={len(vals):,}  |  median {vals.median():.3f}  |  mean {vals.mean():.3f}"
    ax.text(
        0.5, 0.01, stats,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=14, color="#444444",
    )
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.74])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(score_label, fontsize=18, labelpad=10)
    cbar.ax.tick_params(labelsize=13)

    fig.suptitle(
        f"{score_label}  (Full Period)",
        fontsize=28,
        fontweight="bold",
        y=0.97,
    )
    plt.tight_layout(rect=[0, 0, 0.91, 0.93])
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved -> {output_path.name}")



def plot_score_grid(
    gdf_conus: gpd.GeoDataFrame,
    risk_df: pd.DataFrame,
    scores: list,
    output_path: Path,
    nrows: int,
    ncols: int,
    figsize: tuple,
    suptitle: str = None,
    show_stats: bool = True,
) -> None:
    """Draw a grid of choropleth score maps, one subplot per (score_col, score_label)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for ax, (score_col, score_label) in zip(axes, scores):
        merged = gdf_conus.merge(risk_df[["GEOID", score_col]], on="GEOID", how="left")

        vals = risk_df[score_col].dropna()
        if SCALE_MODE == "fixed":
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(vals.quantile(0.01))
            vmax = float(vals.quantile(0.99))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        merged.plot(
            column=score_col,
            ax=ax,
            cmap=CMAP,
            norm=norm,
            linewidth=0,
            missing_kwds={"color": "#d0d0d0", "label": "No data"},
        )

        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
        cbar.ax.tick_params(labelsize=11)

        ax.set_title(score_label, fontsize=20, fontweight="bold", pad=8)
        if show_stats:
            stats = f"median {vals.median():.3f}  |  mean {vals.mean():.3f}"
            ax.text(
                0.5, -0.01, stats,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=13, color="#444444",
            )
        ax.set_axis_off()

    for ax in axes[len(scores):]:
        ax.set_axis_off()

    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize=28,
            fontweight="bold",
            y=0.95,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.91])
    else:
        plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved -> {output_path.name}")




def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gdf_conus = load_tract_shapefile()

    for filename, subdir in [
        ("2006-2024_MinMaxRiskIndex.csv", "MinMaxRiskIndex"),
        ("2006-2024_RankRiskIndex.csv",   "RankRiskIndex"),
    ]:
        print(f"\nGenerating {subdir} maps...")
        risk_df = pd.read_csv(BASE_DIR / filename, dtype={"GEOID": str})
        risk_df["GEOID"] = risk_df["GEOID"].str.zfill(11)

        out_dir = OUTPUT_DIR / subdir
        out_dir.mkdir(exist_ok=True)

        for score_col, score_label in SCORES:
            print(f"  {score_label}...")
            plot_map(
                gdf_conus,
                risk_df,
                score_col,
                score_label,
                out_dir / f"{score_col}.png",
            )

        print("  Combined 2x2 grid (hazard, vulnerability, exposure, combined)...")
        plot_score_grid(
            gdf_conus, risk_df, COMBINED_GRID_SCORES,
            out_dir / "CombinedGrid_2x2.png",
            nrows=2, ncols=2, figsize=(20, 16),
            suptitle=None, show_stats=False,
        )

        print("  Exposure sub-index row (rural, urban, combined)...")
        plot_score_grid(
            gdf_conus, risk_df, EXPOSURE_SUBINDEX_SCORES,
            out_dir / "ExposureSubIndexes_1x3.png",
            nrows=1, ncols=3, figsize=(24, 8),
            suptitle="Exposure Sub-Indexes (Full Period)",
        )

    print(f"\nAll maps saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
