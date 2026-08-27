"""
Generates census-tract choropleth maps of the flood risk index components for
large metropolitan areas, and Albuquerque and Socorro, NM.

Reads the full-period (2006-2024) composite flood risk index from:
  TimeBasedFloodIndex/2006-2024/2006-2024_geometric_MinMaxRiskIndex.csv

For each city a subfolder is created under OUTPUT_DIR and four PNG files are saved:
  {city}_hazard_score.png
  {city}_vulnerability_score.png
  {city}_exposure_score.png
  {city}_combined_risk_score.png
  {city}_combined_overview.png  -- 2x2 grid of all four scores

Requirements:
    pip install geopandas matplotlib pyproj shapely
"""

import os
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from pyproj import Transformer

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True


# Package layout: <package_root>/scripts/06_maps/MetroAreaIndexMaps.py
# The risk index CSV (a result) lives in data/results/; the cartographic
# tract shapefile (an input) lives in data/reconstruction/. City maps go to results/.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH   = os.path.join(PACKAGE_ROOT, "data", "results", "TimeBasedFloodIndex", "2006-2024", "2006-2024_geometric_MinMaxRiskIndex.csv")
TRACTS_SHP = os.path.join(PACKAGE_ROOT, "data", "reconstruction", "CensusTractShapeFiles", "cb_2020_us_tract_500k", "cb_2020_us_tract_500k.shp")
OUTPUT_DIR = os.path.join(PACKAGE_ROOT, "data", "results", "CityMaps")

# (display_name, folder_slug, lat, lon, buffer_radius_km)
# Buffer radius chosen to capture the full urban metro footprint.
CITIES = [
    ("New York City",   "New_York_City",  40.7128,  -74.0060,  60),
    ("Los Angeles",     "Los_Angeles",    34.0522, -118.2437,  60),
    ("Chicago",         "Chicago",        41.8781,  -87.6298,  55),
    ("Houston",         "Houston",        29.7604,  -95.3698,  60),
    ("Phoenix",         "Phoenix",        33.4484, -112.0740,  55),
    ("Philadelphia",    "Philadelphia",   39.9526,  -75.1652,  45),
    ("San Antonio",     "San_Antonio",    29.4241,  -98.4936,  45),
    ("Dallas",          "Dallas",         32.7767,  -96.7970,  55),
    ("San Diego",       "San_Diego",      32.7157, -117.1611,  40),
    ("San Jose",        "San_Jose",       37.3382, -121.8863,  40),
    ("Albuquerque",     "Albuquerque",    35.0844, -106.6504,  45),
    ("Socorro",         "Socorro",        34.0584, -106.8914,  30),
]

TOP10_CITIES = [
    ("New York City", "NY", 40.7128,  -74.0060,  60),
    ("Philadelphia",  "PA", 39.9526,  -75.1652,  45),
    ("Houston",       "TX", 29.7604,  -95.3698,  60),
    ("Los Angeles",   "CA", 34.0522, -118.2437,  60),
    ("Chicago",       "IL", 41.8781,  -87.6298,  55),
    ("Phoenix",       "AZ", 33.4484, -112.0740,  55),
    ("San Antonio",   "TX", 29.4241,  -98.4936,  45),
    ("San Diego",     "CA", 32.7157, -117.1611,  40),
    ("Dallas",        "TX", 32.7767,  -96.7970,  55),
    ("Fort Worth",    "TX", 32.7555,  -97.3308,  45),
]

# Score columns to map (column_name, title)
SCORE_COLS = [
    ("hazard_score",        "Hazard Score"),
    ("vulnerability_score", "Vulnerability Score"),
    ("exposure_score",      "Exposure Score"),
    ("combined_risk_score", "Combined Risk Score"),
]

CMAP = "RdYlGn_r"

# Color scale mode:
#   "quantile" -- stretch color scale to the 1st-99th percentile of each score
#                 (maximizes visual contrast, but scale differs per map)
#   "fixed"    -- fixed 0-1 scale for every map (comparable across maps and
#                 cities, matching the nationwide min-max normalization)
SCALE_MODE = "quantile"



def normalize_geoid(geoid: str) -> str:
    """Strip any 'US' prefix and zero-pad to 11 digits."""
    g = str(geoid).strip()
    if "US" in g:
        g = g[g.index("US") + 2:]
    if "." in g:
        g = g.split(".")[0]
    return g.zfill(11)


def _score_range(risk_df: pd.DataFrame, col: str) -> tuple:
    vals = risk_df[col].dropna()
    if SCALE_MODE == "fixed":
        return 0.0, 1.0
    return float(vals.quantile(0.01)), float(vals.quantile(0.99))


def _city_subset(
    gdf: gpd.GeoDataFrame, transformer: Transformer,
    lat: float, lon: float, radius_km: float,
) -> gpd.GeoDataFrame:
    """Spatially filter gdf (already projected) to tracts within radius_km of (lat, lon)."""
    cx, cy = transformer.transform(lon, lat)
    buffer_geom = Point(cx, cy).buffer(radius_km * 1_000)   # radius in metres
    return gdf[gdf.geometry.intersects(buffer_geom)].copy()


def make_single_map(
    gdf_city: gpd.GeoDataFrame,
    risk_df: pd.DataFrame,
    col: str,
    score_title: str,
    output_path: str,
    title: str = None,
) -> None:
    """Render one choropleth map for a single score column and save as PNG.

    title: optional figure title (e.g. "City, ST"). Omitted by default.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    valid_data = gdf_city[col].dropna()

    if len(valid_data) == 0:
        ax.text(
            0.5, 0.5, "No data available for this area",
            ha="center", va="center", fontsize=14, transform=ax.transAxes,
        )
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    [no data] {os.path.basename(output_path)}")
        return

    vmin, vmax = _score_range(risk_df, col)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)


    gdf_city.plot(
        column=col,
        ax=ax,
        cmap=CMAP,
        norm=norm,
        linewidth=0,
        missing_kwds={"color": "#d0d0d0", "label": "No data"},
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.74])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(score_title, fontsize=18, labelpad=10)
    cbar.ax.tick_params(labelsize=13)

    # Title & stats annotation 
    stats = f"n={len(valid_data):,}  |  median {valid_data.median():.3f}  |  mean {valid_data.mean():.3f}"
    ax.text(
        0.5, 0.01, stats,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=14, color="#444444",
    )
    ax.set_axis_off()

    if title:
        fig.suptitle(title, fontsize=28, fontweight="bold", y=0.97)
        plt.tight_layout(rect=[0, 0, 0.91, 0.93])
    else:
        plt.tight_layout(rect=[0, 0, 0.91, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved: {os.path.basename(output_path)}")


def make_combined_map(
    gdf_city: gpd.GeoDataFrame,
    risk_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Render all four score maps side-by-side in a 2×2 grid and save as PNG."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = np.atleast_1d(axes).flatten()

    for ax, (col, score_title) in zip(axes, SCORE_COLS):
        valid_data = gdf_city[col].dropna()

        if len(valid_data) == 0:
            ax.text(
                0.5, 0.5, "No data available",
                ha="center", va="center", fontsize=12, transform=ax.transAxes,
            )
            ax.set_title(score_title, fontsize=20, fontweight="bold", pad=8)
            ax.axis("off")
            continue

        vmin, vmax = _score_range(risk_df, col)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        gdf_city.plot(
            column=col,
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

        ax.set_title(score_title, fontsize=20, fontweight="bold", pad=8)

        stats = f"median {valid_data.median():.3f}  |  mean {valid_data.mean():.3f}"
        ax.text(
            0.5, -0.01, stats,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=13, color="#444444",
        )
        ax.set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved: {os.path.basename(output_path)}")


def make_top10_grid_map(
    gdf: gpd.GeoDataFrame,
    risk_df: pd.DataFrame,
    transformer: Transformer,
    output_path: str,
) -> None:
    """Render combined_risk_score for TOP10_CITIES in a 2x5 grid, one shared color scale."""
    col = "combined_risk_score"
    vmin, vmax = _score_range(risk_df, col)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 5, figsize=(38, 15))
    axes = axes.flatten()

    for ax, (name, state, lat, lon, radius_km) in zip(axes, TOP10_CITIES):
        city_gdf = _city_subset(gdf, transformer, lat, lon, radius_km)
        valid_data = city_gdf[col].dropna()

        ax.set_title(f"{name}, {state}", fontsize=36, fontweight="bold", pad=16)

        if len(valid_data) == 0:
            ax.text(
                0.5, 0.5, "No data available",
                ha="center", va="center", fontsize=24, transform=ax.transAxes,
            )
            ax.axis("off")
            continue

        city_gdf.plot(
            column=col,
            ax=ax,
            cmap=CMAP,
            norm=norm,
            linewidth=0,
            missing_kwds={"color": "#d0d0d0", "label": "No data"},
        )

        stats = f"median {valid_data.median():.3f}  |  mean {valid_data.mean():.3f}"
        ax.text(
            0.5, -0.01, stats,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=24, color="#444444",
        )
        ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Combined Risk Score", fontsize=32, labelpad=20)
    cbar.ax.tick_params(labelsize=24)

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(output_path)}")


def main() -> None:
    # 1. Load shapefile 
    print("Loading census tract shapefile …")
    tracts = gpd.read_file(TRACTS_SHP, dtype={"GEOID": str})
    tracts["_key"] = tracts["GEOID"].apply(normalize_geoid)
    print(f"  {len(tracts):,} tracts in shapefile")

    #  2. Load index CSV and merge 
    print("Loading index CSV …")
    df = pd.read_csv(CSV_PATH, dtype={"GEOID": str})
    df["_key"] = df["GEOID"].apply(normalize_geoid)
    print(f"  {len(df):,} rows in index CSV")

    gdf = tracts.merge(df, on="_key", how="inner")
    print(f"  {len(gdf):,} tracts matched after merge")

    #  3. Project to EPSG:5070 (Albers Equal Area metres) 
    print("Projecting to EPSG:5070 …")
    gdf = gdf.to_crs("EPSG:5070")

    # Transformer: WGS-84 lon/lat → EPSG:5070
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

    # 4. Build output root 
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #  5. Loop over cities 
    for city_display, city_slug, lat, lon, radius_km in CITIES:
        print(f"\n{'-'*60}")
        print(f"  {city_display}  ({radius_km} km radius)")

        city_gdf = _city_subset(gdf, transformer, lat, lon, radius_km)
        print(f"  {len(city_gdf):,} tracts within buffer")

        if city_gdf.empty:
            print("  WARNING: no tracts found — skipping this city.")
            continue

        # Create per-city output folder
        city_folder = os.path.join(OUTPUT_DIR, city_slug)
        os.makedirs(city_folder, exist_ok=True)

        # Generate one PNG per score column
        for col, score_title in SCORE_COLS:
            fname  = f"{city_slug}_{col}.png"
            fpath  = os.path.join(city_folder, fname)
            make_single_map(city_gdf, df, col, score_title, fpath)

        # Generate combined 2×2 overview image
        combined_path = os.path.join(city_folder, f"{city_slug}_combined_overview.png")
        make_combined_map(city_gdf, df, combined_path)

    # 6. 10 city combined risk score comparison 
    print(f"\n{'-'*60}")
    print("  Top-10 city combined risk score comparison")
    top10_dir = os.path.join(OUTPUT_DIR, "Top10_CombinedRisk")
    os.makedirs(top10_dir, exist_ok=True)

    for name, state, lat, lon, radius_km in TOP10_CITIES:
        city_gdf = _city_subset(gdf, transformer, lat, lon, radius_km)
        slug = name.replace(" ", "_")
        fpath = os.path.join(top10_dir, f"{slug}_combined_risk_score.png")
        make_single_map(
            city_gdf, df, "combined_risk_score", "Combined Risk Score", fpath,
            title=f"{name}, {state}",
        )

    grid_path = os.path.join(top10_dir, "Top10_CombinedRiskGrid.png")
    make_top10_grid_map(gdf, df, transformer, grid_path)

    print(f"\n{'='*60}")
    print(f"All done.  Maps saved to:  {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
