"""
extract_nlcd_farmland.py
------------------------
Calculates the percentage of farmland per census tract from NLCD Annual
land cover rasters for each 5-year time period.

NLCD farmland classes:
  81 = Pasture/Hay
  82 = Cultivated Crops

Output per period:
  TimeBasedFloodIndex/5-year/{period}/Combined Data/farmland.csv
  Columns: GEOID, pct_farmland

Period -> NLCD year -> Census tract vintage:
  2010-2006  ->  NLCD 2010  ->  2010 tracts
  2015-2011  ->  NLCD 2015  ->  2010 tracts
  2019-2015  ->  NLCD 2019  ->  2010 tracts
  2020-2016  ->  NLCD 2020  ->  2020 tracts
  2024-2020  ->  NLCD 2024  ->  2020 tracts

2010 census boundaries cover ACS periods ending in 2010, 2015, and 2019.
2020 census boundaries cover ACS periods ending in 2020 and 2024.

Requirements: rasterstats (pip install rasterstats)

NOTE (replication package): this script is included for methodological
documentation. It requires raw NLCD Annual land cover rasters and the 2010/
2020 TIGER/Line tract shapefiles, which are not shipped in this package's
data/ folder (see the package README for public source citations). Its
output (farmland %, folded into the shipped exposure.csv) is already
reflected in data/TimeBasedFloodIndex/2006-2024/Combined Data/exposure.csv,
so re-running this script is only necessary to rebuild that input from
scratch.
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

try:
    from rasterstats import zonal_stats
except ImportError:
    raise ImportError(
        "rasterstats is required. Install with: pip install rasterstats"
    )

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Package layout: <package_root>/scripts/01_extract/extract_nlcd_farmland.py
# All paths below point into data/reconstruction/ (inputs consumed by the
# pipeline); none of this raw NLCD/shapefile data is shipped -- see README.
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR      = os.path.join(BASE_DIR, "data", "reconstruction")
TIMEBASED_DIR = os.path.join(DATA_DIR, "TimeBasedFloodIndex", "5-year")

TRACT_SHP = {
    2010: os.path.join(DATA_DIR, "CensusTractShapeFiles", "tl_2010_us_tract.shp"),
    2020: os.path.join(DATA_DIR, "CensusTractShapeFiles", "tl_2020_us_tract.shp"),
}

GEOID_LENGTH = 11

# NLCD class values representing agricultural land
FARMLAND_CLASSES = {81, 82}  # 81=Pasture/Hay, 82=Cultivated Crops

# Non-continental states and territories to exclude
EXCLUDE_STATE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}

# folder_name -> (nlcd_year, geo_year)
# geo_year selects which TIGER/Line shapefile vintage to use
TIME_PERIODS = {
    "2010-2006": (2010, 2010),
    "2015-2011": (2015, 2010),
    "2019-2015": (2019, 2010),
    "2020-2016": (2020, 2020),
    "2024-2020": (2024, 2020),
}

# ==============================================================================
# HELPERS
# ==============================================================================

def load_tracts(shp_path: str) -> gpd.GeoDataFrame:
    """
    Load a census tract shapefile, normalize the GEOID column name (the 2010
    TIGER/Line files use the 'GEOID10' suffix), and filter to continental US.
    Returns the GeoDataFrame in its native CRS (WGS84).
    """
    print(f"    Reading shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path)

    # Normalize column names between 2010 ('GEOID10') and 2020 ('GEOID') vintages
    if "GEOID10" in gdf.columns:
        gdf = gdf.rename(columns={"GEOID10": "GEOID"})

    gdf["GEOID"] = gdf["GEOID"].str.zfill(GEOID_LENGTH)

    before = len(gdf)
    gdf = gdf[~gdf["GEOID"].str[:2].isin(EXCLUDE_STATE_FIPS)].reset_index(drop=True)
    print(f"    Continental tracts: {len(gdf):,}  (dropped {before - len(gdf):,} non-continental)")
    return gdf


def get_raster_crs(raster_path: str):
    """Return the CRS of a raster file."""
    with rasterio.open(raster_path) as src:
        return src.crs


def compute_farmland_pct(nlcd_path: str, tracts_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute percentage of farmland per census tract using NLCD categorical
    zonal statistics.

    Steps:
      1. Reproject tracts to the raster CRS (NLCD uses Albers Equal Area).
      2. Run rasterstats.zonal_stats with categorical=True to count pixels per
         NLCD class within each tract polygon.
      3. pct_farmland = (pixels in class 81 + pixels in class 82) /
                        (total valid pixels) * 100
         Nodata pixels are excluded from both numerator and denominator.
         Tracts with zero valid pixels (e.g., entirely offshore) receive NaN.

    Returns a DataFrame with columns ['GEOID', 'pct_farmland'].
    """
    raster_crs = get_raster_crs(nlcd_path)

    # Reproject so that rasterstats does not need to do per-geometry reprojection
    if tracts_gdf.crs.to_epsg() != raster_crs.to_epsg():
        print(f"    Reprojecting tracts ({tracts_gdf.crs.to_epsg()}) -> "
              f"raster CRS ({raster_crs.to_epsg()})...")
        tracts_proj = tracts_gdf.to_crs(raster_crs)
    else:
        tracts_proj = tracts_gdf

    with rasterio.open(nlcd_path) as src:
        nodata_val = src.nodata if src.nodata is not None else 0

    print(f"    Running categorical zonal stats on {len(tracts_proj):,} tracts "
          f"(this may take several minutes for a full-CONUS raster)...")

    stats = zonal_stats(
        tracts_proj,
        nlcd_path,
        categorical=True,
        nodata=nodata_val,
        all_touched=False,  # count only pixels whose centers fall inside the polygon
    )

    records = []
    for geoid, stat in zip(tracts_proj["GEOID"], stats):
        if not stat:
            records.append({"GEOID": geoid, "pct_farmland": np.nan})
            continue

        farm_pixels  = sum(stat.get(cls, 0) for cls in FARMLAND_CLASSES)
        total_pixels = sum(v for v in stat.values() if v is not None)

        if total_pixels == 0:
            pct = np.nan
        else:
            pct = round((farm_pixels / total_pixels) * 100.0, 4)

        records.append({"GEOID": geoid, "pct_farmland": pct})

    return pd.DataFrame(records)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("NLCD Farmland % per Census Tract — 5-year periods")
    print("=" * 60)

    # Load both tract vintages once; each is reused across multiple periods
    print("\nLoading census tract shapefiles...")
    tracts_cache = {}
    for geo_year in (2010, 2020):
        print(f"\n  {geo_year} vintage:")
        tracts_cache[geo_year] = load_tracts(TRACT_SHP[geo_year])

    for folder_name, (nlcd_year, geo_year) in TIME_PERIODS.items():
        print()
        print("=" * 60)
        print(f"Period {folder_name}  |  NLCD {nlcd_year}  |  {geo_year} tract boundaries")
        print("=" * 60)

        nlcd_path = os.path.join(
            TIMEBASED_DIR, folder_name, "Exposure",
            f"Annual_NLCD_LndCov_{nlcd_year}_CU_C1V1.tif",
        )
        if not os.path.exists(nlcd_path):
            print(f"  WARNING: NLCD file not found, skipping: {nlcd_path}")
            continue

        output_dir = os.path.join(TIMEBASED_DIR, folder_name, "Combined Data")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "farmland.csv")

        if os.path.exists(output_path):
            print(f"  Already exists, skipping: {output_path}")
            continue

        result = compute_farmland_pct(nlcd_path, tracts_cache[geo_year])
        result["GEOID"] = result["GEOID"].str.zfill(GEOID_LENGTH)
        result = result[["GEOID", "pct_farmland"]]

        result.to_csv(output_path, index=False)
        nan_count = result["pct_farmland"].isna().sum()
        print(f"  Saved: {output_path}")
        print(f"  Rows: {len(result):,}  |  NaN (no valid pixels): {nan_count:,}")

    print()
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
