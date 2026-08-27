"""

Creates a full-period (2006-2024) composite flood index and writes
Combined Data CSVs into TimeBasedFloodIndex/2006-2024/Combined Data/.

All output uses 2020 census tract boundaries.

 ACS 5-year periods are 2006-2010, 2011-2015, 2016-2020, and 2020-2024.
 Tracts are area weighted crosswalked to 2020 boundaries before averaging.



Hazard Variables:

annual_flood_rate_per_sqkm
  - Flood events 2006-2024 (19 years), spatially joined to 2020 census tracts.
  - Rate = total events / 19 years / tract land area (sq km).

avg_rainfall_mm
  - PRISM average over 2006-2024 (19 years) on 2020 tract boundaries.
  - Uses precomputed annual tract totals; remaps via crosswalk when the PRISM
    data boundary differs from 2020.

pct_100yr_flood_area
  - FEMA 100-year flood zone coverage (static). 2020 tract boundaries.


Vulnerability Variables:

pct_poverty_rate        B17001: below_poverty / total_determined * 100
median_household_income S1903:  all-household median (column located dynamically)
pct_renters             B25003: renter_occupied / total_occupied * 100



Exposure Variables:

pop_density_per_sqkm        B01003: total_pop / area_sqkm
housing_density_per_sqkm    B25034: total_units / area_sqkm
pct_houses_built_pre1980    B25034: sum(pre-1980 decade cols) / total_units * 100
pct_farmland_area           NLCD (81 Pasture/Hay + 82 Cultivated Crops) pixel share
                             per tract, averaged across the four 5-year periods'
                             pre-computed farmland.csv files (from
                             extract_nlcd_farmland.py). Run that script first.

NOTE: this script is included for methodological
documentation. It requires raw per-period ACS 5-year tables, PRISM annual
precipitation totals, flood-event data, Census crosswalk/gazetteer files,
and the derived FEMA flood-zone-area and farmland CSVs, none of which are
shipped in this package's data/ folder (see the package README for public
source citations). Its output is already shipped as
data/TimeBasedFloodIndex/2006-2024/Combined Data/{hazard,vulnerability,exposure}.csv,
which is the direct input to 02_index_construction/fullperiod_pca.py.
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings("ignore")


# CONFIGURATION

# Package layout: <package_root>/scripts/01_extract/extract_fullperiod_index.py
# All raw inputs and the Combined Data output below live under
# data/reconstruction/ (the raw inputs are not shipped -- see README).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR      = os.path.join(BASE_DIR, "data", "reconstruction")

FIVEYEAR_DIR  = os.path.join(DATA_DIR, "TimeBasedFloodIndex", "5-year")
OUTPUT_DIR    = os.path.join(DATA_DIR, "TimeBasedFloodIndex", "2006-2024", "Combined Data")
SHARED_DIR    = os.path.join(FIVEYEAR_DIR, "Shared")

PRISM_DIR        = os.path.join(DATA_DIR, "Precipitation")
PRISM_ANNUAL_DIR = os.path.join(PRISM_DIR, "annual_tract_totals")
CROSSWALK_DIR    = os.path.join(DATA_DIR, "Crosswalks")
FLOOD_EVENTS_DIR = os.path.join(DATA_DIR, "FloodingCombinedData2")

# Static FEMA flood zone coverage on 2020 boundaries
FEMA_CSV = os.path.join(SHARED_DIR, "2020_FEMA_Area.csv")

TRACT_SHP_2020    = os.path.join(DATA_DIR, "CensusTractShapeFiles", "tl_2020_us_tract.shp")
GAZETTEER_2020    = os.path.join(DATA_DIR, "Gazetteer", "2020_gaz_tracts_national.txt")

GEOID_LENGTH = 11
ACS_FILL_THRESHOLD = -100_000
PRE1980_DECADE_KEYWORDS = ["1939", "1940", "1950", "1960", "1970"]
EXCLUDE_STATE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}

START_YEAR = 2006
END_YEAR   = 2024
GEO_YEAR   = 2020  # all output on 2020 tract boundaries

# Four ACS 5-year files that together span 2006-2024.
# (acs_year, period_dir, acs_geo_year)
ACS_PAIRS = [
    (2010, os.path.join(FIVEYEAR_DIR, "2010-2006"), 2010),
    (2015, os.path.join(FIVEYEAR_DIR, "2015-2011"), 2010),
    (2020, os.path.join(FIVEYEAR_DIR, "2020-2016"), 2010),
    (2024, os.path.join(FIVEYEAR_DIR, "2024-2020"), 2020),
]



# Helpers

def strip_geoid(raw: str) -> str:
    s = str(raw).strip()
    if "US" in s:
        s = s.split("US")[-1]
    return s.zfill(GEOID_LENGTH)


def load_acs(path: str):
    """Load ACS CSV with two header rows. Returns (data_df, labels_dict)."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    labels = dict(zip(df.columns, df.iloc[0]))
    df = df.iloc[1:].reset_index(drop=True)
    df["GEOID"] = df["GEO_ID"].apply(strip_geoid)
    return df, labels


def replace_acs_fills(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s[s < ACS_FILL_THRESHOLD] = np.nan
    return s


def is_continental(geoid: str) -> bool:
    return str(geoid).zfill(GEOID_LENGTH)[:2] not in EXCLUDE_STATE_FIPS


def filter_continental(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["GEOID"].apply(is_continental)
    dropped = (~mask).sum()
    if dropped:
        print(f"    Dropped {dropped:,} non-continental tracts.")
    return df[mask].reset_index(drop=True)



# Data loaders 

def load_gazetteer(path: str) -> pd.DataFrame:
    gaz = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    gaz.columns = gaz.columns.str.strip()
    gaz["GEOID"] = gaz["GEOID"].str.strip().str.zfill(GEOID_LENGTH)
    gaz["area_sqkm"] = pd.to_numeric(gaz["ALAND"], errors="coerce") / 1_000_000
    return gaz[["GEOID", "area_sqkm"]]


def load_fema(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df["GEOID"] = df["GEOID"].apply(strip_geoid)
    df["pct_100yr_flood_area"] = pd.to_numeric(df["pct_100yr_flood"], errors="coerce")
    return df[["GEOID", "pct_100yr_flood_area"]]


def load_tract_shapefile(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if "GEOID10" in gdf.columns:
        gdf = gdf.rename(columns={"GEOID10": "GEOID", "ALAND10": "ALAND"})
    gdf["GEOID"] = gdf["GEOID"].str.zfill(GEOID_LENGTH)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


# Crosswalk helpers

def load_crosswalks(crosswalk_dir: str) -> dict:
    """Load 2000->2010 and 2010->2020 census tract crosswalk files."""
    cols_trf = [
        "STATEFP10","COUNTYFP10","TRACTCE10","GEOID10","POP10","HU10","HUCLS10",
        "AREALAND10","AREAWATER10","STATEFP00","COUNTYFP00","TRACTCE00","GEOID00",
        "POP00","HU00","HUCLS00","AREALAND00","AREAWATER00","AREALANDPT","AREAWATERPT",
        "POPPCT10","HUPCT10","POPPCT00","HUPCT00","ZPOP10","ZPOPPCT10","ZPOPPCT00",
        "ZHU10","ZHUPCT10","ZHUPCT00",
    ]
    df_00_10 = pd.read_csv(
        os.path.join(crosswalk_dir, "us2010trf.txt"),
        sep="|", header=None, names=cols_trf, dtype=str, low_memory=False,
    )
    df_00_10["GEOID00"] = df_00_10["GEOID00"].str.zfill(GEOID_LENGTH)
    df_00_10["GEOID10"] = df_00_10["GEOID10"].str.zfill(GEOID_LENGTH)
    areapt  = pd.to_numeric(df_00_10["AREALANDPT"], errors="coerce")
    area10t = pd.to_numeric(df_00_10["AREALAND10"], errors="coerce")
    df_00_10["weight"] = areapt / area10t.replace(0, np.nan)
    cw_00_10 = df_00_10[["GEOID00", "GEOID10", "weight"]].copy()

    df_10_20 = pd.read_csv(
        os.path.join(crosswalk_dir, "tab20_tract20_tract10_natl.txt"),
        sep="|", dtype=str, low_memory=False,
    )
    df_10_20["GEOID_TRACT_10"] = df_10_20["GEOID_TRACT_10"].str.zfill(GEOID_LENGTH)
    df_10_20["GEOID_TRACT_20"] = df_10_20["GEOID_TRACT_20"].str.zfill(GEOID_LENGTH)
    areapt  = pd.to_numeric(df_10_20["AREALAND_PART"],     errors="coerce")
    area20t = pd.to_numeric(df_10_20["AREALAND_TRACT_20"], errors="coerce")
    df_10_20["weight"] = areapt / area20t.replace(0, np.nan)
    cw_10_20 = df_10_20[["GEOID_TRACT_10", "GEOID_TRACT_20", "weight"]].copy()

    return {"2000_to_2010": cw_00_10, "2010_to_2020": cw_10_20}


def crosswalk_intensive_to_2020(
    df: pd.DataFrame, value_cols: list, cw_10_20: pd.DataFrame
) -> pd.DataFrame:
    """
    Area-weighted remap of intensive variables (rates, percentages, medians,
    densities) from 2010->2020 census tract boundaries.
    """
    m = cw_10_20.merge(
        df[["GEOID"] + value_cols],
        left_on="GEOID_TRACT_10", right_on="GEOID",
        how="left",
    )

    result_parts = []
    for col in value_cols:
        valid = m[col].notna()
        wv = np.where(valid, m[col] * m["weight"], np.nan)
        ww = np.where(valid, m["weight"], np.nan)

        tmp = pd.DataFrame({
            "GEOID_TRACT_20": m["GEOID_TRACT_20"],
            "wv": wv,
            "ww": ww,
        })
        grp = tmp.groupby("GEOID_TRACT_20").agg(
            wv_sum=("wv", lambda x: x.sum(min_count=1)),
            ww_sum=("ww", lambda x: x.sum(min_count=1)),
        ).reset_index()
        grp[col] = grp["wv_sum"] / grp["ww_sum"].replace(0, np.nan)
        result_parts.append(grp[["GEOID_TRACT_20", col]])

    result = result_parts[0]
    for rp in result_parts[1:]:
        result = result.merge(rp, on="GEOID_TRACT_20", how="outer")
    result = result.rename(columns={"GEOID_TRACT_20": "GEOID"})
    return result[["GEOID"] + value_cols]


def _crosswalk_precip(
    annual: pd.DataFrame, data_boundary: int, target_boundary: int, crosswalks: dict
) -> pd.DataFrame:
    if data_boundary == target_boundary:
        return annual.rename(columns={"GEOID_src": "GEOID"})

    if data_boundary == 2000 and target_boundary == 2010:
        cw = crosswalks["2000_to_2010"]
        m = cw.merge(annual, left_on="GEOID00", right_on="GEOID_src", how="left")
        m["weighted_ppt"] = m["annual_ppt_mm"] * m["weight"]
        result = m.groupby("GEOID10")["weighted_ppt"].sum(min_count=1).reset_index()
        return result.rename(columns={"GEOID10": "GEOID", "weighted_ppt": "annual_ppt_mm"})

    if data_boundary == 2010 and target_boundary == 2020:
        cw = crosswalks["2010_to_2020"]
        m = cw.merge(annual, left_on="GEOID_TRACT_10", right_on="GEOID_src", how="left")
        m["weighted_ppt"] = m["annual_ppt_mm"] * m["weight"]
        result = m.groupby("GEOID_TRACT_20")["weighted_ppt"].sum(min_count=1).reset_index()
        return result.rename(columns={"GEOID_TRACT_20": "GEOID", "weighted_ppt": "annual_ppt_mm"})

    raise ValueError(f"No crosswalk defined for boundary {data_boundary} -> {target_boundary}")




# Precipitation processing

def _prism_folder(year: int) -> str:
    if year <= 2000:
        return "PRISM_daily_tract_1991-2000_csvs"
    elif year <= 2010:
        return "PRISM_daily_tract_2001-2010_csvs"
    elif year <= 2020:
        return "PRISM_daily_tract_2011-2020_csvs"
    else:
        return "PRISM_daily_tract_2021-2024_csvs"


def _prism_tract_col(year: int) -> str:
    if year <= 2000:
        return "tract_fips90"
    elif year <= 2009:
        return "tract_fips00"
    elif year <= 2019:
        return "tract_fips10"
    else:
        return "tract_fips20"


def _prism_data_boundary(year: int) -> int:
    if year <= 2019:
        return 2010
    else:
        return 2020


def load_annual_precip(year: int) -> pd.DataFrame:
    """
    Return annual PRISM precipitation total per tract for the given year.
    Uses a precomputed cache in PRISM_ANNUAL_DIR; reads the raw daily CSV on
    first call and saves the result for subsequent runs.
    """
    os.makedirs(PRISM_ANNUAL_DIR, exist_ok=True)
    cache_path = os.path.join(PRISM_ANNUAL_DIR, f"annual_ppt_{year}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, dtype={"GEOID_src": str})
        df["GEOID_src"] = df["GEOID_src"].str.zfill(GEOID_LENGTH)
        return df

    folder = _prism_folder(year)
    raw_path = os.path.join(PRISM_DIR, folder, f"nanda_PRISM_daily_Tract_{year}_01P.csv")
    if not os.path.exists(raw_path):
        print(f"    Warning: PRISM file not found: {raw_path}")
        return pd.DataFrame(columns=["GEOID_src", "annual_ppt_mm"])

    tract_col = _prism_tract_col(year)
    print(f"    Reading raw PRISM {year} (large file - one-time, will be cached)...")
    df = pd.read_csv(raw_path, usecols=[tract_col, "ppt"], low_memory=False)
    df["ppt"] = pd.to_numeric(df["ppt"], errors="coerce") / 100.0

    annual = df.groupby(tract_col)["ppt"].sum(min_count=1).reset_index()
    annual.columns = ["GEOID_src", "annual_ppt_mm"]
    annual["GEOID_src"] = annual["GEOID_src"].astype(str).str.zfill(GEOID_LENGTH)

    annual.to_csv(cache_path, index=False)
    print(f"    Cached -> {cache_path}")
    return annual


def compute_avg_precip_range(
    start_year: int, end_year: int, geo_year: int,
    annual_cache: dict, crosswalks: dict
) -> pd.DataFrame:
    """
    Average annual PRISM precipitation over start_year..end_year on geo_year
    tract boundaries.
    """
    years = list(range(start_year, end_year + 1))
    annual_frames = []

    for year in years:
        if year not in annual_cache:
            annual_cache[year] = load_annual_precip(year)
        annual = annual_cache[year]
        if annual.empty:
            continue
        data_bnd = _prism_data_boundary(year)
        mapped = _crosswalk_precip(annual, data_bnd, geo_year, crosswalks)
        mapped = mapped.rename(columns={"annual_ppt_mm": f"ppt_{year}"})
        annual_frames.append(mapped)

    if not annual_frames:
        return pd.DataFrame(columns=["GEOID", "avg_rainfall_mm"])

    combined = annual_frames[0]
    for frame in annual_frames[1:]:
        combined = combined.merge(frame, on="GEOID", how="outer")

    ppt_cols = [c for c in combined.columns if c.startswith("ppt_")]
    combined["avg_rainfall_mm"] = combined[ppt_cols].mean(axis=1)
    return combined[["GEOID", "avg_rainfall_mm"]]



# Hazard flood rate

def load_flood_events(flood_dir: str, start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for year in range(start_year, end_year + 1):
        path = os.path.join(flood_dir, f"Flooding_Combined_Events_{year}.csv")
        if not os.path.exists(path):
            print(f"    Warning: missing flood file for {year}: {path}")
            continue
        frames.append(pd.read_csv(path, low_memory=False))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "coords_source" in combined.columns:
        combined = combined[combined["coords_source"].str.strip().str.lower() != "none"]
    combined = combined.dropna(subset=["BEGIN_LAT", "BEGIN_LON"])
    print(f"    Flood events with usable coordinates: {len(combined):,}")
    return combined


def compute_flood_rate(
    flood_df: pd.DataFrame,
    tracts_gdf: gpd.GeoDataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    n_years = end_year - start_year + 1

    if flood_df.empty:
        print("    No flood events found; flood rate set to 0 for all tracts.")
        result = tracts_gdf[["GEOID"]].copy()
        result["annual_flood_rate_per_sqkm"] = 0.0
        return result

    geometry = [Point(lon, lat)
                for lon, lat in zip(flood_df["BEGIN_LON"], flood_df["BEGIN_LAT"])]
    gdf_floods = gpd.GeoDataFrame(flood_df, geometry=geometry, crs="EPSG:4326")

    print("    Performing spatial join (flood events -> census tracts)...")
    joined = gpd.sjoin(
        gdf_floods,
        tracts_gdf[["GEOID", "ALAND", "geometry"]],
        how="left",
        predicate="within",
    )

    unmatched = joined["GEOID"].isna().sum()
    if unmatched:
        print(f"    {unmatched:,} events outside any tract (coastline/border) -- excluded.")

    flood_counts = (
        joined.dropna(subset=["GEOID"])
        .groupby("GEOID")
        .size()
        .reset_index(name="flood_count")
    )

    result = tracts_gdf[["GEOID", "ALAND"]].copy()
    result = result.merge(flood_counts, on="GEOID", how="left")
    result["flood_count"] = result["flood_count"].fillna(0).astype(int)
    result["annual_avg"] = result["flood_count"] / n_years
    result["area_sqkm"] = result["ALAND"] / 1_000_000
    result["annual_flood_rate_per_sqkm"] = (
        result["annual_avg"] / result["area_sqkm"].replace(0, np.nan)
    )
    return result[["GEOID", "annual_flood_rate_per_sqkm"]]



# Vulnerability

def find_s1903_median_income_col(df: pd.DataFrame, labels: dict) -> str:
    candidates = []
    for code, label in labels.items():
        if not code.endswith("E"):
            continue
        label_str = str(label)
        if "Median income" not in label_str:
            continue
        if not label_str.rstrip().endswith("Households"):
            continue
        if code in df.columns:
            candidates.append(code)

    if not candidates:
        raise RuntimeError(
            "Could not identify the all-household median income column in S1903. "
            "Check the label row of the file."
        )
    if len(candidates) > 1:
        candidates.sort(key=lambda c: len(str(labels[c])))

    chosen = candidates[0]
    print(f"    S1903 median income column: {chosen} -> {labels[chosen]}")
    return chosen


def _extract_vulnerability_single(period_dir: str, acs_year: int) -> pd.DataFrame:
    vuln_dir = os.path.join(period_dir, "Vulnerability")

    b17001, _ = load_acs(os.path.join(vuln_dir, f"ACSDT5Y{acs_year}.B17001-Data.csv"))
    b17001["pct_poverty_rate"] = (
        replace_acs_fills(b17001["B17001_002E"]) /
        replace_acs_fills(b17001["B17001_001E"])
    ) * 100

    s1903, s1903_labels = load_acs(os.path.join(vuln_dir, f"ACSST5Y{acs_year}.S1903-Data.csv"))
    income_col = find_s1903_median_income_col(s1903, s1903_labels)
    s1903["median_household_income"] = replace_acs_fills(s1903[income_col])

    b25003, _ = load_acs(os.path.join(vuln_dir, f"ACSDT5Y{acs_year}.B25003-Data.csv"))
    b25003["pct_renters"] = (
        replace_acs_fills(b25003["B25003_003E"]) /
        replace_acs_fills(b25003["B25003_001E"])
    ) * 100

    df = b17001[["GEOID", "pct_poverty_rate"]].merge(
        s1903[["GEOID", "median_household_income"]], on="GEOID", how="outer"
    ).merge(b25003[["GEOID", "pct_renters"]], on="GEOID", how="outer")
    return df


def extract_vulnerability(
    acs_pairs: list, geo_year: int, crosswalks: dict
) -> pd.DataFrame:
    """
    Load vulnerability variables from four ACS 5-year file sets, crosswalk all
    to geo_year boundaries, and return the simple average across the four years.
    """
    VALUE_COLS = ["pct_poverty_rate", "median_household_income", "pct_renters"]
    frames = []

    for acs_year, period_dir, acs_geo_year in acs_pairs:
        print(f"    Loading ACS {acs_year} vulnerability (boundary vintage {acs_geo_year})...")
        df = _extract_vulnerability_single(period_dir, acs_year)

        if acs_geo_year != geo_year:
            print(f"    Crosswalking ACS {acs_year} vulnerability "
                  f"{acs_geo_year}->{geo_year}...")
            if acs_geo_year == 2010 and geo_year == 2020:
                df = crosswalk_intensive_to_2020(df, VALUE_COLS, crosswalks["2010_to_2020"])
            else:
                raise ValueError(
                    f"No crosswalk defined for ACS boundary {acs_geo_year}->{geo_year}"
                )

        frames.append(df.rename(columns={c: f"{c}_{acs_year}" for c in VALUE_COLS}))

    combined = frames[0]
    for f in frames[1:]:
        combined = combined.merge(f, on="GEOID", how="outer")

    result = combined[["GEOID"]].copy()
    for col in VALUE_COLS:
        year_cols = [f"{col}_{acs_year}" for acs_year, _, _ in acs_pairs]
        result[col] = combined[year_cols].mean(axis=1)

    return result


# Exposure

def find_pre1980_cols(df: pd.DataFrame, labels: dict) -> list:
    pre1980 = []
    for code, label in labels.items():
        label_str = str(label)
        if "Estimate" not in label_str or "Built" not in label_str:
            continue
        for keyword in PRE1980_DECADE_KEYWORDS:
            if keyword in label_str:
                pre1980.append(code)
                break
    return [c for c in pre1980 if c in df.columns]


def _extract_exposure_single(
    period_dir: str, acs_year: int, gaz: pd.DataFrame
) -> pd.DataFrame:
    exp_dir = os.path.join(period_dir, "Exposure")

    b01003, _ = load_acs(os.path.join(exp_dir, f"ACSDT5Y{acs_year}.B01003-Data.csv"))
    b01003["population"] = replace_acs_fills(b01003["B01003_001E"])

    b25034, b25034_labels = load_acs(
        os.path.join(exp_dir, f"ACSDT5Y{acs_year}.B25034-Data.csv")
    )
    b25034["total_housing"] = replace_acs_fills(b25034["B25034_001E"])

    pre1980_cols = find_pre1980_cols(b25034, b25034_labels)
    if not pre1980_cols:
        raise RuntimeError(
            f"Could not identify pre-1980 columns in B25034 for ACS year {acs_year}."
        )
    print(f"    Pre-1980 housing columns ({acs_year}): {pre1980_cols}")

    for col in pre1980_cols:
        b25034[col] = replace_acs_fills(b25034[col])
    b25034["pre1980_units"] = b25034[pre1980_cols].sum(axis=1, min_count=1)
    b25034["pct_houses_built_pre1980"] = (
        b25034["pre1980_units"] / b25034["total_housing"]
    ) * 100

    df = b01003[["GEOID", "population"]].merge(
        b25034[["GEOID", "total_housing", "pct_houses_built_pre1980"]],
        on="GEOID", how="outer"
    ).merge(gaz, on="GEOID", how="left")

    df["pop_density_per_sqkm"] = df["population"] / df["area_sqkm"].replace(0, np.nan)
    df["housing_density_per_sqkm"] = df["total_housing"] / df["area_sqkm"].replace(0, np.nan)

    return df[["GEOID", "pop_density_per_sqkm", "housing_density_per_sqkm",
               "pct_houses_built_pre1980"]]


def extract_exposure(
    acs_pairs: list, geo_year: int, gaz: pd.DataFrame, crosswalks: dict
) -> pd.DataFrame:
    """
    Load exposure variables from four ACS 5-year file sets, crosswalk all to
    geo_year boundaries, and return the simple average across the four years.
    """
    VALUE_COLS = ["pop_density_per_sqkm", "housing_density_per_sqkm",
                  "pct_houses_built_pre1980"]
    frames = []

    for acs_year, period_dir, acs_geo_year in acs_pairs:
        print(f"    Loading ACS {acs_year} exposure (boundary vintage {acs_geo_year})...")
        df = _extract_exposure_single(period_dir, acs_year, gaz)

        if acs_geo_year != geo_year:
            print(f"    Crosswalking ACS {acs_year} exposure "
                  f"{acs_geo_year}->{geo_year}...")
            if acs_geo_year == 2010 and geo_year == 2020:
                df = crosswalk_intensive_to_2020(df, VALUE_COLS, crosswalks["2010_to_2020"])
            else:
                raise ValueError(
                    f"No crosswalk defined for ACS boundary {acs_geo_year}->{geo_year}"
                )

        frames.append(df.rename(columns={c: f"{c}_{acs_year}" for c in VALUE_COLS}))

    combined = frames[0]
    for f in frames[1:]:
        combined = combined.merge(f, on="GEOID", how="outer")

    result = combined[["GEOID"]].copy()
    for col in VALUE_COLS:
        year_cols = [f"{col}_{acs_year}" for acs_year, _, _ in acs_pairs]
        result[col] = combined[year_cols].mean(axis=1)

    return result


# Farmland (rural exposure)


def load_farmland(period_dir: str) -> pd.DataFrame:
    """
    Load the pre-computed farmland percentage CSV produced by
    extract_nlcd_farmland.py. Renames 'pct_farmland' -> 'pct_farmland_area'.
    """
    path = os.path.join(period_dir, "Combined Data", "farmland.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Farmland CSV not found: {path}. Run extract_nlcd_farmland.py first."
        )
    df = pd.read_csv(path, dtype={"GEOID": str})
    df["GEOID"] = df["GEOID"].str.zfill(GEOID_LENGTH)
    return df.rename(columns={"pct_farmland": "pct_farmland_area"})[
        ["GEOID", "pct_farmland_area"]
    ]


def extract_farmland(acs_pairs: list, geo_year: int, crosswalks: dict) -> pd.DataFrame:
    """
    Load the pre-computed farmland percentage for each of the four 5-year
    periods, crosswalk all to geo_year boundaries, and return the simple
    average across the four periods -- i.e. the farmland share over the
    full 2006-2024 timespan.
    """
    VALUE_COLS = ["pct_farmland_area"]
    frames = []

    for acs_year, period_dir, acs_geo_year in acs_pairs:
        print(f"    Loading farmland ({acs_year}, boundary vintage {acs_geo_year})...")
        df = load_farmland(period_dir)

        if acs_geo_year != geo_year:
            print(f"    Crosswalking farmland {acs_year} "
                  f"{acs_geo_year}->{geo_year}...")
            if acs_geo_year == 2010 and geo_year == 2020:
                df = crosswalk_intensive_to_2020(df, VALUE_COLS, crosswalks["2010_to_2020"])
            else:
                raise ValueError(
                    f"No crosswalk defined for boundary {acs_geo_year}->{geo_year}"
                )

        frames.append(df.rename(columns={c: f"{c}_{acs_year}" for c in VALUE_COLS}))

    combined = frames[0]
    for f in frames[1:]:
        combined = combined.merge(f, on="GEOID", how="outer")

    result = combined[["GEOID"]].copy()
    for col in VALUE_COLS:
        year_cols = [f"{col}_{acs_year}" for acs_year, _, _ in acs_pairs]
        result[col] = combined[year_cols].mean(axis=1)

    return result





def main():
    n_years = END_YEAR - START_YEAR + 1

    print("=" * 60)
    print(f"Full-period flood index  {START_YEAR}-{END_YEAR}  ({n_years} years)")
    print(f"Output boundary: {GEO_YEAR} census tracts")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

 
    print("\nLoading shared datasets...")

    print("  Census tract crosswalk files...")
    crosswalks = load_crosswalks(CROSSWALK_DIR)

    print("  Gazetteer (2020 boundaries)...")
    gaz = load_gazetteer(GAZETTEER_2020)

    print("  Census tract shapefile (2020)...")
    tracts_gdf = load_tract_shapefile(TRACT_SHP_2020)
    tracts_gdf = tracts_gdf[
        ~tracts_gdf["GEOID"].str[:2].isin(EXCLUDE_STATE_FIPS)
    ].reset_index(drop=True)
    print(f"  Continental tracts: {len(tracts_gdf):,}")

    print("  FEMA 100-year flood area (2020 boundaries)...")
    fema = load_fema(FEMA_CSV)

    
    print()
    print("[Hazard]")


    annual_precip_cache: dict = {}

    print(f"\n  Computing average precipitation {START_YEAR}-{END_YEAR}...")
    precip = compute_avg_precip_range(
        START_YEAR, END_YEAR, GEO_YEAR, annual_precip_cache, crosswalks
    )

    print(f"\n  Loading flood events {START_YEAR}-{END_YEAR}...")
    flood_df = load_flood_events(FLOOD_EVENTS_DIR, START_YEAR, END_YEAR)

    print("  Computing annual flood rate per sq km...")
    flood_rate = compute_flood_rate(flood_df, tracts_gdf, START_YEAR, END_YEAR)

    hazard = flood_rate.merge(precip, on="GEOID", how="outer")
    hazard = hazard.merge(fema, on="GEOID", how="outer")
    hazard["GEOID"] = hazard["GEOID"].str.zfill(GEOID_LENGTH)
    hazard = hazard[["GEOID", "annual_flood_rate_per_sqkm",
                      "avg_rainfall_mm", "pct_100yr_flood_area"]]
    hazard = filter_continental(hazard)

    hazard_path = os.path.join(OUTPUT_DIR, "hazard.csv")
    hazard.to_csv(hazard_path, index=False)
    print(f"  Saved: {hazard_path}  ({len(hazard):,} rows)")

  

    print()
    print("[Vulnerability]")


    vuln = extract_vulnerability(ACS_PAIRS, GEO_YEAR, crosswalks)
    vuln["GEOID"] = vuln["GEOID"].str.zfill(GEOID_LENGTH)
    vuln = vuln[["GEOID", "pct_poverty_rate", "median_household_income", "pct_renters"]]
    vuln = filter_continental(vuln)

    vuln_path = os.path.join(OUTPUT_DIR, "vulnerability.csv")
    vuln.to_csv(vuln_path, index=False)
    print(f"  Saved: {vuln_path}  ({len(vuln):,} rows)")


    print()
    print("[Exposure]")


    exposure = extract_exposure(ACS_PAIRS, GEO_YEAR, gaz, crosswalks)
    exposure["GEOID"] = exposure["GEOID"].str.zfill(GEOID_LENGTH)

    print("\n  Loading farmland percentages (2006-2024 average)...")
    farmland = extract_farmland(ACS_PAIRS, GEO_YEAR, crosswalks)
    farmland["GEOID"] = farmland["GEOID"].str.zfill(GEOID_LENGTH)
    exposure = exposure.merge(farmland, on="GEOID", how="left")

    exposure = exposure[["GEOID", "pop_density_per_sqkm",
                          "housing_density_per_sqkm", "pct_houses_built_pre1980",
                          "pct_farmland_area"]]
    exposure = filter_continental(exposure)

    exp_path = os.path.join(OUTPUT_DIR, "exposure.csv")
    exposure.to_csv(exp_path, index=False)
    print(f"  Saved: {exp_path}  ({len(exposure):,} rows)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
