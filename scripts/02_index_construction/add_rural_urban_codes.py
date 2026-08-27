"""

Adds a single RUCC_2023 column (USDA Rural-Urban Continuum Code) to the
full-period risk index CSVs. 

Source (long format, one row per FIPS/Attribute):
  Data/Rural-UrbanRating/Ruralurbancontinuumcodes2023.csv
    FIPS, State, County_Name, Attribute, Value
    Attribute in {Population_2020, RUCC_2023, Description}

For each tract, the county FIPS is derived from its GEOID (GEOID[0:5] =
state[0:2] + county[2:5]) and used to look up that county's RUCC_2023 code.

Modifies in place:
  TimeBasedFloodIndex/2006-2024/2006-2024_MinMaxRiskIndex.csv
  TimeBasedFloodIndex/2006-2024/2006-2024_RankRiskIndex.csv
"""

import pandas as pd
from pathlib import Path

GEOID_LENGTH = 11
# Package layout: <package_root>/scripts/02_index_construction/add_rural_urban_codes.py
# RUCC lookup table is an input (data/reconstruction/); the risk index CSVs it
# modifies in place are a result (data/results/).
BASE_DIR    = Path(__file__).resolve().parents[2] / "data"
RUCC_PATH   = BASE_DIR / "reconstruction" / "Rural-UrbanRating" / "Ruralurbancontinuumcodes2023.csv"
TARGET_CSVS = [
    BASE_DIR / "results" / "TimeBasedFloodIndex" / "2006-2024" / "2006-2024_MinMaxRiskIndex.csv",
    BASE_DIR / "results" / "TimeBasedFloodIndex" / "2006-2024" / "2006-2024_RankRiskIndex.csv",
]


def load_rucc_by_county(path: Path) -> pd.Series:
    """Return a Series mapping 5-digit county FIPS -> RUCC_2023 code."""
    long_df = pd.read_csv(path, dtype={"FIPS": str}, encoding="cp1252")
    rucc_rows = long_df[long_df["Attribute"] == "RUCC_2023"]
    return rucc_rows.set_index("FIPS")["Value"]


def add_rucc_column(csv_path: Path, rucc_by_county: pd.Series) -> None:
    df = pd.read_csv(csv_path, dtype={"GEOID": str})
    df["GEOID"] = df["GEOID"].str.strip().str.zfill(GEOID_LENGTH)

    county_fips = df["GEOID"].str[0:5]
    df["RUCC_2023"] = county_fips.map(rucc_by_county)

    n_missing = df["RUCC_2023"].isna().sum()
    if n_missing:
        print(f"    WARNING: {n_missing:,} tracts had no matching county in the RUCC file")

    df.to_csv(csv_path, index=False)
    print(f"  Updated: {csv_path.relative_to(BASE_DIR)}  ({len(df):,} rows)")


def main():
    print(f"Loading RUCC codes from {RUCC_PATH.relative_to(BASE_DIR)} ...")
    rucc_by_county = load_rucc_by_county(RUCC_PATH)
    print(f"  {len(rucc_by_county):,} counties loaded")

    for csv_path in TARGET_CSVS:
        add_rucc_column(csv_path, rucc_by_county)

    print("\nDone.")


if __name__ == "__main__":
    main()
