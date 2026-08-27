"""
NOTE: this script is included for methodological
documentation of how the merged CBP/BDS county-year panel was built. It
requires the raw yearly County Business Patterns (CBP) files and the
Business Dynamics Statistics (BDS) file from the U.S. Census Bureau, which
are not shipped in this package's data/ folder (see the package README for
public source citations). Its output -- the panel underlying the paper's
business-dynamics regressions -- is already shipped as
data/SocioEconomicData/CBPBDSMerged/merged_*.csv.
"""

import pandas as pd
import os
import glob


# CONFIG
# Package layout: <package_root>/scripts/08_business_dynamics_data_prep/CBPBDSMerger.py
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CBP_DIR = os.path.join(PACKAGE_ROOT, "data", "reconstruction", "SocioEconomicData", "CBP")                                    # not shipped -- download from Census CBP
BDS_FILE = os.path.join(PACKAGE_ROOT, "data", "reconstruction", "SocioEconomicData", "bds_1978_2023_st_cty.csv")              # not shipped -- download from Census BDS
OUTPUT_DIR = os.path.join(PACKAGE_ROOT, "data", "reconstruction", "SocioEconomicData", "CBPBDSMerged")

# CBP file naming pattern - adjust to match your files
# Common patterns: "cbp00co.txt", "cbp2000.csv", "CBP2000.CB2000CBP-Data.csv"
# Update this function to match your actual filenames
def get_cbp_filepath(year):
    """Return the filepath for a given year's CBP file. Adjust as needed."""
    # Try common naming patterns
    patterns = [
        os.path.join(CBP_DIR, f"cbp{str(year)[2:]}co.txt"),       # cbp00co.txt
        os.path.join(CBP_DIR, f"cbp{year}co.txt"),                 # cbp2000co.txt
        os.path.join(CBP_DIR, f"cbp{str(year)[2:]}co.csv"),       # cbp00co.csv
        os.path.join(CBP_DIR, f"cbp{year}.csv"),                   # cbp2000.csv
        os.path.join(CBP_DIR, f"CBP{year}.CB{year}CBP-Data.csv"), # Census API format
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    # Try glob as fallback
    matches = glob.glob(os.path.join(CBP_DIR, f"*{year}*"))
    if matches:
        return matches[0]
    matches = glob.glob(os.path.join(CBP_DIR, f"*{str(year)[2:]}*"))
    if matches:
        return matches[0]
    return None

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Columns we want to keep (standardized names, lowercase)
# These are the common columns across all years
COMMON_COLS = [
    "fipstate", "fipscty", "naics",
    "emp", "qp1", "ap", "est",
    "n5_9", "n10_19", "n20_49", "n50_99",
    "n100_249", "n250_499", "n500_999", "n1000",
    "n1000_1", "n1000_2", "n1000_3", "n1000_4",
]

# Columns that changed names across years
RENAME_MAP = {
    "n<5": "n1_4",   # 2017+ renamed n1_4 to n<5
    "n_5":  "n1_4",  # Just in case
}


# Load BDS data
print("Loading BDS data...")
bds = pd.read_csv(BDS_FILE, dtype={"st": str, "cty": str})
bds["st"] = bds["st"].str.zfill(2)
bds["cty"] = bds["cty"].str.zfill(3)
print(f"  BDS loaded: {len(bds):,} rows, years {bds['year'].min()}-{bds['year'].max()}")


# NAICS filter function
def is_3digit_or_broader(code):
    code = str(code).strip()
    if code == "------":
        return True
    n_digits = 0
    for ch in code:
        if ch.isdigit():
            n_digits += 1
        else:
            break
    return n_digits <= 3


# Process each year
for target_year in range(2000, 2024):
    print(f"\n{'='*60}")
    print(f"Processing year {target_year}...")

    # Find CBP file 
    cbp_path = get_cbp_filepath(target_year)
    if cbp_path is None:
        print(f"  CBP file not found for {target_year}, skipping.")
        print(f"  Looked in: {CBP_DIR}/")
        continue
    print(f"  CBP file: {cbp_path}")

    # Get BDS for this year 
    bds_year = bds[bds["year"] == target_year]
    if bds_year.empty:
        print(f"  No BDS data for {target_year}, skipping.")
        continue

    # Read CBP file 
    try:
        cbp = pd.read_csv(
            cbp_path,
            dtype=str,                  # Read everything as string first
            on_bad_lines="warn",
            encoding="latin-1",         # Some older Census files use latin-1
        )
    except Exception as e:
        print(f"  ERROR reading {cbp_path}: {e}")
        continue

    # Standardize column names 
    cbp.columns = cbp.columns.str.lower().str.strip().str.strip('"')

    # Rename changed columns
    cbp.rename(columns=RENAME_MAP, inplace=True)

    print(f"  Columns: {list(cbp.columns)}")
    print(f"  Raw rows: {len(cbp):,}")

    # Filter NAICS 
    if "naics" not in cbp.columns:
        print(f"  ERROR: 'naics' column not found. Skipping.")
        continue

    cbp = cbp[cbp["naics"].apply(is_3digit_or_broader)]
    print(f"  After NAICS filter (<=3-digit): {len(cbp):,} rows")

    # Keep and standardize relevant columns 
    # Add n1_4 column: use n1_4 if it exists, otherwise it was renamed from n<5
    available = [c for c in COMMON_COLS if c in cbp.columns]
    missing = [c for c in COMMON_COLS if c not in cbp.columns]
    if missing:
        print(f"  Note: missing columns (will be NaN): {missing}")

    cbp = cbp.reindex(columns=COMMON_COLS + ["n1_4"])  # include n1_4
    # Deduplicate n1_4 if it already exists in COMMON_COLS
    cbp = cbp.loc[:, ~cbp.columns.duplicated()]

    # Convert numeric columns
    for col in cbp.columns:
        if col not in ["fipstate", "fipscty", "naics"]:
            cbp[col] = pd.to_numeric(cbp[col], errors="coerce")

    # Zero pad FIPS 
    cbp["fipstate"] = cbp["fipstate"].str.strip().str.zfill(2)
    cbp["fipscty"] = cbp["fipscty"].str.strip().str.zfill(3)

    # Add year 
    cbp["year"] = target_year

    # Merge with BDS 
    merged = pd.merge(
        cbp,
        bds_year,
        left_on=["year", "fipstate", "fipscty"],
        right_on=["year", "st", "cty"],
        how="inner"
    )
    merged.drop(columns=["st", "cty"], inplace=True, errors="ignore")

    print(f"  BDS rows: {len(bds_year):,}")
    print(f"  Merged: {len(merged):,} rows")

   
    out_path = os.path.join(OUTPUT_DIR, f"merged_{target_year}.csv")
    merged.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

 
    if target_year == 2000:
        print(f"\n  --- Sample entries from {target_year} ---")
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 250)
        cols_to_show = [
            "year", "fipstate", "fipscty", "naics", "emp", "est",
            "net_job_creation", "net_job_creation_rate",
            "job_creation_rate", "job_destruction_rate"
        ]
        show = [c for c in cols_to_show if c in merged.columns]
        print(merged[show].head(10).to_string(index=False))

print(f"\n{'='*60}")
print("Done! All yearly files saved to:", OUTPUT_DIR)
print("\nIf CBP files were not found, update get_cbp_filepath() with your naming convention.")
print("You can also list your actual files with:")
print(f"  import os; print(os.listdir('{CBP_DIR}'))")