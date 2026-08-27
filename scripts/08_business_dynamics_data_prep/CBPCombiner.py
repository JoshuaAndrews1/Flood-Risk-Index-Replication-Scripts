"""
NOTE (replication package): this script is included for methodological
documentation. It requires the raw yearly County Business Patterns (CBP)
files from the U.S. Census Bureau, which are not shipped in this package's
data/ folder (see the package README for public source citations). The
merged county-year panel it feeds into is already shipped as
data/SocioEconomicData/CBPBDSMerged/merged_*.csv.
"""

import os
import re
import pandas as pd

# ---- EDIT THESE ----
# Package layout: <package_root>/scripts/08_business_dynamics_data_prep/CBPCombiner.py
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FOLDER = os.path.join(PACKAGE_ROOT, "data", "reconstruction", "SocioEconomicData", "CBP")           # not shipped -- download from Census CBP
OUTPUT = os.path.join(PACKAGE_ROOT, "data", "reconstruction", "SocioEconomicData", "CBP_total.csv")
# --------------------

files = []

for filename in os.listdir(FOLDER):
    if not filename.lower().endswith(('.csv', '.txt')):
        continue
    match = re.search(r'(\d{2})', filename)
    if not match:
        print(f"Skipping {filename}: no year found in name")
        continue
    year = 2000 + int(match.group(1))
    files.append((year, filename))

files.sort(key=lambda x: x[0])

header_written = False

for year, filename in files:
    filepath = os.path.join(FOLDER, filename)
    print(f"Processing {filename} -> year {year}")

    df = pd.read_csv(filepath, dtype=str, low_memory=False)
    df.insert(0, 'year', str(year))

    df.to_csv(OUTPUT, mode='a', index=False, header=not header_written)
    header_written = True

print(f"Done. Output saved to {OUTPUT}")