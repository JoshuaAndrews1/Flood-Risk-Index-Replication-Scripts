"""
NOTE: this script is included for methodological
documentation of how the FEMA 100-year flood-zone-area-per-tract CSVs were
built. It requires the raw FEMA NFHL geodatabases (~50GB, one per state,
downloaded from the FEMA Flood Map Service Center) and a TIGER/Line tract
shapefile, neither of which is shipped in this package's data/ folder (see
the package README for public source citations). Its output is already
folded into the shipped hazard.csv (pct_100yr_flood_area column).
"""

import geopandas as gpd
import pandas as pd
import os
import glob
import numpy as np
import time
from shapely.geometry import box

# ── CONFIGURE THESE ─────────────────────────────────────────────────────────
# Package layout: <package_root>/scripts/01_extract/downloadarea.py
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GDB_FOLDER   = os.path.join(PACKAGE_ROOT, "data", "reconstruction", "NFHL")               # not shipped -- download from FEMA
TRACTS_SHP   = os.path.join(PACKAGE_ROOT, "data", "reconstruction", "CensusTractShapeFiles", "tl_2010_us_tract.shp")  # not shipped -- download from Census TIGER/Line
OUTPUT_CSV   = r"2010_all_states_tract_flood_pct.csv"
PROGRESS_LOG = r"processed_gdbs.txt"
TILE_SIZE    = 50
# ────────────────────────────────────────────────────────────────────────────

def load_processed(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def mark_processed(log_path, gdb_path):
    with open(log_path, "a") as f:
        f.write(gdb_path + "\n")

def process_gdb(gdb_path, tracts, geoid_col, tile_size):
    gdb_name = os.path.basename(gdb_path)

    print(f"  Reading S_FLD_HAZ_AR...", flush=True)
    t0 = time.time()
    gdf = gpd.read_file(gdb_path, layer="S_FLD_HAZ_AR")
    print(f"  Read took {time.time()-t0:.1f}s -- {len(gdf)} total features", flush=True)

    sfha = gdf[gdf["SFHA_TF"] == "T"][["geometry"]].copy()
    del gdf
    print(f"  {len(sfha)} SFHA features after filtering", flush=True)

    if sfha.empty:
        print(f"  No SFHA features found -- skipping.", flush=True)
        return None

    print(f"  Reprojecting SFHA to EPSG:5070...", flush=True)
    t0 = time.time()
    sfha = sfha.to_crs("EPSG:5070")
    print(f"  Reproject took {time.time()-t0:.1f}s", flush=True)

    sfha_bounds = sfha.total_bounds
    tracts_in_extent = tracts.cx[
        sfha_bounds[0]:sfha_bounds[2],
        sfha_bounds[1]:sfha_bounds[3]
    ].copy()
    print(f"  {len(tracts_in_extent)} tracts in bounding box", flush=True)

    if tracts_in_extent.empty:
        print(f"  No tracts found in extent -- skipping.", flush=True)
        return None

    print(f"  Building SFHA spatial index...", flush=True)
    t0 = time.time()
    sfha = sfha.reset_index(drop=True)
    sfha_sindex = sfha.sindex
    print(f"  Spatial index built in {time.time()-t0:.1f}s", flush=True)

    tract_indices = np.array_split(
        tracts_in_extent.index.tolist(),
        max(1, len(tracts_in_extent) // tile_size)
    )
    n_tiles = len(tract_indices)
    print(f"  Processing {n_tiles} tiles of ~{tile_size} tracts each...", flush=True)

    flood_accumulator = {}
    tile_times = []

    for i, idx_chunk in enumerate(tract_indices, 1):
        t_tile = time.time()
        tract_tile = tracts_in_extent.loc[idx_chunk]
        tile_bounds = tract_tile.total_bounds
        tile_box = gpd.GeoSeries([box(*tile_bounds)], crs="EPSG:5070")

        candidate_idx = list(sfha_sindex.intersection(tile_bounds))
        print(f"    Tile {i}/{n_tiles} -- {len(candidate_idx)} SFHA candidates...", flush=True)

        if not candidate_idx:
            print(f"      skipped (no candidates)", flush=True)
            continue

        sfha_tile = sfha.iloc[candidate_idx].copy()

        print(f"      Clipping SFHA to tile boundary...", flush=True)
        t_clip = time.time()
        sfha_tile = gpd.clip(sfha_tile, tile_box)
        print(f"      Clip done in {time.time()-t_clip:.1f}s -- {len(sfha_tile)} features remain", flush=True)

        if sfha_tile.empty:
            continue

        try:
            print(f"      Running overlay...", flush=True)
            t_ov = time.time()
            intersection = gpd.overlay(
                sfha_tile[["geometry"]],
                tract_tile[[geoid_col, "tract_area_m2", "geometry"]],
                how="intersection",
                keep_geom_type=True
            )
            print(f"      Overlay done in {time.time()-t_ov:.1f}s -- {len(intersection)} pieces", flush=True)
        except Exception as e:
            print(f"      [WARN] Tile {i} overlay failed: {e} -- skipping tile", flush=True)
            continue

        if not intersection.empty:
            intersection["flood_area_m2"] = intersection.geometry.area
            for geoid, area in intersection.groupby(geoid_col)["flood_area_m2"].sum().items():
                flood_accumulator[geoid] = flood_accumulator.get(geoid, 0) + area

        del intersection, sfha_tile, tract_tile

        elapsed = time.time() - t_tile
        tile_times.append(elapsed)
        avg = sum(tile_times) / len(tile_times)
        remaining_est = avg * (n_tiles - i)
        print(f"    Tile {i}/{n_tiles} complete -- {elapsed:.1f}s -- est. {remaining_est/60:.1f} min remaining", flush=True)

    flood_by_tract = pd.DataFrame(
        list(flood_accumulator.items()),
        columns=[geoid_col, "flood_area_m2"]
    )

    result = tracts_in_extent[[geoid_col, "tract_area_m2"]].merge(
        flood_by_tract, on=geoid_col, how="left"
    )
    result["flood_area_m2"] = result["flood_area_m2"].fillna(0)
    result["pct_100yr_flood"] = (
        (result["flood_area_m2"] / result["tract_area_m2"] * 100)
        .clip(0, 100)
        .round(2)
    )

    output = result[[geoid_col, "tract_area_m2", "flood_area_m2", "pct_100yr_flood"]].copy()
    output.columns = ["GEOID", "tract_area_m2", "flood_area_m2", "pct_100yr_flood"]
    return output



print("Reading census tracts...", flush=True)
tracts = gpd.read_file(TRACTS_SHP)
print(f"  {len(tracts)} tracts loaded", flush=True)

geoid_col = next(
    (c for c in tracts.columns if c.upper() in ("GEOID", "GEOID10", "GEOID20", "AFFGEOID")),
    None
)
if geoid_col is None:
    print("  WARNING: Could not auto-detect GEOID column. Defaulting to 'GEOID'.", flush=True)
    geoid_col = "GEOID"
print(f"  Using GEOID column: '{geoid_col}'", flush=True)

print("Reprojecting tracts to EPSG:5070...", flush=True)
tracts = tracts.to_crs("EPSG:5070")
tracts["tract_area_m2"] = tracts.geometry.area


gdb_files = sorted(glob.glob(os.path.join(GDB_FOLDER, "**", "*.gdb"), recursive=True))
if not gdb_files:
    gdb_files = sorted(glob.glob(os.path.join(GDB_FOLDER, "*.gdb")))

print(f"\nFound {len(gdb_files)} GDB file(s) in '{GDB_FOLDER}'", flush=True)

processed = load_processed(PROGRESS_LOG)
print(f"  Already processed: {len([g for g in gdb_files if g in processed])}", flush=True)
print(f"  Remaining:         {len([g for g in gdb_files if g not in processed])}", flush=True)


for gdb_path in gdb_files:
    gdb_name = os.path.basename(gdb_path)

    if gdb_path in processed:
        print(f"\n[SKIP] {gdb_name} (already processed)", flush=True)
        continue

    print(f"\n{'='*60}", flush=True)
    print(f"[START] {gdb_name}", flush=True)

    try:
        output = process_gdb(gdb_path, tracts, geoid_col, TILE_SIZE)

        if output is not None:
            write_header = not os.path.exists(OUTPUT_CSV)
            output.to_csv(OUTPUT_CSV, mode="a", index=False, header=write_header)
            print(f"  [DONE] {len(output)} tracts written", flush=True)
            print(f"    With any flood zone: {(output['pct_100yr_flood'] > 0).sum()}", flush=True)
            print(f"    Mean pct flood:      {output['pct_100yr_flood'].mean():.2f}%", flush=True)

        mark_processed(PROGRESS_LOG, gdb_path)

    except Exception as e:
        print(f"\n  [ERROR] {gdb_name}: {e}", flush=True)
        print(f"  This GDB was NOT marked as processed and will be retried on next run.", flush=True)
        continue




print(f"\n{'='*60}", flush=True)
print("All GDBs processed.", flush=True)

if os.path.exists(OUTPUT_CSV):
    final = pd.read_csv(OUTPUT_CSV)
    print(f"\nFinal CSV summary ({OUTPUT_CSV}):", flush=True)
    print(f"  Total tract rows:           {len(final)}", flush=True)
    print(f"  Unique GEOIDs:              {final['GEOID'].nunique()}", flush=True)
    print(f"  Tracts with any flood zone: {(final['pct_100yr_flood'] > 0).sum()}", flush=True)
    print(f"  Tracts with >10% flood:     {(final['pct_100yr_flood'] > 10).sum()}", flush=True)
    print(f"  Tracts with >50% flood:     {(final['pct_100yr_flood'] > 50).sum()}", flush=True)
    print(f"  Mean pct flood (all tracts):{final['pct_100yr_flood'].mean():.2f}%", flush=True)