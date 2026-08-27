# flood-risk-index

Code for "Mapping Multidimensional Flood Risk and Socioeconomic Relevance to Local Business Dynamics Across the United States."

This repository contains the scripts needed to reproduce the paper's full-period
(2006-2024) flood risk index, its robustness checks, spatial autocorrelation analysis,
rural-urban continuum comparisons, and city maps. **Data is hosted separately on
Zenodo** (see below) rather than in this repository.

## Getting the data

Download the two archives from Zenodo:

https://doi.org/10.5281/zenodo.22072498

- `data_reconstruction.zip` — inputs needed to (re)run the pipeline (indicator tables,
  2020 census tract shapefiles, USDA RUCC codes, merged CBP/BDS panel)
- `data_results.zip` — outputs already computed by the scripts (risk index CSVs,
  robustness/sensitivity/metro-comparison tables) — the numbers behind the paper's
  tables and figures

Unzip both at the repo root so you end up with:

```
flood-risk-index/
  data/
    reconstruction/   <- from data_reconstruction.zip
    results/           <- from data_results.zip
  scripts/
  ...
```


## Directory layout

```
scripts/
  01_extract/                      raw-data extraction (see "Not directly runnable" below)
  02_index_construction/           PCA dimension scores + combined index (core pipeline)
  03_robustness/                   alternative normalization/aggregation/PCA/leave-one-out
  04_spatial_autocorrelation/      Moran's I, LISA cluster maps, correlation tables
  05_rucc_analysis/                metro/nonmetro t-tests, RUCC bar charts
  06_maps/                         nationwide and city-level choropleths
  07_distributions_summary/        KDE distributions + summary statistics (Table 1, Fig. 2)
  08_business_dynamics_data_prep/  builds the CBP/BDS county-year panel (see gap note below)
```

All scripts resolve their data paths relative to the repo root (via `__file__`), so no
path editing is needed once `data/reconstruction/` and `data/results/` are in place.

## Pipeline order

The scripts in `02_index_construction/` through `07_distributions_summary/` can be run
directly against the downloaded data, in this order:

1. `02_index_construction/fullperiod_pca.py` — builds `2006-2024_MinMaxRiskIndex.csv`
   and `2006-2024_RankRiskIndex.csv` in `data/results/...` from
   `data/reconstruction/.../Combined Data/{hazard,vulnerability,exposure}.csv`.
   Set `COMBINE_METHOD = "multiplicative"` (geometric mean, the paper's baseline) or
   `"equal_weight"` (arithmetic mean) and re-run to regenerate the `geometric_`/
   `arithmetic_`-prefixed files already included in `data/results/` (rename the two
   output files accordingly after each run — this manual rename is not automated in
   the script).
2. `02_index_construction/add_rural_urban_codes.py` — adds the `RUCC_2023` column
   to the result CSVs in place, from the RUCC lookup table in `data/reconstruction/`.
3. `03_robustness/robustness_check.py` and `plot_baseline_vs_arithmetic_scatter.py`
   — Table 3 (robustness comparisons) and the geometric-vs-arithmetic scatter.
4. `04_spatial_autocorrelation/spatial_autocorrelation_fullperiod.py` — Table 2
   (correlations), Global Moran's I, Moran scatterplots (Fig. 3), LISA cluster maps
   (Fig. 4, Table 4-equivalent cluster counts).
5. `05_rucc_analysis/metro_nonmetro_analysis.py` and `plot_rucc_score_comparison.py`
   — Table 5 (metro vs. nonmetro) and Fig. 6 (RUCC bar chart).
6. `06_maps/map_flood_index_fullperiod.py` and `MetroAreaIndexMaps.py` — Fig. 1
   (nationwide component maps) and Fig. 7 (city maps).
7. `07_distributions_summary/plot_index_distributions_fullperiod.py` — Fig. 2 and
   Table 1 (summary statistics, printed to stdout and via the KDE plot).

`01_extract/` and `08_business_dynamics_data_prep/` are included for methodological
documentation but are **not directly runnable** — see below.

## Raw public data sources

`01_extract/` and `08_business_dynamics_data_prep/` document how the data on Zenodo was
built from raw public sources, but those raw sources are not included on Zenodo either
(they are far larger than the derived data actually used downstream). Each script is
annotated in-file with what it needs and where its output already appears in
`data/reconstruction/`. Sources, all public:

- **FEMA NFHL geodatabases** (flood zone geometry, per state) —
  FEMA Flood Map Service Center, https://msc.fema.gov/portal/advanceSearch
- **PRISM daily/annual precipitation rasters** — PRISM Climate Group, Oregon State
  University, https://prism.oregonstate.edu/
- **NLCD Annual land cover rasters** — USGS Multi-Resolution Land Characteristics
  Consortium, https://www.mrlc.gov/
- **ACS 5-year estimates** (poverty, income, renters, housing, population) —
  U.S. Census Bureau American Community Survey, via data.census.gov or the Census API
- **Flood event records** — NOAA Storm Events Database
- **County Business Patterns (CBP)** and **Business Dynamics Statistics (BDS)** —
  U.S. Census Bureau, https://www.census.gov/programs-surveys/cbp.html and
  https://www.census.gov/programs-surveys/bds.html
- **2010/2020 TIGER/Line tract shapefiles** and **Census crosswalk/gazetteer files** —
  U.S. Census Bureau (the 2020 tract shapefile and the 2020 cartographic tract boundary
  file *are* included in the Zenodo `data_reconstruction.zip`, since they are required
  directly by several essential scripts)



## Requirements

```
pip install -r requirements.txt
```


