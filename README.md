# data543_wind_energy_droughts
ERCOT Wind PPA Risk — XGBoost framework predicting high electricity price exposure during wind energy droughts. Models compound risk from low wind output, elevated gas prices, and temperature stress to generate hedging signals for fixed-volume PPA producers. Includes ERA5 reanalysis, futures hedge simulation, and grid-cell P&L analysis.

---

## Repository Structure
 
```
files/
├── 01_download_era5_wind.py
├── 02_download_era5_temperature.py
├── 03_process_electricity_prices.py
├── 04_classify_droughts.py
├── 05_logistic_regression_models.py
├── 06_xgboost_models.py
└── 07_xgboost_zone_models.py
```

---

## Data Sources

| # | Data | Source | Period | Resolution |
|---|------|---------|--------|------------|
| 1 | ERA5 100 m wind (u, v) | [Copernicus CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) | 1980–2025 | Hourly, ~31 km |
| 2 | ERA5 2 m air temperature | [Copernicus CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) | 1980–2025 | Hourly, ~31 km |
| 3 | Vestas V90-2.0 MW power curve | [Vestas Wind Systems](https://en.wind-turbine-models.com/turbines/16-vestas-v90) | — | — |
| 4 | EIA Form EIA-860 (onshore wind) | [U.S. EIA](https://www.eia.gov/electricity/data/eia860/) | 2020–2024 | Annual |
| 5 | ERCOT Settlement Point Prices (NP6-905-CD) | [ERCOT Data Access Portal](https://data.ercot.com/data-product-details/np6-905-cd) | 2020–2024 | 15-min / hourly |
| 6 | Henry Hub natural gas spot prices | [U.S. EIA](https://www.eia.gov/dnav/ng/hist/rngwhhdm.htm) | 2020–2024 | Daily |
| 7 | NYMEX Henry Hub futures (C1–C4) | [U.S. EIA API](https://www.eia.gov/dnav/ng/ng_pri_fut_s1_d.htm) | Oct 2019–Dec 2024 | Daily |

---

## Prerequisites

```bash
pip install cdsapi xarray cfgrib eccodes pandas numpy
```

### CDS API Credentials (ERA5 downloads)

1. Register at <https://cds.climate.copernicus.eu>
2. Accept the ERA5 terms of use on the dataset page
3. Create `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <YOUR_API_KEY>
```

Your key is available on your [CDS profile page](https://cds.climate.copernicus.eu/profile). Never commit credentials to the repository.

---

## Pipeline

### 1 — Download ERA5 Wind Data

**Script:** `files/01_download_era5_wind.py`

Downloads hourly 100 m u- and v-components of wind velocity for the ERCOT / Texas domain (26°–36°N, 94°–107°W) from 1980 through 2025. Output is one GRIB file per year. The script skips years whose output file already exists, so it is safe to re-run after interruptions.

```bash
python files/01_download_era5_wind.py
```

**Output:** `era5_wind_data/era5_wind_{year}.grib`

Wind variables are subsequently converted to wind speeds using the formula $\sqrt{u^2 + v^2}$ and then converting to capacity factors $(0 - 1)$ using the Vestas V90-2.0 MW power curve. A drought event is identified when the zone-level capacity factor falls below 0.30, consistent with typical mid-30% capacity factors observed at Texas utility-scale wind farms.

---

### 2 — Download and Process ERA5 Temperature

**Script:** `files/02_download_era5_temperature.py`

Three sub-steps that can be run individually or together via `--step`.

```bash
# Download GRIB files from CDS (1980–2025)
python files/02_download_era5_temperature.py --step download

# Extract per-cell time series from GRIBs
python files/02_download_era5_temperature.py --step extract

# Compute HDD/CDD and extreme temperature flags
python files/02_download_era5_temperature.py --step degrees

# Run all three steps end-to-end (default)
python files/02_download_era5_temperature.py
```

**Step 2a — Download** fetches hourly 2 m air temperature GRIBs over the same domain and period as the wind data. Output: `raw_temp/{year}_tmm.grib`.

**Step 2b — Extract** reads `data/wind_grid_cells.csv` (produced by the EIA-860 capacity-assignment step) to obtain the 123 ERA5 grid cells that fall inside ERCOT and carry installed wind capacity. For each cell, the nearest-neighbour ERA5 value is selected using vectorised `xarray` operations, converted from Kelvin to Fahrenheit, and written to a per-cell CSV.

**Output:** `tmm_hourly_wind_cells/{lat_idx}_{lon_idx}_tmm.csv`  
**Columns:** `timestamp, Load_Zone, tmm_K, tmm_F, grid_latitude, grid_longitude`

**Step 2c — Degree hours** appends the following columns to each per-cell CSV in place:

| Column | Description |
|--------|-------------|
| `HDD_hourly` | `max(65 − tmm_F, 0)` per hour |
| `CDD_hourly` | `max(tmm_F − 65, 0)` per hour |
| `HDD_daily` | Sum of `HDD_hourly` for the calendar day, broadcast to all hourly rows |
| `CDD_daily` | Sum of `CDD_hourly` for the calendar day, broadcast to all hourly rows |
| `extreme_hdd_hourly` | 1 if `HDD_hourly` exceeds the year-specific 90th percentile |
| `extreme_cdd_hourly` | 1 if `CDD_hourly` exceeds the year-specific 90th percentile |
| `extreme_hdd_daily` | 1 if `HDD_daily` exceeds the year-specific 90th percentile |
| `extreme_cdd_daily` | 1 if `CDD_daily` exceeds the year-specific 90th percentile |
| `extreme_hourly` | 1 if either hourly HDD or CDD threshold is exceeded |
| `extreme_daily` | 1 if either daily HDD or CDD threshold is exceeded |

Year-specific thresholds are computed separately for each year to account for interannual variation in temperature regimes. The final XGBoost model uses `tmm_F` directly alongside a continuous `temp_stress` signal (`|tmm_F − 65| / 70`, clipped to [0, 1]) rather than binary flags.

---

### 3 — Process ERCOT Electricity Prices

**Script:** `files/03_process_electricity_prices.py`

> **Manual download required.** ERCOT does not provide a public bulk-download API. Download 15-minute settlement point price files for 2020–2024 from the [ERCOT Data Access Portal](https://data.ercot.com/data-product-details/np6-905-cd) and place them in `data/raw_prices/`.

```bash
python files/03_process_electricity_prices.py
```

The script aggregates 15-minute intervals to hourly averages, filters to the four load zones with significant installed wind capacity (`LZ_WEST`, `LZ_SOUTH`, `LZ_NORTH`, `LZ_HOUSTON`), and labels each hour `HIGH` or `LOW` using a **year-specific, load-zone-specific 90th-percentile** threshold. This is applied separately to the training and test periods so the `HIGH` label consistently captures the top decile of price stress within each year regardless of fuel price cycles or structural grid changes across 2020–2024.

**Outputs:**

| File | Period | Role |
|------|--------|------|
| `data/processed/ercot_prices_2020_2021.csv` | 2020–2021 | Test set |
| `data/processed/ercot_prices_2022_2024.csv` | 2022–2024 | Training set |

**Output columns:** `hour, load_zone, price, Price Exposure`

---

### 4 — Classify Wind Energy Droughts

**Script:** `files/04_classify_droughts.py`

Classifies every hourly observation in the merged wind + temperature + price files as one of four drought severity tiers, then aggregates to a daily classification using majority-rule logic. Both the training period (2022–2024) and test period (2020–2021) are processed.

```bash
python files/04_classify_droughts.py
```

#### Hourly Classification

A drought hour is any hour where the wind capacity factor falls below 0.30. Each drought hour is further classified by combining instantaneous CF depth with the length of the consecutive drought run it belongs to. Run lengths are computed using vectorised segment IDs that reset whenever the drought flag changes or a timestamp gap exceeds one hour.

| Category | Condition |
|----------|-----------|
| `NO_DROUGHT` | CF ≥ 0.25 (not in a drought run) |
| `MILD` | CF < 0.25 and run < 10 hours |
| `MODERATE` | Run 10–24 hours, OR run ≥ 24 hours with CF ≥ 0.15 |
| `SEVERE` | Run ≥ 24 hours AND CF < 0.15 |

The `drought_run_hours` column — the continuous run length at each hour — is retained as a model feature alongside raw `wind_cf`. A 48-hour drought places considerably more stress on the grid than a 2-hour dip at the same CF level, as reserves deplete over time and cheaper backup options are progressively exhausted.

#### Daily Classification

Each calendar day is assigned the drought category that best represents the majority of its 24 hourly observations, applied in priority order:

1. ≥ 20 hours `NO_DROUGHT` → `NO_DROUGHT`
2. ≥ 12 hours `SEVERE` → `SEVERE`
3. ≥ 12 hours `MODERATE` → `MODERATE`
4. ≥ 12 hours `MILD` → `MILD`
5. Tie-break: category with the most hours that day

**Outputs:**

| Directory | Period | Content |
|-----------|--------|---------|
| `wind_temp_20_21_hourly_FIXED/` | 2020–2021 | Hourly files updated in place with classification columns |
| `wind_temp_22_24_hourly_FIXED/` | 2022–2024 | Hourly files updated in place with classification columns |
| `wind_temp_data_daily_20_21_RECOMPUTED/` | 2020–2021 | One daily summary CSV per grid cell |
| `wind_temp_data_daily_22_24_RECOMPUTED/` | 2022–2024 | One daily summary CSV per grid cell |

**New hourly columns:** `drought, drought_run_hours, hourly_drought_category`

**Daily output columns:** `date, Load_Zone, grid_latitude, grid_longitude, daily_drought_category, daily_drought_hours, daily_non_drought_hours, daily_mild_hours, daily_moderate_hours, daily_severe_hours, daily_mean_wind_cf, daily_min_wind_cf`

---
### 5 — Logistic Regression Baseline Models
 
**Script:** `files/05_logistic_regression_models.py`
 
Three successive logistic regression specifications are evaluated as a baseline before moving to XGBoost. All three use 5-fold stratified cross-validation on the training data and share the same class-balanced setup to handle the ~10:1 LOW:HIGH imbalance.
 
```bash
# Run all three models
python files/05_logistic_regression_models.py
 
# Run a single specification
python files/05_logistic_regression_models.py --model LR-1
python files/05_logistic_regression_models.py --model LR-2
python files/05_logistic_regression_models.py --model LR-3
```
 
#### Model Specifications
 
**LR-1** uses a calendar-day drought category (ordinal 0–3) and a binary extreme-temperature demand flag with one interaction term. The calendar-day label averages over intraday variation in wind output, losing the timing information most relevant to hourly price formation.
 
**LR-2** replaces the calendar-day label with an hourly drought category and adds consecutive drought run hours as a separate feature, along with two interaction terms. Moving to hourly labels improves AUC but the ordinal encoding still discards information at category boundaries — an hour at CF = 0.149 and an hour at CF = 0.051 receive the same label despite representing materially different levels of grid stress.
 
**LR-3** replaces the categorical drought signal entirely with raw `wind_cf` and `drought_run_hours` as continuous inputs, alongside the binary demand flag and two interaction terms. This is the final logistic regression specification. Further search revealed no path to meaningful improvement: AUC had effectively plateaued and precision remained near 0.14–0.15 across all variants.
 
The core limitation of all three models is architectural. Logistic regression learns a single linear decision boundary and can only express the relationship between wind output, temperature stress, and price exposure as a weighted additive combination of inputs. ERCOT price formation is not linear: the marginal impact of a wind shortfall on price depends sharply on what gas costs, how long the drought has persisted, and how much temperature-driven demand is on the system — conditions that compound together in ways no linear model can capture regardless of feature engineering. This motivates the shift to XGBoost.
 
#### Results (Table 3 from paper)
 
| Model | Key Change | CV AUC | CV Recall | CV Precision |
|-------|-----------|--------|-----------|--------------|
| LR-1 | Calendar-day drought category + binary demand flag | 0.583 | 0.536 | 0.134 |
| LR-2 | Hourly drought category + run hours | 0.640 | 0.713 | 0.145 |
| LR-3 | Raw CF + run hours (continuous) | 0.636 | 0.697 | 0.145 |
 
**Outputs:** `results/logistic_regression/{lr_1,lr_2,lr_3}/test_evaluation.png` and `results/logistic_regression/lr_comparison.csv`

## Natural Gas Prices

Henry Hub spot prices and NYMEX futures (C1–C4) are downloaded directly from the EIA API in the model notebooks. No separate script is needed; EIA bulk data is publicly accessible without credentials.

- Spot prices: <https://www.eia.gov/dnav/ng/hist/rngwhhdm.htm>
- Futures (C1–C4): <https://www.eia.gov/dnav/ng/ng_pri_fut_s1_d.htm>

Daily values are forward-filled across weekends and holidays (up to five consecutive non-trading days) and broadcast to the hourly level. Futures prices are converted from `$/MMBtu` to `$/MWh` using a heat rate of 8.978 MMBtu/MWh (38% combined-cycle gas turbine efficiency).

---

### 6 — XGBoost Global Models
 
**Script:** `files/06_xgboost_models.py`
 
Two successive XGBoost specifications are evaluated as the primary prediction models. Both use the same hyperparameter configuration (Table 4 from paper) and 5-fold stratified cross-validation on the training data.
 
```bash
# Run both models end-to-end (default)
python files/06_xgboost_models.py
 
# Run a single specification
python files/06_xgboost_models.py --model XGB-1
python files/06_xgboost_models.py --model XGB-2
```
 
#### Why XGBoost over logistic regression
 
Logistic regression plateaued at CV AUC ~0.636 across all three specifications. The bottleneck was architectural: a single linear decision boundary cannot express the nonlinear, regime-dependent compound interactions between wind output, gas prices, and temperature that drive HIGH price exposure in ERCOT. XGBoost builds an ensemble of decision trees sequentially, each correcting residual errors of the previous ensemble, which allows it to discover these interactions organically without requiring them to be manually specified in advance.
 
#### Hyperparameters (shared across both models)
 
| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 200 | Number of trees in the ensemble |
| `max_depth` | 4 | Maximum tree depth — primary control for overfitting |
| `learning_rate` | 0.1 | Shrinkage factor applied after each tree |
| `subsample` | 0.8 | Fraction of training rows sampled per tree |
| `colsample_bytree` | 0.8 | Fraction of features randomly selected per tree |
| `min_child_weight` | 5 | Minimum instance weight to form a leaf — conservative regulariser |
| `scale_pos_weight` | auto | Computed as n_LOW / n_HIGH from training data to correct for class imbalance |
 
#### Model Specifications
 
**XGB-1** uses eight features: raw `wind_cf`, `drought_run_hours`, the binary `extreme_hourly` demand flag, daily Henry Hub spot price (`gas_price_mmbtu`), and four engineered interaction terms using the binary flag as the demand multiplier. The addition of natural gas spot prices over the logistic regression baseline produced an immediate and substantial improvement in CV AUC (0.636 → 0.791), confirming that gas price was carrying the dominant predictive signal the wind-and-temperature-only models could not access. The binary `extreme_hourly` flag remains a binding constraint — it treats a mildly hot afternoon identically to an extreme heat event.
 
**XGB-2** is the final global model specification. The binary `extreme_hourly` flag is replaced by a continuous `temp_stress` signal (`|tmm_F − 65| / 70`, clipped to [0, 1]), which preserves the nonlinear relationship between temperature deviation and demand intensity. Three additional base features are added: `tmm_F`, `HDD_hourly`, and `CDD_hourly`. All four interaction terms are recomputed using `temp_stress_norm` instead of the binary flag. The dominant feature becomes the three-way `gas_x_low_wind_x_demand` interaction (importance = 0.355), confirming that HIGH price exposure in ERCOT is driven by the simultaneous confluence of low wind output, elevated gas prices, and temperature-driven demand — not any single factor in isolation.
 
#### Results (Table 6 from paper)
 
| Model | CV AUC | CV Recall | CV Precision | Test AUC | Top feature |
|-------|--------|-----------|--------------|----------|-------------|
| XGB-1 | 0.7906 | 0.6728 | 0.2526 | 0.5984 | gas_price_mmbtu (0.271) |
| XGB-2 | 0.8038 | 0.6790 | 0.2684 | 0.6003 | gas_x_low_wind_x_demand (0.355) |
 
The train-test performance gap (CV AUC ~0.80 vs test AUC ~0.60) reflects structural differences between periods rather than overfitting. The 2022–2024 training window was shaped by post-Ukraine war gas price formation and a mature 40,355 MW wind fleet, while the 2020–2021 test period was characterised by COVID-era demand suppression, historically low gas prices ($1.33–$6.37/MMBtu excluding Winter Storm Uri), and a smaller ~29,200 MW installed base. Test AUC improves from 0.546 in 2020 to 0.669 in 2021 as conditions began normalising.
 
**Outputs:** `results/xgboost/{xgb_1,xgb_2}/test_evaluation.png`, `feature_importances.png`, `feature_importances.csv`, and `results/xgboost/xgb_comparison.csv`

### 7 — XGBoost Zone-Specific Models
 
**Script:** `files/07_xgboost_zone_models.py`
 
Extends the global XGBoost analysis by evaluating performance separately for each of the four ERCOT load zones. Two complementary analyses are run in a single script.
 
```bash
python files/07_xgboost_zone_models.py
```
 
#### Analysis A — Global model evaluated per zone
 
The fitted global XGB-2 model is applied to each zone's test data separately. This reveals where the compound stress signal fires most reliably without any retraining.
 
#### Analysis B — Separate XGBoost model trained per zone
 
One XGBoost model is trained exclusively on each zone's 2022–2024 data using the identical 11-feature XGB-2 specification and hyperparameters. This is the more rigorous evaluation: it allows zone-specific feature importance rankings to emerge from zone-specific price formation dynamics rather than being averaged across the full grid.
 
Per-zone models outperform the global specification in three of the four load zones (LZ_HOUSTON, LZ_NORTH, LZ_SOUTH). The global model marginally outperforms in LZ_WEST — the zone with the largest installed capacity and most diffuse wind-price dynamics — where year-over-year variability in conditions makes the statistical relationships learned in training harder to generalise.
 
The 2020-vs-2021 AUC improvement is geographically widespread across all zones, confirming that the train-test gap is driven by COVID-era distributional shift rather than zone-specific model failure.
 
#### Feature importance by zone
 
Feature importance rankings differ meaningfully across zones, consistent with the distinct price formation mechanisms documented in the paper:
 
| Zone | Top feature | Importance | Interpretation |
|------|-------------|------------|----------------|
| LZ_SOUTH | `gas_x_low_wind_x_demand` | 0.500 | Transmission bottlenecks mean local wind shortfalls translate directly into local price stress |
| LZ_NORTH | `gas_x_low_wind_x_demand` | 0.386 | Compound instantaneous shortfalls dominate |
| LZ_HOUSTON | `gas_x_demand` | 0.310 | No local wind — price driven by cost of imported generation under demand stress |
| LZ_WEST | `drought_run_hours` | 0.203 | Duration rather than instantaneous severity predicts price stress, reflecting limited local gas backup and intertie capacity constraints |
 
#### Results (Table 7 from paper)
 
| Zone | CV AUC | Test AUC | Test AUC 2020 | Test AUC 2021 | Test Precision | Test Recall |
|------|--------|----------|---------------|---------------|----------------|-------------|
| LZ_HOUSTON | 0.884 ± 0.008 | 0.626 | 0.589 | 0.682 | 0.204 | 0.284 |
| LZ_NORTH | 0.898 ± 0.001 | 0.644 | 0.593 | 0.724 | 0.262 | 0.232 |
| LZ_SOUTH | 0.882 ± 0.002 | 0.675 | 0.626 | 0.749 | 0.281 | 0.316 |
| LZ_WEST | 0.799 ± 0.001 | 0.565 | 0.522 | 0.629 | 0.127 | 0.319 |
 
**Outputs:** `results/xgboost_zones/global_model_auc_by_zone.png`, `global_vs_zone_model_auc.png`, `zone_feature_importance_heatmap.png`, `zone_results.csv`

### 8 — Grid Cell Performance and Threshold Optimisation
 
**Script:** `files/08_grid_cell_performance_and_threshold.py`
 
Two analyses that bridge the global XGBoost model and the financial simulation.
 
```bash
python files/08_grid_cell_performance_and_threshold.py
```
 
#### Part 1 — Per-grid-cell model performance
 
The fitted global XGB-2 model is evaluated on every individual grid cell in the 2020–2021 test set.  AUC-ROC, precision, and recall are computed per cell, with AUC broken down by year (2020 vs 2021) to isolate the distributional shift effect documented at the zone level.
 
This analysis motivates the selection of grid cell 6_23 (LZ_WEST, 150 MW installed capacity) for the financial simulation.  Despite LZ_WEST having the weakest average cell AUC (mean 0.565), cell 6_23 achieves above-zone-average performance — overall AUC 0.616, 2021 AUC 0.70 — while sitting in the zone with the largest aggregate physical exposure (27,768 MW, 83 grid cells).  The ten best-predicted cells are concentrated in LZ_SOUTH (overall AUC 0.641–0.651, 2021 AUC consistently above 0.70); the ten hardest-to-predict cells are predominantly in LZ_WEST (overall AUC 0.538–0.573).
 
The 2020→2021 AUC improvement is geographically widespread across nearly all grid cells, with the largest gains in LZ_WEST and LZ_NORTH — consistent with the zone-level finding that COVID-era conditions in 2020 suppressed the compound gas-wind-temperature stress signal on which the model was trained.
 
**Outputs:** `results/grid_cell/grid_cell_auc_map.png` (spatial AUC + shift map, Figure 8 from paper), `grid_cell_auc_by_zone.png` (AUC distributions + 2020 vs 2021 scatter), `grid_cell_performance.csv`
 
#### Part 2 — Classification threshold optimisation for cell 6_23
 
Converts the model's continuous probability outputs into binary HIGH / LOW flags by scanning 100 candidate thresholds from 0.25 to 0.74 and selecting the F1-optimal threshold subject to a minimum recall constraint.  MCC is computed but not used as the primary optimisation criterion — it consistently selected thresholds above 0.70 that achieved only 7% recall, providing negligible practical coverage.
 
Two thresholds are derived and compared:
 
| Threshold | Recall floor | Threshold value | F1 | Hours flagged |
|-----------|-------------|-----------------|-----|---------------|
| Conservative | ≥ 55% | 0.3539 | — | 7,215 (41.1%) |
| Aggressive | ≥ 75% | lower | — | 11,212 (63.9%) |
 
The conservative threshold is selected as the primary operating threshold.  The aggressive threshold is economically justified by an economic recall scan (see below) but is operationally impractical: holding active gas futures positions for nearly two-thirds of all operating hours generates substantial hedge premiums even in months with no meaningful price exposure.
 
**Economic recall scan.** For each recall level from 10%–80%, the script finds the F1-optimal threshold satisfying that recall floor and computes net hedge benefit (total actual replacement costs − net costs after hedge) across all four NYMEX Henry Hub futures contracts (C1–C4).  Net benefit rises monotonically to 75% recall, driven primarily by Winter Storm Uri (88% of total replacement costs in February 2021).  This motivates the aggressive threshold as a tail-risk scenario while the conservative threshold remains the default.
 
**Per-year threshold calibration.** An overall threshold (0.3539) is retained rather than recalibrating year-by-year: the year-specific approach improves 2020 performance (lower optimal threshold reflecting the suppressed gas price environment) but uses a more conservative 2021 threshold (0.4084), reducing recall during the Uri period and leaving more extreme loss hours unhedged.
 
**Outputs:** `results/grid_cell/cell_6_23_threshold_scan.png` (Figure 15 from paper), `cell_6_23_recall_scan.png`, `cell_6_23_threshold_summary.csv`, `cell_6_23_recall_scan.csv`

## Citation

Hersbach, H. et al. (2023). ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store. <https://doi.org/10.24381/cds.adbb2d47>
