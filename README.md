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
├── 07_xgboost_zone_models.py
├── 08_grid_cell_performance_and_threshold.py
├── 09_natural_gas_futures.py
├── 10_hazard_analysis.py
├── 11_vulnerability_analysis.py
└── 12_financial_simulation.py
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
 
#### Feature importance by zone (Figure 7 from paper)
 
Feature importance rankings differ meaningfully across zones, consistent with the distinct price formation mechanisms documented in the paper.  The script prints both the top and 2nd-ranked feature per zone, and saves a heatmap (`zone_feature_importance_heatmap.png`) showing the full importance matrix.
 
| Zone | Top feature | Importance | 2nd feature | Interpretation |
|------|-------------|------------|-------------|----------------|
| LZ_SOUTH | `gas_x_low_wind_x_demand` | 0.500 | `gas_x_demand` | Transmission bottlenecks mean local wind shortfalls translate directly into local price stress — the most concentrated compound signal of any zone |
| LZ_NORTH | `gas_x_low_wind_x_demand` | 0.386 | `gas_price_mmbtu` | Compound instantaneous shortfalls dominate; gas price carries independent influence given a 45% natural gas generation mix |
| LZ_HOUSTON | `gas_x_demand` | 0.310 | `gas_x_low_wind_x_demand` | Effectively no local wind (< 0.5% of capacity); price driven by cost of servicing load from imported generation under demand stress |
| LZ_WEST | `drought_run_hours` | 0.203 | `gas_x_low_wind_x_demand` | Duration rather than instantaneous CF predicts price stress — limited local gas backup means prices spike only after reserves are progressively exhausted over multi-day droughts |
 
LZ_WEST is the hardest zone to predict (test AUC 0.565) despite holding 69% of installed wind capacity.  The compound three-way interaction that dominates other zones carries less discriminatory power in LZ_WEST because gas price stress and temperature-driven demand are comparatively weak local price formation mechanisms — wind is dominant but highly volatile, and the model struggles where two of its three signal components are largely absent.
 
#### Results (Table 7 from paper)
 
| Zone | CV AUC | Test AUC | Test AUC 2020 | Test AUC 2021 | Test Precision | Test Recall |
|------|--------|----------|---------------|---------------|----------------|-------------|
| LZ_HOUSTON | 0.884 ± 0.008 | 0.626 | 0.589 | 0.682 | 0.204 | 0.284 |
| LZ_NORTH | 0.898 ± 0.001 | 0.644 | 0.593 | 0.724 | 0.262 | 0.232 |
| LZ_SOUTH | 0.882 ± 0.002 | 0.675 | 0.626 | 0.749 | 0.281 | 0.316 |
| LZ_WEST | 0.799 ± 0.001 | 0.565 | 0.522 | 0.629 | 0.127 | 0.319 |
 
**Outputs:** `results/xgboost_zones/global_model_auc_by_zone.png` (Figure 6 from paper), `global_vs_zone_model_auc.png`, `zone_feature_importance_heatmap.png` (Figure 7), `zone_results.csv`

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

### 9 — Natural Gas Futures Data
 
**Script:** `files/09_natural_gas_futures.py`
 
Downloads NYMEX Henry Hub natural gas futures prices from the EIA API and prepares the hedging instrument data used in the financial simulation.  This step feeds directly into the financial simulation script.
 
```bash
# Set your EIA API key as an environment variable (recommended)
export EIA_API_KEY=your_key_here
python files/09_natural_gas_futures.py
 
# Or pass the key directly
python files/09_natural_gas_futures.py --api-key your_key_here
 
# Re-process an existing download without hitting the API again
python files/09_natural_gas_futures.py --no-download
```
 
A free EIA API key is required. Register at <https://www.eia.gov/opendata/register.php>. Never hard-code credentials in the script.
 
#### What is downloaded
 
Four nearby NYMEX Henry Hub delivery contracts (daily closing prices, October 2019–December 2024):
 
| Series ID | Column | Description |
|-----------|--------|-------------|
| RNGC1 | `NG_C1` | Prompt month |
| RNGC2 | `NG_C2` | One-to-two months ahead |
| RNGC3 | `NG_C3` | Two-to-three months ahead |
| RNGC4 | `NG_C4` | Three-to-four months ahead |
 
Two spread variables are derived: `spread_C2_C1` = C2 − C1 and `spread_C3_C1` = C3 − C1. These represent the shape of the near-term futures curve — positive values indicate contango (market expects higher future prices), negative values indicate backwardation — and were included as XGBoost model features.
 
The start date of October 2019 ensures C1 covers the January 2020 test period start with no gaps.
 
#### Forward-filling
 
Futures markets do not trade on weekends or holidays, leaving gaps in the daily calendar. The script builds a continuous daily calendar and forward-fills across up to five consecutive non-trading days (sufficient to cover Christmas/New Year stretches). A single back-fill step handles any gap at the very start of the calendar.
 
#### Heat rate conversion
 
Futures prices in `$/MMBtu` are converted to electricity-equivalent `$/MWh` for direct comparison with electricity prices using the heat rate derived from a 38% CCGT efficiency assumption — the marginal generation technology most likely to set real-time prices during high-demand or low-wind periods in ERCOT:
 
```
Heat rate = 1 / (0.293071 MWh/MMBtu × 0.381) = 8.978 MMBtu/MWh
 
Hedge quantity per flagged hour = 30 MWh × 8.978 = 269.34 MMBtu
Hedge P&L = (NG spot − NG futures) × 269.34 MMBtu
```
 
Source: [Woodway Energy — Natural Gas Efficiency in Power Generation](https://www.woodwayenergy.com/natural-gas-efficiency-in-power-generation/)
 
**Outputs:** `data/ng_futures/henry_hub_futures_C1_C4_2019_2024.csv` (raw), `data/ng_futures/henry_hub_futures_filled.csv` (forward-filled, simulation-ready)

### 10 — Hazard Analysis
 
**Script:** `files/10_hazard_analysis.py`
 
Produces the two hazard figures that open the Results section of the paper, using the ERA5 reanalysis-based drought event catalogue covering **1980–2025**.
 
```bash
python files/10_hazard_analysis.py
```
 
The script reads two pre-built drought event catalogues from `data/drought_events/`. Both apply the same event definition — zone CF < threshold while installed capacity under production ≤ 50% of the zone total — at different CF thresholds appropriate to each figure's purpose.
 
#### Input files
 
| File | CF threshold | Used for |
|------|-------------|----------|
| `ALL_ZONES_events_1980_2025_CF0.30_cap50pct.csv` | 0.30 | Figure 4 — seasonality and severity |
| `ALL_ZONES_events_1980_2025_CF0.15_cap50pct.csv` | 0.15 | Figure 5 — exceedance probability surfaces |
 
Expected columns: `start_time`, `load_zone`, `duration` (hours), `avg_zone_cf` (0–1).
 
#### Figure 4 — Seasonality of Low-Wind Drought Events (LZ_WEST)
 
Two-panel figure using the CF0.30 catalogue filtered to LZ_WEST.
 
- **Top panel** — count of drought events by meteorological season. Summer produces the highest raw event count (n=3,148), followed by Spring (2,842), Winter (2,684), and Fall (2,614).
- **Bottom panel** — mean and 95th-percentile composite severity score by season. While mean severity scores are broadly similar across seasons, the 95th-percentile scores diverge sharply: Summer droughts reach a severity score of approximately 3.20 at the 95th percentile, the highest of any season, with Fall close behind at ~2.75. This divergence between mean and tail severity is financially significant — it is the worst events, not the average ones, that drive loss exposure under a physical PPA.
Severity score = `duration × max(0.15 − avg_zone_cf, 0)`, weighting events by both duration and depth below the severe CF threshold.
 
#### Figure 5 — Seasonal Low-Wind Event Probabilities (Load Zone West)
 
2×2 grid of heat maps using the CF0.15 catalogue filtered to LZ_WEST. Each panel shows P(≥1 drought event per season-year) across a grid of duration thresholds (x-axis, hours, ≥) and average zone CF thresholds (y-axis, ≤).
 
Probabilities are computed via the Poisson approximation: P = 1 − exp(−λ), where λ = event count / n_season_years.
 
At lenient thresholds all four seasons show near-certain exceedance, confirming that short, moderate wind drought events are a routine structural feature of the West Texas wind resource. The key differentiation emerges as duration thresholds increase: Summer and Fall sustain meaningful exceedance probabilities well past 150 hours, with Summer recording the longest observed event beyond 500 hours. Winter and Spring decay much more rapidly — Spring shows no observed events at the highest duration-severity combinations. The joint probability of long duration and deep CF suppression (avg zone CF < 0.10) remains non-trivial in Summer and Fall up to ~50–75 hours, representing the most financially dangerous drought configuration for PPA producers.
 
**Outputs:** `results/hazard/figure4_seasonality_lz_west.png`, `results/hazard/figure5_exceedance_probabilities_lz_west.png`

### 11 — Vulnerability Analysis
 
**Script:** `files/11_vulnerability_analysis.py`
 
Assesses the financial sensitivity of exposed wind generation assets to real-time electricity price fluctuations during wind drought periods, decomposing vulnerability across three compounding dimensions: wind drought severity, natural gas price, and temperature-driven demand.
 
```bash
python files/11_vulnerability_analysis.py
```
 
#### Stress threshold methodology
 
Three physical stress thresholds are derived via a two-step process consistent with the paper's methodology section.
 
**Step 1 — Marginal thresholds.** Each variable is binned into ten equal-width intervals. The first bin where the HIGH-exposure rate exceeds 1.25× the 10% baseline is identified as the marginal stress threshold for that variable.
 
**Step 2 — Weighted-average thresholds (primary approach).** All combinations of the three variables binned into five intervals each are evaluated jointly. The 83 of 125 combinations (81%) that exceed 1.25× baseline are retained, and hour-weighted averages across these joint stress combinations produce the final thresholds:
 
| Variable | Threshold | Role |
|----------|-----------|------|
| `wind_cf` | < 0.13 | Contractual shortfall zone — producer must purchase replacement power |
| `gas_price_mmbtu` | ≥ $4.15/MMBtu | Primary price amplifier — sets cost floor for backup dispatch |
| `temp_stress` (= `|tmm_F − 65|`) | ≥ 25.8°F | Demand amplifier — tightens gas supply and reinforces price pressure |
 
All three values fall below their respective marginal thresholds, confirming that compounding amplifies price exposure even when no single variable reaches its standalone extreme.
 
#### Figure 9 — Marginal stress threshold discovery
 
Three-panel bar chart showing the HIGH-exposure rate per decile bin for each variable. Red bars exceed the 1.25× lift cutoff. Dashed lines mark the weighted-average thresholds applied throughout the analysis.
 
The marginal wind CF threshold below 0.10 corresponds to a HIGH-exposure rate of 14.1% (1.41× baseline). The gas price threshold captures only 1.8% of test hours and is almost entirely concentrated in the February 2021 Winter Storm Uri event, making it unsuitable as a general structural vulnerability indicator on its own.
 
#### Figure 10 — Conditional HIGH-exposure rates
 
Horizontal bar chart of the conditional HIGH-exposure rate for each combination of the three stress flags. Key finding: when all three conditions coincide, the HIGH-exposure rate reaches **45.7% (4.57× baseline)** — meaning 8,573 of the 18,750 compound stress cell-hours in the test period were actual HIGH-exposure hours. Gas price and temperature stress together (even without wind drought) produce a HIGH rate of 56.1%, the strongest two-variable combination.
 
#### Figure 11 — Model recall by stress condition
 
Demonstrates that the model achieves near-complete recall precisely during compound stress events — the conditions of greatest financial risk. Wind drought alone carries a HIGH-exposure rate of only 14.1%, reinforcing that contractual shortfall is primarily a delivery risk rather than a price risk unless compounded by gas and temperature stress.
 
#### Figure 12 — Seasonal vulnerability profile
 
Four-metric grouped bar chart by season. **Fall emerges as the most financially dangerous period**, with a HIGH-exposure rate of 20.3% during drought hours — nearly double Summer's 10.5%. Despite Summer recording the highest raw drought rate (34.4% of hours), the combination of post-summer gas prices and lingering temperature stress into the cooling season makes Fall the period during which contractual shortfall and price risk most frequently coincide.
 
#### Three-approach comparison
 
| Metric | Percentile (P75) | Weighted Avg | Marginal Lift |
|--------|-----------------|--------------|---------------|
| Gas threshold | yr-specific | ≥ $4.15/MMBtu | ≥ $5.84/MMBtu |
| Triple-stress hours | 54,122 | **18,750** | 5,364 |
| Triple HIGH rate | 27.5% | **45.7%** | ~99% |
| Triple lift vs baseline | 2.75× | **4.57×** | ~10× |
| Model recall on triple | 96.5% | **100%** | 100% |
| Recurring structural signal | Yes | **Yes** | No |
| Recommended use | Robustness | **Primary** | Tail-risk only |
 
**Outputs:** `results/vulnerability/figure9_marginal_stress_thresholds.png`, `figure10_conditional_high_exposure.png`, `figure11_model_recall_by_condition.png`, `figure12_seasonal_vulnerability.png`, `three_approach_comparison.csv`

### 12 — Financial Risk Simulation
 
**Script:** `files/12_financial_simulation.py`
 
Simulates the hourly financial position of a wind energy producer at grid cell 6_23 (LZ_WEST, 100 MW) operating under a fixed-volume physical PPA, and evaluates how well the XGB-2 model's HIGH-exposure flags align with actual financial loss events across the 2020–2021 test period.
 
```bash
python files/12_financial_simulation.py
```
 
#### PPA structure and financial logic
 
| Parameter | Value |
|-----------|-------|
| Nameplate capacity | 100 MW |
| Fixed PPA price | $50 / MWh |
| Delivery obligation | 30 MWh / hr (30% CF floor) |
| Baseline position | $1,500 / hr (no wind variability) |
 
Every hour the simulation computes:
 
- **Shortfall hours** (wind_mwh < 30): producer must buy replacement energy on the spot market. Loss = `max(0, spot − $50) × shortfall_mwh`. No gain if spot < $50 (no-arbitrage constraint).
- **Surplus hours** (wind_mwh > 30): producer sells excess generation at spot. Revenue = `max(0, spot − $50) × excess_mwh`.
- **Net position** = PPA revenue − replacement cost + surplus revenue.
#### Model flagging and loss capture
 
The simulation evaluates the conservative (overall) threshold (0.3539) against the year-specific alternative. The primary metric is the **loss capture rate** — the share of total replacement costs that occurred in model-flagged hours.
 
| Outcome | Definition | Financial interpretation |
|---------|-----------|--------------------------|
| True positive | Flagged AND spot > $50 | Loss correctly anticipated — hedgeable |
| False positive | Flagged AND spot ≤ $50 | Unnecessary flag — small hedge cost |
| False negative | Not flagged AND spot > $50 | Loss missed — unhedged exposure |
| True negative | Not flagged AND spot ≤ $50 | Correctly ignored |
 
#### Results (Table 9 from paper)
 
Over the 2020–2021 test period the model flags 7,215 hours (41.1%), capturing **97% of total replacement costs** ($19.2M of $19.7M) at the conservative threshold. The loss capture rate is dominated by Winter Storm Uri (February 10–20, 2021), which alone accounts for 88.1% of total replacement costs ($17.4M). Uri loss hours are physically distinctive — mean wind CF of 0.093, gas prices at 2.54× normal levels, temperature stress at 3.4× average — producing precisely the compound stress signal the model detects most reliably (Uri AUC = 0.809 vs non-Uri AUC = 0.576). Outside of Uri, the model captures 77.8% of the remaining $2.3M in non-Uri losses.
 
#### Why 97% loss capture despite AUC = 0.616
 
Loss concentration is extreme: a single 10-day event accounts for 88% of total replacement costs. AUC measures the model's ability to rank all hours by risk correctly — including the many moderate-exposure hours in the suppressed 2020 environment where the gas-electricity relationship was disrupted. These hours contribute little to financial losses but substantially depress overall AUC. The contrast between Uri AUC (0.809) and non-Uri AUC (0.576) confirms the model operates in two distinct regimes.
 
#### Figures
 
**Figure 13** — full 2020–2021 period: monthly replacement costs split by model outcome (blue = loss in flagged hours, red = loss in non-flagged hours), surplus revenue, and monthly loss capture rate.
 
**Figure 14** — non-Uri period only: the same panels excluding February 2021, allowing the structure of non-extreme months to be visible without Uri's scale distortion.
 
**Outputs:** `results/financial/figure13_financial_risk_full_period.png`, `figure14_financial_risk_non_uri.png`, `cell_6_23_financial_sim_overall_hourly.csv`, `cell_6_23_financial_sim_overall_monthly.csv`, `cell_6_23_financial_sim_yearspec_hourly.csv`, `cell_6_23_financial_sim_yearspec_monthly.csv`

## Citation

Hersbach, H. et al. (2023). ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store. <https://doi.org/10.24381/cds.adbb2d47>
