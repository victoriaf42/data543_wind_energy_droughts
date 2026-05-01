# data543_wind_energy_droughts
ERCOT Wind PPA Risk — XGBoost framework predicting high electricity price exposure during wind energy droughts. Models compound risk from low wind output, elevated gas prices, and temperature stress to generate hedging signals for fixed-volume PPA producers. Includes ERA5 reanalysis, futures hedge simulation, and grid-cell P&L analysis.

---

## Repository Structure

```
files/
├── 01_download_era5_wind.py
├── 02_download_era5_temperature.py
├── 03_process_electricity_prices.py
└── 04_classify_droughts.py
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

Daily values are forward-filled across weekends and holidays (up to five consecutive non-trading days) and broadcast to the hourly level. Futures prices are converted from \$/MMBtu to \$/MWh using a heat rate of 8.978 MMBtu/MWh (38% combined-cycle gas turbine efficiency).

---

## Citation

Hersbach, H. et al. (2023). ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store. <https://doi.org/10.24381/cds.adbb2d47>
