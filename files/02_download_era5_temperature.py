"""
02_download_era5_temperature.py
================================
Downloads hourly 2-metre air temperature from the Copernicus Climate Data
Store (CDS) for the ERCOT / Texas domain, then extracts the time series for
every ERA5 grid cell that contains installed ERCOT wind capacity, converts
units from Kelvin to Fahrenheit, and writes one CSV per grid cell.

Pipeline
--------
Step 1 — Download
    One GRIB file per year → raw_temp/{year}_tmm.grib
    Domain  : 26°–36°N, 94°–107°W
    Period  : 1980–2025 (configurable via YEARS below)

Step 2 — Extract to wind grid cells
    Reads the grid-cell reference file produced by the capacity assignment
    step (data/wind_grid_cells.csv) to obtain the list of (lat_idx, lon_idx,
    grid_latitude, grid_longitude, Load_Zone) tuples that fall inside ERCOT
    and carry installed wind capacity.

    For each GRIB file and each qualifying grid cell the nearest-neighbour
    ERA5 value is extracted, converted to °F, and appended to a running list.
    After all years are processed, each cell's data are concatenated, sorted,
    de-duplicated, and written to:

        tmm_hourly_wind_cells/{lat_idx}_{lon_idx}_tmm.csv

    Columns: timestamp, Load_Zone, tmm_K, tmm_F, grid_latitude, grid_longitude

Requirements
------------
    pip install cdsapi xarray cfgrib eccodes pandas numpy

CDS API credentials
-------------------
Configure ~/.cdsapirc as described in 01_download_era5_wind.py.

Usage
-----
    # Download GRIB files (Step 1 only)
    python 02_download_era5_temperature.py --step download

    # Extract to per-cell CSVs (Step 2 only, GRIBs already present)
    python 02_download_era5_temperature.py --step extract

    # Run both steps end-to-end (default)
    python 02_download_era5_temperature.py

Reference
---------
Hersbach, H. et al. (2023). ERA5 hourly data on single levels from 1940
to present. Copernicus Climate Change Service (C3S) Climate Data Store.
https://doi.org/10.24381/cds.adbb2d47
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# =============================================================================
# CONFIGURATION
# =============================================================================

YEARS        = list(range(1980, 2026))          # 1980–2025 inclusive
GRIB_DIR     = Path("raw_temp")                 # raw GRIB downloads
OUT_DIR      = Path("tmm_hourly_wind_cells")    # per-cell output CSVs
GRID_FILE    = Path("data/wind_grid_cells.csv") # grid-cell reference table
BASE_TEMP_F  = 65.0                             # comfort baseline for HDD/CDD

DATASET  = "reanalysis-era5-single-levels"
VARIABLE = "2m_temperature"
AREA     = [36, -107, 26, -94]                  # [N, W, S, E]

# =============================================================================
# STEP 1 — DOWNLOAD
# =============================================================================

def build_temp_request(year: int) -> dict:
    return {
        "product_type"   : ["reanalysis"],
        "variable"       : [VARIABLE],
        "year"           : [str(year)],
        "month"          : [f"{m:02d}" for m in range(1, 13)],
        "day"            : [f"{d:02d}" for d in range(1, 32)],
        "time"           : [f"{h:02d}:00" for h in range(24)],
        "data_format"    : "grib",
        "download_format": "unarchived",
        "area"           : AREA,
    }


def download_temperature(years=YEARS, grib_dir=GRIB_DIR):
    """Download annual ERA5 2 m temperature GRIB files from CDS."""
    import cdsapi

    grib_dir.mkdir(parents=True, exist_ok=True)

    print("ERA5 Temperature Download  —  2 m air temperature")
    print("=" * 52)
    print(f"Domain  : {AREA}  [N, W, S, E]")
    print(f"Years   : {years[0]}–{years[-1]}  ({len(years)} years)")
    print(f"Output  : {grib_dir}/")
    print(f"Started : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 52)

    try:
        client = cdsapi.Client()
    except Exception as exc:
        print(f"\n[ERROR] Cannot initialise CDS client: {exc}")
        sys.exit(1)

    ok = failed = skipped = 0
    failed_years = []

    for i, year in enumerate(years, 1):
        out_path = grib_dir / f"{year}_tmm.grib"
        print(f"\n[{i}/{len(years)}] {year}")

        if out_path.exists():
            print(f"  [SKIP] already downloaded: {out_path}")
            skipped += 1
            continue

        print(f"  [REQUEST] submitting to CDS API ...")
        sys.stdout.flush()

        try:
            t0     = time.time()
            result = client.retrieve(DATASET, build_temp_request(year))
            result.download(str(out_path))
            elapsed = (time.time() - t0) / 60
            size_gb = out_path.stat().st_size / (1024 ** 3)
            print(f"  [OK] {size_gb:.2f} GB, {elapsed:.1f} min")
            ok += 1
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            if out_path.exists():
                out_path.unlink()
            failed += 1
            failed_years.append(year)

    print("\n" + "=" * 52)
    print(f"Downloaded: {ok}  |  Skipped: {skipped}  |  Failed: {failed}")
    if failed_years:
        print(f"Failed years: {failed_years}  — re-run to retry.")
    print("=" * 52)


# =============================================================================
# STEP 2 — EXTRACT TO WIND GRID CELLS
# =============================================================================

def _normalize_lon(target_lons: np.ndarray, ds_lon: np.ndarray) -> np.ndarray:
    """Convert target longitudes to 0–360 if the dataset uses 0–360."""
    targ = np.asarray(target_lons, dtype=float)
    if np.nanmax(ds_lon) > 180 and np.nanmin(targ) < 0:
        targ = targ % 360.0
    return targ


def extract_to_cells(grib_dir=GRIB_DIR, out_dir=OUT_DIR, grid_file=GRID_FILE):
    """
    Extract hourly 2 m temperature for every qualifying ERCOT wind grid cell
    and write per-cell CSVs (timestamp, Load_Zone, tmm_K, tmm_F,
    grid_latitude, grid_longitude).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load wind grid-cell reference table
    # Expected columns: lat_idx, lon_idx, grid_latitude, grid_longitude,
    #                   Load_Zone
    # Produced by the capacity-assignment notebook (see data/ directory).
    # ------------------------------------------------------------------
    if not grid_file.exists():
        raise FileNotFoundError(
            f"Grid-cell reference file not found: {grid_file}\n"
            "Run the capacity-assignment step first to generate this file."
        )

    wind = pd.read_csv(grid_file)
    wind.columns = wind.columns.str.strip()

    required = {"lat_idx", "lon_idx", "grid_latitude", "grid_longitude"}
    missing  = required - set(wind.columns)
    if missing:
        raise ValueError(f"Missing columns in {grid_file}: {missing}")

    # Standardise Load Zone column name
    lz_col = next(
        (c for c in ["Load_Zone", "Load Zone", "load_zone", "LOAD_ZONE", "LZ"]
         if c in wind.columns),
        None
    )
    wind["Load_Zone"] = wind[lz_col].astype(str) if lz_col else "UNKNOWN"

    for col in ("lat_idx", "lon_idx"):
        wind[col] = pd.to_numeric(wind[col], errors="coerce").astype("Int64")
    for col in ("grid_latitude", "grid_longitude"):
        wind[col] = pd.to_numeric(wind[col], errors="coerce")

    wind = (
        wind
        .dropna(subset=list(required))
        .drop_duplicates(subset=["grid_latitude", "grid_longitude"])
        .reset_index(drop=True)
    )

    grib_files = sorted(grib_dir.glob("*.grib"))
    if not grib_files:
        raise FileNotFoundError(f"No GRIB files found in {grib_dir}")

    print(f"\nExtraction")
    print(f"  GRIB files     : {len(grib_files)}")
    print(f"  Wind grid cells: {len(wind)}")
    print(f"  Output dir     : {out_dir}/\n")

    target_lats     = wind["grid_latitude"].to_numpy(dtype=float)
    target_lons_raw = wind["grid_longitude"].to_numpy(dtype=float)

    # Accumulate per-cell frames across all GRIB files
    cell_frames: dict[tuple, list] = {
        (int(r.lat_idx), int(r.lon_idx)): []
        for r in wind.itertuples(index=False)
    }

    for gf in grib_files:
        print(f"  Reading {gf.name} ...")
        ds = xr.open_dataset(gf, engine="cfgrib")

        var_name  = "t2m" if "t2m" in ds.data_vars else list(ds.data_vars)[0]
        time_name = (
            "valid_time" if "valid_time" in ds.coords
            else "time"  if "time"       in ds.coords
            else None
        )
        if time_name is None:
            raise ValueError(f"No time coordinate in {gf.name}")

        target_lons = _normalize_lon(target_lons_raw, ds["longitude"].values)

        # Vectorised nearest-neighbour selection across all cells at once
        da = ds[var_name].sel(
            latitude  = xr.DataArray(target_lats, dims="points"),
            longitude = xr.DataArray(target_lons, dims="points"),
            method    = "nearest",
        )

        df = da.to_dataframe(name="tmm_K").reset_index()
        df = df[[time_name, "points", "tmm_K"]].rename(
            columns={time_name: "timestamp"}
        )
        df["tmm_F"] = (df["tmm_K"] - 273.15) * 9 / 5 + 32

        meta = (
            wind.reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "points"})
            [["points", "lat_idx", "lon_idx", "Load_Zone",
              "grid_latitude", "grid_longitude"]]
        )
        df = df.merge(meta, on="points", how="left").drop(columns=["points"])

        for (li, lo), g in df.groupby(["lat_idx", "lon_idx"], sort=False):
            cell_frames[(int(li), int(lo))].append(
                g[["timestamp", "Load_Zone", "tmm_K", "tmm_F",
                   "grid_latitude", "grid_longitude"]].copy()
            )

        ds.close()

    # ------------------------------------------------------------------
    # Write one CSV per cell
    # ------------------------------------------------------------------
    print("\nWriting per-cell CSVs ...")
    written = 0
    for (li, lo), frames in cell_frames.items():
        if not frames:
            continue
        out = pd.concat(frames, ignore_index=True)
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = (
            out
            .dropna(subset=["timestamp", "tmm_K", "tmm_F"])
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="first")
        )
        out.to_csv(out_dir / f"{li}_{lo}_tmm.csv", index=False)
        written += 1

    print(f"Done. Wrote {written} CSV files to {out_dir}/")


# =============================================================================
# STEP 3 — COMPUTE HDD / CDD AND EXTREME FLAGS
# =============================================================================

def compute_degree_hours(out_dir=OUT_DIR, base_temp_f=BASE_TEMP_F):
    """
    For each per-cell temperature CSV, compute:
      - HDD_hourly / CDD_hourly  (hourly deviations from 65 °F base)
      - HDD_daily  / CDD_daily   (same-day totals broadcast to hourly rows)
      - Year-specific 90th-percentile extreme flags for HDD and CDD at
        both hourly and daily resolution
      - Combined extreme_hourly / extreme_daily flags (1 if either HDD or
        CDD threshold is exceeded)

    Files are updated in-place (overwritten with new columns).
    """
    files = sorted(out_dir.glob("*_tmm.csv"))
    print(f"\nComputing HDD/CDD for {len(files)} files ...")

    for fp in files:
        try:
            df = pd.read_csv(fp)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["tmm_F"]     = pd.to_numeric(df["tmm_F"], errors="coerce")
            df = df.dropna(subset=["timestamp", "tmm_F"]).copy()

            df["date"] = df["timestamp"].dt.normalize()

            # --- hourly degree hours ---
            df["HDD_hourly"] = (base_temp_f - df["tmm_F"]).clip(lower=0)
            df["CDD_hourly"] = (df["tmm_F"] - base_temp_f).clip(lower=0)

            # --- daily totals (broadcast back to hourly rows) ---
            lz_col = "Load_Zone" if "Load_Zone" in df.columns else None
            group_cols = ["date"] + ([lz_col] if lz_col else [])

            daily = (
                df.groupby(group_cols, as_index=False)
                  .agg(HDD_daily=("HDD_hourly", "sum"),
                       CDD_daily=("CDD_hourly", "sum"))
            )
            df = df.merge(daily, on=group_cols, how="left")

            # --- year-specific 90th-percentile extreme flags ---
            df["year"] = df["timestamp"].dt.year
            thresh = (
                df.groupby("year", as_index=False)
                  .agg(
                      hh90=("HDD_hourly", lambda x: x.quantile(0.90)),
                      ch90=("CDD_hourly", lambda x: x.quantile(0.90)),
                      hd90=("HDD_daily",  lambda x: x.quantile(0.90)),
                      cd90=("CDD_daily",  lambda x: x.quantile(0.90)),
                  )
            )
            df = df.merge(thresh, on="year", how="left")

            df["extreme_hdd_hourly"] = (df["HDD_hourly"] > df["hh90"]).astype(int)
            df["extreme_cdd_hourly"] = (df["CDD_hourly"] > df["ch90"]).astype(int)
            df["extreme_hdd_daily"]  = (df["HDD_daily"]  > df["hd90"]).astype(int)
            df["extreme_cdd_daily"]  = (df["CDD_daily"]  > df["cd90"]).astype(int)

            df["extreme_hourly"] = (
                (df["extreme_hdd_hourly"] == 1) |
                (df["extreme_cdd_hourly"] == 1)
            ).astype(int)
            df["extreme_daily"] = (
                (df["extreme_hdd_daily"] == 1) |
                (df["extreme_cdd_daily"] == 1)
            ).astype(int)

            df = df.drop(
                columns=["year", "date", "hh90", "ch90", "hd90", "cd90"],
                errors="ignore"
            )
            df.to_csv(fp, index=False)

        except Exception as exc:
            print(f"  [WARN] {fp.name}: {exc}")

    print("Done.")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--step",
        choices=["download", "extract", "degrees", "all"],
        default="all",
        help=(
            "Which step(s) to run.  "
            "'download' fetches GRIBs from CDS.  "
            "'extract' converts GRIBs to per-cell CSVs.  "
            "'degrees' adds HDD/CDD and extreme flags.  "
            "'all' runs all three in sequence (default)."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.step in ("download", "all"):
        download_temperature()

    if args.step in ("extract", "all"):
        extract_to_cells()

    if args.step in ("degrees", "all"):
        compute_degree_hours()


if __name__ == "__main__":
    main()
