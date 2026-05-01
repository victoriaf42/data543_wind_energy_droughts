"""
01_download_era5_wind.py
========================
Downloads hourly 100 m u- and v-components of wind velocity from the
Copernicus Climate Data Store (CDS) for the ERCOT / Texas domain.

    Dataset : ERA5 hourly reanalysis on single levels
    Domain  : 26°–36°N, 94°–107°W  (Texas / ERCOT service territory)
    Period  : 1980–2025  (configurable via YEARS below)
    Output  : One GRIB file per year → era5_wind_data/era5_wind_{year}.grib

Requirements
------------
    pip install cdsapi

CDS API credentials
-------------------
1. Register at https://cds.climate.copernicus.eu
2. Accept the ERA5 terms of use
3. Create ~/.cdsapirc containing:

       url: https://cds.climate.copernicus.eu/api
       key: <YOUR_API_KEY>

   Your key is on your CDS profile page.
   Do NOT hard-code credentials in this script.

Usage
-----
    python 01_download_era5_wind.py

Skips years whose output file already exists, so the script is safe to
re-run after interruptions.

Reference
---------
Hersbach, H. et al. (2023). ERA5 hourly data on single levels from 1940
to present. Copernicus Climate Change Service (C3S) Climate Data Store.
https://doi.org/10.24381/cds.adbb2d47
"""

import os
import sys
import time
from datetime import datetime

import cdsapi

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = "era5_wind_data"          # directory for downloaded GRIB files
YEARS      = list(range(1980, 2026))   # 1980–2025 inclusive

DATASET   = "reanalysis-era5-single-levels"
VARIABLES = ["100m_u_component_of_wind", "100m_v_component_of_wind"]
AREA      = [36, -107, 26, -94]        # [north, west, south, east] in degrees

# =============================================================================
# HELPERS
# =============================================================================

def build_request(year: int) -> dict:
    """Return the CDS API request dictionary for a single year."""
    return {
        "product_type"   : ["reanalysis"],
        "variable"       : VARIABLES,
        "year"           : [str(year)],
        "month"          : [f"{m:02d}" for m in range(1, 13)],
        "day"            : [f"{d:02d}" for d in range(1, 32)],
        "time"           : [f"{h:02d}:00" for h in range(24)],
        "data_format"    : "grib",
        "download_format": "unarchived",
        "area"           : AREA,
    }


def download_year(client: cdsapi.Client, year: int) -> bool:
    """
    Download ERA5 wind data for *year*.

    Returns True on success, False on failure.
    Partial downloads are removed automatically.
    """
    out_path = os.path.join(OUTPUT_DIR, f"era5_wind_{year}.grib")

    if os.path.exists(out_path):
        print(f"  [SKIP] {year} — already downloaded: {out_path}")
        return True

    print(f"  [REQUEST] {year} — submitting to CDS API ...")
    sys.stdout.flush()

    try:
        t0     = time.time()
        result = client.retrieve(DATASET, build_request(year))
        result.download(out_path)
        elapsed = (time.time() - t0) / 60
        size_gb = os.path.getsize(out_path) / (1024 ** 3)
        print(f"  [OK] {year} — {size_gb:.2f} GB, {elapsed:.1f} min")
        return True

    except Exception as exc:
        print(f"  [FAIL] {year} — {exc}")
        if os.path.exists(out_path):
            os.remove(out_path)
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("ERA5 Wind Download  —  100 m u/v components")
    print("=" * 52)
    print(f"Domain  : {AREA}  [N, W, S, E]")
    print(f"Years   : {YEARS[0]}–{YEARS[-1]}  ({len(YEARS)} years)")
    print(f"Output  : {OUTPUT_DIR}/")
    print(f"Started : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 52)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        client = cdsapi.Client()
    except Exception as exc:
        print(f"\n[ERROR] Cannot initialise CDS client: {exc}")
        print("Ensure ~/.cdsapirc is configured correctly.")
        sys.exit(1)

    ok, failed, skipped = 0, 0, 0
    failed_years = []

    for i, year in enumerate(YEARS, 1):
        print(f"\n[{i}/{len(YEARS)}] {year}")
        success = download_year(client, year)

        path = os.path.join(OUTPUT_DIR, f"era5_wind_{year}.grib")
        if success and os.path.exists(path):
            ok += 1
        elif success:
            skipped += 1
        else:
            failed += 1
            failed_years.append(year)

    print("\n" + "=" * 52)
    print("SUMMARY")
    print(f"  Downloaded : {ok}")
    print(f"  Skipped    : {skipped}  (already existed)")
    print(f"  Failed     : {failed}")
    if failed_years:
        print(f"  Failed years: {failed_years}")
        print("  Re-run the script to retry.")
    print(f"  Finished   : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 52)


if __name__ == "__main__":
    main()
