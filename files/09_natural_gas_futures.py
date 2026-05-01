"""
09_natural_gas_futures.py
==========================
Downloads NYMEX Henry Hub natural gas futures prices from the EIA API and
prepares the hedging instrument data used in the financial simulation.

Two outputs are produced:

  1. Raw futures file
     henry_hub_futures_C1_C4_2019_2024.csv
     Daily closing prices for the four nearest delivery contracts (C1–C4),
     covering October 2019 through December 2024.  Includes two spread
     variables: C2−C1 and C3−C1, which represent the shape of the near-term
     futures curve and were included as XGBoost model features.

  2. Simulation-ready futures file
     henry_hub_futures_filled.csv
     The raw file forward-filled across weekends and holidays (up to five
     consecutive non-trading days) and broadcast to a daily calendar spanning
     the full 2020–2024 period, so every calendar day has a valid futures
     price without gaps.

Heat rate conversion
---------------------
Futures prices are reported in $/MMBtu.  To make them directly comparable
with electricity prices ($/MWh), they are converted using a heat rate
derived from a 38% combined-cycle gas turbine (CCGT) efficiency assumption —
the marginal generation technology most likely to set real-time prices during
high-demand or low-wind periods in ERCOT.

    1 MMBtu × 0.293071 MWh/MMBtu × 0.38 efficiency = 0.1114 MWh electricity
    1 MWh electricity requires 1 / 0.1114 = 8.978 MMBtu of gas
    Heat rate = 8.978 MMBtu/MWh

    Hedge quantity per flagged hour = 30 MWh × 8.978 = 269.34 MMBtu

Source: Woodway Energy, Natural Gas Efficiency in Power Generation
        https://www.woodwayenergy.com/natural-gas-efficiency-in-power-generation/

EIA API
-------
Series IDs:
  RNGC1 — prompt month (C1)
  RNGC2 — one-to-two months ahead (C2)
  RNGC3 — two-to-three months ahead (C3)
  RNGC4 — three-to-four months ahead (C4)

API endpoint: https://api.eia.gov/v2/natural-gas/pri/fut/data/

A free EIA API key is required.  Register at:
  https://www.eia.gov/opendata/register.php

Set your key via the EIA_API_KEY environment variable (preferred) or pass
it directly with --api-key.  Never hard-code credentials in this script.

Usage
-----
    # Key from environment variable (recommended)
    export EIA_API_KEY=your_key_here
    python files/09_natural_gas_futures.py

    # Key passed directly
    python files/09_natural_gas_futures.py --api-key your_key_here

    # Skip download if raw file already exists, just re-process
    python files/09_natural_gas_futures.py --no-download

Outputs saved to data/ng_futures/:
    henry_hub_futures_C1_C4_2019_2024.csv   raw daily futures prices
    henry_hub_futures_filled.csv            forward-filled simulation-ready file
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR   = Path("data/ng_futures")
RAW_FILE     = OUTPUT_DIR / "henry_hub_futures_C1_C4_2019_2024.csv"
FILLED_FILE  = OUTPUT_DIR / "henry_hub_futures_filled.csv"

# Coverage: Oct 2019 to Dec 2024
# Start date is Oct 2019 so that C1 covers the Jan 2020 test period start
START_DATE = "2019-10-01"
END_DATE   = "2024-12-31"

# EIA API series IDs for Henry Hub nearby contracts
SERIES_IDS = {
    "RNGC1": "NG_C1",  # prompt month
    "RNGC2": "NG_C2",  # one-to-two months ahead
    "RNGC3": "NG_C3",  # two-to-three months ahead
    "RNGC4": "NG_C4",  # three-to-four months ahead
}
CONTRACTS = list(SERIES_IDS.values())

EIA_ENDPOINT = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"

# Maximum consecutive non-trading days to forward-fill
# 5 covers Christmas/New Year stretches (Fri 24 Dec → Wed 29 Dec)
FFILL_LIMIT = 5

# Heat rate: MMBtu required per MWh of electricity at 38% CCGT efficiency
# = 1 / (0.293071 MWh/MMBtu × 0.381 efficiency)
HEAT_RATE = 1 / (0.293071 * 0.381)   # = 8.978 MMBtu/MWh

# PPA obligation used for hedge quantity annotation (not required for the
# download, but documented here for traceability)
PPA_OBLIGATION_MWH = 30.0
HEDGE_QUANTITY_MMBTU = PPA_OBLIGATION_MWH * HEAT_RATE  # = 269.34 MMBtu per flagged hour

# =============================================================================
# STEP 1 — DOWNLOAD FUTURES FROM EIA API
# =============================================================================

def download_futures(api_key: str) -> pd.DataFrame:
    """
    Retrieve daily closing prices for C1–C4 from the EIA API, pivot to wide
    format, and compute spread variables.

    Returns a DataFrame with columns:
        period, NG_C1, NG_C2, NG_C3, NG_C4, spread_C2_C1, spread_C3_C1
    """
    print(f"Requesting Henry Hub futures from EIA API ...")
    print(f"  Series   : {list(SERIES_IDS.keys())}")
    print(f"  Period   : {START_DATE} → {END_DATE}")

    params = {
        "api_key"               : api_key,
        "frequency"             : "daily",
        "data[0]"               : "value",
        "facets[series][0]"     : "RNGC1",
        "facets[series][1]"     : "RNGC2",
        "facets[series][2]"     : "RNGC3",
        "facets[series][3]"     : "RNGC4",
        "start"                 : START_DATE,
        "end"                   : END_DATE,
        "sort[0][column]"       : "period",
        "sort[0][direction]"    : "asc",
        "offset"                : 0,
        "length"                : 10000,
    }

    try:
        response = requests.get(EIA_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"\n[ERROR] EIA API request failed: {e}")
        print("  Check your API key and ensure it is activated.")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot reach EIA API.  Check network connection.")
        sys.exit(1)

    data = response.json().get("response", {}).get("data", [])
    if not data:
        print("[ERROR] EIA API returned empty data.  Check series IDs and date range.")
        sys.exit(1)

    print(f"  Records received: {len(data):,}")

    df = pd.DataFrame(data)
    df["period"] = pd.to_datetime(df["period"])
    df["value"]  = pd.to_numeric(df["value"], errors="coerce")

    # Pivot from long to wide: one column per contract
    df_wide = (
        df.pivot(index="period", columns="series", values="value")
        .reset_index()
        .sort_values("period")
        .rename(columns=SERIES_IDS)
    )

    # Spread variables: shape of the near-term futures curve
    # Positive spread = contango (futures > spot); negative = backwardation
    df_wide["spread_C2_C1"] = df_wide["NG_C2"] - df_wide["NG_C1"]
    df_wide["spread_C3_C1"] = df_wide["NG_C3"] - df_wide["NG_C1"]

    print(f"\n  Date range  : {df_wide['period'].min().date()} → {df_wide['period'].max().date()}")
    print(f"  Trading days: {len(df_wide):,}")
    print(f"\n  Price summary ($/MMBtu):")
    for c in CONTRACTS:
        print(
            f"    {c}: min=${df_wide[c].min():.2f}  max=${df_wide[c].max():.2f}  "
            f"mean=${df_wide[c].mean():.2f}"
        )

    print(f"\n  Spread summary ($/MMBtu):")
    print(f"    C2−C1: mean={df_wide['spread_C2_C1'].mean():+.4f}  "
          f"std={df_wide['spread_C2_C1'].std():.4f}")
    print(f"    C3−C1: mean={df_wide['spread_C3_C1'].mean():+.4f}  "
          f"std={df_wide['spread_C3_C1'].std():.4f}")

    return df_wide


# =============================================================================
# STEP 2 — FORWARD-FILL AND BUILD SIMULATION-READY FILE
# =============================================================================

def build_filled_file(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a continuous daily calendar from START_DATE to END_DATE and
    forward-fill futures prices across weekends and holidays.

    Also adds electricity-equivalent hedge cost columns ($/MWh) using the
    8.978 MMBtu/MWh heat rate for direct comparison with electricity prices.

    Returns a DataFrame indexed by calendar day with no gaps.
    """
    calendar = pd.DataFrame({
        "date": pd.date_range(START_DATE, END_DATE, freq="D")
    })

    raw_renamed = raw_df.rename(columns={"period": "date"})
    filled = (
        calendar
        .merge(raw_renamed, on="date", how="left")
        .sort_values("date")
    )

    # Forward-fill then back-fill residual gaps at the very start
    filled[CONTRACTS + ["spread_C2_C1", "spread_C3_C1"]] = (
        filled[CONTRACTS + ["spread_C2_C1", "spread_C3_C1"]]
        .ffill(limit=FFILL_LIMIT)
        .bfill(limit=1)
    )

    n_gaps = filled[CONTRACTS[0]].isna().sum()
    if n_gaps > 0:
        print(f"  [WARN] {n_gaps} calendar days still missing after forward-fill.")
        missing_dates = filled.loc[filled[CONTRACTS[0]].isna(), "date"].tolist()
        print(f"         First few: {missing_dates[:5]}")
    else:
        print(f"  Forward-fill complete — full calendar coverage achieved.")

    # Add electricity-equivalent hedge costs ($/MWh)
    for c in CONTRACTS:
        filled[f"{c}_elec_usd_mwh"] = filled[c] * HEAT_RATE

    print(f"\n  Calendar rows  : {len(filled):,}")
    print(f"  Date range     : {filled['date'].min().date()} → {filled['date'].max().date()}")
    print(f"\n  Electricity-equivalent hedge cost ($/MWh at heat rate {HEAT_RATE:.3f} MMBtu/MWh):")
    for c in CONTRACTS:
        col = f"{c}_elec_usd_mwh"
        print(f"    {c}: min=${filled[col].min():.2f}  max=${filled[col].max():.2f}  "
              f"mean=${filled[col].mean():.2f}")

    return filled


# =============================================================================
# STEP 3 — VALIDATE COVERAGE AGAINST SIMULATION PERIODS
# =============================================================================

def validate_coverage(filled_df: pd.DataFrame) -> None:
    """
    Check that the filled futures file covers the key model periods:
    training (2022–2024) and test (2020–2021).
    """
    print(f"\n  Coverage check against model periods:")
    for label, start, end in [
        ("Test  2020–2021", "2020-01-01", "2021-12-31"),
        ("Train 2022–2024", "2022-01-01", "2024-12-31"),
    ]:
        mask      = (filled_df["date"] >= start) & (filled_df["date"] <= end)
        n_days    = mask.sum()
        n_missing = filled_df.loc[mask, CONTRACTS[0]].isna().sum()
        c1_range  = (
            f"${filled_df.loc[mask, 'NG_C1'].min():.2f}–"
            f"${filled_df.loc[mask, 'NG_C1'].max():.2f}/MMBtu"
        )
        print(f"    {label}: {n_days} days, {n_missing} missing, C1 range {c1_range}")

    # Hedge quantity annotation
    print(f"\n  Hedge quantity per flagged hour:")
    print(f"    PPA obligation : {PPA_OBLIGATION_MWH:.1f} MWh/hr")
    print(f"    Heat rate      : {HEAT_RATE:.3f} MMBtu/MWh  (38% CCGT efficiency)")
    print(f"    Hedge volume   : {HEDGE_QUANTITY_MMBTU:.2f} MMBtu per flagged hour")
    print(f"    Hedge P&L      : (NG spot − NG futures) × {HEDGE_QUANTITY_MMBTU:.2f} MMBtu")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("EIA_API_KEY", ""),
        help="EIA API key.  Defaults to EIA_API_KEY environment variable.",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Skip the API download and re-process an existing raw file.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: download or load raw file ---
    if args.no_download and RAW_FILE.exists():
        print(f"[--no-download] Loading existing raw file: {RAW_FILE}")
        raw_df = pd.read_csv(RAW_FILE, parse_dates=["period"])
    else:
        if not args.api_key:
            print(
                "[ERROR] No API key provided.\n"
                "  Set the EIA_API_KEY environment variable or pass --api-key.\n"
                "  Register for a free key at https://www.eia.gov/opendata/register.php"
            )
            sys.exit(1)
        raw_df = download_futures(args.api_key)
        raw_df.to_csv(RAW_FILE, index=False)
        print(f"\n  Saved raw file: {RAW_FILE}")

    # --- Step 2: forward-fill to continuous calendar ---
    print(f"\n{'=' * 60}")
    print("  STEP 2 — FORWARD-FILLING TO CONTINUOUS CALENDAR")
    print(f"{'=' * 60}")
    filled_df = build_filled_file(raw_df)
    filled_df.to_csv(FILLED_FILE, index=False)
    print(f"\n  Saved filled file: {FILLED_FILE}")

    # --- Step 3: validate coverage ---
    print(f"\n{'=' * 60}")
    print("  STEP 3 — COVERAGE VALIDATION")
    print(f"{'=' * 60}")
    validate_coverage(filled_df)

    print(f"\n{'=' * 60}")
    print("  DONE")
    print(f"{'=' * 60}")
    print(f"  Outputs saved to {OUTPUT_DIR}/")
    print(f"    {RAW_FILE.name:<45} raw daily futures prices")
    print(f"    {FILLED_FILE.name:<45} forward-filled, simulation-ready")
    print()
    print("  Next step: 10_financial_simulation.py")
    print("  The filled file is read automatically by the financial simulation.")


if __name__ == "__main__":
    main()
