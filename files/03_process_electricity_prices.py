"""
03_process_electricity_prices.py
=================================
Processes raw ERCOT real-time settlement point prices (15-minute intervals)
into hourly averages, then labels each hour HIGH or LOW based on a
year-specific, load-zone-specific 90th-percentile threshold.

The same logic is applied separately to the training period (2022–2024) and
the test period (2020–2021), producing two output files that are used
directly as model targets.

Input
-----
Place the raw ERCOT settlement-point price CSVs in:

    data/raw_prices/

The expected file format (from ERCOT NP6-905-CD) is:
    Columns: DeliveryDate, HourEnding, SettlementPoint, SettlementPointPrice
    (or equivalent column names — see COLUMN_MAP below to remap if needed)

The script accepts either 15-minute or hourly data; 15-minute records are
aggregated to hourly averages before labelling.

Data source
-----------
ERCOT Settlement Point Prices at Resource Nodes, Hubs, and Load Zones
(NP6-905-CD):
    https://data.ercot.com/data-product-details/np6-905-cd

Only the four load zones with significant installed wind capacity are
retained: LZ_WEST, LZ_SOUTH, LZ_NORTH, LZ_HOUSTON.

Output
------
    data/processed/ercot_prices_2020_2021.csv  (test period)
    data/processed/ercot_prices_2022_2024.csv  (training period)

Both files share the same schema:
    hour          datetime (UTC hour start)
    load_zone     str
    price         float ($/MWh, hourly average)
    Price Exposure  str  ("HIGH" if price > year-specific P90, else "LOW")

Usage
-----
    python 03_process_electricity_prices.py
"""

from pathlib import Path

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_DIR  = Path("data/raw_prices")
OUT_DIR  = Path("data/processed")

LOAD_ZONES = ["LZ_WEST", "LZ_SOUTH", "LZ_NORTH", "LZ_HOUSTON"]

# Map raw column names to standard names.
# Update these values if your ERCOT download uses different column headers.
COLUMN_MAP = {
    "DeliveryDate"         : "date",
    "HourEnding"           : "hour_ending",
    "SettlementPoint"      : "load_zone",
    "SettlementPointPrice" : "price",
    # Alternative flat-file headers (comment out one set):
    "hour"                 : "hour",      # if already a datetime column
    "price"                : "price",
}

# Output files by period
PERIODS = {
    "2020_2021": ("2020-01-01", "2021-12-31"),
    "2022_2024": ("2022-01-01", "2024-12-31"),
}

# =============================================================================
# HELPERS
# =============================================================================

def load_raw_prices(raw_dir: Path) -> pd.DataFrame:
    """
    Read all CSV files under raw_dir and concatenate into a single DataFrame
    with columns [hour, load_zone, price].

    Handles two common ERCOT file formats:
      (a) Flat files with a pre-built 'hour' datetime column
      (b) Raw NP6-905-CD files with DeliveryDate + HourEnding columns
    """
    frames = []
    for fp in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(fp, low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    df = pd.concat(frames, ignore_index=True)

    # --- detect format ---
    if "hour" in df.columns and "load_zone" in df.columns and "price" in df.columns:
        # Already in target format
        df["hour"]  = pd.to_datetime(df["hour"],  errors="coerce")
        df["price"] = pd.to_numeric(df["price"],  errors="coerce")

    elif "DeliveryDate" in df.columns and "HourEnding" in df.columns:
        # Raw NP6-905-CD format
        df = df.rename(columns={
            "SettlementPoint"      : "load_zone",
            "SettlementPointPrice" : "price",
        })
        # Construct a UTC hour timestamp from DeliveryDate + HourEnding
        # HourEnding is 1-based (1 = 00:00–01:00), convert to 0-based hour start
        df["hour_num"] = pd.to_numeric(
            df["HourEnding"].astype(str).str.replace(":00", "").str.strip(),
            errors="coerce"
        ) - 1
        df["hour"] = pd.to_datetime(df["DeliveryDate"], errors="coerce") + \
                     pd.to_timedelta(df["hour_num"], unit="h")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    else:
        raise ValueError(
            "Unrecognised column layout. Expected either:\n"
            "  ['hour', 'load_zone', 'price']  or\n"
            "  ['DeliveryDate', 'HourEnding', 'SettlementPoint', 'SettlementPointPrice']\n"
            f"Got: {list(df.columns)}"
        )

    return df[["hour", "load_zone", "price"]].dropna(subset=["hour"])


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the data contain sub-hourly records (e.g. 15-minute intervals),
    aggregate to hourly averages.  Hourly data is passed through unchanged.
    """
    df["hour"] = df["hour"].dt.floor("h")
    return (
        df.groupby(["hour", "load_zone"], as_index=False)["price"]
          .mean()
    )


def label_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary 'Price Exposure' column.
    HIGH = price above the year- and load-zone-specific 90th percentile.
    LOW  = otherwise.

    Using year-specific thresholds accounts for interannual variation in
    ERCOT price levels (fuel cycles, demand growth, structural grid changes).
    """
    df = df.copy()
    df["year"] = df["hour"].dt.year

    p90 = (
        df.groupby(["year", "load_zone"])["price"]
          .quantile(0.90)
          .reset_index()
          .rename(columns={"price": "P90"})
    )
    df = df.merge(p90, on=["year", "load_zone"], how="left")
    df["Price Exposure"] = (df["price"] > df["P90"]).map({True: "HIGH", False: "LOW"})
    df = df.drop(columns=["P90", "year"])
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw ERCOT prices ...")
    raw = load_raw_prices(RAW_DIR)

    print("Aggregating to hourly averages ...")
    hourly = aggregate_to_hourly(raw)

    # Retain only the four wind-capacity load zones
    hourly = hourly[hourly["load_zone"].isin(LOAD_ZONES)].copy()

    print(f"Total hourly records (all zones, all years): {len(hourly):,}")

    for label, (start, end) in PERIODS.items():
        subset = hourly[
            (hourly["hour"] >= start) &
            (hourly["hour"] <= end)
        ].copy()

        subset = label_exposure(subset)

        out_path = OUT_DIR / f"ercot_prices_{label}.csv"
        subset.to_csv(out_path, index=False)

        high_pct = (subset["Price Exposure"] == "HIGH").mean() * 100
        print(
            f"  {label}: {len(subset):,} rows, "
            f"{high_pct:.1f}% HIGH  →  {out_path}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
