"""
04_classify_droughts.py
========================
Classifies wind energy drought severity at the hourly and daily level for
every ERCOT wind grid cell, for both the training period (2022–2024) and the
test period (2020–2021).

The classification scheme follows a two-step process:

  Step 1 — Hourly classification
  --------------------------------
  Each hour is assigned one of four categories based on the combination of
  instantaneous capacity factor (CF) and the length of the consecutive
  drought run it belongs to (drought = CF < 0.30):

    NO_DROUGHT  CF 0.25–0.30 and run < 10 hours
    MILD        CF < 0.25    and run < 10 hours
    MODERATE    Run 10–24 hours, OR longer runs where CF >= 0.15
    SEVERE      Run >= 24 hours AND CF < 0.15

  Consecutive run lengths are computed using vectorised segment IDs that
  reset whenever the drought flag changes or a timestamp gap exceeds 1 hour.
  `drought_run_hours` (the run length at each hour) is retained as a
  continuous feature for the XGBoost models.

  Step 2 — Daily classification (majority-rule aggregation)
  ----------------------------------------------------------
  Each calendar day is assigned the drought category that best represents
  the majority of its hourly observations:

    NO_DROUGHT  if >= 20 of 24 hours are NO_DROUGHT
    SEVERE      elif >= 12 hours are SEVERE
    MODERATE    elif >= 12 hours are MODERATE
    MILD        elif >= 12 hours are MILD
    tie-break   otherwise: the category with the most hours that day

Input
-----
Per-cell wind + temperature + price CSV files produced by earlier pipeline
steps.  Two period directories are processed:

    wind_temp_20_21_hourly_FIXED/   (test period)
    wind_temp_22_24_hourly_FIXED/   (training period)

Expected columns include: time, wind_cf, Load_Zone, grid_latitude,
grid_longitude, and whatever temperature / price columns were merged in
earlier steps.

Output
------
Hourly files (in-place update):
    Same directories — drought classification columns appended.

Daily files (new outputs):
    wind_temp_data_daily_20_21_RECOMPUTED/{lat_idx}_{lon_idx}_daily.csv
    wind_temp_data_daily_22_24_RECOMPUTED/{lat_idx}_{lon_idx}_daily.csv

Daily output columns:
    date, Load_Zone, grid_latitude, grid_longitude,
    daily_drought_category, daily_drought_hours,
    daily_non_drought_hours, daily_mild_hours,
    daily_moderate_hours, daily_severe_hours,
    daily_mean_wind_cf, daily_min_wind_cf

Usage
-----
    python files/04_classify_droughts.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input: merged wind + temperature + price hourly files
HOURLY_DIRS = {
    "20_21": Path("wind_temp_20_21_hourly_FIXED"),
    "22_24": Path("wind_temp_22_24_hourly_FIXED"),
}

# Output: recomputed daily classification files
DAILY_OUT_DIRS = {
    "20_21": Path("wind_temp_data_daily_20_21_RECOMPUTED"),
    "22_24": Path("wind_temp_data_daily_22_24_RECOMPUTED"),
}

# Drought threshold — consistent with ~mid-30% CF at Texas utility-scale farms
DROUGHT_CF_THRESHOLD = 0.30

# =============================================================================
# HOURLY CLASSIFICATION
# =============================================================================

def classify_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append hourly drought classification columns to *df*.

    New columns
    -----------
    drought               int  1 if wind_cf < DROUGHT_CF_THRESHOLD, else 0
    drought_run_hours     int  length of the consecutive drought run this
                               hour belongs to (0 for non-drought hours)
    hourly_drought_category  str  NO_DROUGHT | MILD | MODERATE | SEVERE
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # --- parse and sort by time ---
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    df["wind_cf"] = pd.to_numeric(df["wind_cf"], errors="coerce")
    df = df.dropna(subset=["wind_cf"]).copy()

    # --- base drought flag ---
    df["drought"] = (df["wind_cf"] < DROUGHT_CF_THRESHOLD).astype(int)

    # --- consecutive run detection ---
    # A new segment starts when drought status changes OR timestamp gap != 1 h
    time_diff_h = df["time"].diff().dt.total_seconds().div(3600)
    new_segment = (
        df["drought"].ne(df["drought"].shift(1))
        | time_diff_h.ne(1)
        | time_diff_h.isna()
    )
    df["_segment_id"] = new_segment.cumsum()
    seg_size = df.groupby("_segment_id")["_segment_id"].transform("size")

    # drought_run_hours is 0 for non-drought hours
    df["drought_run_hours"] = np.where(df["drought"] == 1, seg_size, 0)
    df = df.drop(columns=["_segment_id"])

    # --- four-tier hourly classification ---
    conditions = [
        # NO_DROUGHT: CF in [0.25, 0.30) with run < 10 h
        (df["drought"] == 0),
        # MILD: CF < 0.25, run < 10 h
        (df["drought"] == 1) & (df["drought_run_hours"] < 10) & (df["wind_cf"] < 0.25),
        # MODERATE: run 10–24 h, OR run >= 24 h with CF >= 0.15
        (
            (df["drought"] == 1)
            & (df["drought_run_hours"] >= 10)
            & (df["drought_run_hours"] < 24)
        )
        | (
            (df["drought"] == 1)
            & (df["drought_run_hours"] >= 24)
            & (df["wind_cf"] >= 0.15)
        ),
        # SEVERE: run >= 24 h AND CF < 0.15
        (df["drought"] == 1)
        & (df["drought_run_hours"] >= 24)
        & (df["wind_cf"] < 0.15),
    ]
    choices = ["NO_DROUGHT", "MILD", "MODERATE", "SEVERE"]
    df["hourly_drought_category"] = np.select(conditions, choices, default="NO_DROUGHT")

    return df


# =============================================================================
# DAILY CLASSIFICATION
# =============================================================================

def _classify_one_day(day_df: pd.DataFrame) -> str:
    """
    Majority-rule daily classification from hourly labels.

    Rules (applied in priority order):
      1. >= 20 hours NO_DROUGHT  →  NO_DROUGHT
      2. >= 12 hours SEVERE      →  SEVERE
      3. >= 12 hours MODERATE    →  MODERATE
      4. >= 12 hours MILD        →  MILD
      5. tie-break: category with the most hours (SEVERE > MODERATE > MILD >
         NO_DROUGHT in case of a true tie)
    """
    counts = day_df["hourly_drought_category"].value_counts()
    nd = counts.get("NO_DROUGHT", 0)
    sv = counts.get("SEVERE", 0)
    mo = counts.get("MODERATE", 0)
    mi = counts.get("MILD", 0)

    if nd >= 20:
        return "NO_DROUGHT"
    if sv >= 12:
        return "SEVERE"
    if mo >= 12:
        return "MODERATE"
    if mi >= 12:
        return "MILD"

    # tie-break: most hours wins
    return max(
        [("SEVERE", sv), ("MODERATE", mo), ("MILD", mi), ("NO_DROUGHT", nd)],
        key=lambda x: x[1],
    )[0]


def build_daily_labels(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate an hourly classified DataFrame to one row per calendar day.

    Returns a DataFrame with columns:
        date, Load_Zone, grid_latitude, grid_longitude,
        daily_drought_category, daily_drought_hours,
        daily_non_drought_hours, daily_mild_hours,
        daily_moderate_hours, daily_severe_hours,
        daily_mean_wind_cf, daily_min_wind_cf
    """
    df = hourly_df.copy()

    if "date" not in df.columns:
        df["date"] = df["time"].dt.date
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    group_keys = ["date", "Load_Zone", "grid_latitude", "grid_longitude"]
    missing = [c for c in group_keys if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for daily aggregation: {missing}")

    records = []
    for keys, g in df.groupby(group_keys):
        date, load_zone, grid_lat, grid_lon = keys
        records.append({
            "date"                    : date,
            "Load_Zone"               : load_zone,
            "grid_latitude"           : grid_lat,
            "grid_longitude"          : grid_lon,
            "daily_drought_category"  : _classify_one_day(g),
            "daily_drought_hours"     : int((g["drought"] == 1).sum()),
            "daily_non_drought_hours" : int((g["hourly_drought_category"] == "NO_DROUGHT").sum()),
            "daily_mild_hours"        : int((g["hourly_drought_category"] == "MILD").sum()),
            "daily_moderate_hours"    : int((g["hourly_drought_category"] == "MODERATE").sum()),
            "daily_severe_hours"      : int((g["hourly_drought_category"] == "SEVERE").sum()),
            "daily_mean_wind_cf"      : round(g["wind_cf"].mean(), 6),
            "daily_min_wind_cf"       : round(g["wind_cf"].min(), 6),
        })

    return pd.DataFrame(records)


# =============================================================================
# MAIN LOOP
# =============================================================================

def process_period(label: str, hourly_dir: Path, daily_out_dir: Path) -> None:
    daily_out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(hourly_dir.glob("*.csv"))
    print(f"\n[{label}] {len(files)} files in {hourly_dir}")

    for fp in files:
        try:
            df = pd.read_csv(fp)

            # --- hourly classification ---
            df = classify_hourly(df)

            # overwrite hourly file in-place with new columns
            df.to_csv(fp, index=False)

            # --- daily aggregation ---
            daily = build_daily_labels(df)

            # derive output filename from the hourly filename stem
            stem = fp.stem  # e.g. "6_23_wind_temp_hourly"
            daily_path = daily_out_dir / f"{stem}_daily.csv"
            daily.to_csv(daily_path, index=False)

        except Exception as exc:
            print(f"  [WARN] {fp.name}: {exc}")

    print(f"  Done. Daily files written to {daily_out_dir}/")


def main():
    for label, hourly_dir in HOURLY_DIRS.items():
        daily_out = DAILY_OUT_DIRS[label]
        process_period(label, hourly_dir, daily_out)

    print("\nAll periods complete.")


if __name__ == "__main__":
    main()
