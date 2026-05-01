"""
08_grid_cell_performance_and_threshold.py
==========================================
Two analyses that bridge the global XGBoost model and the financial
simulation:

  Part 1 — Per-grid-cell model performance
  -----------------------------------------
  Evaluates the fitted XGB-2 model on every individual grid cell in the
  2020–2021 test set.  Computes AUC-ROC, precision, and recall per cell,
  breaks AUC down by year (2020 vs 2021), and produces:
    - Spatial scatter map of overall AUC and 2020→2021 AUC shift
    - AUC distribution histograms by load zone
    - 2020 vs 2021 AUC scatter (one dot per grid cell)
    - Full per-cell metrics CSV

  This analysis motivates the selection of grid cell 6_23 (LZ_WEST, 150 MW)
  for the financial simulation.  Despite LZ_WEST having the weakest average
  AUC, cell 6_23 achieves above-zone-average performance (overall AUC 0.616,
  2021 AUC 0.70) while sitting in the zone with the largest aggregate
  physical exposure (27,768 MW, 83 grid cells).

  Part 2 — Classification threshold optimisation for cell 6_23
  -------------------------------------------------------------
  Converts the model's continuous probability outputs into binary HIGH / LOW
  flags by scanning 100 candidate thresholds from 0.25 to 0.74 and selecting
  the F1-optimal threshold subject to a minimum recall constraint.

  Two thresholds are derived:
    Conservative (recall ≥ 55%) — F1-optimal for normal market conditions
    Aggressive   (recall ≥ 75%) — justified by economic recall scan showing
                                   net hedge benefit rises monotonically to
                                   75%, driven by Winter Storm Uri (88% of
                                   total replacement costs in Feb 2021)

  The economic recall scan evaluates, for each recall level from 10%–80%:
    1. The F1-optimal threshold satisfying that recall floor
    2. Monthly natural gas futures hedge gains and costs under a fixed-volume
       strategy (hedge quantity = full PPA obligation × heat rate)
    3. Net hedge benefit = total actual losses − net losses after hedge
  across all four NYMEX Henry Hub futures contracts (C1–C4).

  The overall threshold (0.3539) is retained as the primary operating
  threshold over the year-specific alternative (which improves 2020
  performance but reduces Uri-period recall in 2021).

  Physical parameters (consistent with financial simulation)
    Capacity      : 100 MW nameplate
    PPA price     : $50 / MWh
    Obligation    : 30 MWh / hr (30% CF floor)
    Heat rate     : 8.978 MMBtu / MWh  (38% CCGT efficiency)

Usage
-----
    python files/08_grid_cell_performance_and_threshold.py

Outputs saved to results/grid_cell/:
    grid_cell_auc_map.png               spatial AUC map + shift map
    grid_cell_auc_by_zone.png           AUC distribution + 2020 vs 2021 scatter
    grid_cell_performance.csv           full per-cell metrics table
    cell_6_23_threshold_scan.png        threshold optimisation scan (Figure 15)
    cell_6_23_recall_scan.png           economic recall constraint scan
    cell_6_23_threshold_summary.csv     threshold candidates comparison
"""

import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_DIR      = Path("wind_temp_22_24_hourly_FIXED")
TEST_DIR       = Path("wind_temp_20_21_hourly_FIXED")
GAS_PATH       = Path("data/NG_prices.csv")
NG_FUTURES_PATH = Path("data/henry_hub_futures_C1_C4_2019_2024.csv")
OUTPUT_DIR     = Path("results/grid_cell")

TARGET_COL = "Price Exposure"
DATE_COL   = "time"

# Minimum HIGH samples per cell to compute AUC reliably
MIN_HIGH_SAMPLES = 10

# Cell selected for financial simulation (paper Section 4)
SIM_CELL = ("6", "23")       # (lat_idx, lon_idx)
SIM_CAPACITY_MW  = 100.0     # nameplate capacity used in simulation
SIM_FIXED_PRICE  = 50.0      # $/MWh PPA fixed price
SIM_OBLIGATION   = 30.0      # MWh/hr delivery obligation (30% CF floor)
HEAT_RATE        = 1 / (0.293071 * 0.381)  # 8.978 MMBtu/MWh (38% CCGT efficiency)

# Threshold scan range (100 thresholds)
THRESH_MIN, THRESH_MAX, N_THRESH = 0.25, 0.74, 100

# Recall constraint floors
RECALL_CONSERVATIVE = 0.55   # primary operating threshold
RECALL_AGGRESSIVE   = 0.75   # tail-risk threshold (Uri-motivated)
RECALL_SCAN_LEVELS  = np.arange(0.10, 0.85, 0.05)

# XGB-2 hyperparameters (match 06_xgboost_models.py)
XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    eval_metric="auc", random_state=42, n_jobs=-1,
)
FEATURE_COLS = [
    "wind_cf", "drought_run_hours", "tmm_F", "HDD_hourly", "CDD_hourly",
    "temp_stress", "gas_price_mmbtu", "low_wind_x_demand",
    "run_hours_x_demand", "gas_x_demand", "gas_x_low_wind_x_demand",
]

ZONE_COLORS = {
    "LZ_WEST": "#185FA5", "LZ_NORTH": "#0F6E56",
    "LZ_SOUTH": "#993C1D", "LZ_HOUSTON": "#854F0B",
}

# =============================================================================
# DATA PIPELINE  (shared helpers — mirrors 06_xgboost_models.py)
# =============================================================================

def load_period(directory: Path, label: str) -> pd.DataFrame:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {directory}")
    chunks = []
    for fp in files:
        df = pd.read_csv(fp, parse_dates=[DATE_COL])
        parts = fp.stem.split("_")
        df["lat_idx"] = parts[0]
        df["lon_idx"] = parts[1]
        chunks.append(df)
    combined = pd.concat(chunks, ignore_index=True)
    print(f"  [{label}] {len(files)} files, {len(combined):,} rows")
    return combined


def load_gas(path: Path) -> pd.DataFrame:
    gas = pd.read_csv(path)
    gas.columns = gas.columns.str.strip()
    date_col  = "observation_date" if "observation_date" in gas.columns else gas.columns[0]
    price_col = "$/MMBtu"          if "$/MMBtu"          in gas.columns else gas.columns[1]
    gas = gas[[date_col, price_col]].copy()
    gas.columns = ["date_key", "gas_price_mmbtu"]
    gas["date_key"] = pd.to_datetime(gas["date_key"]).dt.normalize()
    gas = gas.dropna(subset=["gas_price_mmbtu"]).sort_values("date_key")
    gas["gas_price_mmbtu"] = gas["gas_price_mmbtu"].replace(0, np.nan).ffill()
    return gas


def merge_gas(df: pd.DataFrame, gas_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_key"] = pd.to_datetime(df[DATE_COL]).dt.normalize()
    df = df.merge(gas_df, on="date_key", how="left").sort_values(DATE_COL)
    df["gas_price_mmbtu"] = df["gas_price_mmbtu"].ffill().bfill()
    return df


def label_exposure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df[DATE_COL].dt.year
    p90 = (
        df.groupby(["Load_Zone", "year"])["price"]
        .quantile(0.90).rename("price_p90").reset_index()
    )
    df = df.merge(p90, on=["Load_Zone", "year"], how="left")
    df[TARGET_COL] = (df["price"] >= df["price_p90"]).map({True: "HIGH", False: "LOW"})
    df["target"] = (df[TARGET_COL] == "HIGH").astype(int)
    return df.drop(columns=["price_p90", "year"], errors="ignore")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["wind_cf"]        = df["wind_cf"].clip(0.0, 1.0)
    df["temp_stress"]    = np.abs(df["tmm_F"] - 65.0)
    df["temp_stress_norm"] = (df["temp_stress"] / 70.0).clip(0.0, 1.0)
    df["low_wind_x_demand"]       = (1.0 - df["wind_cf"]) * df["temp_stress_norm"]
    df["run_hours_x_demand"]      = (df["drought_run_hours"] / 24.0) * df["temp_stress_norm"]
    df["gas_x_demand"]            = df["gas_price_mmbtu"] * df["temp_stress_norm"]
    df["gas_x_low_wind_x_demand"] = (
        df["gas_price_mmbtu"] * (1.0 - df["wind_cf"]) * df["temp_stress_norm"]
    )
    return df


def fit_global_model(train_df: pd.DataFrame) -> Pipeline:
    X = train_df[FEATURE_COLS]
    y = train_df["target"]
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]
    spw = round((y == 0).sum() / (y == 1).sum())
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)),
    ])
    pipeline.fit(X, y)
    print(f"  Global XGB-2 fitted on {len(X):,} training rows  (spw={spw})")
    return pipeline

# =============================================================================
# PART 1 — PER-GRID-CELL MODEL PERFORMANCE
# =============================================================================

def compute_cell_metrics(pipeline: Pipeline, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the global model to every grid cell in the test set individually.
    Returns a DataFrame with one row per cell containing AUC, precision,
    recall, and year-specific AUC.
    """
    X_all = test_df[FEATURE_COLS]
    valid  = X_all.notna().all(axis=1)
    df     = test_df[valid].copy().reset_index(drop=True)
    X_all  = X_all[valid].reset_index(drop=True)

    df["prob_high"]  = pipeline.predict_proba(X_all)[:, 1]
    df["pred_label"] = pipeline.predict(X_all)
    df["true_label"] = df["target"]
    df["year"]       = pd.to_datetime(df[DATE_COL]).dt.year

    overall_auc = roc_auc_score(df["true_label"], df["prob_high"])
    print(f"  Overall test AUC (all cells): {overall_auc:.4f}")

    records = []
    for (lat_idx, lon_idx), g in df.groupby(["lat_idx", "lon_idx"]):
        n_high = int(g["true_label"].sum())
        grid_lat  = g["grid_latitude"].iloc[0]  if "grid_latitude"  in g.columns else np.nan
        grid_lon  = g["grid_longitude"].iloc[0] if "grid_longitude" in g.columns else np.nan
        load_zone = g["Load_Zone"].iloc[0]       if "Load_Zone"      in g.columns else "Unknown"

        rec = {
            "lat_idx": lat_idx, "lon_idx": lon_idx,
            "grid_lat": grid_lat, "grid_lon": grid_lon,
            "Load_Zone": load_zone,
            "n_total": len(g), "n_high": n_high,
            "pct_high": round(n_high / len(g) * 100, 2),
            "auc_overall": np.nan, "auc_2020": np.nan, "auc_2021": np.nan,
            "precision": np.nan, "recall": np.nan,
            "insufficient_data": n_high < MIN_HIGH_SAMPLES,
        }

        if n_high >= MIN_HIGH_SAMPLES:
            rec["auc_overall"] = round(roc_auc_score(g["true_label"], g["prob_high"]), 4)
            rec["precision"]   = round(precision_score(g["true_label"], g["pred_label"], zero_division=0), 4)
            rec["recall"]      = round(recall_score(g["true_label"], g["pred_label"], zero_division=0), 4)
            for yr in [2020, 2021]:
                yr_g = g[g["year"] == yr]
                if yr_g["true_label"].sum() >= MIN_HIGH_SAMPLES:
                    rec[f"auc_{yr}"] = round(
                        roc_auc_score(yr_g["true_label"], yr_g["prob_high"]), 4
                    )

        records.append(rec)

    results = pd.DataFrame(records)
    valid_r = results[~results["insufficient_data"]]

    print(f"  Cells evaluated  : {(~results['insufficient_data']).sum()}")
    print(f"  Cells skipped    : {results['insufficient_data'].sum()} (<{MIN_HIGH_SAMPLES} HIGH samples)")
    print(f"\n  AUC distribution:")
    print(f"    Mean   : {valid_r['auc_overall'].mean():.4f}")
    print(f"    Median : {valid_r['auc_overall'].median():.4f}")
    print(f"    Std    : {valid_r['auc_overall'].std():.4f}")
    print(f"    Min    : {valid_r['auc_overall'].min():.4f}")
    print(f"    Max    : {valid_r['auc_overall'].max():.4f}")

    print(f"\n  AUC by load zone:")
    for zone in sorted(valid_r["Load_Zone"].unique()):
        zdf = valid_r[valid_r["Load_Zone"] == zone]
        print(
            f"    {zone:<12}  mean={zdf['auc_overall'].mean():.4f}  "
            f"std={zdf['auc_overall'].std():.4f}  "
            f"n={len(zdf)}  "
            f"2020→2021 shift={zdf['auc_2021'].mean()-zdf['auc_2020'].mean():+.4f}"
        )

    # Selected cell summary
    sel = valid_r[
        (valid_r["lat_idx"].astype(str) == SIM_CELL[0]) &
        (valid_r["lon_idx"].astype(str) == SIM_CELL[1])
    ]
    if len(sel) > 0:
        r = sel.iloc[0]
        print(f"\n  Selected simulation cell {SIM_CELL[0]}_{SIM_CELL[1]} ({r['Load_Zone']}):")
        print(f"    Overall AUC : {r['auc_overall']:.4f}  "
              f"(zone mean: {valid_r[valid_r['Load_Zone']==r['Load_Zone']]['auc_overall'].mean():.4f})")
        print(f"    AUC 2020    : {r['auc_2020']:.4f}")
        print(f"    AUC 2021    : {r['auc_2021']:.4f}")

    return results


def plot_spatial_maps(results: pd.DataFrame, output_dir: Path) -> None:
    """Spatial AUC map (overall + 2020→2021 shift) and AUC distributions."""
    valid_r  = results[~results["insufficient_data"]].copy()
    insuff_r = results[results["insufficient_data"]].copy()
    shift_r  = valid_r.dropna(subset=["auc_2020", "auc_2021"]).copy()
    shift_r["auc_shift"] = shift_r["auc_2021"] - shift_r["auc_2020"]

    cmap = plt.cm.RdYlGn

    # --- Figure 1: spatial maps ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")

    for ax, data, col, label, vmin, vmax, title in [
        (axes[0], valid_r,  "auc_overall", "AUC-ROC",           0.45, 0.75,
         "Grid cell AUC-ROC — overall (2020–2021)"),
        (axes[1], shift_r, "auc_shift",   "AUC shift (2021−2020)", None, None,
         "Distribution shift by grid cell\n(green = 2021 better than 2020)"),
    ]:
        ax.set_facecolor("#f8f8f8")
        if col == "auc_shift":
            vmin = shift_r["auc_shift"].quantile(0.05)
            vmax = shift_r["auc_shift"].quantile(0.95)

        if len(insuff_r) > 0 and insuff_r["grid_lat"].notna().any():
            ax.scatter(insuff_r["grid_lon"], insuff_r["grid_lat"],
                       c="#cccccc", s=60, marker="s", zorder=2,
                       label="Insufficient data", alpha=0.6)

        sc = ax.scatter(
            data["grid_lon"], data["grid_lat"], c=data[col],
            cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
            s=80, marker="s", zorder=3, edgecolors="white", linewidths=0.3,
        )
        plt.colorbar(sc, ax=ax, label=label, shrink=0.8, pad=0.02)

        for zone, color in ZONE_COLORS.items():
            zdf = data[data["Load_Zone"] == zone]
            if len(zdf) > 0:
                ax.annotate(
                    zone.replace("LZ_", ""),
                    (zdf["grid_lon"].mean(), zdf["grid_lat"].mean()),
                    fontsize=8, color=color, fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.6, edgecolor="none"),
                )

        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("XGBoost model performance by grid cell — test data 2020–2021",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out = output_dir / "grid_cell_auc_map.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")

    # --- Figure 2: AUC distributions ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.patch.set_facecolor("white")
    zones = sorted(valid_r["Load_Zone"].unique())

    # histogram by zone
    ax3 = axes2[0]
    for zone in zones:
        color = ZONE_COLORS.get(zone, "#888888")
        aucs  = valid_r[valid_r["Load_Zone"] == zone]["auc_overall"].dropna()
        ax3.hist(aucs, bins=15, alpha=0.6, color=color,
                 label=f"{zone} (n={len(aucs)})", edgecolor="white", linewidth=0.5)
    ax3.axvline(0.5, color="#999999", linestyle="--", linewidth=1, label="Random (0.50)")
    ax3.set_xlabel("AUC-ROC", fontsize=11)
    ax3.set_ylabel("Number of grid cells", fontsize=11)
    ax3.set_title("AUC distribution by load zone", fontsize=12)
    ax3.legend(frameon=False, fontsize=9)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # 2020 vs 2021 scatter
    ax4 = axes2[1]
    for zone in zones:
        color  = ZONE_COLORS.get(zone, "#888888")
        zdf    = valid_r[valid_r["Load_Zone"] == zone].dropna(subset=["auc_2020", "auc_2021"])
        ax4.scatter(zdf["auc_2020"], zdf["auc_2021"],
                    c=color, s=50, alpha=0.75, label=zone,
                    edgecolors="white", linewidths=0.3)

    lims = [0.4, 0.85]
    ax4.plot(lims, lims, color="#999999", linestyle="--", linewidth=1, label="Equal (2020=2021)")
    ax4.fill_between(lims, lims, [0.85, 0.85], alpha=0.04, color="green")
    ax4.fill_between(lims, [0.4, 0.4], lims, alpha=0.04, color="red")
    ax4.text(0.42, 0.78, "2021 better", fontsize=8, color="#3B6D11", style="italic")
    ax4.text(0.65, 0.43, "2020 better", fontsize=8, color="#A32D2D", style="italic")
    ax4.set_xlabel("AUC-ROC 2020", fontsize=11)
    ax4.set_ylabel("AUC-ROC 2021", fontsize=11)
    ax4.set_title("Per-cell AUC: 2020 vs 2021\n(each dot = one grid cell)", fontsize=12)
    ax4.legend(frameon=False, fontsize=9, loc="upper right")
    ax4.set_xlim(0.40, 0.75)
    ax4.set_ylim(0.40, 0.85)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    fig2.suptitle("Grid cell AUC distribution and year-over-year shift",
                  fontsize=14, y=1.01)
    plt.tight_layout()
    out2 = output_dir / "grid_cell_auc_by_zone.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out2}")

# =============================================================================
# PART 2 — THRESHOLD OPTIMISATION FOR CELL 6_23
# =============================================================================

def scan_thresholds(cell_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scan 100 candidate thresholds (0.25–0.74) and compute F1, MCC,
    precision, recall, and flagging rate at each threshold.
    """
    scan_thresholds = np.linspace(THRESH_MIN, THRESH_MAX, N_THRESH)
    rows = []
    for t in scan_thresholds:
        preds = (cell_df["prob_high"] >= t).astype(int)
        true  = cell_df["true_label"]
        tp = int(((preds == 1) & (true == 1)).sum())
        fp = int(((preds == 1) & (true == 0)).sum())
        fn = int(((preds == 0) & (true == 1)).sum())
        tn = int(((preds == 0) & (true == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        mcc  = float(matthews_corrcoef(true, preds))
        rows.append({
            "threshold": round(t, 4), "f1": f1, "mcc": mcc,
            "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_flagged": int(preds.sum()),
        })
    return pd.DataFrame(rows)


def select_threshold(scan_df: pd.DataFrame, min_recall: float) -> pd.Series:
    """Return the F1-optimal row subject to recall >= min_recall."""
    constrained = scan_df[scan_df["recall"] >= min_recall]
    if len(constrained) == 0:
        return scan_df.loc[scan_df["f1"].idxmax()]
    return constrained.loc[constrained["f1"].idxmax()]


def economic_recall_scan(
    cell_df: pd.DataFrame,
    futures_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each recall level in RECALL_SCAN_LEVELS, find the F1-optimal
    threshold and compute net hedge benefit under a fixed-volume strategy
    across all four futures contracts.

    Hedge quantity per flagged hour = OBLIGATION_MWH × HEAT_RATE (MMBtu).
    Hedge P&L = (spot − futures) × hedge_quantity.
    Net benefit = total actual replacement cost − net loss after hedge.
    """
    rows = []
    for min_rec in RECALL_SCAN_LEVELS:
        thresh_row = None
        best_f1 = -1.0
        for t in np.linspace(THRESH_MIN, THRESH_MAX, 200):
            preds = (cell_df["prob_high"] >= t).astype(int)
            true  = cell_df["true_label"]
            tp = ((preds == 1) & (true == 1)).sum()
            fp = ((preds == 1) & (true == 0)).sum()
            fn = ((preds == 0) & (true == 1)).sum()
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if rec >= min_rec and f1 > best_f1:
                best_f1, thresh_row = f1, {
                    "threshold": t, "f1": f1, "recall": rec,
                    "precision": prec, "n_flagged": int(preds.sum()),
                }

        if thresh_row is None:
            continue

        thresh = thresh_row["threshold"]
        df     = cell_df.copy()
        df["flagged"]       = (df["prob_high"] >= thresh).astype(int)
        df["wind_mwh"]      = SIM_CAPACITY_MW * df["wind_cf"]
        df["shortfall_mwh"] = np.maximum(0, SIM_OBLIGATION - df["wind_mwh"])
        df["replacement_cost"] = df["shortfall_mwh"] * np.maximum(
            0, df["price"] - SIM_FIXED_PRICE
        )
        df["month"] = pd.to_datetime(df["time"]).dt.to_period("M")

        total_losses = df["replacement_cost"].sum()

        for contract in ["NG_C1", "NG_C2", "NG_C3", "NG_C4"]:
            if contract not in df.columns:
                continue
            flagged = df[df["flagged"] == 1]
            hedge_mwh    = SIM_OBLIGATION * len(flagged)
            hedge_mmbtu  = hedge_mwh * HEAT_RATE
            futures_avg  = flagged[contract].mean() if len(flagged) > 0 else 0.0
            spot_avg     = flagged["gas_price_mmbtu"].mean() if len(flagged) > 0 else 0.0
            hedge_pnl    = (spot_avg - futures_avg) * hedge_mmbtu
            net_loss     = total_losses - hedge_pnl
            net_benefit  = total_losses - net_loss

            rows.append({
                "min_recall"   : min_rec,
                "actual_recall": thresh_row["recall"],
                "threshold"    : round(thresh, 4),
                "f1"           : round(thresh_row["f1"], 4),
                "n_flagged"    : thresh_row["n_flagged"],
                "contract"     : contract,
                "total_losses" : total_losses,
                "hedge_pnl"    : hedge_pnl,
                "net_benefit"  : net_benefit,
                "pct_protected": round(net_benefit / total_losses * 100, 2)
                                  if total_losses > 0 else 0.0,
            })

    return pd.DataFrame(rows)


def plot_threshold_scan(
    scan_df: pd.DataFrame,
    optimal_thresh: float,
    output_dir: Path,
) -> None:
    """Three-panel threshold scan plot (Figure 15 from paper)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")

    # Panel 1: F1 and MCC
    ax1 = axes[0]
    ax1.plot(scan_df["threshold"], scan_df["f1"],
             color="#185FA5", linewidth=2, label="F1 score")
    ax1.plot(scan_df["threshold"], scan_df["mcc"],
             color="#0F6E56", linewidth=2, label="MCC")
    ax1.axvline(optimal_thresh, color="#993C1D", linestyle="--",
                linewidth=1.5, label=f"Optimal ({optimal_thresh:.4f})")
    ax1.axvline(0.50, color="#888888", linestyle=":", linewidth=1.2, label="Default (0.50)")
    ax1.set_xlabel("Threshold", fontsize=10)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_title("F1 and MCC by threshold\n(overall 2020–2021)", fontsize=11)
    ax1.legend(frameon=False, fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", color="#eeeeee", linewidth=0.6)

    # Panel 2: recall vs precision tradeoff
    ax2 = axes[1]
    ax2.plot(scan_df["threshold"], scan_df["recall"] * 100,
             color="#185FA5", linewidth=2, label="Recall")
    ax2.plot(scan_df["threshold"], scan_df["precision"] * 100,
             color="#993C1D", linewidth=2, label="Precision")
    ax2.axvline(optimal_thresh, color="#0F6E56", linestyle="--",
                linewidth=1.5, label=f"Optimal ({optimal_thresh:.4f})")
    ax2.set_xlabel("Threshold", fontsize=10)
    ax2.set_ylabel("Percent (%)", fontsize=10)
    ax2.set_title("Recall vs precision tradeoff\nacross 100 thresholds", fontsize=11)
    ax2.legend(frameon=False, fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", color="#eeeeee", linewidth=0.6)

    # Panel 3: hours flagged vs caught
    ax3 = axes[2]
    ax3.plot(scan_df["threshold"], scan_df["n_flagged"],
             color="#185FA5", linewidth=2, label="Total hours flagged")
    ax3.plot(scan_df["threshold"], scan_df["tp"],
             color="#0F6E56", linewidth=2, label="HIGH hours caught (TP)")
    ax3.plot(scan_df["threshold"], scan_df["fn"],
             color="#993C1D", linewidth=2, linestyle="--", label="HIGH hours missed (FN)")
    ax3.axvline(optimal_thresh, color="#888888", linestyle="--",
                linewidth=1.5, label=f"Optimal ({optimal_thresh:.4f})")
    ax3.set_xlabel("Threshold", fontsize=10)
    ax3.set_ylabel("Number of hours", fontsize=10)
    ax3.set_title("Hours flagged vs caught\nacross 100 thresholds", fontsize=11)
    ax3.legend(frameon=False, fontsize=9)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(axis="y", color="#eeeeee", linewidth=0.6)

    fig.suptitle(
        f"Cell 6_23 — Threshold optimisation scan "
        f"({N_THRESH} thresholds, {THRESH_MIN}–{THRESH_MAX})",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out = output_dir / "cell_6_23_threshold_scan.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")


def plot_recall_scan(recall_df: pd.DataFrame, output_dir: Path) -> None:
    """Net hedge benefit vs recall level by futures contract (C1–C4)."""
    if recall_df.empty:
        print("  [SKIP] No recall scan data (futures file missing?)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    contract_colors = {
        "NG_C1": "#185FA5", "NG_C2": "#0F6E56",
        "NG_C3": "#993C1D", "NG_C4": "#854F0B",
    }
    contract_labels = {
        "NG_C1": "C1 (1-month ahead)",    "NG_C2": "C2 (1–2 months ahead)",
        "NG_C3": "C3 (2–3 months ahead)", "NG_C4": "C4 (3–4 months ahead)",
    }
    for contract in ["NG_C1", "NG_C2", "NG_C3", "NG_C4"]:
        sub = recall_df[recall_df["contract"] == contract].sort_values("min_recall")
        if sub.empty:
            continue
        ax.plot(
            sub["min_recall"] * 100, sub["net_benefit"] / 1000,
            color=contract_colors[contract], linewidth=2,
            label=contract_labels[contract],
        )

    ax.axvline(RECALL_CONSERVATIVE * 100, color="#333333", linestyle="--",
               linewidth=1.5, label=f"Conservative threshold ({RECALL_CONSERVATIVE*100:.0f}% recall)")
    ax.axvline(RECALL_AGGRESSIVE * 100, color="#333333", linestyle=":",
               linewidth=1.5, label=f"Aggressive threshold ({RECALL_AGGRESSIVE*100:.0f}% recall)")
    ax.axhline(0, color="#aaaaaa", linewidth=0.8)
    ax.set_xlabel("Minimum recall constraint (%)", fontsize=11)
    ax.set_ylabel("Net hedge benefit ($000s)", fontsize=11)
    ax.set_title(
        "Cell 6_23 — Economic recall scan\n"
        "Net hedge benefit by recall floor and futures contract",
        fontsize=12,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.6)
    plt.tight_layout()
    out = output_dir / "cell_6_23_recall_scan.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")


def run_threshold_analysis(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Run the full threshold analysis for cell 6_23:
      1. Extract cell predictions
      2. Scan thresholds → conservative and aggressive cutoffs
      3. Per-year threshold calibration
      4. Economic recall scan (if futures data available)
      5. Save plots and summary CSV
    """
    print(f"\n{'=' * 65}")
    print(f"  PART 2 — THRESHOLD OPTIMISATION FOR CELL {SIM_CELL[0]}_{SIM_CELL[1]}")
    print(f"{'=' * 65}")

    # --- extract cell predictions ---
    X_all  = test_df[FEATURE_COLS]
    valid  = X_all.notna().all(axis=1)
    df_val = test_df[valid].copy().reset_index(drop=True)
    X_val  = X_all[valid].reset_index(drop=True)

    df_val["prob_high"]  = pipeline.predict_proba(X_val)[:, 1]
    df_val["true_label"] = df_val["target"]
    df_val["year"]       = pd.to_datetime(df_val[DATE_COL]).dt.year

    cell_df = df_val[
        (df_val["lat_idx"].astype(str) == SIM_CELL[0]) &
        (df_val["lon_idx"].astype(str) == SIM_CELL[1])
    ].copy().reset_index(drop=True)

    if len(cell_df) == 0:
        print(f"  [SKIP] Cell {SIM_CELL[0]}_{SIM_CELL[1]} not found in test data.")
        return

    print(f"  Cell rows : {len(cell_df):,}")
    print(f"  HIGH hours: {int(cell_df['true_label'].sum()):,}  "
          f"({cell_df['true_label'].mean()*100:.1f}%)")
    print(f"  AUC-ROC   : {roc_auc_score(cell_df['true_label'], cell_df['prob_high']):.4f}")
    print(f"  Prob separation — LOW mean: "
          f"{cell_df[cell_df['true_label']==0]['prob_high'].mean():.4f}  "
          f"HIGH mean: {cell_df[cell_df['true_label']==1]['prob_high'].mean():.4f}")

    # --- threshold scan ---
    scan_df = scan_thresholds(cell_df)

    thresh_conservative = select_threshold(scan_df, RECALL_CONSERVATIVE)
    thresh_aggressive   = select_threshold(scan_df, RECALL_AGGRESSIVE)
    OPTIMAL_THRESHOLD   = float(thresh_conservative["threshold"])

    print(f"\n  Conservative threshold (recall ≥ {RECALL_CONSERVATIVE*100:.0f}%):")
    print(f"    Threshold   : {thresh_conservative['threshold']:.4f}")
    print(f"    F1          : {thresh_conservative['f1']:.4f}")
    print(f"    Recall      : {thresh_conservative['recall']*100:.1f}%")
    print(f"    Precision   : {thresh_conservative['precision']*100:.1f}%")
    print(f"    Hours flagged: {thresh_conservative['n_flagged']:,}  "
          f"({thresh_conservative['n_flagged']/len(cell_df)*100:.1f}%)")

    print(f"\n  Aggressive threshold (recall ≥ {RECALL_AGGRESSIVE*100:.0f}%):")
    print(f"    Threshold   : {thresh_aggressive['threshold']:.4f}")
    print(f"    F1          : {thresh_aggressive['f1']:.4f}")
    print(f"    Recall      : {thresh_aggressive['recall']*100:.1f}%")
    print(f"    Precision   : {thresh_aggressive['precision']*100:.1f}%")
    print(f"    Hours flagged: {thresh_aggressive['n_flagged']:,}  "
          f"({thresh_aggressive['n_flagged']/len(cell_df)*100:.1f}%)")

    # --- per-year threshold calibration ---
    print(f"\n  Per-year threshold calibration (recall ≥ {RECALL_CONSERVATIVE*100:.0f}%):")
    year_thresholds = {}
    for yr in [2020, 2021]:
        yr_df   = cell_df[cell_df["year"] == yr]
        yr_scan = scan_thresholds(yr_df)
        yr_best = select_threshold(yr_scan, RECALL_CONSERVATIVE)
        year_thresholds[yr] = float(yr_best["threshold"])
        print(f"    {yr}: threshold={yr_best['threshold']:.4f}  "
              f"F1={yr_best['f1']:.4f}  recall={yr_best['recall']*100:.1f}%  "
              f"precision={yr_best['precision']*100:.1f}%")

    print(f"\n  Threshold gap 2021−2020: "
          f"{year_thresholds[2021]-year_thresholds[2020]:+.4f}")
    print(f"  Overall threshold ({OPTIMAL_THRESHOLD:.4f}) retained as primary —")
    print(f"  year-specific calibration improves 2020 but reduces 2021 Uri recall.")

    # --- summary CSV ---
    summary = pd.DataFrame([
        {"approach": "conservative", "min_recall": RECALL_CONSERVATIVE,
         **thresh_conservative[["threshold","f1","recall","precision","n_flagged"]].to_dict()},
        {"approach": "aggressive",   "min_recall": RECALL_AGGRESSIVE,
         **thresh_aggressive[["threshold","f1","recall","precision","n_flagged"]].to_dict()},
        {"approach": "year_2020",    "min_recall": RECALL_CONSERVATIVE,
         "threshold": year_thresholds[2020]},
        {"approach": "year_2021",    "min_recall": RECALL_CONSERVATIVE,
         "threshold": year_thresholds[2021]},
    ])
    summary.to_csv(output_dir / "cell_6_23_threshold_summary.csv", index=False)

    # --- threshold scan plot ---
    plot_threshold_scan(scan_df, OPTIMAL_THRESHOLD, output_dir)

    # --- economic recall scan (requires futures file) ---
    if NG_FUTURES_PATH.exists():
        futures_df = pd.read_csv(NG_FUTURES_PATH)
        futures_df["period"] = pd.to_datetime(
            futures_df[futures_df.columns[0]], errors="coerce"
        ).dt.normalize()
        cell_df["date_key"] = pd.to_datetime(cell_df[DATE_COL]).dt.normalize()
        cell_enriched = cell_df.merge(
            futures_df.rename(columns={futures_df.columns[0]: "period"}),
            left_on="date_key", right_on="period", how="left",
        )
        recall_df = economic_recall_scan(cell_enriched, futures_df)
        plot_recall_scan(recall_df, output_dir)
        recall_df.to_csv(output_dir / "cell_6_23_recall_scan.csv", index=False)
    else:
        print(f"  [SKIP] Futures file not found at {NG_FUTURES_PATH} — "
              f"economic recall scan skipped.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    train_raw = load_period(TRAIN_DIR, "TRAIN 2022–2024")
    test_raw  = load_period(TEST_DIR,  "TEST  2020–2021")

    gas_df    = load_gas(GAS_PATH)
    train_raw = merge_gas(train_raw, gas_df)
    test_raw  = merge_gas(test_raw,  gas_df)

    train_df = label_exposure(engineer_features(train_raw))
    test_df  = label_exposure(engineer_features(test_raw))

    print("\nFitting global XGB-2 model ...")
    pipeline = fit_global_model(train_df)

    # --- Part 1: per-cell performance ---
    print(f"\n{'=' * 65}")
    print("  PART 1 — PER-GRID-CELL MODEL PERFORMANCE")
    print(f"{'=' * 65}")
    results = compute_cell_metrics(pipeline, test_df)

    results_path = OUTPUT_DIR / "grid_cell_performance.csv"
    results[[
        "lat_idx", "lon_idx", "grid_lat", "grid_lon", "Load_Zone",
        "n_total", "n_high", "pct_high",
        "auc_overall", "auc_2020", "auc_2021",
        "precision", "recall", "insufficient_data",
    ]].sort_values(
        ["Load_Zone", "auc_overall"], ascending=[True, False]
    ).to_csv(results_path, index=False, float_format="%.4f")
    print(f"  Saved: {results_path}")

    valid_r = results[~results["insufficient_data"]]
    print(f"\n  Top 10 best-predicted cells:")
    print(valid_r.nlargest(10, "auc_overall")[
        ["lat_idx", "lon_idx", "Load_Zone", "auc_overall", "auc_2020", "auc_2021"]
    ].to_string(index=False, float_format="%.4f"))

    print(f"\n  Bottom 10 hardest-to-predict cells:")
    print(valid_r.nsmallest(10, "auc_overall")[
        ["lat_idx", "lon_idx", "Load_Zone", "auc_overall", "auc_2020", "auc_2021"]
    ].to_string(index=False, float_format="%.4f"))

    print("\nSaving spatial maps ...")
    plot_spatial_maps(results, OUTPUT_DIR)

    # --- Part 2: threshold optimisation ---
    run_threshold_analysis(pipeline, test_df, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
