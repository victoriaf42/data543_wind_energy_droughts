"""
07_xgboost_zone_models.py
==========================
Zone-specific XGBoost analysis for ERCOT price exposure prediction.

Extends the global XGBoost models (06_xgboost_models.py) by evaluating
performance separately for each of the four ERCOT load zones with installed
wind capacity.  Two complementary analyses are run:

  Analysis A — Global model evaluated per zone
  ---------------------------------------------
  The already-fitted XGB-2 global model is applied to each zone's test data
  separately.  This reveals where the global model's compound stress signal
  fires most reliably and where it underperforms, without any retraining.

  Analysis B — Separate XGBoost model trained per zone
  ------------------------------------------------------
  One XGBoost model is trained exclusively on each zone's 2022–2024 data
  using the identical 11-feature XGB-2 specification and hyperparameters.
  This is the more rigorous evaluation: it allows zone-specific feature
  importance rankings to emerge from zone-specific price formation dynamics.

Key findings (from paper, Table 7)
------------------------------------
Zone-specific models outperform the global specification in three of the
four load zones (LZ_HOUSTON, LZ_NORTH, LZ_SOUTH), while the global model
marginally outperforms in LZ_WEST — the zone with the largest installed
capacity and most complex wind resource dynamics.

Feature importance rankings differ meaningfully across zones:
  LZ_SOUTH    gas_x_low_wind_x_demand dominates (importance 0.500) —
              transmission bottlenecks mean local wind shortfalls translate
              directly into local price stress
  LZ_NORTH    gas_x_low_wind_x_demand dominant (0.386), consistent with
              compound instantaneous shortfalls
  LZ_HOUSTON  gas_x_demand leads (0.310) — no local wind; price formation
              is driven by the cost of imported generation under demand stress
  LZ_WEST     drought_run_hours leads (0.203) — duration rather than
              instantaneous severity predicts price stress, reflecting limited
              local gas backup and intertie capacity constraints

The 2020-vs-2021 AUC improvement is widespread across all zones, confirming
that the train-test gap is driven by COVID-era distributional shift rather
than zone-specific model failure.

Results summary (from paper, Table 7)
---------------------------------------
Zone         Grid  Train   Train   CV AUC       Test AUC  Test AUC  Test       Test      Top
             cells rows    HIGH%   ± std        overall   2020      2021       Precision Recall  feature
LZ_HOUSTON   1     25,966  10.1%   0.884±0.008  0.626     0.589     0.682      0.204     0.284   gas_x_demand
LZ_NORTH     15    387,930 10.2%   0.898±0.001  0.644     0.593     0.724      0.262     0.232   gas_x_low_wind_x_demand
LZ_SOUTH     24    616,032 10.2%   0.882±0.002  0.675     0.626     0.749      0.281     0.316   gas_x_low_wind_x_demand
LZ_WEST      83    1,998,142 10.9% 0.799±0.001  0.565     0.522     0.629      0.127     0.319   drought_run_hours

Usage
-----
    python files/07_xgboost_zone_models.py

This script loads data, fits the global XGB-2 model, then runs both zone
analyses end-to-end.  It does not require the global model to be pre-fitted
in memory.

Outputs saved to results/xgboost_zones/:
    global_model_auc_by_zone.png        — Option A AUC + 2020/2021 by zone
    global_vs_zone_model_auc.png        — Global vs per-zone AUC comparison
    zone_feature_importance_heatmap.png — Feature importance heatmap by zone
    zone_results.csv                    — Full metrics table
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION  —  must match 06_xgboost_models.py
# =============================================================================

TRAIN_DIR  = Path("wind_temp_22_24_hourly_FIXED")
TEST_DIR   = Path("wind_temp_20_21_hourly_FIXED")
GAS_PATH   = Path("data/NG_prices.csv")
OUTPUT_DIR = Path("results/xgboost_zones")

TARGET_COL = "Price Exposure"
DATE_COL   = "time"

ZONES = ["LZ_HOUSTON", "LZ_NORTH", "LZ_SOUTH", "LZ_WEST"]

# XGB-2 hyperparameters (Table 4 from paper — identical to global model)
XGB_PARAMS = dict(
    n_estimators     = 200,
    max_depth        = 4,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    eval_metric      = "auc",
    random_state     = 42,
    n_jobs           = -1,
)
CV_FOLDS = 5

# XGB-2 feature set (11 features — final global specification)
FEATURE_COLS = [
    "wind_cf",
    "drought_run_hours",
    "tmm_F",
    "HDD_hourly",
    "CDD_hourly",
    "temp_stress",
    "gas_price_mmbtu",
    "low_wind_x_demand",
    "run_hours_x_demand",
    "gas_x_demand",
    "gas_x_low_wind_x_demand",
]

# =============================================================================
# DATA LOADING  (shared with 06_xgboost_models.py)
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
    print(
        f"  [{label}] {len(files)} files, {len(combined):,} rows, "
        f"{combined[DATE_COL].dt.year.min()}–{combined[DATE_COL].dt.year.max()}"
    )
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
        .quantile(0.90)
        .rename("price_p90")
        .reset_index()
    )
    df = df.merge(p90, on=["Load_Zone", "year"], how="left")
    df[TARGET_COL] = (df["price"] >= df["price_p90"]).map({True: "HIGH", False: "LOW"})
    df["target"] = (df[TARGET_COL] == "HIGH").astype(int)
    return df.drop(columns=["price_p90", "year"], errors="ignore")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """XGB-2 feature engineering — identical to 06_xgboost_models.py."""
    df = df.copy()
    df["wind_cf"]       = df["wind_cf"].clip(0.0, 1.0)
    df["temp_stress"]   = np.abs(df["tmm_F"] - 65.0)
    df["temp_stress_norm"] = (df["temp_stress"] / 70.0).clip(0.0, 1.0)
    df["low_wind_x_demand"]       = (1.0 - df["wind_cf"]) * df["temp_stress_norm"]
    df["run_hours_x_demand"]      = (df["drought_run_hours"] / 24.0) * df["temp_stress_norm"]
    df["gas_x_demand"]            = df["gas_price_mmbtu"] * df["temp_stress_norm"]
    df["gas_x_low_wind_x_demand"] = (
        df["gas_price_mmbtu"] * (1.0 - df["wind_cf"]) * df["temp_stress_norm"]
    )
    return df


def build_pipeline(spw: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("xgb",   XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)),
    ])

# =============================================================================
# ANALYSIS A — GLOBAL MODEL EVALUATED PER ZONE
# =============================================================================

def evaluate_global_by_zone(
    global_pipeline: Pipeline,
    test_df: pd.DataFrame,
    valid_mask: pd.Series,
) -> pd.DataFrame:
    """
    Apply the already-fitted global XGB-2 model to each zone's test data
    separately.  Returns a DataFrame with per-zone metrics including
    year-specific AUC for 2020 and 2021.
    """
    print(f"\n{'=' * 65}")
    print("  ANALYSIS A — GLOBAL MODEL EVALUATED PER ZONE")
    print(f"{'=' * 65}")

    test_valid = test_df[valid_mask].copy().reset_index(drop=True)
    X_test = test_valid[FEATURE_COLS]
    y_test = test_valid["target"]

    records = []
    for zone in ZONES:
        mask = test_valid["Load_Zone"] == zone
        X_z  = X_test[mask]
        y_z  = y_test[mask]

        if y_z.sum() < 10:
            print(f"  {zone}: skipped — insufficient HIGH samples")
            continue

        prob_z = global_pipeline.predict_proba(X_z)[:, 1]
        pred_z = global_pipeline.predict(X_z)
        auc_z  = roc_auc_score(y_z, prob_z)
        prec_z = precision_score(y_z, pred_z, zero_division=0)
        rec_z  = recall_score(y_z, pred_z, zero_division=0)

        # year-specific AUC
        auc_by_year = {}
        for yr in [2020, 2021]:
            yr_mask = test_valid.loc[mask, DATE_COL].dt.year == yr
            X_zy = X_z[yr_mask.values]
            y_zy = y_z[yr_mask.values]
            if len(y_zy) > 0 and y_zy.sum() >= 5:
                auc_by_year[yr] = roc_auc_score(
                    y_zy, global_pipeline.predict_proba(X_zy)[:, 1]
                )
            else:
                auc_by_year[yr] = np.nan

        print(
            f"  {zone:<12}  AUC={auc_z:.4f}  "
            f"2020={auc_by_year[2020]:.4f}  2021={auc_by_year[2021]:.4f}  "
            f"Prec={prec_z:.4f}  Rec={rec_z:.4f}"
        )
        records.append({
            "Load_Zone"       : zone,
            "test_rows"       : int(mask.sum()),
            "high_pct"        : round(y_z.mean() * 100, 1),
            "global_auc"      : round(auc_z, 4),
            "global_auc_2020" : round(auc_by_year[2020], 4) if not np.isnan(auc_by_year[2020]) else np.nan,
            "global_auc_2021" : round(auc_by_year[2021], 4) if not np.isnan(auc_by_year[2021]) else np.nan,
            "global_precision": round(prec_z, 4),
            "global_recall"   : round(rec_z, 4),
        })

    return pd.DataFrame(records)


# =============================================================================
# ANALYSIS B — SEPARATE XGBoost MODEL PER ZONE
# =============================================================================

def train_zone_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    valid_mask: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    """
    Train one XGBoost model per zone on 2022–2024 zone data using the
    identical XGB-2 11-feature specification and hyperparameters.
    Returns (zone_pipelines dict, results DataFrame).
    """
    print(f"\n{'=' * 65}")
    print("  ANALYSIS B — SEPARATE XGBOOST MODEL PER ZONE")
    print(f"{'=' * 65}")

    test_valid = test_df[valid_mask].copy().reset_index(drop=True)
    X_test_all = test_valid[FEATURE_COLS]
    y_test_all = test_valid["target"]

    zone_pipelines = {}
    records = []

    for zone in ZONES:
        print(f"\n  --- {zone} ---")

        # --- training data for this zone ---
        z_train = train_df[train_df["Load_Zone"] == zone].copy()
        X_zt    = z_train[FEATURE_COLS]
        y_zt    = z_train["target"]
        valid_tr = X_zt.notna().all(axis=1)
        X_zt, y_zt = X_zt[valid_tr], y_zt[valid_tr]

        n_high = int(y_zt.sum())
        n_low  = int((y_zt == 0).sum())
        spw    = round(n_low / max(n_high, 1))

        print(f"    Train rows       : {len(y_zt):,}  ({y_zt.mean()*100:.1f}% HIGH)")
        print(f"    scale_pos_weight : {spw}")

        pipeline = build_pipeline(spw)

        # --- 5-fold CV ---
        if n_high >= CV_FOLDS * 2:
            cv       = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
            cv_aucs  = cross_val_score(pipeline, X_zt, y_zt,
                                       cv=cv, scoring="roc_auc", n_jobs=-1)
            cv_prec  = cross_val_score(pipeline, X_zt, y_zt,
                                       cv=cv, scoring="precision", n_jobs=-1)
            cv_rec   = cross_val_score(pipeline, X_zt, y_zt,
                                       cv=cv, scoring="recall", n_jobs=-1)
            print(
                f"    CV AUC           : {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}  "
                f"Prec={cv_prec.mean():.4f}  Rec={cv_rec.mean():.4f}"
            )
        else:
            cv_aucs = np.array([np.nan])
            cv_prec = np.array([np.nan])
            cv_rec  = np.array([np.nan])
            print("    CV AUC           : skipped (insufficient HIGH samples)")

        # --- fit final zone model ---
        pipeline.fit(X_zt, y_zt)
        zone_pipelines[zone] = pipeline

        # --- feature importances ---
        imp_df = pd.DataFrame({
            "feature"   : FEATURE_COLS,
            "importance": pipeline.named_steps["xgb"].feature_importances_,
        }).sort_values("importance", ascending=False)
        top  = imp_df.iloc[0]
        top2 = imp_df.iloc[1]
        print(f"    Top feature      : '{top['feature']}' ({top['importance']:.3f})")
        print(f"    2nd feature      : '{top2['feature']}' ({top2['importance']:.3f})")

        # --- test evaluation ---
        mask_te = test_valid["Load_Zone"] == zone
        X_zte   = X_test_all[mask_te]
        y_zte   = y_test_all[mask_te]

        if len(y_zte) == 0 or y_zte.sum() < 5:
            print("    Test AUC         : skipped (insufficient test samples)")
            continue

        prob_te = pipeline.predict_proba(X_zte)[:, 1]
        pred_te = pipeline.predict(X_zte)
        auc_te  = roc_auc_score(y_zte, prob_te)
        prec_te = precision_score(y_zte, pred_te, zero_division=0)
        rec_te  = recall_score(y_zte, pred_te, zero_division=0)
        print(f"    Test AUC         : {auc_te:.4f}  Prec={prec_te:.4f}  Rec={rec_te:.4f}")

        # year-specific test AUC
        auc_by_year = {}
        for yr in [2020, 2021]:
            yr_mask = test_valid.loc[mask_te, DATE_COL].dt.year.values == yr
            X_zy = X_zte[yr_mask]
            y_zy = y_zte[yr_mask]
            if len(y_zy) > 0 and y_zy.sum() >= 5:
                auc_by_year[yr] = round(
                    roc_auc_score(y_zy, pipeline.predict_proba(X_zy)[:, 1]), 4
                )
                print(f"    Test AUC {yr}    : {auc_by_year[yr]:.4f}")
            else:
                auc_by_year[yr] = np.nan

        records.append({
            "Load_Zone"      : zone,
            "grid_cells"     : len(z_train[["lat_idx", "lon_idx"]].drop_duplicates()),
            "train_rows"     : len(y_zt),
            "train_high_pct" : round(y_zt.mean() * 100, 1),
            "cv_auc"         : round(cv_aucs.mean(), 4),
            "cv_auc_std"     : round(cv_aucs.std(),  4),
            "cv_precision"   : round(cv_prec.mean(), 4),
            "cv_recall"      : round(cv_rec.mean(),  4),
            "test_auc"       : round(auc_te,   4),
            "test_auc_2020"  : auc_by_year.get(2020, np.nan),
            "test_auc_2021"  : auc_by_year.get(2021, np.nan),
            "test_precision" : round(prec_te, 4),
            "test_recall"    : round(rec_te,  4),
            "top_feature"    : top["feature"],
            "top_importance" : round(top["importance"], 3),
            "second_feature" : top2["feature"],
            "second_importance": round(top2["importance"], 3),
            "importances"    : imp_df.set_index("feature")["importance"].to_dict(),
        })

    return zone_pipelines, pd.DataFrame(records)


# =============================================================================
# PLOTS
# =============================================================================

def plot_global_auc_by_zone(global_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Three-panel figure: overall AUC by zone, 2020 vs 2021 AUC by zone,
    and HIGH exposure rate by zone.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Global Model (XGB-2) Performance by Load Zone — Test 2020–2021")

    # overall AUC
    axes[0].bar(global_df["Load_Zone"], global_df["global_auc"],
                color="steelblue", alpha=0.85)
    axes[0].axhline(0.5, color="red", linestyle="--", label="Random (0.50)")
    axes[0].axhline(global_df["global_auc"].mean(), color="orange", linestyle="--",
                    label=f"Mean ({global_df['global_auc'].mean():.3f})")
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("Overall Test AUC by Zone")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)

    # 2020 vs 2021
    x     = np.arange(len(global_df))
    width = 0.35
    axes[1].bar(x - width / 2, global_df["global_auc_2020"], width,
                label="2020", color="steelblue", alpha=0.85)
    axes[1].bar(x + width / 2, global_df["global_auc_2021"], width,
                label="2021", color="coral", alpha=0.85)
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(global_df["Load_Zone"])
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("Test AUC by Zone: 2020 vs 2021")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    # HIGH exposure rate
    axes[2].bar(global_df["Load_Zone"], global_df["high_pct"],
                color="coral", alpha=0.85)
    axes[2].axhline(10, color="gray", linestyle="--", label="Expected ~10%")
    axes[2].set_ylabel("% HIGH Exposure Hours")
    axes[2].set_title("HIGH Exposure Rate by Zone")
    axes[2].legend()

    plt.tight_layout()
    out = output_dir / "global_model_auc_by_zone.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_global_vs_zone(
    global_df: pd.DataFrame,
    zone_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Side-by-side comparison of global vs per-zone model test AUC."""
    merged = global_df[["Load_Zone", "global_auc"]].merge(
        zone_df[["Load_Zone", "test_auc"]], on="Load_Zone"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(merged))
    width = 0.35
    ax.bar(x - width / 2, merged["global_auc"], width,
           label="Global Model (XGB-2)", color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, merged["test_auc"],   width,
           label="Per-Zone Model",        color="coral",     alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["Load_Zone"])
    ax.set_ylabel("Test AUC-ROC (2020–2021)")
    ax.set_title("Global vs Per-Zone XGBoost AUC by Load Zone")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    out = output_dir / "global_vs_zone_model_auc.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_feature_importance_heatmap(
    zone_pipelines: dict,
    output_dir: Path,
) -> None:
    """Feature importance heatmap — each column is a load zone."""
    imp_matrix = pd.DataFrame(
        {zone: pipe.named_steps["xgb"].feature_importances_
         for zone, pipe in zone_pipelines.items()},
        index=FEATURE_COLS,
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(imp_matrix.values, aspect="auto", cmap="Blues", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(imp_matrix.columns)))
    ax.set_xticklabels(imp_matrix.columns, fontsize=10)
    ax.set_yticks(range(len(imp_matrix.index)))
    ax.set_yticklabels(imp_matrix.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="Feature Importance")
    ax.set_title("Feature Importance by Load Zone — Per-Zone XGBoost Models", fontsize=12)

    for i in range(len(imp_matrix.index)):
        for j in range(len(imp_matrix.columns)):
            v = imp_matrix.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if v > 0.25 else "black")

    plt.tight_layout()
    out = output_dir / "zone_feature_importance_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def print_comparison_table(
    global_df: pd.DataFrame,
    zone_df: pd.DataFrame,
) -> None:
    """Print the paper's Table 7 comparison."""
    merged = global_df[["Load_Zone", "global_auc", "global_auc_2020", "global_auc_2021"]].merge(
        zone_df[["Load_Zone", "cv_auc", "cv_auc_std", "test_auc",
                 "test_auc_2020", "test_auc_2021",
                 "test_precision", "test_recall", "top_feature"]],
        on="Load_Zone",
    )
    merged["auc_improvement"] = (merged["test_auc"] - merged["global_auc"]).round(4)

    print(f"\n{'=' * 72}")
    print("  ZONE MODEL COMPARISON (Table 7 from paper)")
    print(f"{'=' * 72}")
    print(
        f"  {'Zone':<12} {'CV AUC':>8} {'Test AUC':>9} {'vs Global':>10} "
        f"{'2020':>7} {'2021':>7}  Top feature"
    )
    print("  " + "-" * 70)
    for _, r in merged.iterrows():
        direction = "▲" if r["auc_improvement"] > 0 else "▼"
        print(
            f"  {r['Load_Zone']:<12} "
            f"{r['cv_auc']:>8.4f} "
            f"{r['test_auc']:>9.4f} "
            f"{direction}{abs(r['auc_improvement']):>8.4f} "
            f"{r['test_auc_2020']:>7.4f} "
            f"{r['test_auc_2021']:>7.4f}  "
            f"{r['top_feature']}"
        )
    improved = (merged["auc_improvement"] > 0).sum()
    print(f"\n  Per-zone models outperformed global in {improved}/{len(merged)} zones.")
    print()
    print("  Top and 2nd features by zone:")
    for _, r in merged.iterrows():
        top2 = zone_df.loc[zone_df["Load_Zone"] == r["Load_Zone"], "second_feature"]
        top2_str = top2.values[0] if len(top2) > 0 else "—"
        print(f"    {r['Load_Zone']:<12}: 1st={r['top_feature']}  2nd={top2_str}")
    print()
    print("  Key findings:")
    lz_west = merged[merged["Load_Zone"] == "LZ_WEST"]
    others  = merged[merged["Load_Zone"] != "LZ_WEST"]
    if len(lz_west) > 0 and len(others) > 0:
        west_auc   = lz_west["test_auc"].values[0]
        others_mean = others["test_auc"].mean()
        print(f"    LZ_WEST per-zone AUC   : {west_auc:.4f}")
        print(f"    Other zones mean AUC   : {others_mean:.4f}")
        if west_auc < others_mean:
            print(
                "    LZ_WEST is the hardest zone to predict despite holding 69% of\n"
                "    installed wind capacity.  Duration (drought_run_hours) rather than\n"
                "    instantaneous CF drives its price signal — reflecting limited local\n"
                "    gas backup and the progressive exhaustion of cheaper reserves over\n"
                "    multi-day low-wind periods."
            )
        else:
            print("    LZ_WEST is more predictable — wind concentration amplifies compound signal.")
    print()
    print("  Paper-reported values (Table 7):")
    print("  Zone         CV AUC       Test AUC  2020    2021    Top feature")
    print("  LZ_HOUSTON   0.884±0.008  0.626     0.589   0.682   gas_x_demand")
    print("  LZ_NORTH     0.898±0.001  0.644     0.593   0.724   gas_x_low_wind_x_demand")
    print("  LZ_SOUTH     0.882±0.002  0.675     0.626   0.749   gas_x_low_wind_x_demand")
    print("  LZ_WEST      0.799±0.001  0.565     0.522   0.629   drought_run_hours")
    print(f"{'=' * 72}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- load and prepare data ---
    print("Loading data ...")
    train_raw = load_period(TRAIN_DIR, "TRAIN 2022–2024")
    test_raw  = load_period(TEST_DIR,  "TEST  2020–2021")

    gas_df    = load_gas(GAS_PATH)
    train_raw = merge_gas(train_raw, gas_df)
    test_raw  = merge_gas(test_raw,  gas_df)

    train_df = label_exposure(engineer_features(train_raw))
    test_df  = label_exposure(engineer_features(test_raw))

    X_train_all = train_df[FEATURE_COLS]
    y_train_all = train_df["target"]
    valid_train = X_train_all.notna().all(axis=1)
    X_train_all = X_train_all[valid_train]
    y_train_all = y_train_all[valid_train]

    X_test_all  = test_df[FEATURE_COLS]
    y_test_all  = test_df["target"]
    valid_test  = X_test_all.notna().all(axis=1)

    # --- fit global XGB-2 model ---
    print("\nFitting global XGB-2 model on all training data ...")
    spw_global = round((y_train_all == 0).sum() / (y_train_all == 1).sum())
    global_pipeline = build_pipeline(spw_global)
    global_pipeline.fit(X_train_all, y_train_all)

    # --- Analysis A: global model per zone ---
    global_df = evaluate_global_by_zone(global_pipeline, test_df, valid_test)

    # --- Analysis B: per-zone models ---
    zone_pipelines, zone_df = train_zone_models(train_df, test_df, valid_test)

    # --- comparison table ---
    print_comparison_table(global_df, zone_df)

    # --- plots ---
    print("\nSaving plots ...")
    plot_global_auc_by_zone(global_df, OUTPUT_DIR)
    plot_global_vs_zone(global_df, zone_df, OUTPUT_DIR)
    plot_feature_importance_heatmap(zone_pipelines, OUTPUT_DIR)

    # --- save results CSV ---
    results_path = OUTPUT_DIR / "zone_results.csv"
    # drop the importances dict before saving
    zone_df.drop(columns=["importances"], errors="ignore").to_csv(results_path, index=False)
    print(f"  Saved: {results_path}")


if __name__ == "__main__":
    main()
