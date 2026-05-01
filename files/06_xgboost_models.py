"""
06_xgboost_models.py
=====================
XGBoost binary classification models for predicting hourly HIGH / LOW
electricity price exposure in ERCOT.  Two successive global model
specifications are evaluated — XGB-1 and XGB-2 — before the analysis
moves to zone-specific models and financial simulation.

Both models share the same hyperparameter configuration and training setup:
  - Training data  : 2022–2024  (wind_temp_22_24_hourly_FIXED/)
  - Test data      : 2020–2021  (wind_temp_20_21_hourly_FIXED/)
  - Target         : Price Exposure — HIGH if price > year- and zone-specific
                     90th percentile, LOW otherwise  (~10% class balance)
  - CV             : 5-fold stratified cross-validation on training data
  - Class balance  : scale_pos_weight = n_LOW / n_HIGH  (computed from data)

Why XGBoost over logistic regression
--------------------------------------
Logistic regression plateaued at CV AUC ~0.636 across all three
specifications (LR-1 through LR-3).  The bottleneck was architectural, not
the features: a single linear decision boundary cannot express the nonlinear,
regime-dependent compound interactions between wind output, gas prices, and
temperature that drive HIGH price exposure in ERCOT.  XGBoost builds an
ensemble of decision trees sequentially, each correcting residual errors of
the previous ensemble, which allows it to discover these interactions
organically without requiring them to be manually specified in advance.

Model specifications
---------------------
XGB-1  Eight features: raw wind_cf, drought_run_hours, binary extreme_hourly
       demand flag, daily Henry Hub gas spot price ($/MMBtu), and four
       engineered interaction terms using the binary demand flag.
       The binary extreme_hourly flag is a binding constraint: it treats a
       mildly hot afternoon identically to an extreme heat event, discarding
       all information about the degree of temperature deviation.
       Reported CV AUC: 0.7906   Top feature: gas_price_mmbtu (0.271)

XGB-2  Eleven features: the eight features from XGB-1, with the binary
       extreme_hourly flag replaced by a continuous temp_stress signal
       (|tmm_F − 65| / 70, clipped to [0, 1]), plus tmm_F, HDD_hourly, and
       CDD_hourly as additional base features.  All four interaction terms
       are recomputed using temp_stress_norm instead of extreme_hourly,
       preserving the nonlinear relationship between temperature deviation
       and demand intensity that a binary flag cannot express.
       The dominant feature becomes the three-way gas × low_wind × temp_stress
       interaction, confirming that HIGH price exposure in ERCOT is driven by
       the simultaneous confluence of low wind output, elevated gas prices, and
       temperature-driven demand — not any single factor in isolation.
       Reported CV AUC: 0.8038   Top feature: gas_x_low_wind_x_demand (0.355)

Results summary (from paper, Table 6)
---------------------------------------
Model   CV AUC   CV Recall   CV Precision   Test AUC   Top feature
XGB-1   0.7906   0.6728      0.2526         0.5984     gas_price_mmbtu (0.271)
XGB-2   0.8038   0.6790      0.2684         0.6003     gas_x_low_wind_x_demand (0.355)

The train-test performance gap (CV AUC ~0.80 vs test AUC ~0.60) reflects
structural differences between periods rather than overfitting.  The 2022–2024
training window was shaped by post-Ukraine war gas price formation and a mature
40,355 MW wind fleet, while the 2020–2021 test period was characterised by
COVID-era demand suppression, historically low gas prices ($1.33–$6.37/MMBtu
excluding Winter Storm Uri), and a smaller ~29,200 MW installed base —
conditions that materially shifted the wind-gas-price relationships the model
learned to recognise.

Usage
-----
    # Run both models end-to-end (default)
    python files/06_xgboost_models.py

    # Run a single specification
    python files/06_xgboost_models.py --model XGB-1
    python files/06_xgboost_models.py --model XGB-2
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_DIR   = Path("wind_temp_22_24_hourly_FIXED")
TEST_DIR    = Path("wind_temp_20_21_hourly_FIXED")
GAS_PATH    = Path("data/NG_prices.csv")   # daily Henry Hub spot, columns: date, $/MMBtu
OUTPUT_DIR  = Path("results/xgboost")

TARGET_COL = "Price Exposure"
DATE_COL   = "time"

# Shared XGBoost hyperparameters (Table 4 from paper)
XGB_PARAMS = dict(
    n_estimators      = 200,
    max_depth         = 4,
    learning_rate     = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 5,
    eval_metric       = "auc",
    random_state      = 42,
    n_jobs            = -1,
)

CV_FOLDS = 5

# =============================================================================
# DATA LOADING
# =============================================================================

def load_period(directory: Path, label: str) -> pd.DataFrame:
    """Load and concatenate all per-cell hourly CSV files in *directory*."""
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {directory}.\n"
            "Run pipeline steps 01–04 first."
        )
    chunks = []
    for fp in files:
        df = pd.read_csv(fp, parse_dates=[DATE_COL])
        parts = fp.stem.split("_")
        df["lat_idx"] = parts[0]
        df["lon_idx"] = parts[1]
        chunks.append(df)
    combined = pd.concat(chunks, ignore_index=True)
    print(
        f"  [{label}] {len(files)} files, "
        f"{len(combined):,} rows, "
        f"{combined[DATE_COL].dt.year.min()}–{combined[DATE_COL].dt.year.max()}"
    )
    return combined


def load_gas_prices(path: Path) -> pd.DataFrame:
    """
    Load daily Henry Hub spot prices and forward-fill weekends / holidays.
    Expects columns: date (parseable), $/MMBtu (Henry Hub spot in $/MMBtu).
    """
    gas = pd.read_csv(path)
    gas.columns = gas.columns.str.strip()

    # Accept either 'observation_date' or the first column as the date
    date_col = "observation_date" if "observation_date" in gas.columns else gas.columns[0]
    price_col = "$/MMBtu" if "$/MMBtu" in gas.columns else gas.columns[1]

    gas = gas[[date_col, price_col]].copy()
    gas.columns = ["date_key", "gas_price_mmbtu"]
    gas["date_key"] = pd.to_datetime(gas["date_key"]).dt.normalize()
    gas = gas.dropna(subset=["gas_price_mmbtu"]).sort_values("date_key")

    # Replace any zero prices (data artefacts) with forward-filled values
    gas["gas_price_mmbtu"] = gas["gas_price_mmbtu"].replace(0, np.nan).ffill()

    print(
        f"  Gas prices: {len(gas):,} days, "
        f"${gas['gas_price_mmbtu'].min():.2f}–"
        f"${gas['gas_price_mmbtu'].max():.2f}/MMBtu"
    )
    return gas


def merge_gas(df: pd.DataFrame, gas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily gas prices onto hourly grid-cell data by calendar date.
    Forward-fills up to five consecutive non-trading days (weekends/holidays).
    """
    df = df.copy()
    df["date_key"] = pd.to_datetime(df[DATE_COL]).dt.normalize()
    df = df.merge(gas_df, on="date_key", how="left")
    # Forward-fill then back-fill residual gaps
    df = df.sort_values(DATE_COL)
    df["gas_price_mmbtu"] = df["gas_price_mmbtu"].ffill().bfill()
    return df


def label_price_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute year- and load-zone-specific 90th-percentile price thresholds and
    assign HIGH / LOW labels.  Adds a binary 'target' column (1 = HIGH).
    """
    df = df.copy()
    df["year"] = df[DATE_COL].dt.year
    p90 = (
        df.groupby(["Load_Zone", "year"])["price"]
        .quantile(0.90)
        .rename("price_p90")
        .reset_index()
    )
    df = df.merge(p90, on=["Load_Zone", "year"], how="left")
    df[TARGET_COL] = (df["price"] >= df["price_p90"]).map(
        {True: "HIGH", False: "LOW"}
    )
    df["target"] = (df[TARGET_COL] == "HIGH").astype(int)
    return df.drop(columns=["price_p90", "year"], errors="ignore")


# =============================================================================
# FEATURE ENGINEERING  —  one function per model specification
# =============================================================================

def features_xgb1(df: pd.DataFrame) -> pd.DataFrame:
    """
    XGB-1: Eight features using the binary extreme_hourly demand flag.

    Base features: wind_cf, drought_run_hours, extreme_hourly,
                   gas_price_mmbtu
    Interactions (using binary extreme_hourly as the demand multiplier):
      low_wind_x_demand       = (1 − wind_cf) × extreme_hourly
      run_hours_x_demand      = (drought_run_hours / 24) × extreme_hourly
      gas_x_demand            = gas_price_mmbtu × extreme_hourly
      gas_x_low_wind_x_demand = gas_price_mmbtu × (1 − wind_cf) × extreme_hourly

    Limitation: the binary flag treats a mildly hot afternoon and an extreme
    heat event identically, discarding information about the degree of
    temperature deviation that a continuous signal would preserve.
    """
    df = df.copy()
    df["wind_cf"] = df["wind_cf"].clip(0.0, 1.0)
    df["low_wind_x_demand"]       = (1.0 - df["wind_cf"]) * df["extreme_hourly"]
    df["run_hours_x_demand"]      = (df["drought_run_hours"] / 24.0) * df["extreme_hourly"]
    df["gas_x_demand"]            = df["gas_price_mmbtu"] * df["extreme_hourly"]
    df["gas_x_low_wind_x_demand"] = df["gas_price_mmbtu"] * (1.0 - df["wind_cf"]) * df["extreme_hourly"]
    return df


def features_xgb2(df: pd.DataFrame) -> pd.DataFrame:
    """
    XGB-2: Eleven features — final global model specification.

    Replaces the binary extreme_hourly demand flag with a continuous
    temp_stress signal: |tmm_F − 65| / 70, clipped to [0, 1].
    Normalising by 70 (the maximum plausible deviation from the 65°F comfort
    baseline) preserves scale consistency with wind_cf and ensures the
    interaction terms remain bounded.

    Additional base features: tmm_F, HDD_hourly, CDD_hourly

    All four interaction terms are recomputed using temp_stress_norm,
    capturing nonlinear temperature-demand relationships that a binary flag
    discards.  The dominant feature (gas × low_wind × temp_stress) confirms
    that HIGH exposure is driven by simultaneous compound stress across all
    three dimensions — gas price, wind shortfall, and temperature-driven
    demand — rather than any single variable in isolation.
    """
    df = df.copy()
    df["wind_cf"] = df["wind_cf"].clip(0.0, 1.0)

    # Continuous temperature stress signal (normalised, symmetric around 65°F)
    df["temp_stress"]      = np.abs(df["tmm_F"] - 65.0)
    df["temp_stress_norm"] = (df["temp_stress"] / 70.0).clip(0.0, 1.0)

    # Interaction terms (demand multiplier = temp_stress_norm)
    df["low_wind_x_demand"]       = (1.0 - df["wind_cf"]) * df["temp_stress_norm"]
    df["run_hours_x_demand"]      = (df["drought_run_hours"] / 24.0) * df["temp_stress_norm"]
    df["gas_x_demand"]            = df["gas_price_mmbtu"] * df["temp_stress_norm"]
    df["gas_x_low_wind_x_demand"] = (
        df["gas_price_mmbtu"] * (1.0 - df["wind_cf"]) * df["temp_stress_norm"]
    )
    return df


# Feature column lists for each specification
MODEL_FEATURES = {
    "XGB-1": [
        "wind_cf",
        "drought_run_hours",
        "extreme_hourly",
        "gas_price_mmbtu",
        "low_wind_x_demand",
        "run_hours_x_demand",
        "gas_x_demand",
        "gas_x_low_wind_x_demand",
    ],
    "XGB-2": [
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
    ],
}

FEATURE_ENGINEERS = {
    "XGB-1": features_xgb1,
    "XGB-2": features_xgb2,
}

# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

def run_model(
    model_id: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Train XGBoost model *model_id*, run 5-fold CV on training data, evaluate
    on the held-out test set, save plots and feature importances.
    Returns a dict of summary metrics.
    """
    print(f"\n{'=' * 65}")
    print(f"  {model_id}")
    print(f"{'=' * 65}")

    engineer = FEATURE_ENGINEERS[model_id]
    features = MODEL_FEATURES[model_id]

    train_eng = engineer(train_df)
    test_eng  = engineer(test_df)

    missing = [c for c in features if c not in train_eng.columns]
    if missing:
        raise ValueError(f"[{model_id}] Missing features in training data: {missing}")

    X_train = train_eng[features].copy()
    y_train = train_eng["target"].copy()
    X_test  = test_eng[features].copy()
    y_test  = test_eng["target"].copy()

    mask_tr = X_train.notna().all(axis=1)
    mask_te = X_test.notna().all(axis=1)
    X_train, y_train = X_train[mask_tr], y_train[mask_tr]
    X_test,  y_test  = X_test[mask_te],  y_test[mask_te]

    # scale_pos_weight = n_LOW / n_HIGH (computed from training data)
    spw = round((y_train == 0).sum() / (y_train == 1).sum())

    print(f"  Training rows      : {len(X_train):,}   HIGH rate: {y_train.mean()*100:.1f}%")
    print(f"  Test rows          : {len(X_test):,}   HIGH rate: {y_test.mean()*100:.1f}%")
    print(f"  scale_pos_weight   : {spw}")
    print(f"  Features ({len(features)}): {features}")

    # XGBoost does not require scaling but we include it for consistency
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)),
    ])

    # --- 5-fold stratified cross-validation ---
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_auc       = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc",   n_jobs=-1)
    cv_precision = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="precision", n_jobs=-1)
    cv_recall    = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="recall",    n_jobs=-1)

    print(f"\n  Cross-validation results (train, 2022–2024):")
    for i, s in enumerate(cv_auc, 1):
        print(f"    Fold {i}: {s:.4f}")
    print(f"    AUC-ROC   mean={cv_auc.mean():.4f}   std={cv_auc.std():.4f}")
    print(f"    Precision mean={cv_precision.mean():.4f}   std={cv_precision.std():.4f}")
    print(f"    Recall    mean={cv_recall.mean():.4f}   std={cv_recall.std():.4f}")

    # --- fit final model on all training data ---
    pipeline.fit(X_train, y_train)
    xgb_model = pipeline.named_steps["xgb"]

    # --- feature importances ---
    importance_df = pd.DataFrame({
        "feature":    features,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\n  Feature importances:")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] * 40)
        print(f"    {row['feature']:35s}: {row['importance']:.4f}  {bar}")

    # --- test evaluation ---
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    test_auc    = roc_auc_score(y_test, y_pred_prob)

    print(f"\n  Test AUC-ROC (2020–2021): {test_auc:.4f}")
    print(f"  Train-test gap           : {cv_auc.mean() - test_auc:.4f}")
    print(f"\n  Classification report (test):")
    print(classification_report(y_test, y_pred, target_names=["LOW", "HIGH"]))

    # --- year-by-year test breakdown ---
    test_eng_valid = test_eng[mask_te].copy().reset_index(drop=True)
    X_test_r = X_test.reset_index(drop=True)
    y_test_r = y_test.reset_index(drop=True)
    print("  Test AUC by year:")
    for yr in [2020, 2021]:
        yr_mask = test_eng_valid[DATE_COL].dt.year == yr
        if yr_mask.sum() == 0:
            continue
        auc_yr = roc_auc_score(
            y_test_r[yr_mask],
            pipeline.predict_proba(X_test_r[yr_mask])[:, 1],
        )
        print(f"    {yr}: AUC={auc_yr:.4f}   n={yr_mask.sum():,}")

    # --- outputs ---
    model_out = output_dir / model_id.lower().replace("-", "_")
    model_out.mkdir(parents=True, exist_ok=True)

    # save feature importances CSV
    importance_df.to_csv(model_out / "feature_importances.csv", index=False)

    # evaluation plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{model_id} — Test Evaluation (2020–2021)", fontsize=13)

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["LOW", "HIGH"]).plot(
        ax=axes[0], colorbar=False
    )
    axes[0].set_title("Confusion Matrix")

    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "k--", label="Random (AUC=0.50)")
    axes[1].set_title(f"ROC Curve  (AUC={test_auc:.3f})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(model_out / "test_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # feature importance bar chart
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1],
        color="steelblue", alpha=0.85,
    )
    ax2.set_xlabel("Feature Importance")
    ax2.set_title(f"{model_id} — Feature Importances")
    plt.tight_layout()
    plt.savefig(model_out / "feature_importances.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Outputs saved to: {model_out}/")

    return {
        "model"         : model_id,
        "n_features"    : len(features),
        "cv_auc_mean"   : round(cv_auc.mean(),       4),
        "cv_auc_std"    : round(cv_auc.std(),         4),
        "cv_precision"  : round(cv_precision.mean(),  4),
        "cv_recall"     : round(cv_recall.mean(),     4),
        "test_auc"      : round(test_auc,             4),
        "train_test_gap": round(cv_auc.mean() - test_auc, 4),
        "top_feature"   : importance_df.iloc[0]["feature"],
        "top_importance": round(importance_df.iloc[0]["importance"], 3),
    }


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison(results: list[dict]) -> None:
    """Print a summary comparison table against the paper's reported values."""
    print(f"\n{'=' * 72}")
    print("  XGBOOST MODEL COMPARISON")
    print(f"{'=' * 72}")
    header = (
        f"  {'Model':<8} {'CV AUC':>8} {'CV Recall':>10} "
        f"{'CV Prec':>8} {'Test AUC':>9} {'Top Feature':<35}"
    )
    print(header)
    print("  " + "-" * 70)
    for r in results:
        print(
            f"  {r['model']:<8} "
            f"{r['cv_auc_mean']:>8.4f} "
            f"{r['cv_recall']:>10.4f} "
            f"{r['cv_precision']:>8.4f} "
            f"{r['test_auc']:>9.4f} "
            f"{r['top_feature']:<35}"
        )
    print()
    print("  Paper-reported values (Table 6):")
    print("  Model    CV AUC   CV Recall  CV Prec   Test AUC  Top feature")
    print("  XGB-1    0.7906   0.6728     0.2526    0.5984    gas_price_mmbtu (0.271)")
    print("  XGB-2    0.8038   0.6790     0.2684    0.6003    gas_x_low_wind_x_demand (0.355)")
    print()
    print("  Train-test gap note:")
    print("  The ~0.20 gap between CV AUC (~0.80) and test AUC (~0.60) reflects")
    print("  structural differences between periods, not overfitting.  The 2022–2024")
    print("  training window was shaped by post-Ukraine war gas prices and a 40,355 MW")
    print("  wind fleet; the 2020–2021 test period had COVID-era demand suppression,")
    print("  gas prices of $1.33–$6.37/MMBtu (ex-Uri), and ~29,200 MW installed.")
    print("  Test AUC improves from 0.546 in 2020 to 0.669 in 2021 as conditions")
    print("  began normalising.")
    print(f"{'=' * 72}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=["XGB-1", "XGB-2", "all"],
        default="all",
        help="Which model(s) to run (default: all).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    train_raw = load_period(TRAIN_DIR, "TRAIN 2022–2024")
    test_raw  = load_period(TEST_DIR,  "TEST  2020–2021")

    print("\nLoading natural gas prices ...")
    gas_df = load_gas_prices(GAS_PATH)

    print("\nMerging gas prices ...")
    train_raw = merge_gas(train_raw, gas_df)
    test_raw  = merge_gas(test_raw,  gas_df)

    print("\nLabelling price exposure (year- and zone-specific P90) ...")
    train_df = label_price_exposure(train_raw)
    test_df  = label_price_exposure(test_raw)

    models_to_run = ["XGB-1", "XGB-2"] if args.model == "all" else [args.model]

    results = []
    for model_id in models_to_run:
        metrics = run_model(model_id, train_df, test_df, OUTPUT_DIR)
        results.append(metrics)

    if len(results) > 1:
        print_comparison(results)

    results_path = OUTPUT_DIR / "xgb_comparison.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
