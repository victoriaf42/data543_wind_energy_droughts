"""
05_logistic_regression_models.py
==================================
Logistic regression baseline for predicting hourly HIGH / LOW electricity
price exposure in ERCOT.  Three successive model specifications are evaluated,
each documented in the paper as LR-1, LR-2, and LR-3.

All three models share the same setup:
  - Training data  : 2022–2024  (wind_temp_22_24_hourly_FIXED/)
  - Test data      : 2020–2021  (wind_temp_20_21_hourly_FIXED/)
  - Target         : Price Exposure — HIGH if price > year- and zone-specific
                     90th percentile, LOW otherwise  (~10% class balance)
  - CV             : 5-fold stratified cross-validation on training data
  - Class weight   : balanced  (compensates for ~10:1 LOW:HIGH imbalance)

Model specifications
--------------------
LR-1  Calendar-day drought category (ordinal 0–3) + binary extreme-temp flag
      + one interaction term.
      Limitation: calendar-day label averages over intraday variation;
      too coarse a signal for hourly price prediction.
      Reported CV AUC: 0.583

LR-2  Hourly drought category (ordinal 0–3) + consecutive drought run hours
      + demand interaction terms.
      Improvement over LR-1 by moving to hourly drought labels.
      Limitation: ordinal encoding discards information at category boundaries
      (CF = 0.149 and CF = 0.051 receive the same label despite different
      levels of grid stress).
      Reported CV AUC: 0.640

LR-3  Raw wind CF (continuous) + drought run hours (continuous) + binary
      extreme-temperature demand flag + two interaction terms.
      Final logistic regression specification — avoids binning artefacts.
      Limitation: AUC plateaus here because logistic regression cannot
      express the nonlinear, regime-dependent compound interactions between
      wind output, gas price, and temperature that drive HIGH price exposure.
      This motivates the shift to XGBoost in the next stage.
      Reported CV AUC: 0.636

Results summary (from paper, Table 3)
--------------------------------------
Model   CV AUC   CV Recall   CV Precision
LR-1    0.583    0.536       0.134
LR-2    0.640    0.713       0.145
LR-3    0.636    0.697       0.145

Usage
-----
    python files/05_logistic_regression_models.py

    To run a single model specification:
    python files/05_logistic_regression_models.py --model LR-1
    python files/05_logistic_regression_models.py --model LR-2
    python files/05_logistic_regression_models.py --model LR-3
"""

import argparse
import glob
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_DIR  = Path("wind_temp_22_24_hourly_FIXED")
TEST_DIR   = Path("wind_temp_20_21_hourly_FIXED")
OUTPUT_DIR = Path("results/logistic_regression")

TARGET_COL = "Price Exposure"
DATE_COL   = "time"

# Shared hyperparameters across all three LR models
LR_PARAMS = dict(
    penalty      = "l2",
    solver       = "lbfgs",
    max_iter     = 1000,
    class_weight = "balanced",
    random_state = 42,
    C            = 1.0,
)
CV_FOLDS = 5

# =============================================================================
# DATA LOADING
# =============================================================================

def load_period(directory: Path, label: str) -> pd.DataFrame:
    """
    Load all per-cell hourly CSV files from *directory* and concatenate
    into a single DataFrame.  Grid-cell indices are parsed from filenames.
    """
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {directory}.\n"
            "Run the earlier pipeline steps first."
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


def label_price_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add / recompute the Price Exposure binary label using year- and
    load-zone-specific 90th percentile thresholds.
    HIGH = top decile within each year × load zone combination.
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

def features_lr1(df: pd.DataFrame) -> pd.DataFrame:
    """
    LR-1: Calendar-day drought category (ordinal 0–3) + binary extreme-
    temperature flag + one interaction term.

    Drought ordinal encoding:
        0 = NO_DROUGHT
        1 = MILD
        2 = MODERATE
        3 = SEVERE

    The calendar-day label is the same for every hour within a day, which
    averages over intraday variation and loses the timing information most
    relevant to real-time price formation.
    """
    df = df.copy()
    category_map = {"NO_DROUGHT": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}
    df["drought_category_numeric"] = (
        df["daily_drought_category"]
        .map(category_map)
        .fillna(0)
        .astype(int)
    )
    # Single interaction: ordinal drought severity × binary demand flag
    df["drought_x_demand"] = (
        df["drought_category_numeric"] * df["extreme_hourly"]
    )
    return df


def features_lr2(df: pd.DataFrame) -> pd.DataFrame:
    """
    LR-2: Hourly drought category (ordinal 0–3) + consecutive drought run
    hours + demand interaction terms.

    Moves to hourly drought labels, capturing timing more precisely than LR-1.
    Still uses an ordinal encoding that compresses CF depth within each tier.
    """
    df = df.copy()
    category_map = {"NO_DROUGHT": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}
    df["hourly_drought_category_numeric"] = (
        df["hourly_drought_category"]
        .map(category_map)
        .fillna(0)
        .astype(int)
    )
    # Interaction 1: hourly drought severity × demand
    df["drought_x_demand"] = (
        df["hourly_drought_category_numeric"] * df["extreme_hourly"]
    )
    # Interaction 2: drought run duration (in days) × demand
    df["run_hours_x_demand"] = (
        (df["drought_run_hours"] / 24.0) * df["extreme_hourly"]
    )
    return df


def features_lr3(df: pd.DataFrame) -> pd.DataFrame:
    """
    LR-3: Raw continuous wind_cf + drought_run_hours + binary extreme-
    temperature demand flag + two interaction terms.

    Final logistic regression specification.  Replaces the categorical
    drought encoding entirely with continuous inputs, avoiding the boundary
    artefacts that discard physically meaningful information at tier edges.
    """
    df = df.copy()
    df["wind_cf"] = df["wind_cf"].clip(0.0, 1.0)

    # Interaction 1: wind shortfall magnitude × demand
    # (1 − wind_cf) = 0 at full output, 1 at complete failure
    df["low_wind_x_demand"] = (1.0 - df["wind_cf"]) * df["extreme_hourly"]

    # Interaction 2: drought run duration (in days) × demand
    df["run_hours_x_demand"] = (
        (df["drought_run_hours"] / 24.0) * df["extreme_hourly"]
    )
    return df


# Feature column lists for each specification
MODEL_FEATURES = {
    "LR-1": [
        "drought_category_numeric",
        "extreme_hourly",
        "drought_x_demand",
    ],
    "LR-2": [
        "hourly_drought_category_numeric",
        "drought_run_hours",
        "extreme_hourly",
        "drought_x_demand",
        "run_hours_x_demand",
    ],
    "LR-3": [
        "wind_cf",
        "drought_run_hours",
        "extreme_hourly",
        "low_wind_x_demand",
        "run_hours_x_demand",
    ],
}

FEATURE_ENGINEERS = {
    "LR-1": features_lr1,
    "LR-2": features_lr2,
    "LR-3": features_lr3,
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
    Train logistic regression model *model_id*, run cross-validation on
    training data, evaluate on the held-out test set, and save outputs.

    Returns a dict of summary metrics.
    """
    print(f"\n{'=' * 60}")
    print(f"  {model_id}")
    print(f"{'=' * 60}")

    engineer = FEATURE_ENGINEERS[model_id]
    features = MODEL_FEATURES[model_id]

    # --- feature engineering ---
    train_eng = engineer(train_df)
    test_eng  = engineer(test_df)

    missing_train = [c for c in features if c not in train_eng.columns]
    if missing_train:
        raise ValueError(f"[{model_id}] Missing features in training data: {missing_train}")

    X_train = train_eng[features].copy()
    y_train = train_eng["target"].copy()
    X_test  = test_eng[features].copy()
    y_test  = test_eng["target"].copy()

    # drop rows with NaN in features
    mask_train = X_train.notna().all(axis=1)
    mask_test  = X_test.notna().all(axis=1)
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_test,  y_test  = X_test[mask_test],   y_test[mask_test]

    print(f"  Training rows : {len(X_train):,}  |  Test rows: {len(X_test):,}")
    print(f"  Features      : {features}")
    high_pct = y_train.mean() * 100
    print(f"  HIGH rate (train): {high_pct:.1f}%")

    # --- sklearn pipeline: scale → logistic ---
    pipeline = Pipeline([
        ("scaler",   StandardScaler()),
        ("logistic", LogisticRegression(**LR_PARAMS)),
    ])

    # --- 5-fold stratified cross-validation ---
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_auc       = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc",   n_jobs=-1)
    cv_precision = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="precision", n_jobs=-1)
    cv_recall    = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="recall",    n_jobs=-1)

    print(f"\n  Cross-validation results (train, 2022–2024):")
    print(f"    AUC-ROC   mean={cv_auc.mean():.4f}   std={cv_auc.std():.4f}")
    print(f"    Precision mean={cv_precision.mean():.4f}   std={cv_precision.std():.4f}")
    print(f"    Recall    mean={cv_recall.mean():.4f}   std={cv_recall.std():.4f}")

    # --- fit final model on all training data ---
    pipeline.fit(X_train, y_train)

    # --- test evaluation ---
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    test_auc    = roc_auc_score(y_test, y_pred_prob)

    print(f"\n  Test AUC-ROC (2020–2021): {test_auc:.4f}")
    print(f"  Train-test gap           : {cv_auc.mean() - test_auc:.4f}")
    print(f"\n  Classification report (test):")
    print(classification_report(y_test, y_pred, target_names=["LOW", "HIGH"]))

    # --- coefficients ---
    lr_step = pipeline.named_steps["logistic"]
    coef_df = pd.DataFrame({
        "feature":     features,
        "coefficient": lr_step.coef_[0],
    }).sort_values("coefficient", ascending=False)
    print("  Coefficients (scaled — comparable across features):")
    for _, row in coef_df.iterrows():
        direction = "↑ P(HIGH)" if row["coefficient"] > 0 else "↓ P(HIGH)"
        print(f"    {row['feature']:30s}: {row['coefficient']:+.4f}  {direction}")

    # --- save evaluation plots ---
    model_out = output_dir / model_id.lower().replace("-", "_")
    model_out.mkdir(parents=True, exist_ok=True)

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
    plot_path = model_out / "test_evaluation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {plot_path}")

    return {
        "model"         : model_id,
        "cv_auc_mean"   : round(cv_auc.mean(),       4),
        "cv_auc_std"    : round(cv_auc.std(),         4),
        "cv_precision"  : round(cv_precision.mean(),  4),
        "cv_recall"     : round(cv_recall.mean(),     4),
        "test_auc"      : round(test_auc,             4),
        "train_test_gap": round(cv_auc.mean() - test_auc, 4),
    }


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison(results: list[dict]) -> None:
    """Print a summary comparison table and note the paper's reported values."""
    print(f"\n{'=' * 70}")
    print("  LOGISTIC REGRESSION MODEL COMPARISON")
    print(f"{'=' * 70}")
    header = f"  {'Model':<8} {'CV AUC':>8} {'CV Recall':>10} {'CV Precision':>13} {'Test AUC':>9}"
    print(header)
    print("  " + "-" * 50)
    for r in results:
        print(
            f"  {r['model']:<8} "
            f"{r['cv_auc_mean']:>8.3f} "
            f"{r['cv_recall']:>10.3f} "
            f"{r['cv_precision']:>13.3f} "
            f"{r['test_auc']:>9.3f}"
        )
    print()
    print("  Paper-reported values (Table 3):")
    print("  Model    CV AUC   CV Recall  CV Precision")
    print("  LR-1     0.583    0.536      0.134")
    print("  LR-2     0.640    0.713      0.145")
    print("  LR-3     0.636    0.697      0.145")
    print()
    print("  Note: AUC plateaued across all three LR specifications.")
    print("  The architectural ceiling of logistic regression — a single linear")
    print("  decision boundary — cannot express the nonlinear, regime-dependent")
    print("  compound interactions between wind output, gas price, and temperature")
    print("  that drive HIGH price exposure in ERCOT. This motivates the shift to")
    print("  XGBoost in 06_xgboost_models.py.")
    print(f"{'=' * 70}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        choices=["LR-1", "LR-2", "LR-3", "all"],
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

    print("\nLabelling price exposure (year- and zone-specific P90) ...")
    train_df = label_price_exposure(train_raw)
    test_df  = label_price_exposure(test_raw)

    models_to_run = ["LR-1", "LR-2", "LR-3"] if args.model == "all" else [args.model]

    results = []
    for model_id in models_to_run:
        metrics = run_model(model_id, train_df, test_df, OUTPUT_DIR)
        results.append(metrics)

    if len(results) > 1:
        print_comparison(results)

    # Save results table
    results_path = OUTPUT_DIR / "lr_comparison.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
