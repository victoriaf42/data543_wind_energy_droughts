"""
12_financial_simulation.py
===========================
Financial risk simulation for grid cell 6_23 (LZ_WEST, 100 MW).

Simulates the hourly financial position of a wind energy producer operating
under a fixed-volume physical PPA and evaluates how well the XGB-2 model's
HIGH-exposure flags align with actual financial loss events.

PPA structure
-------------
  Capacity        : 100 MW nameplate
  Fixed PPA price : $50 / MWh
  Obligation      : 30 MWh / hr  (30% CF floor)

  Shortfall hours  — wind_mwh < 30 MWh → buy replacement energy on spot market
    Replacement cost = max(0, spot − $50) × shortfall_mwh

  Surplus hours    — wind_mwh > 30 MWh → sell excess on spot market
    Surplus revenue  = max(0, spot − $50) × excess_mwh

  Net position = PPA revenue − replacement cost + surplus revenue
  Baseline     = $50 × 30 MWh = $1,500/hr  (no wind variability, no spot exposure)

  No-arbitrage constraint: buying replacement energy when spot < $50 generates
  no gain because the replacement merely allows the producer to honour the
  fixed-price obligation, not to profit from below-market spot prices.

Model flagging framework
------------------------
  The XGB-2 model assigns each hour a probability of HIGH price exposure.
  Two threshold approaches are compared:

    Conservative / overall  : threshold = 0.3539 (F1-optimal, recall ≥ 55%)
    Year-specific           : separate F1-optimal threshold per year,
                              calibrated on each year's probability distribution

  True positive  (TP) : flagged AND spot > $50 → loss correctly anticipated
  False positive (FP) : flagged AND spot ≤ $50 → unnecessary flag
  False negative (FN) : not flagged AND spot > $50 → unhedged exposure
  True negative  (TN) : not flagged AND spot ≤ $50 → correctly ignored

  Loss capture rate: share of total replacement costs in flagged hours.

Input
-----
Per-hour CSV files from the threshold calibration step (08_grid_cell_...py).
Expected at:
  data/cell_6_23/cell_6_23_ALL_HOURS_overall_threshold.csv
  data/cell_6_23/cell_6_23_ALL_HOURS_year_specific_threshold.csv

Required columns: time, wind_cf, price, gas_price_mmbtu, tmm_F, temp_stress,
                  flagged, true_label, Price_Exposure, prob_high, outcome_code

Outputs (results/financial/)
-----------------------------
  cell_6_23_financial_sim_overall_hourly.csv    hourly simulation (overall thresh)
  cell_6_23_financial_sim_yearspec_hourly.csv   hourly simulation (year-specific)
  cell_6_23_financial_sim_overall_monthly.csv   monthly aggregation (overall)
  cell_6_23_financial_sim_yearspec_monthly.csv  monthly aggregation (year-specific)
  figure13_financial_risk_full_period.png       Figure 13 — full 2020-2021 time series
  figure14_financial_risk_non_uri.png           Figure 14 — non-Uri period only

Usage
-----
    python files/12_financial_simulation.py
"""

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR  = Path("data/cell_6_23")
OUTPUT_DIR = Path("results/financial")

FILE_OVERALL  = INPUT_DIR / "cell_6_23_ALL_HOURS_overall_threshold.csv"
FILE_YEARSPEC = INPUT_DIR / "cell_6_23_ALL_HOURS_year_specific_threshold.csv"

# PPA parameters
CAPACITY_MW    = 100.0   # MW nameplate
FIXED_PRICE    = 50.0    # $/MWh — fixed PPA price
OBLIGATION_MWH = 30.0    # MWh/hr — delivery obligation (30% CF floor)

# Classification thresholds (from 08_grid_cell_performance_and_threshold.py)
OPTIMAL_THRESHOLD = 0.3539   # overall conservative, recall ≥ 55%

# Winter Storm Uri period
URI_START = "2021-02-10"
URI_END   = "2021-02-20"

DATE_COL = "time"

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [FILE_OVERALL, FILE_YEARSPEC]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}\n"
                "Run 08_grid_cell_performance_and_threshold.py first to generate\n"
                "the per-hour model output files for cell 6_23."
            )

    df_o = pd.read_csv(FILE_OVERALL,  parse_dates=[DATE_COL])
    df_y = pd.read_csv(FILE_YEARSPEC, parse_dates=[DATE_COL])

    for label, df in [("Overall", df_o), ("Year-specific", df_y)]:
        print(f"  {label}: {len(df):,} rows  "
              f"{df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")

    return df_o, df_y

# =============================================================================
# STEP 2 — FINANCIAL SIMULATION
# =============================================================================

def simulate_ppa(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Compute hourly PPA financial position for every row in df.

    Financial logic:
      wind_mwh         = CAPACITY_MW × wind_cf
      shortfall_mwh    = max(0, OBLIGATION_MWH − wind_mwh)
      excess_mwh       = max(0, wind_mwh − OBLIGATION_MWH)
      ppa_revenue      = min(wind_mwh, OBLIGATION_MWH) × FIXED_PRICE
      replacement_cost = shortfall_mwh × max(0, spot − FIXED_PRICE)
      surplus_revenue  = excess_mwh   × max(0, spot − FIXED_PRICE)
      net_position     = ppa_revenue − replacement_cost + surplus_revenue
      baseline         = OBLIGATION_MWH × FIXED_PRICE  [no spot exposure]
    """
    sim = df.copy()

    sim["wind_mwh"]         = CAPACITY_MW * sim["wind_cf"]
    sim["shortfall_mwh"]    = np.maximum(0, OBLIGATION_MWH - sim["wind_mwh"])
    sim["excess_mwh"]       = np.maximum(0, sim["wind_mwh"] - OBLIGATION_MWH)
    sim["delivered_mwh"]    = np.minimum(sim["wind_mwh"], OBLIGATION_MWH)

    sim["ppa_revenue"]      = sim["delivered_mwh"] * FIXED_PRICE
    sim["replacement_cost"] = sim["shortfall_mwh"] * np.maximum(0, sim["price"] - FIXED_PRICE)
    sim["surplus_revenue"]  = sim["excess_mwh"]    * np.maximum(0, sim["price"] - FIXED_PRICE)

    sim["net_position"]     = sim["ppa_revenue"] - sim["replacement_cost"] + sim["surplus_revenue"]
    sim["baseline_position"] = OBLIGATION_MWH * FIXED_PRICE

    sim["net_exposure"]     = sim["surplus_revenue"] - sim["replacement_cost"]
    sim["is_loss_hour"]     = (sim["replacement_cost"] > 0).astype(int)
    sim["is_surplus_hour"]  = (sim["surplus_revenue"]  > 0).astype(int)

    sim["threshold_label"]  = label
    return sim


def print_simulation_summary(sim: pd.DataFrame, label: str) -> None:
    total_loss    = sim["replacement_cost"].sum()
    total_surplus = sim["surplus_revenue"].sum()
    flagged_loss  = sim.loc[sim["flagged"] == 1, "replacement_cost"].sum()
    missed_loss   = sim.loc[sim["flagged"] == 0, "replacement_cost"].sum()
    n_flagged     = int(sim["flagged"].sum())
    n_total       = len(sim)

    print(f"\n  {label}:")
    print(f"    PPA revenue            : ${sim['ppa_revenue'].sum():>15,.0f}")
    print(f"    Replacement cost (loss): ${total_loss:>15,.0f}")
    print(f"    Surplus revenue        : ${total_surplus:>15,.0f}")
    print(f"    Baseline position      : ${sim['baseline_position'].sum():>15,.0f}")
    print(f"    Net vs baseline        : ${total_surplus - total_loss:>15,.0f}")
    print(f"    Loss hours             : {sim['is_loss_hour'].sum():>10,}")
    print(f"    Max single-hour loss   : ${sim['replacement_cost'].max():>15,.0f}")
    print(f"\n    Model flags:  {n_flagged:,} / {n_total:,} hours  ({n_flagged/n_total*100:.1f}%)")
    print(f"    Loss in flagged hours  : ${flagged_loss:>12,.0f}  "
          f"({flagged_loss/total_loss*100:.1f}% of total — loss capture rate)")
    print(f"    Loss in non-flagged    : ${missed_loss:>12,.0f}  "
          f"({missed_loss/total_loss*100:.1f}% of total — unhedged exposure)")

# =============================================================================
# STEP 3 — MONTHLY AGGREGATION
# =============================================================================

def monthly_aggregate(sim: pd.DataFrame) -> pd.DataFrame:
    sim = sim.copy()
    sim["month"] = sim[DATE_COL].dt.to_period("M")

    monthly = sim.groupby("month").agg(
        ppa_revenue       = ("ppa_revenue",       "sum"),
        replacement_cost  = ("replacement_cost",  "sum"),
        surplus_revenue   = ("surplus_revenue",   "sum"),
        net_position      = ("net_position",      "sum"),
        baseline_position = ("baseline_position", "sum"),
        net_exposure      = ("net_exposure",       "sum"),
        loss_hours        = ("is_loss_hour",       "sum"),
        surplus_hours     = ("is_surplus_hour",    "sum"),
        flagged_hours     = ("flagged",            "sum"),
        total_hours       = ("flagged",            "count"),
        mean_wind_cf      = ("wind_cf",            "mean"),
        mean_price        = ("price",              "mean"),
        mean_gas_price    = ("gas_price_mmbtu",    "mean"),
        flagged_loss      = ("replacement_cost",
                             lambda x: x[sim.loc[x.index, "flagged"] == 1].sum()),
        missed_loss       = ("replacement_cost",
                             lambda x: x[sim.loc[x.index, "flagged"] == 0].sum()),
    ).reset_index()

    monthly["month_dt"] = monthly["month"].dt.to_timestamp()
    monthly["year"]     = monthly["month"].dt.year
    monthly["pct_loss_captured"] = np.where(
        monthly["replacement_cost"] > 0,
        monthly["flagged_loss"] / monthly["replacement_cost"] * 100, 0
    )
    return monthly

# =============================================================================
# STEP 4 — URI ANALYSIS
# =============================================================================

def uri_analysis(sim: pd.DataFrame) -> None:
    """
    Decomposes loss concentration around Winter Storm Uri (Feb 2021) and
    explains why the model captures 97% of total losses despite an overall
    test AUC of 0.6157.
    """
    uri_mask  = (sim[DATE_COL] >= URI_START) & (sim[DATE_COL] <= URI_END)
    uri_df    = sim[uri_mask].copy()
    non_uri   = sim[~uri_mask].copy()

    total_loss   = sim["replacement_cost"].sum()
    uri_loss     = uri_df["replacement_cost"].sum()
    non_uri_loss = non_uri["replacement_cost"].sum()

    uri_flagged   = uri_df[uri_df["flagged"] == 1]["replacement_cost"].sum()
    non_uri_flagged = non_uri[non_uri["flagged"] == 1]["replacement_cost"].sum()
    uri_missed    = uri_df[uri_df["flagged"] == 0]["replacement_cost"].sum()
    non_uri_missed  = non_uri[non_uri["flagged"] == 0]["replacement_cost"].sum()

    print(f"\n  Loss concentration:")
    print(f"    Total losses (2020-2021)  : ${total_loss:>15,.0f}")
    print(f"    Uri losses (Feb 10-20)    : ${uri_loss:>15,.0f}  ({uri_loss/total_loss*100:.1f}% of total)")
    print(f"    Non-Uri losses            : ${non_uri_loss:>15,.0f}  ({non_uri_loss/total_loss*100:.1f}% of total)")

    print(f"\n  Model capture by period:")
    print(f"    Uri captured   : ${uri_flagged:>12,.0f}  ({uri_flagged/uri_loss*100:.1f}% of Uri losses)")
    print(f"    Uri missed     : ${uri_missed:>12,.0f}  ({uri_missed/uri_loss*100:.1f}% of Uri losses)")
    print(f"    Non-Uri captured: ${non_uri_flagged:>11,.0f}  ({non_uri_flagged/non_uri_loss*100:.1f}% of non-Uri losses)")
    print(f"    Non-Uri missed : ${non_uri_missed:>12,.0f}  ({non_uri_missed/non_uri_loss*100:.1f}% of non-Uri losses)")

    # AUC during Uri vs outside
    uri_auc     = roc_auc_score(uri_df["true_label"],   uri_df["prob_high"])
    non_uri_auc = roc_auc_score(non_uri["true_label"],  non_uri["prob_high"])
    print(f"\n  AUC during Uri (Feb 2021)  : {uri_auc:.4f}")
    print(f"  AUC outside Uri            : {non_uri_auc:.4f}")
    print(f"  Overall test AUC           : {roc_auc_score(sim['true_label'], sim['prob_high']):.4f}")

    # Uri physical characteristics
    uri_loss_hrs = uri_df[uri_df["replacement_cost"] > 0]
    non_loss_hrs = non_uri[non_uri["replacement_cost"] > 0]
    print(f"\n  Uri loss hour physical characteristics:")
    print(f"    Mean wind CF             : {uri_loss_hrs['wind_cf'].mean():.4f}  "
          f"(non-Uri: {non_loss_hrs['wind_cf'].mean():.4f})")
    print(f"    % hours wind_cf < 0.05   : {(uri_loss_hrs['wind_cf'] < 0.05).mean()*100:.1f}%")
    print(f"    % hours wind_cf = 0.00   : {(uri_loss_hrs['wind_cf'] == 0.0).mean()*100:.1f}%")
    print(f"    Mean gas price ($/MMBtu) : {uri_loss_hrs['gas_price_mmbtu'].mean():.2f}  "
          f"(non-Uri: {non_loss_hrs['gas_price_mmbtu'].mean():.2f})")
    print(f"    Mean temp stress (deg F) : {uri_loss_hrs['temp_stress'].mean():.1f}  "
          f"(non-Uri: {non_loss_hrs['temp_stress'].mean():.1f})")
    print(f"    Max spot price ($/MWh)   : ${uri_loss_hrs['price'].max():,.2f}")
    print(f"    Mean model prob_high     : {uri_loss_hrs['prob_high'].mean():.4f}")
    print(f"    % above threshold 0.3539 : {(uri_loss_hrs['prob_high'] >= 0.3539).mean()*100:.1f}%")

    print(f"""
  Why 97% loss capture despite AUC = 0.616:

  1. CONCENTRATION: {uri_loss/total_loss*100:.1f}% of all losses occurred during a
     single 10-day event (Uri, Feb 10-20 2021).

  2. URI IS PHYSICALLY DISTINCTIVE: Wind CF collapsed to a mean
     of {uri_loss_hrs['wind_cf'].mean():.3f} during Uri loss hours ({(uri_loss_hrs['wind_cf']<0.05).mean()*100:.0f}% of hours
     below 0.05 CF).  Gas prices reached ${uri_loss_hrs['gas_price_mmbtu'].mean():.1f}/MMBtu
     ({uri_loss_hrs['gas_price_mmbtu'].mean()/non_loss_hrs['gas_price_mmbtu'].mean():.1f}x non-Uri levels) and temperature
     stress reached {uri_loss_hrs['temp_stress'].mean():.1f} deg F from comfort.  These
     conditions produced precisely the compound signal the model was
     trained to detect — {(uri_loss_hrs['prob_high'] >= 0.3539).mean()*100:.1f}% of Uri loss hours were flagged.

  3. AUC vs LOSS CAPTURE: AUC measures the model's ability to rank
     ALL hours by risk correctly.  Loss capture measures whether the
     model flags hours containing LARGE losses.  A model can have
     mediocre AUC but excellent loss capture if extreme events that
     dominate financial losses have distinctive features.
     Uri AUC = {uri_auc:.3f} vs Non-Uri AUC = {non_uri_auc:.3f} confirms
     the model operates in two distinct regimes.
    """)

# =============================================================================
# STEP 5 — FIGURES 13 AND 14
# =============================================================================

def _fmt_dollars(ax, ymax: float) -> None:
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"${abs(x)/1e6:.1f}M" if abs(x) >= 1e6 else f"${abs(x)/1e3:.0f}K"
        )
    )


def _style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)


def plot_financial_figures(
    sim_o: pd.DataFrame,
    sim_y: pd.DataFrame,
    monthly_o: pd.DataFrame,
    monthly_y: pd.DataFrame,
    output_dir: Path,
    uri_start: str,
    uri_end: str,
) -> None:
    """
    Figure 13 — Full 2020-2021 period: monthly replacement costs with TP
                (flagged, loss hours, shown in green) and FN (missed, red).
    Figure 14 — Non-Uri period only: same layout, allowing non-extreme
                months to be visible without Uri's scale distortion.
    """

    def _make_figure(
        sim: pd.DataFrame,
        monthly: pd.DataFrame,
        title_suffix: str,
        out_name: str,
        exclude_uri: bool = False,
    ) -> None:
        if exclude_uri:
            uri_mask  = (sim[DATE_COL] >= uri_start) & (sim[DATE_COL] <= uri_end)
            sim       = sim[~uri_mask].copy()
            mo        = monthly[monthly["month_dt"].dt.strftime("%Y-%m") != "2021-02"].copy()
        else:
            mo        = monthly.copy()

        uri_flag = (sim[DATE_COL] >= uri_start) & (sim[DATE_COL] <= uri_end)

        # Classify each loss hour by model outcome
        sim["is_tp"] = ((sim["flagged"] == 1) & (sim["replacement_cost"] > 0)).astype(int)
        sim["is_fn"] = ((sim["flagged"] == 0) & (sim["replacement_cost"] > 0)).astype(int)

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.patch.set_facecolor("white")

        # Panel 1: monthly replacement cost with model coverage
        ax1 = axes[0]
        ax1.bar(mo["month_dt"], mo["flagged_loss"] * -1,
                width=20, color="#185FA5", alpha=0.85,
                label="Loss in flagged hours (TP — hedgeable)")
        ax1.bar(mo["month_dt"], mo["missed_loss"] * -1,
                width=20, color="#993C1D", alpha=0.85,
                bottom=mo["flagged_loss"] * -1,
                label="Loss in non-flagged hours (FN — unhedged)")
        ax1.bar(mo["month_dt"], mo["surplus_revenue"],
                width=20, color="#0F6E56", alpha=0.75,
                label="Surplus revenue")
        ax1.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax1.set_ylabel("$ / month")
        ax1.set_title(
            f"Monthly financial exposure — loss capture by model flag\n"
            f"Blue = loss in flagged hours  |  Red = loss in non-flagged hours"
        )
        ax1.legend(frameon=False, fontsize=9)
        _fmt_dollars(ax1, mo["replacement_cost"].max())
        _style_ax(ax1)

        # Panel 2: monthly loss capture rate
        ax2 = axes[1]
        ax2.plot(mo["month_dt"], mo["pct_loss_captured"],
                 color="#185FA5", linewidth=2, marker="o", markersize=5,
                 label=f"Overall threshold ({OPTIMAL_THRESHOLD:.4f})")
        ax2.axhline(50, color="#888888", linestyle=":", linewidth=0.8, label="50% reference")
        ax2.axhline(100, color="#0F6E56", linestyle="--", linewidth=0.8, alpha=0.5, label="100%")
        ax2.set_ylabel("% of monthly loss captured")
        ax2.set_title("Monthly loss capture rate — % of replacement costs in model-flagged hours")
        ax2.set_ylim(0, 115)
        ax2.legend(frameon=False, fontsize=9)
        _style_ax(ax2)

        total = sim["replacement_cost"].sum()
        caught = sim.loc[sim["flagged"] == 1, "replacement_cost"].sum()
        fig.suptitle(
            f"Cell 6_23 (LZ_WEST, 100 MW) — PPA Financial Risk Simulation {title_suffix}\n"
            f"Fixed price: ${FIXED_PRICE}/MWh  |  Obligation: {OBLIGATION_MWH} MWh/hr  |  "
            f"Total loss: ${total/1e6:.2f}M  |  Captured: {caught/total*100:.1f}%",
            fontsize=12, y=1.01,
        )
        plt.tight_layout()
        out = output_dir / out_name
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {out}")

    # Figure 13 — full period (Figure 13 from paper)
    _make_figure(
        sim_o.copy(), monthly_o.copy(),
        title_suffix="(2020–2021 full period)",
        out_name="figure13_financial_risk_full_period.png",
        exclude_uri=False,
    )

    # Figure 14 — non-Uri only (Figure 14 from paper)
    _make_figure(
        sim_o.copy(), monthly_o.copy(),
        title_suffix="(2020–2021 excluding Winter Storm Uri Feb 2021)",
        out_name="figure14_financial_risk_non_uri.png",
        exclude_uri=True,
    )

# =============================================================================
# STEP 6 — SAVE CSVS
# =============================================================================

def save_csvs(
    sim_o: pd.DataFrame, monthly_o: pd.DataFrame,
    sim_y: pd.DataFrame, monthly_y: pd.DataFrame,
    output_dir: Path,
) -> None:
    hourly_cols = [
        DATE_COL, "wind_cf", "wind_mwh", "price", "gas_price_mmbtu",
        "tmm_F", "temp_stress",
        "shortfall_mwh", "excess_mwh", "delivered_mwh",
        "ppa_revenue", "replacement_cost", "surplus_revenue",
        "net_position", "baseline_position", "net_exposure",
        "is_loss_hour", "is_surplus_hour",
        "flagged", "true_label", "Price_Exposure", "prob_high",
        "outcome_code", "threshold_label",
    ]

    for label, sim, fname in [
        ("overall",   sim_o, "cell_6_23_financial_sim_overall_hourly.csv"),
        ("yearspec",  sim_y, "cell_6_23_financial_sim_yearspec_hourly.csv"),
    ]:
        cols = [c for c in hourly_cols if c in sim.columns]
        path = output_dir / fname
        sim[cols].sort_values(DATE_COL).to_csv(path, index=False, float_format="%.4f")
        print(f"  Saved hourly ({label}): {path.name}")

    for label, monthly, fname in [
        ("overall",  monthly_o, "cell_6_23_financial_sim_overall_monthly.csv"),
        ("yearspec", monthly_y, "cell_6_23_financial_sim_yearspec_monthly.csv"),
    ]:
        path = output_dir / fname
        monthly.drop(columns=["month"], errors="ignore").to_csv(
            path, index=False, float_format="%.2f"
        )
        print(f"  Saved monthly ({label}): {path.name}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary_table(sim: pd.DataFrame, monthly: pd.DataFrame, label: str) -> None:
    total_loss   = sim["replacement_cost"].sum()
    total_surplus = sim["surplus_revenue"].sum()
    flagged_loss = sim.loc[sim["flagged"] == 1, "replacement_cost"].sum()
    missed_loss  = sim.loc[sim["flagged"] == 0, "replacement_cost"].sum()

    worst_month = monthly.loc[monthly["replacement_cost"].idxmax()]
    best_month  = monthly.loc[monthly["surplus_revenue"].idxmax()]

    print(f"\n  {label}:")
    print(f"    Total replacement cost    : ${total_loss:>15,.0f}")
    print(f"    Total surplus revenue     : ${total_surplus:>15,.0f}")
    print(f"    Net vs baseline           : ${total_surplus-total_loss:>15,.0f}")
    print(f"    Loss in flagged hours     : ${flagged_loss:>15,.0f}  "
          f"({flagged_loss/total_loss*100:.1f}% loss capture rate)")
    print(f"    Loss in non-flagged hours : ${missed_loss:>15,.0f}  "
          f"({missed_loss/total_loss*100:.1f}% unhedged)")
    print(f"    Worst month (loss)        : ${worst_month['replacement_cost']:>12,.0f}  "
          f"({worst_month['month_dt'].strftime('%b %Y')})")
    print(f"    Best month (surplus)      : ${best_month['surplus_revenue']:>12,.0f}  "
          f"({best_month['month_dt'].strftime('%b %Y')})")

# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CELL 6_23 — PPA FINANCIAL RISK SIMULATION")
    print("=" * 60)
    print(f"  Capacity       : {CAPACITY_MW} MW")
    print(f"  Fixed price    : ${FIXED_PRICE}/MWh")
    print(f"  Obligation     : {OBLIGATION_MWH} MWh/hr")
    print(f"  Threshold      : {OPTIMAL_THRESHOLD} (overall conservative)")

    print("\nStep 1 — Loading data ...")
    df_overall, df_yearspec = load_data()

    print("\nStep 2 — Running financial simulation ...")
    sim_o = simulate_ppa(df_overall,  "overall_optimal")
    sim_y = simulate_ppa(df_yearspec, "year_specific")

    for label, sim in [("Overall threshold", sim_o),
                        ("Year-specific threshold", sim_y)]:
        print_simulation_summary(sim, label)

    print("\nStep 3 — Monthly aggregation ...")
    monthly_o = monthly_aggregate(sim_o)
    monthly_y = monthly_aggregate(sim_y)

    print("\nStep 4 — Summary table (Table 9 from paper) ...")
    print(f"\n{'=' * 60}")
    for label, sim, monthly in [
        ("Overall threshold",       sim_o, monthly_o),
        ("Year-specific threshold", sim_y, monthly_y),
    ]:
        print_summary_table(sim, monthly, label)
    print(f"{'=' * 60}")

    print("\nStep 5 — Uri analysis (explains loss concentration) ...")
    print(f"\n{'=' * 60}")
    print("URI ANALYSIS — LOSS CONCENTRATION AND MODEL CAPTURE")
    print(f"{'=' * 60}")
    uri_analysis(sim_o)

    print("\nStep 6 — Saving figures ...")
    plot_financial_figures(
        sim_o, sim_y, monthly_o, monthly_y,
        OUTPUT_DIR, URI_START, URI_END,
    )

    print("\nStep 7 — Saving CSVs ...")
    save_csvs(sim_o, monthly_o, sim_y, monthly_y, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
