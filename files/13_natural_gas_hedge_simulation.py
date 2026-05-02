"""
13_natural_gas_hedge_simulation.py
====================================
Natural gas futures hedge simulation for grid cell 6_23 (LZ_WEST, 100 MW).

Evaluates whether buying NYMEX Henry Hub natural gas futures contracts on
model-flagged hours can partially offset the electricity price replacement
costs incurred under the fixed-volume PPA when wind falls short.

How the hedge works
-------------------
For each hour the XGB-2 model flags as HIGH price exposure:
  1. Buy NG futures at the prevailing futures price (C1–C4)
  2. At settlement, realise: P&L = (NG spot − NG futures) × hedge quantity
  3. This P&L partially offsets replacement cost when spot > futures
     (i.e. when gas spiked above the locked-in price)

Hedge quantity = 30 MWh × 8.978 MMBtu/MWh = 269.34 MMBtu per flagged hour
(full contract obligation × heat rate derived from 38% CCGT efficiency)

Three strategies are benchmarked:
  UNHEDGED — no hedge, full spot exposure
  MODEL    — hedge every hour flagged by XGB-2 (conservative threshold 0.3539)
  ORACLE   — hedge only hours with actual replacement cost > 0 (perfect hindsight)

The Oracle strategy is the theoretical performance ceiling and cannot be
replicated in practice.  Comparing it to the Model strategy isolates the
financial cost of the model's classification errors.

Basis risk
----------
Henry Hub gas futures explain approximately 25% of ERCOT electricity price
variance (R² = 0.257 from log-price regression over 2020–2024, clipped at
$15/MMBtu to limit Uri influence).  This basis risk is the primary limitation
of the hedge — gas futures are a natural but imperfect hedge instrument, and
their effectiveness varies materially month-to-month.

Inputs
------
  results/financial/cell_6_23_financial_sim_overall_hourly.csv
    (from 12_financial_simulation.py — requires this to be run first)

  data/ng_futures/henry_hub_futures_filled.csv
    (from 09_natural_gas_futures.py — forward-filled daily futures prices)

  data/NG_prices.csv
    Henry Hub daily spot prices (same file used in model training)

Outputs (results/hedge/)
------------------------
  figure16_futures_vs_spot.png          avg futures price vs realised spot
                                         per contract (Figure 16 from paper)
  figure17_monthly_hedge_pnl.png        monthly hedge P&L over 2020–2021
                                         for each contract (Figure 17)
  figure18_strategy_net_position.png    total net position: unhedged vs
                                         model vs oracle (Figure 18)
  figure19_monthly_strategy.png         monthly net position for all three
                                         strategies (Figure 19)
  hedge_summary.csv                     contract-level summary table

Usage
-----
    python files/13_natural_gas_hedge_simulation.py

Run 09_natural_gas_futures.py and 12_financial_simulation.py first.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

SIM_PATH     = Path("results/financial/cell_6_23_financial_sim_overall_hourly.csv")
FUTURES_PATH = Path("data/ng_futures/henry_hub_futures_filled.csv")
SPOT_PATH    = Path("data/NG_prices.csv")
OUTPUT_DIR   = Path("results/hedge")

# Hedge parameters (match 09_natural_gas_futures.py)
HEAT_RATE    = 8.978    # MMBtu/MWh  (38% CCGT efficiency)
CONTRACT_MWH = 30.0     # MWh/hr — full PPA obligation hedged every flagged hour
HEDGE_QTY    = CONTRACT_MWH * HEAT_RATE   # = 269.34 MMBtu per flagged hour

CONTRACTS = ["NG_C1", "NG_C2", "NG_C3", "NG_C4"]
CONTRACT_LABELS = {
    "NG_C1": "C1 (1-mo ahead)",
    "NG_C2": "C2 (2-mo ahead)",
    "NG_C3": "C3 (3-mo ahead)",
    "NG_C4": "C4 (4-mo ahead)",
}

# Colour palette
GREEN  = "#1D9E75"
RED    = "#E24B4A"
BLUE   = "#378ADD"
GRAY   = "#888780"
NAVY   = "#2E4A7A"
AMBER  = "#BA7517"
PURPLE = "#534AB7"

plt.rcParams.update({
    "font.family"       : "sans-serif",
    "font.size"         : 11,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.linewidth"    : 0.6,
    "axes.grid"         : True,
    "axes.grid.axis"    : "y",
    "grid.color"        : "#E0E0E0",
    "grid.linewidth"    : 0.5,
    "axes.axisbelow"    : True,
    "figure.dpi"        : 150,
    "savefig.dpi"       : 150,
    "savefig.bbox"      : "tight",
    "savefig.facecolor" : "white",
})

# =============================================================================
# HELPERS
# =============================================================================

def dollar_fmt(x, _):
    if abs(x) >= 1e6:
        return f"${x/1e6:.1f}M"
    if abs(x) >= 1e3:
        sign = "-" if x < 0 else ""
        return f"{sign}${abs(x)/1e3:.0f}K"
    return f"${x:.0f}"


def save_fig(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.name}")

# =============================================================================
# STEP 1 — LOAD AND MERGE DATA
# =============================================================================

def load_data() -> pd.DataFrame:
    """
    Load the financial simulation output, merge daily futures prices and
    daily spot gas prices, and forward-fill weekend/holiday gaps.
    """
    for path in [SIM_PATH, FUTURES_PATH, SPOT_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}\n"
                "Ensure 09_natural_gas_futures.py and 12_financial_simulation.py"
                " have been run first."
            )

    sim = pd.read_csv(SIM_PATH)
    sim.columns = sim.columns.str.strip()
    sim["time"] = pd.to_datetime(sim["time"], errors="coerce")
    sim["date"] = sim["time"].dt.normalize()
    sim["year_month"] = sim["time"].dt.to_period("M")

    # Futures (forward-filled file from 09_natural_gas_futures.py)
    fut = pd.read_csv(FUTURES_PATH)
    fut.columns = fut.columns.str.strip()
    date_col = "date" if "date" in fut.columns else fut.columns[0]
    fut[date_col] = pd.to_datetime(fut[date_col], errors="coerce").dt.normalize()
    fut = fut.rename(columns={date_col: "date"})

    # Additional forward-fill for any residual gaps across simulation dates
    all_dates = pd.DataFrame({"date": pd.date_range(sim["date"].min(), sim["date"].max(), freq="D")})
    fut_filled = (
        all_dates
        .merge(fut[["date"] + CONTRACTS], on="date", how="left")
        .sort_values("date")
    )
    fut_filled[CONTRACTS] = fut_filled[CONTRACTS].ffill(limit=5).bfill(limit=1)

    df = sim.merge(fut_filled, on="date", how="left")

    # NG spot prices (for hedge P&L settlement)
    spot = pd.read_csv(SPOT_PATH)
    spot.columns = spot.columns.str.strip()
    date_col_s  = "observation_date" if "observation_date" in spot.columns else spot.columns[0]
    price_col_s = "$/MMBtu"          if "$/MMBtu"          in spot.columns else spot.columns[1]
    spot = spot[[date_col_s, price_col_s]].copy()
    spot.columns = ["date", "ng_spot_mmbtu"]
    spot["date"] = pd.to_datetime(spot["date"]).dt.normalize()
    spot["ng_spot_mmbtu"] = spot["ng_spot_mmbtu"].replace(0, np.nan).ffill()

    df = df.merge(spot, on="date", how="left")
    df["ng_spot_mmbtu"] = df["ng_spot_mmbtu"].ffill().bfill()

    # Keep only rows with valid futures prices
    df = df.dropna(subset=[CONTRACTS[0]]).reset_index(drop=True)

    # Financial quadrant labels (based on actual replacement cost, not P90 label)
    df["actual_loss"] = (df["replacement_cost"] > 0).astype(int)
    df["fin_quadrant"] = np.select(
        [
            (df["flagged"] == 1) & (df["actual_loss"] == 1),
            (df["flagged"] == 1) & (df["actual_loss"] == 0),
            (df["flagged"] == 0) & (df["actual_loss"] == 1),
            (df["flagged"] == 0) & (df["actual_loss"] == 0),
        ],
        ["TP_fin", "FP_fin", "FN_fin", "TN_fin"],
        default="UNKNOWN",
    )

    print(f"  Total hours          : {len(df):,}")
    print(f"  Model-flagged hours  : {int(df['flagged'].sum()):,}  "
          f"({df['flagged'].mean()*100:.1f}%)")
    print(f"  Actual loss hours    : {int(df['actual_loss'].sum()):,}  "
          f"({df['actual_loss'].mean()*100:.1f}%)")
    print(f"  TP (flagged+loss)    : {int((df['fin_quadrant']=='TP_fin').sum()):,}")
    print(f"  FP (flagged+no loss) : {int((df['fin_quadrant']=='FP_fin').sum()):,}")
    print(f"  FN (missed+loss)     : {int((df['fin_quadrant']=='FN_fin').sum()):,}")

    return df

# =============================================================================
# STEP 2 — HEDGE CALCULATIONS: C1 THROUGH C4
# =============================================================================

def compute_hedge_stress_test(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    For each contract (C1–C4), compute hedge P&L on all model-flagged hours.
    Returns the flagged-hours DataFrame with hedge columns and a summary dict.
    """
    flagged = df[df["flagged"] == 1].copy()
    flagged["hedge_mmbtu"] = HEDGE_QTY   # fixed per flagged hour

    results = {}
    for c in CONTRACTS:
        # Hedge P&L = (NG spot − NG futures) × hedge quantity (MMBtu)
        flagged[f"{c}_hedge_pnl"] = (
            (flagged["ng_spot_mmbtu"] - flagged[c]) * flagged["hedge_mmbtu"]
        )
        flagged[f"{c}_net_pos"] = (
            flagged["net_position"] + flagged[f"{c}_hedge_pnl"]
        )
        flagged[f"{c}_hedge_cost"] = (-flagged[f"{c}_hedge_pnl"]).clip(lower=0)

        results[c] = {
            "label"              : CONTRACT_LABELS[c],
            "n_flagged"          : len(flagged),
            "avg_futures_mmbtu"  : flagged[c].mean(),
            "avg_spot_mmbtu"     : flagged["ng_spot_mmbtu"].mean(),
            "total_replacement"  : flagged["replacement_cost"].sum(),
            "total_hedge_pnl"    : flagged[f"{c}_hedge_pnl"].sum(),
            "total_hedge_cost"   : flagged[f"{c}_hedge_cost"].sum(),
            "total_hedge_gain"   : flagged[f"{c}_hedge_pnl"].clip(lower=0).sum(),
            "net_pos_unhedged"   : flagged["net_position"].sum(),
            "net_pos_hedged"     : flagged[f"{c}_net_pos"].sum(),
            "pct_hours_gain"     : (flagged[f"{c}_hedge_pnl"] > 0).mean() * 100,
            "pct_hours_cost"     : (flagged[f"{c}_hedge_pnl"] < 0).mean() * 100,
            "hedge_efficiency"   : (
                flagged[f"{c}_hedge_pnl"].sum() / len(flagged)
                if len(flagged) > 0 else 0
            ),
        }

    print(f"\n  Hedge stress test summary (all flagged hours):")
    print(f"  {'Contract':<20}  {'Avg Fut':>9}  {'Avg Spot':>9}  "
          f"{'Hedge P&L':>12}  {'Hedge Cost':>11}  {'$/hr':>8}")
    print("  " + "─" * 78)
    for c, r in results.items():
        print(
            f"  {r['label']:<20}  "
            f"${r['avg_futures_mmbtu']:>7.2f}  "
            f"${r['avg_spot_mmbtu']:>7.2f}  "
            f"${r['total_hedge_pnl']:>11,.0f}  "
            f"${r['total_hedge_cost']:>10,.0f}  "
            f"${r['hedge_efficiency']:>7.1f}"
        )

    return flagged, results

# =============================================================================
# STEP 3 — THREE-STRATEGY COMPARISON  (C1 only, per paper)
# =============================================================================

def compute_strategies(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Compare three hedging strategies on ALL hours using C1 futures:
      UNHEDGED — no hedge
      MODEL    — hedge every flagged hour
      ORACLE   — hedge every hour with actual replacement cost > 0
    """
    df = df.copy()

    df["hedge_model"]  = np.where(
        df["flagged"] == 1,
        (df["ng_spot_mmbtu"] - df["NG_C1"]) * HEDGE_QTY,
        0.0,
    )
    df["hedge_oracle"] = np.where(
        df["replacement_cost"] > 0,
        (df["ng_spot_mmbtu"] - df["NG_C1"]) * HEDGE_QTY,
        0.0,
    )

    df["net_unhedged"] = df["net_position"]
    df["net_model"]    = df["net_position"] + df["hedge_model"]
    df["net_oracle"]   = df["net_position"] + df["hedge_oracle"]

    flagged_hours = int(df["flagged"].sum())
    oracle_hours  = int((df["replacement_cost"] > 0).sum())
    total_hours   = len(df)
    total_repl    = df["replacement_cost"].sum()
    unhedged_net  = df["net_unhedged"].sum()

    strategies = {
        "Unhedged": {
            "net": unhedged_net,
            "hedge_pnl": 0.0,
            "hours_hedged": 0,
        },
        "Model": {
            "net": df["net_model"].sum(),
            "hedge_pnl": df["hedge_model"].sum(),
            "hours_hedged": flagged_hours,
        },
        "Oracle": {
            "net": df["net_oracle"].sum(),
            "hedge_pnl": df["hedge_oracle"].sum(),
            "hours_hedged": oracle_hours,
        },
    }

    print(f"\n  Three-strategy comparison (C1 futures):")
    print(f"  {'Strategy':<12}  {'Hours Hedged':>13}  {'Hedge P&L':>14}  "
          f"{'Net Position':>14}  {'vs Unhedged':>13}  {'$/hr':>8}")
    print("  " + "─" * 80)
    for name, s in strategies.items():
        diff = s["net"] - unhedged_net
        eff  = s["hedge_pnl"] / max(s["hours_hedged"], 1)
        print(
            f"  {name:<12}  {s['hours_hedged']:>13,}  "
            f"${s['hedge_pnl']:>13,.0f}  ${s['net']:>13,.0f}  "
            f"${diff:>+12,.0f}  ${eff:>7.1f}"
        )

    model_benefit  = strategies["Model"]["net"]  - unhedged_net
    oracle_benefit = strategies["Oracle"]["net"] - unhedged_net
    if oracle_benefit != 0:
        pct = model_benefit / oracle_benefit * 100
        print(f"\n  Model captures {pct:.1f}% of the oracle hedge benefit")
        print(f"  Model commits {flagged_hours/oracle_hours:.1f}× as many hedged hours as oracle")
        print(f"  Model efficiency: ${model_benefit/flagged_hours:.1f}/hr hedged  "
              f"(oracle: ${oracle_benefit/oracle_hours:.1f}/hr)")

    # FP cost and FN exposure
    fp_mask = df["fin_quadrant"] == "FP_fin"
    fn_mask = df["fin_quadrant"] == "FN_fin"
    fp_hedge_cost = (-df.loc[fp_mask, "hedge_model"]).clip(lower=0).sum()
    fn_repl_cost  = df.loc[fn_mask, "replacement_cost"].sum()
    print(f"\n  False positive hedge costs  : ${fp_hedge_cost:>12,.0f}")
    print(f"  False negative unhedged loss: ${fn_repl_cost:>12,.0f}")

    # Monthly breakdown
    df["month_str"] = df["year_month"].astype(str)
    monthly = (
        df.groupby("month_str")
        .agg(
            net_unhedged   = ("net_unhedged",    "sum"),
            net_model      = ("net_model",       "sum"),
            net_oracle     = ("net_oracle",      "sum"),
            replacement_cost = ("replacement_cost", "sum"),
            hedge_model    = ("hedge_model",     "sum"),
            hedge_oracle   = ("hedge_oracle",    "sum"),
        )
        .reset_index()
        .sort_values("month_str")
    )

    return strategies, monthly, df

# =============================================================================
# FIGURES 16–19
# =============================================================================

def plot_futures_vs_spot(results: dict, output_dir: Path) -> None:
    """
    Figure 16 — Average futures price vs realised NG spot price per contract
    (all model-flagged hours).  Contango = futures > spot = hedge cost;
    backwardation = futures < spot = hedge gain.
    """
    labels  = [CONTRACT_LABELS[c] for c in CONTRACTS]
    fut_avg = [results[c]["avg_futures_mmbtu"] for c in CONTRACTS]
    spt_avg = [results[c]["avg_spot_mmbtu"]    for c in CONTRACTS]

    x     = np.arange(len(CONTRACTS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, fut_avg, width, label="Avg futures price", color=BLUE,   alpha=0.85)
    b2 = ax.bar(x + width / 2, spt_avg, width, label="Avg realised spot",  color=GREEN, alpha=0.85)

    for bar, val in zip(list(b1) + list(b2), fut_avg + spt_avg):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                f"${val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Price ($/MMBtu)")
    ax.set_title(
        "Average futures price vs realised NG spot — flagged hours\n"
        "C1 closest to spot; contango (futures > spot) generates hedge cost",
        fontsize=11,
    )
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}"))
    fig.tight_layout()
    save_fig(fig, output_dir / "figure16_futures_vs_spot.png")


def plot_monthly_hedge_pnl(flagged: pd.DataFrame, output_dir: Path) -> None:
    """
    Figure 17 — Monthly hedge P&L over the 2020–2021 test period,
    one line per futures contract (C1–C4).
    """
    flagged = flagged.copy()
    flagged["month_str"] = flagged["time"].dt.to_period("M").astype(str)

    monthly_pnl = (
        flagged.groupby("month_str")
        [[f"{c}_hedge_pnl" for c in CONTRACTS]]
        .sum()
        .reset_index()
        .sort_values("month_str")
    )

    x      = np.arange(len(monthly_pnl))
    colors = [BLUE, GREEN, AMBER, PURPLE]

    fig, ax = plt.subplots(figsize=(13, 5))
    for c, color in zip(CONTRACTS, colors):
        ax.plot(x, monthly_pnl[f"{c}_hedge_pnl"],
                color=color, linewidth=2, marker="o", markersize=4,
                label=CONTRACT_LABELS[c])

    ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.set_xticks(x[::2])
    ax.set_xticklabels(monthly_pnl["month_str"].tolist()[::2],
                       rotation=35, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(dollar_fmt))
    ax.set_ylabel("Hedge P&L ($)")
    ax.set_title(
        "Monthly hedge P&L by futures contract — flagged hours only\n"
        "Positive = spot exceeded futures price (hedge effective)",
        fontsize=11,
    )
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_fig(fig, output_dir / "figure17_monthly_hedge_pnl.png")


def plot_strategy_net_position(strategies: dict, flagged_hours: int,
                                oracle_hours: int, output_dir: Path) -> None:
    """
    Figure 18 — Total net position for the three strategies.
    Unhedged / Model-hedged / Oracle-hedged.
    """
    names   = list(strategies.keys())
    nets    = [strategies[n]["net"] for n in names]
    colors  = [GRAY, BLUE, GREEN]
    unhedged_net = strategies["Unhedged"]["net"]

    hours_labels = [
        f"Unhedged\n(0 hrs hedged)",
        f"Model\n({flagged_hours:,} hrs flagged)",
        f"Oracle\n({oracle_hours:,} hrs actual loss)",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, nets, color=colors, width=0.5, zorder=3)

    for bar, val in zip(bars, nets):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val - abs(val) * 0.02,
                f"${val/1e6:.2f}M",
                ha="center", va="top", fontsize=10, fontweight="500", color="white")

    for i, (name, col) in enumerate(zip(["Model", "Oracle"], [BLUE, GREEN])):
        diff = strategies[name]["net"] - unhedged_net
        sign = "+" if diff >= 0 else ""
        ax.annotate(
            f"{sign}${diff/1e3:.0f}K vs unhedged",
            xy=(i + 1, strategies[name]["net"]),
            xytext=(i + 1, strategies[name]["net"] + abs(unhedged_net) * 0.04),
            ha="center", fontsize=9, color=col, fontweight="500",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.8),
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(dollar_fmt))
    ax.axhline(0, color="#cccccc", linewidth=0.5)
    ax.set_xticklabels(hours_labels, fontsize=9)
    ax.set_title(
        "Total net position by hedging strategy\n"
        "(C1 futures, full 30 MWh/hr contract, all test hours)",
        fontsize=11,
    )
    ax.set_ylabel("Net position ($)")
    fig.tight_layout()
    save_fig(fig, output_dir / "figure18_strategy_net_position.png")


def plot_monthly_strategy(monthly: pd.DataFrame, output_dir: Path) -> None:
    """
    Figure 19 — Monthly net position for all three strategies.
    Green shading where model beats unhedged; red where model is worse.
    Includes a secondary axis with the model-minus-oracle delta and
    its cumulative sum to show where basis risk concentrated.
    """
    x = np.arange(len(monthly))
    labels = monthly["month_str"].tolist()

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    # Panel 1: monthly net position by strategy
    ax1 = axes[0]
    ax1.plot(x, monthly["net_unhedged"], color=GRAY,  linewidth=1.6,
             linestyle="--", label="Unhedged", marker=".", markersize=3)
    ax1.plot(x, monthly["net_model"],   color=BLUE,  linewidth=1.8,
             linestyle="solid", label="Model hedge (C1)", marker="o", markersize=3)
    ax1.plot(x, monthly["net_oracle"],  color=GREEN, linewidth=1.8,
             linestyle=(0, (4, 2)), label="Oracle hedge (C1)", marker="s", markersize=3)
    ax1.axhline(0, color="#cccccc", linewidth=0.6)
    ax1.fill_between(x, monthly["net_model"], monthly["net_unhedged"],
                     where=monthly["net_model"] > monthly["net_unhedged"],
                     alpha=0.12, color=GREEN, label="Model improves on unhedged")
    ax1.fill_between(x, monthly["net_model"], monthly["net_unhedged"],
                     where=monthly["net_model"] < monthly["net_unhedged"],
                     alpha=0.12, color=RED, label="Model worse than unhedged")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(labels[::2], rotation=35, ha="right", fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(dollar_fmt))
    ax1.set_title(
        "Monthly net position: Unhedged vs Model vs Oracle (C1 futures)\n"
        "Green shading = model hedge improves position  |  Red = model hedge hurts",
        fontsize=11,
    )
    ax1.set_ylabel("Net position ($)")
    ax1.legend(frameon=False, fontsize=9, ncol=3, loc="lower left")

    # Panel 2: model − oracle delta with cumulative
    ax2 = axes[1]
    delta = (monthly["net_model"] - monthly["net_oracle"]).tolist()
    colors_d = [GREEN if v >= 0 else RED for v in delta]
    ax2.bar(x, delta, color=colors_d, zorder=3, width=0.7)
    ax2.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels(labels[::2], rotation=35, ha="right", fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(dollar_fmt))
    ax2.set_title(
        "Monthly net position: Model minus Oracle (C1 futures)\n"
        "Green = model outperforms oracle  |  Red = model misses relative to oracle",
        fontsize=11,
    )
    ax2.set_ylabel("Model − Oracle ($)")

    ax2r = ax2.twinx()
    ax2r.spines["right"].set_visible(True)
    ax2r.spines["top"].set_visible(False)
    ax2r.plot(x, np.cumsum(delta), color=NAVY, linewidth=1.8,
              marker="o", markersize=3, label="Cumulative delta")
    ax2r.yaxis.set_major_formatter(mticker.FuncFormatter(dollar_fmt))
    ax2r.set_ylabel("Cumulative delta ($)", color=NAVY)
    ax2r.tick_params(axis="y", colors=NAVY)
    ax2r.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout(h_pad=3)
    save_fig(fig, output_dir / "figure19_monthly_strategy.png")

# =============================================================================
# STEP 4 — SUMMARY TABLE AND PRINT
# =============================================================================

def print_and_save_summary(
    results: dict, strategies: dict,
    flagged_hours: int, oracle_hours: int,
    output_dir: Path,
) -> None:
    """Print the summary table and save hedge_summary.csv."""
    unhedged_net = strategies["Unhedged"]["net"]

    rows = []
    for c, r in results.items():
        improvement = r["net_pos_hedged"] - r["net_pos_unhedged"]
        rows.append({
            "contract"            : r["label"],
            "avg_futures_mmbtu"   : round(r["avg_futures_mmbtu"], 4),
            "avg_spot_mmbtu"      : round(r["avg_spot_mmbtu"], 4),
            "total_hedge_pnl"     : round(r["total_hedge_pnl"], 2),
            "total_hedge_gain"    : round(r["total_hedge_gain"], 2),
            "total_hedge_cost"    : round(r["total_hedge_cost"], 2),
            "net_pos_unhedged"    : round(r["net_pos_unhedged"], 2),
            "net_pos_hedged"      : round(r["net_pos_hedged"], 2),
            "improvement"         : round(improvement, 2),
            "pct_hours_gain"      : round(r["pct_hours_gain"], 2),
            "pct_hours_cost"      : round(r["pct_hours_cost"], 2),
            "hedge_efficiency_per_hr": round(r["hedge_efficiency"], 2),
        })

    summary_df = pd.DataFrame(rows)
    out = output_dir / "hedge_summary.csv"
    summary_df.to_csv(out, index=False)
    print(f"  Saved: {out.name}")

    print(f"\n  Full contract-level summary:")
    print(f"  {'Contract':<22}  {'Hedge P&L':>12}  {'Hedge Gain':>11}  "
          f"{'Hedge Cost':>11}  {'Net Improv':>11}  {'$/hr':>8}")
    print("  " + "─" * 82)
    for _, r_df in summary_df.iterrows():
        print(
            f"  {r_df['contract']:<22}  "
            f"${r_df['total_hedge_pnl']:>11,.0f}  "
            f"${r_df['total_hedge_gain']:>10,.0f}  "
            f"${r_df['total_hedge_cost']:>10,.0f}  "
            f"${r_df['improvement']:>10,.0f}  "
            f"${r_df['hedge_efficiency_per_hr']:>7.1f}"
        )

    print(f"\n  Three-strategy summary (C1, all hours):")
    print(f"  {'Strategy':<12}  {'Hours':>8}  {'Hedge P&L':>12}  "
          f"{'Net Position':>14}  {'vs Unhedged':>13}")
    print("  " + "─" * 65)
    for name, s in strategies.items():
        diff = s["net"] - unhedged_net
        print(
            f"  {name:<12}  {s['hours_hedged']:>8,}  "
            f"${s['hedge_pnl']:>11,.0f}  ${s['net']:>13,.0f}  "
            f"${diff:>+12,.0f}"
        )

    model_b  = strategies["Model"]["net"]  - unhedged_net
    oracle_b = strategies["Oracle"]["net"] - unhedged_net

    print(f"\n  Model hedge efficiency   : ${model_b/flagged_hours:.1f} per hour hedged")
    print(f"  Oracle hedge efficiency  : ${oracle_b/oracle_hours:.1f} per hour hedged")
    if oracle_b != 0:
        print(f"  Model captures {model_b/oracle_b*100:.1f}% of oracle hedge benefit")
        print(f"  Model commits {flagged_hours/oracle_hours:.1f}x more hedged hours than oracle")

    total_repl = summary_df["total_hedge_pnl"].iloc[0] + results["NG_C1"]["total_replacement"]
    recovery   = results["NG_C1"]["total_hedge_pnl"]
    print(f"\n  C1 hedge P&L recovery rate:")
    print(f"    Total replacement costs     : ${results['NG_C1']['total_replacement']:>12,.0f}")
    print(f"    C1 net hedge P&L            : ${results['NG_C1']['total_hedge_pnl']:>12,.0f}")
    if results["NG_C1"]["total_replacement"] > 0:
        print(f"    Recovery rate               : "
              f"{results['NG_C1']['total_hedge_pnl']/results['NG_C1']['total_replacement']*100:.2f}%")
    print(
        f"\n  Note: The gas-electricity basis risk (R²=0.257 over 2020-2024) means\n"
        f"  NG futures are a natural but incomplete hedge.  Gas price movements\n"
        f"  explain only ~25% of electricity price variance, so gas futures alone\n"
        f"  cannot meaningfully absorb extreme electricity price spikes.  Meaningful\n"
        f"  risk reduction would require parametric insurance products, geographic\n"
        f"  diversification of wind assets, or a combination of instruments."
    )

# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NATURAL GAS FUTURES HEDGE SIMULATION")
    print("=" * 60)
    print(f"  Hedge quantity : {HEDGE_QTY:.2f} MMBtu / flagged hour")
    print(f"  (30 MWh × {HEAT_RATE} MMBtu/MWh heat rate)")

    print("\nStep 1 — Loading and merging data ...")
    df = load_data()

    print("\nStep 2 — Hedge stress test (C1–C4) ...")
    flagged, results = compute_hedge_stress_test(df)

    print("\nStep 3 — Three-strategy comparison (C1) ...")
    strategies, monthly, df_with_strategies = compute_strategies(df)

    flagged_hours = int(df["flagged"].sum())
    oracle_hours  = int((df["replacement_cost"] > 0).sum())

    print("\nStep 4 — Saving figures ...")
    plot_futures_vs_spot(results, OUTPUT_DIR)
    plot_monthly_hedge_pnl(flagged, OUTPUT_DIR)
    plot_strategy_net_position(strategies, flagged_hours, oracle_hours, OUTPUT_DIR)
    plot_monthly_strategy(monthly, OUTPUT_DIR)

    print("\nStep 5 — Saving summary table ...")
    print_and_save_summary(results, strategies, flagged_hours, oracle_hours, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
