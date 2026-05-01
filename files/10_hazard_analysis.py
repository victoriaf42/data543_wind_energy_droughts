"""
10_hazard_analysis.py
======================
Hazard analysis for wind energy droughts in ERCOT, 1980–2025.

Produces two figures from the Results section of the paper:

  Figure 4 — Seasonality of Low-Wind Drought Events (LZ_WEST)
  -------------------------------------------------------------
  Top panel  : count of drought events by meteorological season
  Bottom panel: mean and 95th-percentile composite severity score by season

  Severity score = duration (hours) × max(0.15 − avg_zone_cf, 0)
  This weights events by both duration and depth below the severe CF
  threshold (0.15), giving higher scores to long events with deeply
  suppressed wind output — the combination that most stresses grid
  operations and creates the greatest financial exposure for PPA producers.

  Figure 5 — Seasonal Low-Wind Event Probabilities (Load Zone West)
  -----------------------------------------------------------------
  2×2 grid of heat maps, one panel per meteorological season.
  Each cell shows P(≥1 drought event per season-year) meeting or exceeding
  a given combination of:
    x-axis: event duration threshold (hours, ≥)
    y-axis: average zone CF threshold (≤)

  Probabilities are computed via the Poisson approximation:
    P = 1 − exp(−λ)   where λ = event count / n_season_years

Input
-----
Pre-built drought event catalogues covering 1980–2025, one per CF threshold:

  data/drought_events/ALL_ZONES_events_1980_2025_CF0.30_cap50pct.csv
    Used for Figure 4 (seasonality and severity).
    Events identified when zone CF < 0.30 AND installed capacity under
    production ≤ 50% of zone total.

  data/drought_events/ALL_ZONES_events_1980_2025_CF0.15_cap50pct.csv
    Used for Figure 5 (exceedance probability surfaces).
    Same event definition applied at the more severe CF0.15 threshold,
    which isolates the deeply suppressed wind conditions most likely
    to generate replacement cost losses under a physical PPA.

Expected columns in both files:
  start_time   datetime  event start timestamp
  load_zone    str       LZ_WEST | LZ_NORTH | LZ_SOUTH | LZ_HOUSTON
  duration     float     event duration in hours
  avg_zone_cf  float     mean capacity factor across the zone during event

Season assignment
-----------------
Meteorological seasons (Northern Hemisphere):
  Winter  Dec / Jan / Feb
  Spring  Mar / Apr / May
  Summer  Jun / Jul / Aug
  Fall    Sep / Oct / Nov

December observations are credited to the following year's Winter
season-year count so each Winter season-year is complete.

Usage
-----
    python files/10_hazard_analysis.py

Outputs saved to results/hazard/:
  figure4_seasonality_lz_west.png
  figure5_exceedance_probabilities_lz_west.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input catalogues — CF0.30 for Figure 4, CF0.15 for Figure 5
FILE_CF030 = Path("data/drought_events/ALL_ZONES_events_1980_2025_CF0.30_cap50pct.csv")
FILE_CF015 = Path("data/drought_events/ALL_ZONES_events_1980_2025_CF0.15_cap50pct.csv")

OUTPUT_DIR = Path("results/hazard")

# Load zone for both figures (the zone with largest installed wind capacity)
ZONE = "LZ_WEST"

# Severity score reference threshold (Figure 4)
SEVERITY_CF_THRESHOLD = 0.15

# Threshold grid resolution (Figure 5)
DUR_GRID_N = 120   # number of duration threshold steps
CF_GRID_N  = 80    # number of CF threshold steps

SEASONS_ORDER = ["Winter", "Spring", "Summer", "Fall"]

# =============================================================================
# SHARED HELPERS
# =============================================================================

def load_catalogue(path: Path, zone: str) -> pd.DataFrame:
    """Load and clean a drought event catalogue, filtered to *zone*."""
    df = pd.read_csv(path)
    df["duration"]    = pd.to_numeric(df["duration"],    errors="coerce")
    df["avg_zone_cf"] = pd.to_numeric(df["avg_zone_cf"], errors="coerce")
    df["start_time"]  = pd.to_datetime(df["start_time"], errors="coerce")

    df = df[
        (df["duration"]    > 0) &
        df["avg_zone_cf"].notna() &
        df["start_time"].notna()
    ].copy()

    if zone is not None:
        df = df[df["load_zone"] == zone].copy()

    yr_min = int(df["start_time"].dt.year.min())
    yr_max = int(df["start_time"].dt.year.max())
    print(f"  Loaded {len(df):,} events  ({yr_min}–{yr_max})  zone={zone or 'ALL'}")
    return df


def assign_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'season' and 'season_year' columns."""
    df = df.copy()
    m = df["start_time"].dt.month
    df["season"] = np.select(
        [m.isin([12, 1, 2]), m.isin([3, 4, 5]),
         m.isin([6, 7, 8]),  m.isin([9, 10, 11])],
        SEASONS_ORDER,
        default=np.nan,
    )
    df = df[df["season"].notna()].copy()

    # December belongs to the following year's Winter season-year
    df["season_year"] = df["start_time"].dt.year
    dec_mask = (df["season"] == "Winter") & (df["start_time"].dt.month == 12)
    df.loc[dec_mask, "season_year"] = df.loc[dec_mask, "season_year"] + 1
    return df


def season_year_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return the number of distinct season-years for each season."""
    return df.groupby("season")["season_year"].nunique().to_dict()


# =============================================================================
# FIGURE 4 — SEASONALITY OF DROUGHT EVENTS (LZ_WEST)
# =============================================================================

def plot_seasonality(output_dir: Path) -> None:
    """
    Two-panel figure for LZ_WEST.

    Top    — count of drought events by season (CF0.30 catalogue)
    Bottom — mean and 95th-percentile composite severity score by season

    Severity = duration × max(0.15 − avg_zone_cf, 0)
    """
    print("\nFigure 4 — Seasonality of Drought Events (LZ_WEST)")
    df = load_catalogue(FILE_CF030, ZONE)
    df = assign_seasons(df)

    # Composite severity score
    df["severity_score"] = df["duration"] * np.maximum(
        SEVERITY_CF_THRESHOLD - df["avg_zone_cf"], 0.0
    )

    season_counts   = df["season"].value_counts().reindex(SEASONS_ORDER)
    season_mean_sev = df.groupby("season")["severity_score"].mean().reindex(SEASONS_ORDER)
    season_p95_sev  = df.groupby("season")["severity_score"].quantile(0.95).reindex(SEASONS_ORDER)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Seasonality of Low-Wind Drought Events ({ZONE})", y=0.98)

    # Top panel — event counts
    axes[0].bar(SEASONS_ORDER, season_counts.values, color="#3A7EBF")
    axes[0].set_ylabel("Number of Events")
    axes[0].set_title("Seasonal Distribution of Drought Events")
    count_max = season_counts.max()
    for i, v in enumerate(season_counts.values):
        axes[0].text(
            i, v + count_max * 0.02, f"{int(v)}", ha="center", va="bottom"
        )

    # Bottom panel — severity
    x     = np.arange(len(SEASONS_ORDER))
    width = 0.35
    axes[1].bar(x - width / 2, season_mean_sev.values,
                width=width, label="Mean Severity",          color="#3A7EBF")
    axes[1].bar(x + width / 2, season_p95_sev.values,
                width=width, label="95th Percentile Severity", color="#E07B2A")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(SEASONS_ORDER)
    axes[1].set_ylabel("Severity Score")
    axes[1].set_title("Seasonal Severity of Drought Events")
    axes[1].legend()

    plt.tight_layout()
    out = output_dir / "figure4_seasonality_lz_west.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # Print summary statistics
    print(f"\n  Season counts: {season_counts.to_dict()}")
    print(f"  95th-pct severity: {season_p95_sev.round(2).to_dict()}")


# =============================================================================
# FIGURE 5 — SEASONAL EXCEEDANCE PROBABILITY SURFACES (LZ_WEST)
# =============================================================================

def _prob_surface(
    dur: np.ndarray,
    cf: np.ndarray,
    dur_grid: np.ndarray,
    cf_grid: np.ndarray,
    n_season_years: int,
) -> np.ndarray:
    """
    Compute P(≥1 event per season-year) on a (duration × CF) threshold grid.

    For each (dur_threshold, cf_threshold) pair:
      λ = count(duration ≥ dur_thr AND avg_zone_cf ≤ cf_thr) / n_season_years
      P = 1 − exp(−λ)   [Poisson approximation]

    Returns array of shape (len(dur_grid), len(cf_grid)).
    """
    prob = np.zeros((len(dur_grid), len(cf_grid)), dtype=float)
    for i, d_thr in enumerate(dur_grid):
        dur_mask = dur >= d_thr
        if not dur_mask.any():
            continue
        cf_sub = cf[dur_mask]
        for j, c_thr in enumerate(cf_grid):
            count = int(np.sum(cf_sub <= c_thr))
            if count > 0:
                lam = count / n_season_years
                prob[i, j] = 1.0 - np.exp(-lam)
    return prob


def plot_exceedance_probabilities(output_dir: Path) -> None:
    """
    2×2 heat-map grid — one panel per meteorological season.

    Each panel shows P(≥1 drought event per season-year) across a grid of
    event duration (x-axis, hours, ≥) and average zone CF (y-axis, ≤)
    thresholds, using the CF0.15 catalogue for LZ_WEST.
    """
    print("\nFigure 5 — Seasonal Exceedance Probabilities (LZ_WEST)")
    df = load_catalogue(FILE_CF015, ZONE)
    df = assign_seasons(df)

    sy_counts = season_year_counts(df)

    # Shared threshold grids across all four panels
    dur_max = float(df["duration"].max())
    cf_lo   = max(0.0, float(df["avg_zone_cf"].min()))
    cf_hi   = min(1.0, float(df["avg_zone_cf"].max()))

    dur_grid = np.linspace(1.0, dur_max, DUR_GRID_N)
    cf_grid  = np.linspace(cf_lo, cf_hi, CF_GRID_N)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    im_ref = None

    for ax, season in zip(axes, SEASONS_ORDER):
        s_df = df[df["season"] == season].copy()
        n_sy = max(sy_counts.get(season, 0), 1)

        dur = s_df["duration"].to_numpy()
        cf  = s_df["avg_zone_cf"].to_numpy()

        prob = _prob_surface(dur, cf, dur_grid, cf_grid, n_sy)

        im = ax.imshow(
            prob.T,
            origin="lower",
            aspect="auto",
            extent=[dur_grid[0], dur_grid[-1], cf_grid[0], cf_grid[-1]],
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        im_ref = im

        ax.set_title(
            f"{season} (n={len(s_df):,}, season-years={n_sy})"
        )
        ax.set_xlim(dur_grid[0], dur_grid[-1])
        ax.set_ylim(cf_lo, cf_hi)

    axes[0].set_ylabel("Average Zone CF Threshold (≤)")
    axes[2].set_ylabel("Average Zone CF Threshold (≤)")
    axes[2].set_xlabel("Event Duration Threshold (hours, ≥)")
    axes[3].set_xlabel("Event Duration Threshold (hours, ≥)")

    fig.suptitle("Seasonal Low-Wind Event Probabilities (Load Zone West)", y=0.97)
    fig.subplots_adjust(right=0.88, top=0.88)

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar    = fig.colorbar(im_ref, cax=cbar_ax)
    cbar.set_label("Probability of ≥1 Event per Season-Year")

    out = output_dir / "figure5_exceedance_probabilities_lz_west.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # Print season-year counts for reference
    print(f"\n  Season-years: {sy_counts}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for f in [FILE_CF030, FILE_CF015]:
        if not f.exists():
            raise FileNotFoundError(
                f"Drought event catalogue not found: {f}\n"
                "Generate this file from the ERA5 wind capacity factor time\n"
                "series (1980–2025) using the drought identification methodology\n"
                "described in the paper (CF threshold + 50% capacity cap)."
            )

    plot_seasonality(OUTPUT_DIR)
    plot_exceedance_probabilities(OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
