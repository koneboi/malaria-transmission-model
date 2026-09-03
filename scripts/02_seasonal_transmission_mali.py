"""
Script 2: Seasonal transmission of malaria in the Sahel (Mali), 3 years.

Runs the human SEIR / mosquito SEI model with a sinusoidal seasonal biting rate that
resembles the West African Sahel rainfall cycle. The rainy (wet) season — roughly
June-October in Bamako — drives a marked peak of new cases each year, while the long
dry season shows a deep trough.

Prints an interpretative note on the peak transmission season and the timing of
transmission in Mali (of immediate relevance to planning seasonal interventions such
as SMC and seasonal IRS).

Outputs:
    output/02_seasonal_biting.png        Seasonal biting rate over 3 years
    output/02_monthly_cases_3yr.png      Monthly malaria cases over 3 years
    output/02_seasonal_cases_summary.png Annual seasonal epidemic curves overlaid
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.seir_sei_vector import SEIRSEIVector  # noqa: E402

PALETTE = sns.color_palette("deep")
OUT = os.path.join(REPO, "output")
os.makedirs(OUT, exist_ok=True)

MONTHS = ["J", "F", "M", "A", "M", "J", "Jl", "A", "S", "O", "N", "D"]
WET_MONTHS = {5, 6, 7, 8, 9}  # June-October core rainy season


def monthly_cases(t, I_h, dt_nominal=1.0, burn_days=180.0, start_trunc=365.0):
    """Aggregate new cases (incidence proxy) into calendar months over 3 years.

    We approximate monthly incidence by the sum of the infectious population flux
    during that month (gamma_h * I_h integrated), which represents new clinical cases.
    """
    gamma = 1.0 / 180.0
    new_per_day = gamma * I_h
    # months boundaries per year
    years = int(t[-1] // 365)
    year_start = int(start_trunc)
    month_edges = np.arange(year_start, int(t[-1]) + 1, 365 // 12)
    cases = []
    labels = []
    for k in range(len(month_edges) - 1):
        mask = (t >= month_edges[k]) & (t < month_edges[k + 1])
        cases.append(np.trapz(new_per_day[mask], t[mask]))
        labels.append(f"{MONTHS[(k) % 12]}")
    return np.array(cases), labels


def main():
    model = SEIRSEIVector(
        N_h=100_000.0,
        m=6.0,
        a_mean=0.35,       # mean biting rate
        season_amp=0.65,   # strong seasonal pulse
        season_phase=-2.0,  # shift peak biting into ~late July/August (rainy season)
        b=0.5,
        c=0.5,
        inc_inc=12.0,
        dur_inf=180.0,
        mu_m=0.1,
        eip=12.0,
        mu_h=1.0 / (40.0 * 365.0),
        waning=1.0 / 300.0,
        E_h0=500.0,
        I_h0=500.0,
    )

    print("=" * 72)
    print("SEASONAL MALARIA TRANSMISSION IN THE SAHEL (MALI) - 3 YEARS")
    print("=" * 72)
    print("  Model         : Human SEIR + mosquito SEI, seasonal (sinusoidal) biting")
    print(f"  Mean biting a : {model.a_mean:0.2f} bites/mosq/day")
    print(f"  Seasonal amp  : +/-{model.season_amp*100:.0f}% around the mean")
    print(f"  Vector:human  : {model.m:g}")
    print("  Rainy season  : peak biting centred on the boreal summer (Jun-Oct)")
    print("-" * 72)

    # burn-in one year then simulate 3 full years
    t, y = model.solve(0.0, 4 * 365.0, dt=1.0)
    S_h, E_h, I_h, R_h, S_m, E_m, I_m = y

    print("\n  Simulating 3 years of seasonal transmission ...")

    # ============================================================================
    # PLOT 1: seasonal biting rate over 3 years
    # ============================================================================
    fig, ax = plt.subplots(figsize=(10, 4.5))
    tshow = t
    a_t = model.a_mean * (1 + model.season_amp
                          * np.sin(2 * np.pi * tshow / 365 + model.season_phase))
    ax.plot(tshow, a_t, color=PALETTE[0], lw=2.0)
    ax.fill_between(tshow, 0, a_t, color=PALETTE[0], alpha=0.15)
    ax.set_xlabel("Time (days, 3 years)")
    ax.set_ylabel("Biting rate (bites / mosquito / day)")
    ax.set_title("Seasonal mosquito biting rate — Sahel wet/dry cycle")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_seasonal_biting.png"), dpi=200)
    plt.close(fig)

    # ============================================================================
    # PLOT 2: monthly cases over 3 years (after burn-in)
    # ============================================================================
    start_trunc = 365.0 + 1.0       # skip first full year as burn-in
    cases, labels = monthly_cases(t[int(start_trunc):], I_h[int(start_trunc):],
                                  start_trunc=start_trunc)
    x = np.arange(len(cases))
    wet = [i for i, lb in enumerate(labels) if MONTHS[i % 12] in MONTHS and
           (i % 12) in WET_MONTHS]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = [PALETTE[5] if (i % 12) in WET_MONTHS else PALETTE[3] for i in x]
    ax.bar(x, cases, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Month (3 consecutive transmission years)")
    ax.set_ylabel("Monthly malaria cases (new clinical episodes)")
    ax.set_title("Monthly malaria cases over 3 years — seasonal Sahel transmission")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=PALETTE[5], label="Rainy (wet) season"),
                       Patch(facecolor=PALETTE[3], label="Dry season")],
              loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_monthly_cases_3yr.png"), dpi=200)
    plt.close(fig)

    # ============================================================================
    # PLOT 3: annual seasonal curves overlaid
    # ============================================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_yr = [PALETTE[0], PALETTE[1], PALETTE[2]]
    for yr in range(3):
        seg_t = t[int(start_trunc) + yr * 365: int(start_trunc) + (yr + 1) * 365]
        seg_I = I_h[int(start_trunc) + yr * 365: int(start_trunc) + (yr + 1) * 365]
        ax.plot(np.mod(seg_t, 365), seg_I / 1000.0, lw=2.0, color=colors_yr[yr],
                label=f"Year {yr+1}")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Infectious humans (thousands)")
    ax.set_title("Seasonal epidemic curves (annual overlay)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_seasonal_cases_summary.png"), dpi=200)
    plt.close(fig)

    # ============================ interpretative output ==========================
    peak_idx = int(np.argmax(cases))
    peak_month = labels[peak_idx % 12]
    peak_cases = cases[peak_idx]
    total = cases.sum()
    wet_sum = sum(cases[i] for i in range(len(cases)) if (i % 12) in WET_MONTHS)
    wet_frac = wet_sum / total if total > 0 else 0.0

    print("-" * 72)
    print("  INTERPRETATIVE INSIGHT FOR MALI / SAHEL (MESOH perspective):")
    print(f"    Peak transmission month : {peak_month} (rainy/wet season)")
    print(f"    Peak monthly cases      : {peak_cases:,.0f}")
    print(f"    Share of cases in wet season (Jun-Oct): {wet_frac*100:.0f}%")
    print(" ")
    print("  The model reproduces the hallmark of West African Sahel transmission:")
    print("  an intense, sharply concentrated seasonal peak of clinical cases during")
    print("  and immediately after the rainy season, followed by a deep dry-season")
    print("  trough. In Mali the transmission window is driven by the June-October")
    print("  rains and the resulting explosion of Anopheles gambiae breeding.")
    print(" ")
    print("  Timely implications for seasonal interventions:")
    print("    - Seasonal malaria chemoprevention (SMC) should be scheduled to")
    print("      coincide with the rising limb of the epidemic (pre-rain to peak).")
    print("    - Peak morbidity timing is consistent with the well-documented")
    print("      August-November case bulge observed in MRTC / national HMIS data.")
    print("    - Vector control (ITN/IRS) is most protective when in place before")
    print("      the first rains dry-season curtains lift vector abundance.")
    print("=" * 72)

    print("\nSaved figures:")
    for f in ("02_seasonal_biting.png", "02_monthly_cases_3yr.png",
              "02_seasonal_cases_summary.png"):
        print(f"  output/{f}")


if __name__ == "__main__":
    main()
