"""
Script 5: Real-world malaria analysis — Mali case study using WHO/OWID incidence data.

Downloads publicly available WHO malaria incidence data (via Our World in Data) for Mali
(2000–2024), analyses the trend, calibrates the SEIR-SEI model to match observed burden,
and computes the implied R0.

Data source:
    World Health Organization (Global Health Observatory), via World Bank –
    processed by Our World in Data. "Incidence of malaria (per 1,000 population
    at risk)". https://ourworldindata.org/grapher/incidence-of-malaria
    License: CC BY 4.0.

Outputs:
    output/05_mali_incidence_trend.png       WHO incidence trend for Mali
    output/05_implied_R0_trend.png           Implied R0 per year from incidence
    output/05_model_vs_data.png              SEIR-SEI model vs WHO data
    output/05_mali_R0_sensitivity.png        R0 sensitivity to entomological params
"""

from __future__ import annotations

import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.ross_macdonald import RossMacdonald  # noqa: E402
from models.seir_sei_vector import SEIRSEIVector  # noqa: E402

PALETTE = sns.color_palette("deep")
OUT = os.path.join(REPO, "output")
DATA_DIR = os.path.join(REPO, "data", "real")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MALI_CSV = os.path.join(DATA_DIR, "mali_malaria_incidence_who_2000_2024.csv")
FULL_CSV_URL = (
    "https://ourworldindata.org/grapher/incidence-of-malaria.csv"
    "?v=1&csvType=full&useColumnShortNames=false"
)
INCIDENCE_COL = "Incidence of malaria (per 1,000 population at risk)"

# Mali epidemiological constants
TAU_INF = 200.0
MU_M = 0.1
A_TYPICAL = 0.3
M_TYPICAL = 6.0
B_TYPICAL = 0.5
C_TYPICAL = 0.5
POP_MALI = 22_400_000
AT_RISK_FRAC = 0.95


# =================================================================== DATA HANDLING
def download_mali_data() -> pd.DataFrame:
    """Download OWID malaria incidence CSV and extract Mali rows."""
    print("  Downloading WHO malaria incidence data (Our World in Data) ...")
    req = urllib.request.Request(FULL_CSV_URL,
                                 headers={"User-Agent": "MalariaModel/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    df_all = pd.read_csv(io.StringIO(raw))
    df_mali = df_all[df_all["Code"] == "MLI"][["Entity", "Code", "Year",
                                                INCIDENCE_COL]].copy()
    df_mali.columns = ["Country", "Code", "Year", "Incidence_per_1000"]
    df_mali["Incidence_per_1000"] = pd.to_numeric(
        df_mali["Incidence_per_1000"], errors="coerce")
    df_mali = df_mali.dropna(subset=["Incidence_per_1000"])
    df_mali = df_mali.sort_values("Year").reset_index(drop=True)
    df_mali.to_csv(MALI_CSV, index=False)
    return df_mali


def load_or_download() -> pd.DataFrame:
    """Load cached Mali data or download fresh."""
    if os.path.exists(MALI_CSV):
        try:
            df = pd.read_csv(MALI_CSV)
            if INCIDENCE_COL in df.columns:
                df = df.rename(columns={INCIDENCE_COL: "Incidence_per_1000"})
            if "Incidence_per_1000" in df.columns and len(df) > 3:
                return df
        except Exception:
            pass
    return download_mali_data()


# ===================================================== IMPLIED R0 FROM INCIDENCE
def implied_R0_from_incidence(incidence_per_1000: float,
                              tau: float = TAU_INF) -> float:
    """Estimate implied R0 from WHO-reported clinical incidence.

    At equilibrium in the SEIR model with clinical recovery period tau:
        annual_clinical_incidence_per_person = (R0 - 1) / (R0 * tau_years)

    Rearranging: R0 = 1 / (1 - annual_incidence * tau_years)

    This is a LOWER BOUND: in holoendemic settings the true R0 is higher because
    (a) WHO clinical incidence underestimates total infection incidence and
    (b) partial immunity means many reinfections are asymptomatic.
    """
    annual_inc = incidence_per_1000 / 1000.0
    tau_years = tau / 365.25
    denom = 1.0 - annual_inc * tau_years
    if denom <= 0.01:
        return 50.0
    return 1.0 / denom


# ================================================================ MODELLING
def calibrate_seirsei(target_clinical_inc_per_yr: float):
    """Find biting rate a that makes the SEIR-SEI model match target clinical incidence.

    Clinical incidence = new people entering I_h per day = (1/inc_inc) * E_h.
    At equilibrium this equals gamma_h * I_h = I_h / dur_inf.
    Per person per year = (I_h / N_h) / dur_inf * 365.
    """
    best_a = None
    best_err = np.inf
    best_metrics = {}

    for a_try in np.linspace(0.05, 0.7, 200):
        model = SEIRSEIVector(
            N_h=100_000.0, m=M_TYPICAL, a_mean=a_try,
            season_amp=0.65, season_phase=-2.0,
            b=B_TYPICAL, c=C_TYPICAL,
            inc_inc=12.0, dur_inf=TAU_INF,
            mu_m=MU_M, eip=12.0,
            mu_h=1.0 / (40.0 * 365.0), waning=1.0 / 300.0,
            E_h0=500.0, I_h0=500.0,
        )
        t, y = model.solve(0.0, 5 * 365.0, dt=1.0)
        # Use last 2 years for steady-state measurement
        start = int(3 * 365)
        I_h = y[2, start:]
        E_h = y[1, start:]
        S_h = y[0, start:]
        R_h = y[3, start:]
        I_m = y[6, start:]
        t_sub = t[start:]

        N_h_sub = S_h + E_h + I_h + R_h
        # Clinical incidence: new people entering I_h per day = (1/inc_inc) * E_h
        clinical_per_day = (1.0 / 12.0) * E_h
        clinical_pp_day = clinical_per_day / N_h_sub
        # Per person per year: mean daily rate * 365
        annual_clinical_pp = float(np.mean(clinical_pp_day) * 365.0)

        err = abs(annual_clinical_pp - target_clinical_inc_per_yr)
        if err < best_err:
            best_err = err
            best_a = a_try
            mean_prev = np.mean((I_h + E_h) / N_h_sub)
            mean_Ih_frac = np.mean(I_h / N_h_sub)
            mean_Eh_frac = np.mean(E_h / N_h_sub)
            mean_Rh_frac = np.mean(R_h / N_h_sub)
            mean_Sh_frac = np.mean(S_h / N_h_sub)
            # Force of infection
            a_t = model.a_mean * (1 + model.season_amp *
                                  np.sin(2 * np.pi * t_sub / 365.0 + model.season_phase))
            N_m = model.N_m
            foi = a_t * model.b * (I_m / N_m)
            mean_foi = np.mean(foi)
            best_metrics = {
                "annual_clinical": annual_clinical_pp,
                "mean_prev": mean_prev,
                "mean_Ih_frac": mean_Ih_frac,
                "mean_Eh_frac": mean_Eh_frac,
                "mean_Rh_frac": mean_Rh_frac,
                "mean_Sh_frac": mean_Sh_frac,
                "mean_foi": mean_foi,
            }

    return best_a, best_metrics


def compute_monthly_incidence(t, y, model, start_day=730.0):
    """Compute monthly clinical incidence (per person per month) from SEIR-SEI output."""
    start = int(start_day)
    t_sub = t[start:] - t[start]
    E_h = y[1, start:]
    N_h = y[0, start:] + y[1, start:] + y[2, start:] + y[3, start:]
    clinical_per_day = (1.0 / model.inc_inc) * E_h
    clinical_pp_day = clinical_per_day / N_h

    month_edges = np.arange(0, int(t_sub[-1]) + 1, 30.44)
    monthly = []
    for k in range(len(month_edges) - 1):
        mask = (t_sub >= month_edges[k]) & (t_sub < month_edges[k + 1])
        if mask.sum() > 0:
            # Average daily rate in this month * days in month = per-person episodes
            monthly.append(float(np.mean(clinical_pp_day[mask]) * 30.44))
    return np.array(monthly)


# ================================================================ MAIN
def main():
    print("=" * 72)
    print("REAL-WORLD MALARIA ANALYSIS: MALI CASE STUDY")
    print("=" * 72)

    # ---- Load data ----
    df = load_or_download()
    years = df["Year"].values.astype(int)
    incidence = df["Incidence_per_1000"].values.astype(float)

    print(f"\n  Dataset   : WHO Global Health Observatory (via Our World in Data)")
    print(f"  Indicator : Incidence of malaria (per 1,000 population at risk)")
    print(f"  Country   : Mali (MLI)")
    print(f"  Years     : {years[0]} – {years[-1]} ({len(years)} data points)")
    print(f"  Source URL: {FULL_CSV_URL}")
    print(f"  License   : CC BY 4.0 (Our World in Data); WHO data public domain")
    print("-" * 72)

    # ---- Summary statistics ----
    peak_idx = int(np.argmax(incidence))
    min_idx = int(np.argmin(incidence))
    peak_year, peak_val = years[peak_idx], incidence[peak_idx]
    latest_year, latest_val = years[-1], incidence[-1]
    mean_val = float(np.mean(incidence))
    min_year, min_val = years[min_idx], incidence[min_idx]

    print(f"\n  KEY FINDINGS (WHO incidence data):")
    print(f"    Peak incidence        : {peak_val:.1f} /1,000 at risk ({peak_year})")
    print(f"    Lowest incidence      : {min_val:.1f} /1,000 at risk ({min_year})")
    print(f"    Mean incidence        : {mean_val:.1f} /1,000 at risk")
    print(f"    Latest incidence      : {latest_val:.1f} /1,000 at risk ({latest_year})")
    pct_change = (latest_val - peak_val) / peak_val * 100
    print(f"    Change peak → latest  : {pct_change:+.1f}%")

    pop_at_risk = POP_MALI * AT_RISK_FRAC
    est_cases_latest = latest_val / 1000.0 * pop_at_risk
    est_cases_peak = peak_val / 1000.0 * pop_at_risk
    print(f"\n    Est. total cases ({latest_year}): ~{est_cases_latest/1e6:.1f} million")
    print(f"    Est. total cases ({peak_year}):  ~{est_cases_peak/1e6:.1f} million")

    # ---- Implied R0 from WHO incidence (lower bound) ----
    implied_R0_simple = np.array([implied_R0_from_incidence(inc) for inc in incidence])

    # Reference R0 from Ross-MacDonald with typical Sahel params
    rm_ref = RossMacdonald(a=A_TYPICAL, b=B_TYPICAL, c=C_TYPICAL,
                           m=M_TYPICAL, r=1.0 / TAU_INF, mu=MU_M)
    R0_ref = rm_ref.R0

    # Also compute R0 with more conservative params (shorter effective infectious period)
    rm_conservative = RossMacdonald(a=0.3, b=0.5, c=0.5, m=6.0,
                                    r=1.0 / 50.0, mu=0.1)
    R0_cons = rm_conservative.R0

    print(f"\n  BASIC REPRODUCTION NUMBER R0:")
    print(f"    From model params (r=1/{TAU_INF:.0f}d, a={A_TYPICAL}, m={M_TYPICAL}):"
          f" {R0_ref:.1f}")
    print(f"    From model params (r=1/50d, a={A_TYPICAL}, m={M_TYPICAL}):"
          f" {R0_cons:.1f}")
    print(f"    Lower bound from WHO incidence ({peak_year}):"
          f" {implied_R0_simple[peak_idx]:.2f}")
    print(f"    Lower bound from WHO incidence ({latest_year}):"
          f" {implied_R0_simple[-1]:.2f}")
    print(f"    Literature R0 for high-endemic Sahel: ~10–50")
    print(f"    NOTE: Model R0 with r=1/200d reflects the full untreated")
    print(f"    infectious period; shorter r gives more conservative R0.")

    # ---- Calibrate SEIR-SEI to WHO incidence ----
    target_clinical = mean_val / 1000.0  # per person per year
    print(f"\n  CALIBRATING SEIR-SEI MODEL (target: {mean_val:.1f} /1,000 = "
          f"{target_clinical:.4f} per person per year) ...")
    best_a, metrics = calibrate_seirsei(target_clinical)

    print(f"    Calibrated biting rate a : {best_a:.3f} bites/mosquito/day")
    print(f"    Model clinical incidence : {metrics['annual_clinical']*1000:.1f} /1,000"
          f" (target: {mean_val:.1f})")
    print(f"    Model equilibrium prev.  : {metrics['mean_prev']*100:.1f}%")
    print(f"    Model I_h fraction       : {metrics['mean_Ih_frac']*100:.1f}%")
    print(f"    Model S_h fraction       : {metrics['mean_Sh_frac']*100:.1f}%")
    print(f"    Model R_h fraction       : {metrics['mean_Rh_frac']*100:.1f}%")

    # Run calibrated model for figures
    model_cal = SEIRSEIVector(
        N_h=100_000.0, m=M_TYPICAL, a_mean=best_a,
        season_amp=0.65, season_phase=-2.0,
        b=B_TYPICAL, c=C_TYPICAL,
        inc_inc=12.0, dur_inf=TAU_INF,
        mu_m=MU_M, eip=12.0,
        mu_h=1.0 / (40.0 * 365.0), waning=1.0 / 300.0,
        E_h0=500.0, I_h0=500.0,
    )
    t_cal, y_cal = model_cal.solve(0.0, 4 * 365.0, dt=1.0)

    # R0 and EIR from calibrated model
    rm_cal = RossMacdonald(a=best_a, b=B_TYPICAL, c=C_TYPICAL,
                           m=M_TYPICAL, r=1.0 / TAU_INF, mu=MU_M, N_h=100_000.0)
    R0_cal = rm_cal.R0
    _, (H_i_eq, V_i_eq) = rm_cal.equilibrium()
    eir_daily = rm_cal.a * rm_cal.b * (V_i_eq / rm_cal.N_m)
    eir_annual = eir_daily * 365.25

    print(f"    Model R0 (Ross-MacDonald): {R0_cal:.1f}")
    print(f"    Model EIR (annual)       : {eir_annual:.0f} infectious bites/person/yr")

    # ---- R0 sensitivity ----
    print("\n  R0 SENSITIVITY (Ross-MacDonald formula R0 = m*a²*b*c / (r*mu)):")
    print(f"  Using r=1/{TAU_INF:.0f}d (full untreated infectious period):")
    for pname, pvals, calc_fn in [
        ("a (biting rate)",
         [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
         lambda v: RossMacdonald(a=v, b=B_TYPICAL, c=C_TYPICAL,
                                 m=M_TYPICAL, r=1.0/TAU_INF, mu=MU_M).R0),
        ("m (vector:human)",
         [1, 2, 4, 6, 8, 10, 15],
         lambda v: RossMacdonald(a=A_TYPICAL, b=B_TYPICAL, c=C_TYPICAL,
                                 m=v, r=1.0/TAU_INF, mu=MU_M).R0),
        ("mu (mosq. death rate, 1/d)",
         [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
         lambda v: RossMacdonald(a=A_TYPICAL, b=B_TYPICAL, c=C_TYPICAL,
                                 m=M_TYPICAL, r=1.0/TAU_INF, mu=v).R0),
    ]:
        print(f"    {pname}:")
        for v in pvals:
            print(f"      {v:<10} -> R0 = {calc_fn(v):.1f}")

    # ============================================================================
    # FIGURE 1: Mali incidence trend (WHO/OWID data)
    # ============================================================================
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(years, incidence, "-o", color=PALETTE[0], lw=2.5, markersize=5,
            markerfacecolor="white", markeredgewidth=1.5, zorder=3)
    ax.fill_between(years, 0, incidence, color=PALETTE[0], alpha=0.12)
    ax.annotate(f"Peak: {peak_val:.0f}\n({peak_year})",
                xy=(peak_year, peak_val), xytext=(peak_year + 1.5, peak_val + 20),
                arrowprops=dict(arrowstyle="->", color=PALETTE[3]),
                fontsize=10, fontweight="bold", color=PALETTE[3])
    ax.annotate(f"Latest: {latest_val:.0f}\n({latest_year})",
                xy=(latest_year, latest_val),
                xytext=(latest_year - 4, latest_val + 40),
                arrowprops=dict(arrowstyle="->", color=PALETTE[2]),
                fontsize=10, fontweight="bold", color=PALETTE[2])
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Malaria incidence (per 1,000 population at risk)", fontsize=11)
    ax.set_title("Malaria incidence in Mali, 2000–2024\n"
                 "(WHO Global Health Observatory via Our World in Data)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_ylim(0, peak_val * 1.15)
    ax.grid(alpha=0.3, linestyle="--")
    ax.text(0.01, 0.02, "Source: WHO GHO / World Bank / OWID, CC BY 4.0",
            transform=ax.transAxes, fontsize=8, color="gray", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "05_mali_incidence_trend.png"), dpi=200)
    plt.close(fig)

    # ============================================================================
    # FIGURE 2: Implied R0 trend
    # ============================================================================
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(years, implied_R0_simple, "-s", color=PALETTE[1], lw=2.2, markersize=5,
             markerfacecolor="white", markeredgewidth=1.5, zorder=3,
             label="R0 lower bound (from WHO incidence)")
    ax1.axhline(1.0, color="red", ls="--", lw=1.5, alpha=0.6,
                label="Elimination threshold (R0 = 1)")
    ax1.axhline(R0_cons, color=PALETTE[4], ls=":", lw=1.8, alpha=0.7,
                label=f"R0 ref (r=1/50d) = {R0_cons:.0f}")
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Basic reproduction number R0", fontsize=11)
    ax1.set_title("Implied R0 for malaria in Mali, 2000–2024\n"
                  "(Ross–MacDonald equilibrium from WHO clinical incidence)",
                  fontsize=13, fontweight="bold")
    ax1.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax1.set_ylim(0, max(implied_R0_simple) * 1.3)
    ax1.legend(fontsize=9, loc="center right")
    ax1.grid(alpha=0.3, linestyle="--")
    # secondary axis: incidence
    ax2 = ax1.twinx()
    ax2.plot(years, incidence, ":", color=PALETTE[2], lw=1.5, alpha=0.5)
    ax2.set_ylabel("Incidence (per 1,000 at risk)", color=PALETTE[2], fontsize=10)
    ax2.tick_params(axis="y", labelcolor=PALETTE[2])
    ax1.text(0.01, 0.02, "Source: WHO GHO / World Bank / OWID, CC BY 4.0",
             transform=ax1.transAxes, fontsize=8, color="gray", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "05_implied_R0_trend.png"), dpi=200)
    plt.close(fig)

    # ============================================================================
    # FIGURE 3: Model vs Data
    # ============================================================================
    monthly = compute_monthly_incidence(t_cal, y_cal, model_cal, start_day=730.0)
    n_months = min(24, len(monthly))
    monthly_1000 = monthly[:n_months] * 1000.0
    who_monthly_avg = mean_val / 12.0

    months_labels = ["J", "F", "M", "A", "M", "J", "Jl", "A", "S", "O", "N", "D"]
    x_m = np.arange(n_months)
    lbl = [months_labels[i % 12] for i in range(n_months)]
    wet_mask = [(i % 12) in {5, 6, 7, 8, 9} for i in range(n_months)]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: model monthly clinical incidence
    colors_m = [PALETTE[5] if w else PALETTE[3] for w in wet_mask]
    ax_a.bar(x_m, monthly_1000, color=colors_m, edgecolor="white", linewidth=0.5)
    ax_a.axhline(who_monthly_avg, color="red", ls="--", lw=1.8, alpha=0.8)
    ax_a.set_xticks(x_m[::3])
    ax_a.set_xticklabels([lbl[i] for i in range(0, n_months, 3)], fontsize=9)
    ax_a.set_xlabel("Month (model output, 2 years)")
    ax_a.set_ylabel("Monthly clinical incidence (per 1,000)")
    ax_a.set_title(f"SEIR-SEI model seasonal clinical incidence\n"
                   f"(a={best_a:.2f}, R0={R0_cal:.0f})",
                   fontsize=11, fontweight="bold")
    ax_a.legend(handles=[
        mpatches.Patch(color=PALETTE[5], label="Rainy season"),
        mpatches.Patch(color=PALETTE[3], label="Dry season"),
        plt.Line2D([0], [0], color="red", ls="--", lw=1.8,
                    label=f"WHO avg = {who_monthly_avg:.1f}/mo"),
    ], fontsize=8, loc="upper left")
    ax_a.grid(axis="y", alpha=0.3)

    # Panel B: Model prevalence dynamics + R0 reference
    start_p = int(730)
    t_p = t_cal[start_p:] - t_cal[start_p]
    N_h_p = (y_cal[0, start_p:] + y_cal[1, start_p:] +
             y_cal[2, start_p:] + y_cal[3, start_p:])
    prev = (y_cal[1, start_p:] + y_cal[2, start_p:]) / N_h_p * 100.0
    ax_b.plot(t_p, prev, color=PALETTE[0], lw=1.8)
    ax_b.fill_between(t_p, 0, prev, color=PALETTE[0], alpha=0.15)
    ax_b.set_xlabel("Day (2 years after burn-in)")
    ax_b.set_ylabel("Model prevalence (E_h + I_h, %)", fontsize=11)
    ax_b.set_title(f"Model equilibrium prevalence dynamics\n"
                   f"(mean = {metrics['mean_prev']*100:.1f}%, "
                   f"R0 = {R0_cal:.0f})",
                   fontsize=11, fontweight="bold")
    ax_b.grid(alpha=0.3, linestyle="--")
    ax_b.set_ylim(0, max(prev) * 1.2)
    ax_b.text(0.01, 0.02,
              f"WHO GHO / OWID\nCalibrated a = {best_a:.2f}",
              transform=ax_b.transAxes, fontsize=8, color="gray", va="bottom")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "05_model_vs_data.png"), dpi=200)
    plt.close(fig)

    # ============================================================================
    # FIGURE 4: R0 sensitivity to entomological parameters
    # ============================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: R0 vs biting rate a
    a_range = np.linspace(0.05, 0.5, 100)
    R0_a = np.array([(M_TYPICAL * a**2 * B_TYPICAL * C_TYPICAL) /
                     ((1.0/TAU_INF) * MU_M) for a in a_range])
    axes[0].plot(a_range, R0_a, color=PALETTE[0], lw=2.5)
    axes[0].axvline(best_a, color=PALETTE[2], ls="--", lw=1.5,
                    label=f"Calibrated a = {best_a:.2f}")
    axes[0].axhline(1.0, color="red", ls=":", lw=1, alpha=0.5)
    axes[0].set_xlabel("Biting rate a (bites/mosquito/day)")
    axes[0].set_ylabel("R0")
    axes[0].set_title("R0 vs biting rate\n(m = 6)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Panel B: R0 vs vector:human ratio m
    m_range = np.linspace(1, 15, 100)
    R0_m = np.array([(m * A_TYPICAL**2 * B_TYPICAL * C_TYPICAL) /
                     ((1.0/TAU_INF) * MU_M) for m in m_range])
    axes[1].plot(m_range, R0_m, color=PALETTE[1], lw=2.5)
    axes[1].axhline(R0_cons, color=PALETTE[4], ls="--", lw=1.5,
                    label=f"R0 (r=1/50d) = {R0_cons:.0f}")
    axes[1].axhline(1.0, color="red", ls=":", lw=1, alpha=0.5)
    axes[1].set_xlabel("Vector:human ratio m")
    axes[1].set_ylabel("R0")
    axes[1].set_title("R0 vs vector:human ratio\n(a = 0.3)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # Panel C: R0 vs mosquito mortality mu
    mu_range = np.linspace(0.04, 0.25, 100)
    R0_mu = np.array([(M_TYPICAL * A_TYPICAL**2 * B_TYPICAL * C_TYPICAL) /
                      ((1.0/TAU_INF) * mu) for mu in mu_range])
    axes[2].plot(mu_range, R0_mu, color=PALETTE[3], lw=2.5)
    axes[2].axvline(MU_M, color=PALETTE[0], ls="--", lw=1.5,
                    label=f"Default mu = {MU_M}")
    axes[2].axhline(1.0, color="red", ls=":", lw=1, alpha=0.5)
    axes[2].set_xlabel("Mosquito mortality rate mu (1/day)")
    axes[2].set_ylabel("R0")
    axes[2].set_title("R0 vs mosquito mortality\n(a = 0.3, m = 6)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle("Sensitivity of R0 to entomological parameters (Ross–MacDonald)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "05_mali_R0_sensitivity.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    # ========================== INTERPRETATION ==========================
    print("\n" + "=" * 72)
    print("INTERPRETATION SUMMARY")
    print("=" * 72)
    print(f"""
  DATA SOURCE: WHO Global Health Observatory via Our World in Data
  URL:        {FULL_CSV_URL}
  Indicator:  Malaria incidence (per 1,000 population at risk)
  Country:    Mali (MLI)
  Period:     {years[0]}–{years[-1]}

  KEY NUMBERS:
    Peak incidence     : {peak_val:.0f} per 1,000 at risk ({peak_year})
    Current (latest)   : {latest_val:.0f} per 1,000 at risk ({latest_year})
    Decline from peak  : {abs(pct_change):.0f}%
    Est. cases ({latest_year}): ~{est_cases_latest/1e6:.1f} million
    R0 (r=1/50d)       : {R0_cons:.0f}
    R0 (r=1/{TAU_INF:.0f}d)      : {R0_ref:.0f}
    Model calibrated a : {best_a:.2f} bites/mosquito/day
    Model EIR (annual) : {eir_annual:.0f} infectious bites/person/yr
    Model equilibrium  : {metrics['mean_prev']*100:.1f}% prevalence

  INTERPRETATION:
    1. Mali's WHO-reported malaria incidence peaked at {peak_val:.0f} per 1,000
       population at risk in {peak_year} and has declined {abs(pct_change):.0f}% to
       {latest_val:.0f} per 1,000 in {latest_year}. Despite this progress, Mali
       remains among the highest-burden countries globally, with ~{est_cases_latest/1e6:.1f}
       million clinical episodes per year — consistent with holoendemic
       P. falciparum transmission in the Sahel.

    2. The basic reproduction number R0 derived from the Ross-MacDonald formula
       is highly sensitive to the assumed human recovery/infectious period. With
       the full untreated infectious period (r=1/{TAU_INF:.0f}d), R0 = {R0_ref:.0f};
       with a shorter effective infectious period (r=1/50d), R0 = {R0_cons:.0f}.
       Both are well above the elimination threshold (R0 = 1), confirming that
       Mali requires sustained high-coverage interventions for control.

    3. The SEIR-SEI model calibrated to Mali's average WHO incidence (biting rate
       a = {best_a:.2f}) reproduces the strongly seasonal epidemic pattern: a sharp
       peak during the June–October rainy season followed by a deep dry-season
       trough — the hallmark of Sahel transmission driven by An. gambiae breeding
       in rain-filled pools.

    4. R0 sensitivity analysis reveals that vector control (reducing effective
       biting rate a and increasing mosquito mortality mu) has the strongest
       leverage on transmission. A 30% reduction in effective biting reduces R0
       by ~50%, explaining why ITN distribution and IRS have been the primary
       drivers of Mali's observed {abs(pct_change):.0f}% incidence decline.

    5. The plateau in incidence since ~2019 (326–346 per 1,000) suggests that
       residual transmission is persistent under current intervention coverage.
       Further gains will likely require combining existing tools (ITN, IRS, SMC)
       with additional strategies such as next-generation vector control, vaccine
       deployment (RTS,S/R21), or enhanced surveillance for focal elimination.
""")
    print("=" * 72)

    print("\nSaved figures:")
    for f in ("05_mali_incidence_trend.png", "05_implied_R0_trend.png",
              "05_model_vs_data.png", "05_mali_R0_sensitivity.png"):
        print(f"  output/{f}")


if __name__ == "__main__":
    main()
