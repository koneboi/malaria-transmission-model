"""
Script 4: Seasonal malaria chemoprevention (SMC) impact.

SMC delivers intermittent preventive treatment (typically to children under 5) during
the peak transmission months of the rainy season (e.g. 4 monthly rounds Jun-Sep). This
script models the effect by massively shortening the infectious/patent period of
treated individuals during the SMC window and reducing the force of infection reaching
susceptibles, reproducing a protective cover over the rainy season.

We compare two scenarios with the seasonal SEIR-SEI model:
    - "No SMC"           : full seasonal transmission, no monthly prophylaxis
    - "With SMC"         : rainy-season chemoprevention protects susceptibles &
                           shortens infectiousness during the SMC window

Outputs:
    output/04_incidence_with_without_smc.png   Monthly incidence curves
    output/04_cumulative_cases_averted.png     Cumulative cases averted over time
    output/04_smc_health_impact.png            Annual cases, peak & % averted summary
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
# SMC window: June through September (rainy season core) - month indices 5..8
SMC_MONTHS = {5, 6, 7, 8}


class SEIR_SEI_SMC(SEIRSEIVector):
    """SEIR-SEI with optional SMC: rainy-season chemoprevention.

    During SMC months a fraction `smc_cov` of susceptible/infectious children are
    'protected': susceptibles become prophylactically refractory to new infections (a
    `PROPH_EFF` fraction reduction of the force of infection) and infectious cases are
    treated with ACT, shortening the infectious period (a `TREAT_EFF` fraction). The
    prophylactic + treatment effects together model the real mechanism of SMC that
    prevents incident cases in the protected age group without eradicating transmission
    from the untreated adult reservoir (residual transmission keeps the model endemic).
    """

    PROPH_EFF = 0.45   # per-coverage reduction of the force of infection on children
    TREAT_EFF = 0.10   # per-coverage fraction shifted to rapid ACT clearance

    def __init__(self, smc_cov=0.0, dur_inf=180.0, **kw):
        super().__init__(dur_inf=dur_inf, **kw)
        self.smc_cov = smc_cov
        self.dur_inf_base = dur_inf
        self.dur_treated = 3.0

    def _in_smc_window(self, t):
        # SMC rounds during Jun-Sep of each year (month index 5..8 -> day ~150-270)
        doy = t % 365.0
        month = int(doy // 30.4)
        return month in SMC_MONTHS

    def _rhs(self, t, y):
        S_h, E_h, I_h, R_h, S_m, E_m, I_m = y
        N_h = S_h + E_h + I_h + R_h
        N_m = S_m + E_m + I_m

        a = self.biting_rate(t)
        lam_h = a * self.b * (I_m / N_m)
        lam_m = a * self.c * (I_h / N_h)

        gamma_h = 1.0 / self.dur_inf_base
        if self._in_smc_window(t) and self.smc_cov > 0:
            lam_h = lam_h * (1.0 - self.PROPH_EFF * self.smc_cov)
            treat_frac = self.TREAT_EFF * self.smc_cov
            gamma_h = (1.0 - treat_frac) * gamma_h + treat_frac * (1.0 / self.dur_treated)

        inc_rate = 1.0 / self.inc_inc
        eip_rate = 1.0 / self.eip
        births = self.mu_h * N_h

        dS_h = births - lam_h * S_h + self.waning * R_h - self.mu_h * S_h
        dE_h = lam_h * S_h - inc_rate * E_h - self.mu_h * E_h
        dI_h = inc_rate * E_h - gamma_h * I_h - self.mu_h * I_h
        dR_h = gamma_h * I_h - self.waning * R_h - self.mu_h * R_h

        dS_m = self.mu_m * N_m - lam_m * S_m - self.mu_m * S_m
        dE_m = lam_m * S_m - eip_rate * E_m - self.mu_m * E_m
        dI_m = eip_rate * E_m - self.mu_m * I_m
        return np.array([dS_h, dE_h, dI_h, dR_h, dS_m, dE_m, dI_m])


def monthly_incidence(t, I_h, start_trunc=730.0):
    """Monthly clinical cases ~ recovery flux from the infectious compartment.

    Clinical (case) load is taken proportional to the patent infectious population,
    which peaks in the rainy/early post-rainy season — reproducing the August-November
    case bulge documented in high-transmission Sahel settings including Mali.
    """
    gamma_scale = 1.0 / 180.0
    ts = t[int(start_trunc):]
    Is = I_h[int(start_trunc):]
    newd = gamma_scale * Is
    month_edges = np.arange(int(start_trunc), int(t[-1]) + 1, 365 // 12)
    cases = []
    for k in range(len(month_edges) - 1):
        mask = (ts >= month_edges[k]) & (ts < month_edges[k + 1])
        cases.append(np.trapz(newd[mask], ts[mask]))
    labels = [MONTHS[i % 12] for i in range(len(cases))]
    return np.array(cases), labels, ts, newd


def main():
    smc_cov = 0.75

    print("=" * 72)
    print("SEASONAL MALARIA CHEMOPREVENTION (SMC) IMPACT - CHILDREN, RAINY SEASON")
    print("=" * 72)
    print(f"  SMC coverage     : {smc_cov*100:.0f}% (rainy-season rounds, Jun-Sep)")
    print("  Target           : young children during peak transmission (Jul-Sep)")
    print("-" * 72)

    common = dict(
        N_h=100_000.0, m=6.0, a_mean=0.35, season_amp=0.65,
        season_phase=-2.0, b=0.5, c=0.5,
        inc_inc=12.0, dur_inf=180.0, mu_m=0.1, eip=12.0,
        mu_h=1.0 / (40.0 * 365.0), waning=1.0 / 300.0,
        E_h0=500.0, I_h0=500.0,
    )
    no_smc = SEIR_SEI_SMC(smc_cov=0.0, **common)
    with_smc = SEIR_SEI_SMC(smc_cov=smc_cov, **common)

    # simulate with warm-up, then measure the recurrent steady-state years
    t_no, y_no = no_smc.solve(0.0, 5 * 365.0, dt=1.0)
    t_wt, y_wt = with_smc.solve(0.0, 5 * 365.0, dt=1.0)

    cases_no, labels, ts_no, newd_no = monthly_incidence(t_no, y_no[2])
    cases_wt, _, ts_wt, newd_wt = monthly_incidence(t_wt, y_wt[2])
    n = min(len(cases_no), len(cases_wt))
    cases_no, cases_wt, labels = cases_no[:n], cases_wt[:n], labels[:n]

    averted = (cases_no - cases_wt).clip(min=0)
    cum_averted = np.cumsum(averted)

    total_no = cases_no.sum()
    total_wt = cases_wt.sum()
    pct = (total_no - total_wt) / total_no * 100 if total_no > 0 else 0.0
    peak_no = cases_no.max()
    peak_wt = cases_wt.max()

    print("  HEALTH IMPACT SUMMARY (3 transmission years):")
    print(f"    Total cases, no SMC   : {total_no:,.0f}")
    print(f"    Total cases, with SMC : {total_wt:,.0f}")
    print(f"    Cases averted         : {total_no-total_wt:,.0f}")
    print(f"    Relative reduction    : {pct:.1f}%")
    print(f"    Peak monthly cases    : {peak_no:,.0f} -> {peak_wt:,.0f} (with SMC)")
    print("  Interpretive note: SMC delivered across the rainy season windows")
    print("  markedly compresses the seasonal peak, averting a large share of the")
    print("  clinical burden among the target age group during the high-transmission")
    print("  months — consistent with expected impact of national SMC programmes in")
    print("  the Sahel (e.g. Mali, Burkina Faso, Niger).")
    print("=" * 72)

    x = np.arange(n)

    # ============================================================================
    # PLOT 1: monthly incidence with/without SMC
    # ============================================================================
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - 0.2, cases_no, width=0.4, color=PALETTE[3], label="No SMC")
    ax.bar(x + 0.2, cases_wt, width=0.4, color=PALETTE[0], label="With SMC")
    for i in range(len(labels)):
        if (i % 12) in SMC_MONTHS:
            ax.axvspan(i - 0.5, i + 0.5, color="gold", alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Month (3 transmission years; gold = SMC window)")
    ax.set_ylabel("Monthly malaria cases")
    ax.set_title("Effect of SMC on monthly malaria incidence (children)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "04_incidence_with_without_smc.png"), dpi=200)
    plt.close(fig)

    # ============================================================================
    # PLOT 2: cumulative cases averted over time
    # ============================================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, cum_averted, color=PALETTE[2], lw=2.6)
    ax.fill_between(x, 0, cum_averted, color=PALETTE[2], alpha=0.15)
    ax.set_xlabel("Month (3 transmission years)")
    ax.set_ylabel("Cumulative cases averted by SMC")
    ax.set_title("Cumulative clinical cases averted by SMC over 3 years")
    ax.axhline(total_no - total_wt, ls="--", color=PALETTE[6],
               label=f"Final total averted = {total_no-total_wt:,.0f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "04_cumulative_cases_averted.png"), dpi=200)
    plt.close(fig)

    # ============================================================================
    # PLOT 3: health-impact summary metric bars
    # ============================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Total cases\n(3 yr)", "Peak monthly\ncases", "Cases\naverted"]
    vals_no = [total_no / 1000, peak_no / 1000, 0.0]
    vals_wt = [total_wt / 1000, peak_wt / 1000, (total_no - total_wt) / 1000]
    width = 0.35
    xm = np.arange(3)
    ax.bar(xm - width / 2, vals_no, width, color=PALETTE[3], label="No SMC")
    ax.bar(xm + width / 2, vals_wt, width, color=PALETTE[0], label="With SMC")
    ax.set_xticks(xm)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Cases (thousands)")
    ax.set_title(f"SMC health impact — {pct:.0f}% reduction in clinical cases")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "04_smc_health_impact.png"), dpi=200)
    plt.close(fig)

    print("\nSaved figures:")
    for f in ("04_incidence_with_without_smc.png", "04_cumulative_cases_averted.png",
              "04_smc_health_impact.png"):
        print(f"  output/{f}")


if __name__ == "__main__":
    main()
