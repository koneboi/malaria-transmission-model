"""
Script 3: ITN & IRS impact on transmission — coverage sweeps and elimination threshold.

Sweeps bed-net (ITN) coverage 0-90% and indoor-residual-spray (IRS) coverage 0-60%,
computing the reduction in new infections and the effective reproduction number Rc.
Generates:
    - a reduction-in-new-infections heatmap vs (ITN, IRS) coverage
    - a contour map of the effective reproduction number Rc vs net coverage and net
      efficacy, showing the elimination threshold Rc < 1
    - a line sweep of Rc vs ITN coverage at fixed IRS levels showing threshold crossings
Prints the coverage combination required to push Rc below the elimination threshold.

Outputs:
    output/03_reduction_heatmap.png     New-infection reduction vs coverage
    output/03_rc_contour.png            Rc contour (net coverage x net efficacy)
    output/03_rc_itn_sweep.png          Rc vs ITN coverage, various IRS levels
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

from models.control_measures import ControlMeasures  # noqa: E402
from models.ross_macdonald import RossMacdonald  # noqa: E402

PALETTE = sns.color_palette("deep")
OUT = os.path.join(REPO, "output")
os.makedirs(OUT, exist_ok=True)

BASE = RossMacdonald(a=0.20, b=0.4, c=0.3, m=2.0, r=1.0 / 200.0, mu=0.12, N_h=100_000.0)


def main():
    cm = ControlMeasures(base=BASE)

    print("=" * 72)
    print("ITN / IRS CONTROL IMPACT ON MALARIA TRANSMISSION")
    print("=" * 72)
    print(f"  Baseline R0 (no control) : {BASE.R0:.2f}")
    print("-" * 72)

    itn_covs = np.linspace(0.0, 0.90, 46)
    irs_covs = np.linspace(0.0, 0.60, 31)

    # ---- reduction grid (ITN x IRS)
    reduction = np.empty((len(irs_covs), len(itn_covs)))
    Rc_grid = np.empty((len(irs_covs), len(itn_covs)))
    for i, irs in enumerate(irs_covs):
        for j, itn in enumerate(itn_covs):
            reduction[i, j] = cm.transmission_reduction(itn_cov=itn, irs_cov=irs)
            Rc_grid[i, j] = cm.rc(itn_cov=itn, irs_cov=irs)

    # ---- Rc contour vs (net coverage, net efficacy) at moderate IRS
    net_covs = np.linspace(0.0, 0.90, 46)
    effs = np.linspace(0.0, 0.95, 40)
    Rc_ce = np.empty((len(effs), len(net_covs)))
    fixed_irs = 0.30
    for i, eff in enumerate(effs):
        for j, ncov in enumerate(net_covs):
            Rc_ce[i, j] = cm.rc(itn_cov=ncov, irs_cov=fixed_irs,
                                personal_protect=eff)

    # ------------------------------------------------------------------ threshold
    thr = cm.elimination_threshold(itn_covs, irs_covs, net_efficacy=0.75)
    print("  ELIMINATION THRESHOLD (net personal efficacy 75%):")
    if thr is not None:
        print(f"    ITN coverage {thr[0]*100:.0f}% + IRS {thr[1]*100:.0f}% "
              f"=> Rc = {thr[2]:.2f}  (< 1)")
    else:
        print("    Not reachable within the swept coverage range 0-90% / 0-60%.")
    print("=" * 72)

    # ============================================================================
    # PLOT 1: reduction heatmap vs (ITN, IRS)
    # ============================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(itn_covs * 100, irs_covs * 100, reduction * 100,
                       shading="auto", cmap="YlGnBu")
    cs = ax.contour(itn_covs * 100, irs_covs * 100, reduction * 100,
                    levels=[25, 50, 75, 90, 95], colors="k", linewidths=0.8)
    ax.clabel(cs, fmt="%d%%", fontsize=9)
    ax.set_xlabel("ITN (bed net) coverage (%)")
    ax.set_ylabel("IRS (indoor residual spray) coverage (%)")
    ax.set_title("Reduction in new infections (%) vs vector-control coverage")
    fig.colorbar(im, ax=ax, label="Reduction in new infections (%)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_reduction_heatmap.png"), dpi=200)
    plt.close(fig)

    # ============================================================================
    # PLOT 2: Rc contour vs net coverage & net efficacy
    # ============================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(net_covs * 100, effs * 100, Rc_ce, shading="auto",
                       cmap="viridis")
    levels = [0.5, 1.0, 2.0, 4.0, 8.0]
    cs = ax.contour(net_covs * 100, effs * 100, Rc_ce, levels=levels, colors="w",
                    linewidths=1.2)
    ax.clabel(cs, fmt="%0.1f", fontsize=9, colors="white")
    # mark elimination region Rc < 1
    ax.contour(net_covs * 100, effs * 100, Rc_ce < 1.0, levels=[0.5],
               colors="red", linewidths=2.0)
    ax.text(0.05, 0.95, "Elimination region\n(Rc < 1)", transform=ax.transAxes,
            ha="left", va="top", fontsize=11, color="red", fontweight="bold")
    ax.set_xlabel("Net (ITN) coverage (%)")
    ax.set_ylabel("Net personal protection efficacy (%)")
    ax.set_title(f"Effective reproduction number Rc  (IRS = {fixed_irs*100:.0f}%)")
    fig.colorbar(im, ax=ax, label="Effective reproduction Rc")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_rc_contour.png"), dpi=200)
    plt.close(fig)

    # ============================================================================
    # PLOT 3: Rc vs ITN coverage at several IRS levels
    # ============================================================================
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for irs, clr in zip([0.0, 0.2, 0.4, 0.6], PALETTE):
        line = [cm.rc(itn_cov=c, irs_cov=irs) for c in itn_covs]
        ax.plot(itn_covs * 100, line, lw=2.2, color=clr, label=f"IRS = {irs*100:.0f}%")
    ax.axhline(1.0, color="k", ls="--", lw=1.5, label="Elimination threshold (Rc = 1)")
    ax.set_xlabel("ITN (bed net) coverage (%)")
    ax.set_ylabel("Effective reproduction number Rc")
    ax.set_title("Rc vs ITN coverage at increasing IRS coverage")
    ax.set_ylim(0, max(3.0, BASE.R0 * 1.1))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_rc_itn_sweep.png"), dpi=200)
    plt.close(fig)

    print("\nSaved figures:")
    for f in ("03_reduction_heatmap.png", "03_rc_contour.png", "03_rc_itn_sweep.png"):
        print(f"  output/{f}")


if __name__ == "__main__":
    main()
