"""
Script 1: Classic Ross-MacDonald simulation to equilibrium.

Runs the Ross-MacDonald model until steady state and visualises the transient
dynamics of infected humans and infectious mosquitoes. Prints the basic reproduction
number R0 and the entomological inoculation rate (EIR, here per person per year) at
equilibrium. Saves three publication-quality PNG figures to output/.

Outputs:
    output/01_rm_infected_dynamics.png   Infected humans & mosquitoes over time
    output/01_rm_phase_equilibrium.png   Phase portrait toward equilibrium
    output/01_rm_eir_R0.png              EIR trajectory and R0 annotation
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

from models.ross_macdonald import RossMacdonald  # noqa: E402

PALETTE = sns.color_palette("deep")
OUT = os.path.join(REPO, "output")
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------- params
A = 0.3       # bites / mosquito / day  (Anopheles gambiae, Sahel)
B = 0.5       # human infectibility
C = 0.5       # mosquito infectibility
M = 6.0       # vector:human ratio
R = 1.0 / 200.0
MU = 0.1
N_H = 100_000.0


def main():
    model = RossMacdonald(a=A, b=B, c=C, m=M, r=R, mu=MU, N_h=N_H)

    print("=" * 72)
    print("CLASSIC ROSS-MACDONALD MALARIA MODEL")
    print("=" * 72)
    print(f"  Biting rate a            : {A:0.2f} bites/mosq/day")
    print(f"  Vector:human ratio (m)   : {M:g}")
    print(f"  Human recovery rate (r)  : {R:0.4f}/day  (~{1/R:.0f} days infectious)")
    print(f"  Mosquito mortality (mu)  : {MU:0.2f}/day  (~{1/MU:.0f} day lifespan)")
    print(f"  Human population         : {N_H:,.0f}")

    R0 = model.R0
    print("-" * 72)
    print(f"  BASIC REPRODUCTION NUMBER  R0 = {R0:.2f}")
    print("  (average secondary human infections per infected human in a")
    print("   fully susceptible population; R0 > 1 implies endemic transmission)")
    print("-" * 72)

    # ------------------------------------------------------------------ simulate
    t, y = model.solve((0.0, 2000.0), t_eval=np.linspace(0, 2000, 2001))
    H_s, H_i, V_s, V_i = y

    # equilibrium values
    _, (H_i_eq, V_i_eq) = model.equilibrium()
    eir_daily = model.a * model.b * (V_i_eq / model.N_m)
    eir_annual = eir_daily * 365.25

    print(f"  EQUILIBRIUM (after ~2000 days):")
    print(f"    Infected humans       : {H_i_eq:,.0f} ({H_i_eq/N_H*100:.1f}% of population)")
    print(f"    Infectious mosquitoes : {V_i_eq:,.0f} ({V_i_eq/model.N_m*100:.1f}% of vectors)")
    print(f"    EIR (daily)           : {eir_daily:.2f} infectious bites/person/day")
    print(f"    EIR (annual)          : {eir_annual:.0f} infectious bites/person/year")
    print(f"  Interpretive note: an annual EIR of ~{eir_annual:.0f} infectious bites")
    print("  per person is consistent with the intense, often highly seasonal")
    print("  transmission measured in parts of the Sahel, including Mali. With the")
    print("  large R0 here, the parasite invades essentially the whole population at")
    print("  a quasi-stable, high-prevalence equilibrium (so-called stable endemic,")
    print("  holo-endemic transmission).")
    print("=" * 72)

    # ============================================================================
    # PLOT 1: transient dynamics of infected humans and mosquitoes
    # ============================================================================
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t, H_i / 1000.0, color=PALETTE[0], lw=2.5,
             label="Infected humans (thousands)")
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Infected humans (thousands)", color=PALETTE[0])
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])
    ax1.axhline(H_i_eq / 1000.0, ls="--", color=PALETTE[0], alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(t, V_i / 1000.0, color=PALETTE[3], lw=2.5,
             label="Infectious mosquitoes (thousands)")
    ax2.set_ylabel("Infectious mosquitoes (thousands)", color=PALETTE[3])
    ax2.tick_params(axis="y", labelcolor=PALETTE[3])
    ax2.axhline(V_i_eq / 1000.0, ls="--", color=PALETTE[3], alpha=0.5)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", framealpha=0.9)
    ax1.set_title("Ross-MacDonald: infection dynamics to equilibrium")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_rm_infected_dynamics.png"), dpi=200)
    plt.close(fig)

    # ============================================================================
    # PLOT 2: phase portrait of infected humans vs infectious mosquitoes
    # ============================================================================
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(V_i / 1000.0, H_i / 1000.0, color=PALETTE[1], lw=2.0)
    ax.scatter([V_i_eq / 1000.0], [H_i_eq / 1000.0], color=PALETTE[2], s=120,
               zorder=5, label=f"Equilibrium (R0={R0:.1f})")
    ax.set_xlabel("Infectious mosquitoes (thousands)")
    ax.set_ylabel("Infected humans (thousands)")
    ax.set_title("Phase portrait of the Ross–MacDonald model")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_rm_phase_equilibrium.png"), dpi=200)
    plt.close(fig)

    # ============================================================================
    # PLOT 3: EIR trajectory + R0 annotation
    # ============================================================================
    eir_t = A * B * (V_i / model.N_m) * 365.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, eir_t, color=PALETTE[4], lw=2.5)
    ax.axhline(eir_annual, ls="--", color=PALETTE[4], alpha=0.5,
               label=f"Equilibrium EIR = {eir_annual:.0f}/yr")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("EIR (infectious bites / person / year)")
    ax.set_title(f"Entomological Inoculation Rate trajectory  (R0 = {R0:.2f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_rm_eir_R0.png"), dpi=200)
    plt.close(fig)

    print("Saved figures:")
    for f in ("01_rm_infected_dynamics.png", "01_rm_phase_equilibrium.png",
              "01_rm_eir_R0.png"):
        print(f"  output/{f}")


if __name__ == "__main__":
    main()
