"""
Intervention / control measure impact on malaria transmission.

Implements the effect of the three principal malaria control tools:

- **ITN (insecticide-treated bed nets):** coverage `itn_cov` with personal-protection
  efficacy `personal_protect` (fraction of bites prevented) and mosquito mortality
  effect `mosq_kill`. Net use reduces the effective biting rate reaching humans and
  adds to mosquito mortality where vectors bite humans indoors at night.

- **IRS (indoor residual spraying):** coverage `irs_cov` adding mosquito mortality for
  resting endophilic vectors.

- **ACT treatment:** coverage `act_cov` of clinical cases shortens the infectious period
  (recovery rate r is increased), reducing onward transmission.

The combined effect is summarised as the **effective reproduction number**

    Rc = R0 * (1 - effectiveness_bite) * (mu / (mu + added_mortality)) ...

where each coverage feeds a fractional reduction factor. We provide:

    reduce_transmission(...) -> fractional reduction, Rc, EIR modifier
    rc_over_grid(...)        -> effective reproduction contour over coverage space
    elimination_threshold(...) -> lowest coverage combination driving Rc < 1
"""

from __future__ import annotations

import numpy as np
from .ross_macdonald import RossMacdonald


def itn_reduction(itn_cov: float, personal_protect: float = 0.75,
                  mosq_kill: float = 0.20, biting_indoors: float = 0.80) -> float:
    """Fractional reduction in the human biting rate from ITN coverage.

    Assumes biting is indoor-biased (k = biting_indoors), that nets protect a fraction
    `personal_protect` of bites, and that a fraction `mosq_kill` of mosquitoes biting a
    net user are killed. Coverage is per-person bed net use.
    """
    # share of bites that would otherwise land on a net user (extrapolated from coverage)
    bite_share = itn_cov * biting_indoors
    return bite_share * personal_protect


def itn_mortality_effect(itn_cov: float, mosq_kill: float = 0.20,
                         biting_indoors: float = 0.80, mu: float = 0.1) -> float:
    """Additional per-day mosquito mortality induced by ITN coverage."""
    return itn_cov * biting_indoors * mosq_kill * np.clip(bite_share_factor(itn_cov), 0, 1)


def bite_share_factor(itn_cov: float) -> float:
    return itn_cov


def reduction_factor_from_mortality(mu_base: float, mu_added: float) -> float:
    """Survival-proportion reduction due to added mosquito mortality.

    Added mortality shortens adult survival; the reduction factor in vectorial capacity
    scales roughly as (mu / (mu + mu_added))^2 for the bite-squared terms, and linearly
    for the life-time of infectivity. We use the conservative power-1.5 compromise.
    """
    f = mu_base / (mu_base + mu_added)
    return f ** 1.5


class ControlMeasures:
    """Utility bundle to compute intervention impact and elimination thresholds."""

    def __init__(self, base: RossMacdonald | None = None, **kwargs):
        self.base = base if base is not None else RossMacdonald(**kwargs)

    def effective_biting_rate(self, itn_cov: float,
                              personal_protect: float = 0.75) -> float:
        """Reduced biting reaching humans under ITN coverage."""
        red = itn_reduction(itn_cov, personal_protect)
        return self.base.a * (1.0 - red)

    def effective_recovery_rate(self, act_cov: float,
                                dur_treated: float = 3.0) -> float:
        """Recovery rate (1/d) after scaling by ACT treatment coverage.

        Treated cases clear faster; the population recovery rate is a coverage-weighted
        blend of treated (fast) and untreated (natural) recovery.
        """
        nat = self.base.r
        fast = 1.0 / dur_treated
        return (1 - act_cov) * nat + act_cov * fast

    def rc(self, itn_cov: float = 0.0, irs_cov: float = 0.0, act_cov: float = 0.0,
           personal_protect: float = 0.75, mosq_kill: float = 0.20,
           irs_kill: float = 0.30) -> float:
        """Effective reproduction number Rc under combined interventions."""
        # biting reduction from ITN
        bite_red = itn_reduction(itn_cov, personal_protect, mosq_kill)
        a_eff = self.base.a * (1.0 - bite_red)

        # added mosquito mortality from ITN + IRS
        mu_add = itn_cov * 0.80 * mosq_kill + irs_cov * irs_kill
        surv_factor = reduction_factor_from_mortality(self.base.mu, mu_add)

        # human recovery altered by ACT
        r_eff = self.effective_recovery_rate(act_cov)

        # effective reproduction number: biting squared, mortality (with additional
        # ITN/IRS mortality), and shortened infectious period from ACT treatment.
        rc = (self.base.m * a_eff**2 * self.base.b * self.base.c) / (r_eff * (self.base.mu + mu_add))
        return rc * surv_factor

    def transmission_reduction(self, itn_cov: float = 0.0, irs_cov: float = 0.0,
                               act_cov: float = 0.0) -> float:
        """Fractional reduction of new infections vs the no-intervention baseline.

        Reduction of Rc translates approximately into reduction of new infections.
        """
        rc0 = self.rc(0, 0, 0)
        rc1 = self.rc(itn_cov, irs_cov, act_cov)
        if rc0 <= 0:
            return 0.0
        return float(np.clip(1.0 - rc1 / rc0, 0.0, 1.0))

    def rc_grid(self, itn_covs: np.ndarray, irs_covs: np.ndarray,
                net_efficacy: float = 0.75, act_cov: float = 0.0) -> np.ndarray:
        """Effective reproduction contour over (itn_cov, irs_cov) grid."""
        Rc = np.empty((len(irs_covs), len(itn_covs)))
        for i, irs in enumerate(irs_covs):
            for j, itn in enumerate(itn_covs):
                Rc[i, j] = self.rc(itn_cov=itn, irs_cov=irs, act_cov=act_cov,
                                   personal_protect=net_efficacy)
        return Rc

    def elimination_threshold(self, itn_grid, irs_grid, net_efficacy: float = 0.75,
                              target: float = 1.0):
        """Find the minimum combined coverage that drives Rc < target."""
        Rc = self.rc_grid(itn_grid, irs_grid, net_efficacy=net_efficacy)
        # among combinations achieving Rc < target, pick the one with least total coverage
        eligible = np.argwhere(Rc < target)
        if len(eligible) == 0:
            return None
        best = min(eligible, key=lambda ij: itn_grid[ij[1]] + irs_grid[ij[0]])
        i, j = best
        return itn_grid[j], irs_grid[i], Rc[i, j]
