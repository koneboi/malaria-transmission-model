"""
Human SEIR + mosquito SEI vector-borne malaria model with seasonal forcing.

Humans retain full Susceptible-Exposed-Infectious-Removed structure (SEIR), capturing
the intrinsic incubation period in humans. Mosquitoes follow Susceptible-Exposed-
Infectious (SEI): they acquire the parasite, undergo an extrinsic incubation period
(sporogony), become infectious and — crucially — **never recover**, remaining infectious
for their (short) adult life.

The biting rate is a sinusoidal function of time to represent the Sahel wet/dry seasonal
cycle:

    a(t) = a_mean * (1 + amp * sin(2*pi*(t - phase)/365))

This produces a distinct seasonal epidemic curve: a sharp pulse of cases during and just
after the rainy season and a long dry-season trough — characteristic of high seasonal
transmission in Mali.

Equation set (per-day derivatives):
    dS_h = - (a(t) * b * I_m/N_m) * S_h + sigma_h * E_h_recover... (see below)
    dE_h = lambda_h * S_h - (1/inc_inc) * E_h
    dI_h = (1/inc_inc) * E_h - gamma_h * I_h
    dR_h = gamma_h * I_h
    dS_m = mu_m * N_m - lambda_m * S_m - mu_m * S_m
    dE_m = lambda_m * S_m - (1/eip) * E_m - mu_m * E_m
    dI_m = (1/eip) * E_m - mu_m * I_m   # no recovery

lambda_h = a(t) * b * (I_m / N_m)          # force of infection on humans
lambda_m = a(t) * c * (I_h / N_h)          # force of infection on mosquitoes
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


class SEIRSEIVector:
    """Human SEIR + mosquito SEI model with seasonal (Sahel) biting forcing."""

    # state indices
    S_h, E_h, I_h, R_h = 0, 1, 2, 3
    S_m, E_m, I_m = 4, 5, 6

    def __init__(
        self,
        N_h: float = 100_000.0,
        m: float = 6.0,            # vector:human ratio
        a_mean: float = 0.35,      # mean biting rate (bites/mosquito/day)
        season_amp: float = 0.65,  # fractional amplitude of seasonal biting
        season_phase: float = 0.0, # phase (rad); peak near day ~150-200 (rainy season)
        b: float = 0.5,            # human infectibility per infectious bite
        c: float = 0.5,            # mosquito infectibility per bite on infectious human
        inc_inc: float = 12.0,     # human intrinsic incubation period (days)
        dur_inf: float = 180.0,    # human infectious period (days) -> gamma_h
        mu_m: float = 0.1,         # mosquito death rate (1/d)
        eip: float = 12.0,         # extrinsic incubation period (sporogony, days)
        mu_h: float = 1.0 / (40.0 * 365.0),  # human birth/death rate (replenishes susceptibles)
        waning: float = 0.0,       # rate of immunity waning R -> S (0 = lifelong immunity)
        E_h0: float = 500.0,
        I_h0: float = 500.0,
    ):
        self.N_h = N_h
        self.m = m
        self.N_m = m * N_h
        self.a_mean = a_mean
        self.season_amp = season_amp
        self.season_phase = season_phase
        self.b = b
        self.c = c
        self.inc_inc = inc_inc
        self.dur_inf = dur_inf
        self.gamma_h = 1.0 / dur_inf
        self.mu_m = mu_m
        self.eip = eip
        self.mu_h = mu_h
        self.waning = waning
        self.E_h0 = E_h0
        self.I_h0 = I_h0

    # ------------------------------------------------------------------ forcing
    def biting_rate(self, t: float) -> float:
        """Seasonal biting rate: sinusoidal Sahel wet/dry cycle."""
        return self.a_mean * (1.0 + self.season_amp
                              * np.sin(2.0 * np.pi * t / 365.0 + self.season_phase))

    def force_of_infection(self, t: float, y: np.ndarray) -> float:
        """Daily per-capita force of infection applied to susceptible humans."""
        _, _, I_h, _, S_m, E_m, I_m = y
        return self.biting_rate(t) * self.b * (I_m / (S_m + E_m + I_m))

    def incidence_daily(self, t: float, y: np.ndarray) -> float:
        """Daily number of NEW human infections at time t (clinical-case incidence)."""
        return self.force_of_infection(t, y) * y[self.S_h]

    # ------------------------------------------------------------------ dynamics
    def _rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        S_h, E_h, I_h, R_h, S_m, E_m, I_m = y
        N_h = S_h + E_h + I_h + R_h
        N_m = S_m + E_m + I_m

        a = self.biting_rate(t)
        lam_h = a * self.b * (I_m / N_m)          # force of infection on humans
        lam_m = a * self.c * (I_h / N_h)          # force of infection on mosquitoes

        inc_rate = 1.0 / self.inc_inc
        eip_rate = 1.0 / self.eip
        births = self.mu_h * N_h  # constant inflow of new susceptibles (demographic)

        dS_h = births - lam_h * S_h + self.waning * R_h - self.mu_h * S_h
        dE_h = lam_h * S_h - inc_rate * E_h - self.mu_h * E_h
        dI_h = inc_rate * E_h - self.gamma_h * I_h - self.mu_h * I_h
        dR_h = self.gamma_h * I_h - self.waning * R_h - self.mu_h * R_h

        dS_m = self.mu_m * N_m - lam_m * S_m - self.mu_m * S_m
        dE_m = lam_m * S_m - eip_rate * E_m - self.mu_m * E_m
        dI_m = eip_rate * E_m - self.mu_m * I_m

        return np.array([dS_h, dE_h, dI_h, dR_h, dS_m, dE_m, dI_m])

    def solve(self, t_start: float, t_end: float, dt: float = 1.0):
        """Integrate from (t_start, t_end) in units of days; return (t, state)."""
        y0 = np.array([
            self.N_h - self.E_h0 - self.I_h0,
            self.E_h0,
            self.I_h0,
            0.0,
            self.N_m * 0.2,
            self.N_m * 0.1,
            self.N_m * 0.1,
        ])
        t_eval = np.arange(t_start, t_end + dt, dt)
        sol = solve_ivp(self._rhs, (t_start, t_end), y0, t_eval=t_eval,
                        method="LSODA", rtol=1e-6, atol=1e-9)
        return sol.t, sol.y
