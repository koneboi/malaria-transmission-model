"""
Classic Ross-MacDonald malaria model.

The Ross-MacDonald model couples a susceptible/infectious human compartment with a
susceptible/infectious mosquito compartment. It is the foundational mathematical
description of Plasmodium transmission and underpins key entomological metrics such as
the basic reproduction number R0, the entomological inoculation rate (EIR) and the
vectorial capacity.

Humans:
    dH_s/dt = - b * m * a * (H_i / H_total) * H_s + r * H_i
    dH_i/dt = + b * m * a * (H_i / H_total) * H_s - r * H_i          (nonlinear)

   (linearised for new infections)
    dH_i/dt = a * b * m * V_i - r * H_i
    dV_s/dt = mu * V_total - a * c * (H_i / H_total) * V_s - mu * V_s
    dV_i/dt = a * c * (H_i / H_total) * V_s - mu * V_i

where:
    a  : mosquito biting rate (bites per mosquito per day)
    b  : probability a bite on an infectious mosquito transmits (human infectibility)
    c  : probability a bite on an infectious human infects a mosquito
    m  : vector:human ratio (mosquitoes per human)
    r  : human recovery rate (1/duration of infectiousness)
    mu : mosquito death rate (1/average mosquito lifespan)

R0 (Ross-MacDonald / Macdonald form):
    R0 = (m * a^2 * b * c) / (r * mu)

EIR equilibrium:
    EIR = a * b * m * (V_i / V_total)   [infectious bites per person per day]
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


class RossMacdonald:
    """Classic Ross-MacDonald susceptible/infectious human + mosquito model."""

    def __init__(
        self,
        a: float = 0.3,
        b: float = 0.5,
        c: float = 0.5,
        m: float = 5.0,
        r: float = 1.0 / 200.0,
        mu: float = 0.1,
        N_h: float = 100_000.0,
    ):
        """
        Parameters
        ----------
        a : mosquito biting rate (bites/mosquito/day), 0.2-0.5 in the Sahel
        b : probability of human infection from an infectious bite
        c : probability a mosquito becomes infected from an infectious human
        m : vector:human ratio (mosquitoes per human), 2-20 typical
        r : human recovery rate (1/d)
        mu: mosquito death rate (1/d)
        N_h: total human population size
        """
        self.a = a
        self.b = b
        self.c = c
        self.m = m
        self.r = r
        self.mu = mu
        self.N_h = N_h
        self.N_m = m * N_h  # total mosquito population

    # ------------------------------------------------------------------ metrics
    @property
    def R0(self) -> float:
        """Basic reproduction number (Macdonald form)."""
        return (self.m * self.a**2 * self.b * self.c) / (self.r * self.mu)

    @property
    def vectorial_capacity(self) -> float:
        """Daily rate of future inoculations from a currently infective human."""
        return self.m * self.a**2 * self.b * self.c / self.mu

    def eir_equilibrium(self) -> float:
        """Equilibrium EIR (infectious bites per person per year) of the ODE model."""
        r0 = self.R0
        if r0 <= 1:
            return 0.0
        p_inf_m = (r0 - 1.0) / (r0 * (self.a * self.c / self.mu + self.r / (self.m * self.mu)) /
                                (self.a * self.c / self.mu) + self.r / self.mu)
        # Simpler & robust direct steady-state solve:
        # At equilibrium with R0>1, V_i/V_m = (a*c*H_i/H)/(a*c*H_i/H + mu)
        # and H_i dominates. We solve numerically via the ODE instead.
        _, y_eq = self.equilibrium()
        H_i, V_i = y_eq
        return self.a * self.b * (V_i / self.N_m)

    # ------------------------------------------------------------------ dynamics
    def _rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        H_s, H_i, V_s, V_i = y
        N_h = H_s + H_i
        # new infections among humans
        new_h = self.a * self.b * (V_i / self.N_m) * H_s
        # new infections among mosquitoes
        new_m = self.a * self.c * (H_i / N_h) * V_s
        dH_s = -new_h + self.r * H_i
        dH_i = +new_h - self.r * H_i
        dV_s = self.mu * self.N_m - new_m - self.mu * V_s
        dV_i = +new_m - self.mu * V_i
        return np.array([dH_s, dH_i, dV_s, dV_i])

    def equilibrium(self, seed: float = 1e-4):
        """Integrate until steady state; return (final state array, final state tuple)."""
        N_h = self.N_h
        y0 = np.array([N_h * (1 - seed), N_h * seed, self.N_m * seed, self.N_m * seed])
        sol = solve_ivp(
            self._rhs,
            (0.0, 10_000.0),
            y0,
            method="LSODA",
            rtol=1e-6,
            atol=1e-8,
        )
        H_s, H_i, V_s, V_i = sol.y[:, -1]
        return np.array([H_s, H_i, V_s, V_i]), (H_i, V_i)

    def solve(self, t_span, y0=None, t_eval=None):
        """Integrate the model over `t_span`, returning (t, state)."""
        if y0 is None:
            N_h = self.N_h
            y0 = np.array([N_h * 0.999, N_h * 0.001, self.N_m / 50.0, self.N_m / 50.0])
        sol = solve_ivp(self._rhs, t_span, y0, method="LSODA", t_eval=t_eval,
                        rtol=1e-6, atol=1e-9)
        return sol.t, sol.y
