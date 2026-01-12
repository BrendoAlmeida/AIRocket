from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .derive import grain_propellant_volume, derive_nozzle_exit_r

G0 = 9.80665


@dataclass(frozen=True)
class MotorParams:
    prop_density: float
    isp_eff_s: float
    eta_thrust: float
    burn_rate: float
    curve_alpha: float
    curve_beta: float
    throat_r: float
    expansion_ratio: float


@dataclass(frozen=True)
class MotorDerived:
    burn_time_s: float
    prop_mass_kg: float
    total_impulse_ns: float
    nozzle_exit_r: float


def _beta_log_norm(a: float, b: float) -> float:
    # log(Beta(a,b)) via lgamma for numeric stability.
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_shape(tau: np.ndarray, a: float, b: float) -> np.ndarray:
    # PDF of Beta(a,b) on [0,1], normalized to integrate to 1.
    tau = np.clip(tau, 0.0, 1.0)
    # Avoid 0^(-) with a,b>=1.5 in our sampling.
    log_pdf = (a - 1.0) * np.log(np.clip(tau, 1e-12, 1.0)) + (b - 1.0) * np.log(np.clip(1.0 - tau, 1e-12, 1.0))
    log_pdf -= _beta_log_norm(a, b)
    return np.exp(log_pdf)


def derive_motor_from_geometry(
    *,
    num_grains: int,
    grain_h: float,
    grain_out_r: float,
    grain_in_r: float,
    params: MotorParams,
    burn_time_min: float = 1.5,
    burn_time_max: float = 6.0,
) -> MotorDerived:
    web = max(0.0, grain_out_r - grain_in_r)
    burn_time = web / max(params.burn_rate, 1e-9)
    burn_time = float(min(max(burn_time, burn_time_min), burn_time_max))

    prop_vol = grain_propellant_volume(grain_out_r, grain_in_r, grain_h, num_grains)
    prop_mass = float(prop_vol * params.prop_density)

    total_impulse = float(params.eta_thrust * prop_mass * G0 * params.isp_eff_s)

    nozzle_exit_r = derive_nozzle_exit_r(params.throat_r, params.expansion_ratio)

    return MotorDerived(
        burn_time_s=burn_time,
        prop_mass_kg=prop_mass,
        total_impulse_ns=total_impulse,
        nozzle_exit_r=nozzle_exit_r,
    )


def generate_thrust_curve(
    derived: MotorDerived,
    params: MotorParams,
    n_points: int = 200,
    baseline_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    burn_time = float(derived.burn_time_s)
    if burn_time <= 0:
        raise ValueError("burn_time_s must be > 0")
    if n_points < 20:
        n_points = 20

    t = np.linspace(0.0, burn_time, n_points)
    tau = t / burn_time

    shape = beta_shape(tau, params.curve_alpha, params.curve_beta)

    # Add a small baseline so thrust doesn't stay near-zero at ignition for peaked beta shapes.
    # This preserves total impulse after normalization and improves rail-exit behavior.
    if baseline_fraction > 0:
        shape = (shape + float(baseline_fraction)) / (1.0 + float(baseline_fraction))

    # Convert normalized shape (integral ~1 over [0,1]) into thrust (N) so integral over time is total_impulse.
    # F(t) = I_total * f(t/burn_time) / burn_time
    thrust = (derived.total_impulse_ns * shape / burn_time).astype(float)

    # Ensure endpoints are ~0 for numerical friendliness.
    thrust[0] = 0.0
    thrust[-1] = 0.0

    # RocketPy expects non-negative.
    thrust = np.clip(thrust, 0.0, None)

    return t, thrust
