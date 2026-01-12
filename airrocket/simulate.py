from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from rocketpy import Rocket, SolidMotor, Flight, Environment

from .constraints import check_constraints
from .derive import derive_tube_d, derive_motor_casing_length
from .motor_param import MotorParams, derive_motor_from_geometry, generate_thrust_curve


@dataclass(frozen=True)
class SimConfig:
    rail_length_m: float = 5.0
    ode_solver: str = "LSODA"
    terminate_on_apogee: bool = True
    max_time_s: float = 300.0

    # Simple mass model knobs (can be upgraded)
    airframe_mass_per_length_kg_m: float = 1.5
    payload_mass_kg: float = 0.5

    # Tube derivation
    tube_clearance_m: float = 0.004
    tube_wall_thickness_m: float = 0.002


@dataclass(frozen=True)
class MonteCarloConfig:
    # How many simulations to run.
    n_runs: int = 30
    random_state: int = 42

    # Relative (multiplicative) 1-sigma uncertainties.
    # Example: 0.05 means ~5% sigma.
    sigma_isp: float = 0.04
    sigma_eta_thrust: float = 0.03
    sigma_burn_rate: float = 0.06

    # Absolute (additive) 1-sigma uncertainties.
    sigma_airframe_mass_per_length: float = 0.25  # kg/m
    sigma_payload_mass: float = 0.15  # kg

    # Small geometry / build tolerances (additive, meters).
    sigma_fin_span: float = 0.003
    sigma_fin_sweep: float = 0.003
    sigma_fin_root: float = 0.003
    sigma_fin_tip: float = 0.003
    sigma_throat_r: float = 0.0003

    # Wind uncertainty model.
    # - "constant": constant wind with altitude.
    # - "linear": linear interpolation between ground and a reference altitude.
    wind_profile: str = "constant"  # "constant" | "linear"
    wind_ref_alt_m: float = 2000.0

    # Wind components at ground (m/s)
    wind_u_bias_m_s: float = 0.0
    wind_v_bias_m_s: float = 0.0
    wind_u_sigma_m_s: float = 2.0
    wind_v_sigma_m_s: float = 2.0

    # Wind components at reference altitude (m/s) - used when wind_profile == "linear"
    wind_u_ref_bias_m_s: float = 0.0
    wind_v_ref_bias_m_s: float = 0.0
    wind_u_ref_sigma_m_s: float = 3.0
    wind_v_ref_sigma_m_s: float = 3.0

    # Optional gust (added via Environment.add_wind_gust). Set to 0 to disable.
    wind_gust_sigma_m_s: float = 0.0

    # Percentiles to report.
    p_lo: float = 5.0
    p_hi: float = 95.0


def _clamp_positive(x: float, min_value: float = 1e-9) -> float:
    return float(max(min_value, x))


def _lognormal_factor(rng: np.random.Generator, sigma_rel: float) -> float:
    """Multiplicative perturbation with approx relative sigma (small-sigma)."""
    if sigma_rel <= 0:
        return 1.0
    # For small sigma, lognormal with mean ~1.
    s2 = float(sigma_rel) ** 2
    mu = -0.5 * s2
    return float(rng.lognormal(mean=mu, sigma=float(sigma_rel)))


def perturb_design_for_mc(design: dict, rng: np.random.Generator, mc: MonteCarloConfig) -> dict:
    """Return a perturbed copy of the design (manufacturing + motor variability)."""
    d = dict(design)

    # Motor performance (multiplicative)
    d["isp_eff_s"] = _clamp_positive(float(d["isp_eff_s"]) * _lognormal_factor(rng, mc.sigma_isp))
    d["eta_thrust"] = float(np.clip(float(d["eta_thrust"]) * _lognormal_factor(rng, mc.sigma_eta_thrust), 0.05, 1.0))
    d["burn_rate"] = _clamp_positive(float(d["burn_rate"]) * _lognormal_factor(rng, mc.sigma_burn_rate))

    # Small geometry tolerances (additive)
    d["fin_span"] = _clamp_positive(float(d["fin_span"]) + float(rng.normal(0.0, mc.sigma_fin_span)))
    d["fin_sweep"] = float(max(0.0, float(d["fin_sweep"]) + float(rng.normal(0.0, mc.sigma_fin_sweep))))

    fin_root = _clamp_positive(float(d["fin_root"]) + float(rng.normal(0.0, mc.sigma_fin_root)))
    fin_tip = _clamp_positive(float(d["fin_tip"]) + float(rng.normal(0.0, mc.sigma_fin_tip)))
    d["fin_root"] = fin_root
    d["fin_tip"] = float(min(fin_tip, fin_root))

    d["throat_r"] = _clamp_positive(float(d["throat_r"]) + float(rng.normal(0.0, mc.sigma_throat_r)))

    return d


def clone_env_with_wind_profile(
    base_env: Environment,
    *,
    wind_u_source,
    wind_v_source,
    gust_x: float | None = None,
    gust_y: float | None = None,
) -> Environment:
    """Create a new Environment matching base_env but with custom wind profiles.

    Notes:
    - RocketPy's public API sets constant winds via set_atmospheric_model(wind_u, wind_v),
      but for profiles we override the wind velocity functions directly.
    """
    env = Environment(
        latitude=float(getattr(base_env, "latitude", 0.0)),
        longitude=float(getattr(base_env, "longitude", 0.0)),
        elevation=float(getattr(base_env, "elevation", 0.0)),
    )
    base_date = getattr(base_env, "date", None)
    if base_date is not None:
        env.set_date(base_date)

    atm_type = getattr(base_env, "atmospheric_model_type", None) or "standard_atmosphere"

    # Initialize atmosphere model first.
    env.set_atmospheric_model(type=atm_type)

    # Override wind profiles (private setters, stable across RocketPy 1.11).
    getattr(env, "_Environment__set_wind_velocity_x_function")(wind_u_source)
    getattr(env, "_Environment__set_wind_velocity_y_function")(wind_v_source)

    if gust_x is not None or gust_y is not None:
        env.add_wind_gust(float(gust_x or 0.0), float(gust_y or 0.0))

    return env


def simulate_metrics_safe(design: dict, env: Environment, cfg: SimConfig) -> dict:
    """Wrapper that never raises; returns ok=False with fail_reason on exception."""
    try:
        return simulate_metrics(design, env, cfg)
    except Exception as e:
        return {
            "ok": False,
            "fail_reason": f"exception:{type(e).__name__}",
            **design,
        }


def _summarize_metric(values: np.ndarray, *, p_lo: float, p_hi: float) -> dict:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            f"p{int(p_lo)}": float("nan"),
            "p50": float("nan"),
            f"p{int(p_hi)}": float("nan"),
        }
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        f"p{int(p_lo)}": float(np.percentile(v, p_lo)),
        "p50": float(np.percentile(v, 50.0)),
        f"p{int(p_hi)}": float(np.percentile(v, p_hi)),
    }


def simulate_monte_carlo(
    design: dict,
    env: Environment,
    cfg: SimConfig,
    mc: MonteCarloConfig,
    *,
    return_runs: bool = False,
) -> dict:
    """Run repeated RocketPy sims with perturbed inputs; return uncertainty statistics."""
    n = int(max(1, mc.n_runs))
    rng = np.random.default_rng(int(mc.random_state))

    wind_u0_used = np.zeros(n, dtype=float)
    wind_v0_used = np.zeros(n, dtype=float)
    wind_u_ref_used = np.zeros(n, dtype=float)
    wind_v_ref_used = np.zeros(n, dtype=float)
    gust_x_used = np.zeros(n, dtype=float)
    gust_y_used = np.zeros(n, dtype=float)

    runs: list[dict] = []
    for i in range(n):
        d = perturb_design_for_mc(design, rng, mc)

        wind_u0 = float(mc.wind_u_bias_m_s + rng.normal(0.0, mc.wind_u_sigma_m_s))
        wind_v0 = float(mc.wind_v_bias_m_s + rng.normal(0.0, mc.wind_v_sigma_m_s))

        if str(mc.wind_profile).lower() == "linear":
            wind_u_ref = float(mc.wind_u_ref_bias_m_s + rng.normal(0.0, mc.wind_u_ref_sigma_m_s))
            wind_v_ref = float(mc.wind_v_ref_bias_m_s + rng.normal(0.0, mc.wind_v_ref_sigma_m_s))
        else:
            wind_u_ref = wind_u0
            wind_v_ref = wind_v0

        gust_x = float(rng.normal(0.0, mc.wind_gust_sigma_m_s)) if mc.wind_gust_sigma_m_s > 0 else 0.0
        gust_y = float(rng.normal(0.0, mc.wind_gust_sigma_m_s)) if mc.wind_gust_sigma_m_s > 0 else 0.0

        wind_u0_used[i] = wind_u0
        wind_v0_used[i] = wind_v0
        wind_u_ref_used[i] = wind_u_ref
        wind_v_ref_used[i] = wind_v_ref
        gust_x_used[i] = gust_x
        gust_y_used[i] = gust_y

        elev = float(getattr(env, "elevation", 0.0))
        ref_alt = float(max(1.0, mc.wind_ref_alt_m))

        def wind_u_func(hasl_m: float) -> float:
            rel = float(hasl_m) - elev
            frac = float(np.clip(rel / ref_alt, 0.0, 1.0))
            return float(wind_u0 + (wind_u_ref - wind_u0) * frac)

        def wind_v_func(hasl_m: float) -> float:
            rel = float(hasl_m) - elev
            frac = float(np.clip(rel / ref_alt, 0.0, 1.0))
            return float(wind_v0 + (wind_v_ref - wind_v0) * frac)

        env_run = clone_env_with_wind_profile(
            base_env=env,
            wind_u_source=wind_u_func,
            wind_v_source=wind_v_func,
            gust_x=gust_x,
            gust_y=gust_y,
        )

        # Per-run mass model uncertainty lives in SimConfig.
        cfg_run = replace(
            cfg,
            airframe_mass_per_length_kg_m=float(max(0.05, cfg.airframe_mass_per_length_kg_m + rng.normal(0.0, mc.sigma_airframe_mass_per_length))),
            payload_mass_kg=float(max(0.0, cfg.payload_mass_kg + rng.normal(0.0, mc.sigma_payload_mass))),
        )

        m = simulate_metrics_safe(d, env_run, cfg_run)
        runs.append(m)

    ok_mask = np.array([bool(r.get("ok", False)) for r in runs], dtype=bool)
    n_ok = int(np.sum(ok_mask))
    n_fail = int(n - n_ok)

    # Failure reason histogram (useful when ok_rate is low).
    fail_reason_counts: dict[str, int] = {}
    for r in runs:
        if not bool(r.get("ok", False)):
            reason = str(r.get("fail_reason", "unknown"))
            fail_reason_counts[reason] = fail_reason_counts.get(reason, 0) + 1

    def arr(key: str) -> np.ndarray:
        out = np.array([float(r.get(key, np.nan)) for r in runs], dtype=float)
        # Only summarize successful runs to avoid mixing fail sentinels.
        out[~ok_mask] = np.nan
        return out

    summary = {
        "mc_n": n,
        "mc_ok": n_ok,
        "mc_fail": n_fail,
        "mc_ok_rate": float(n_ok / n),
        "mc_fail_reason_counts": fail_reason_counts,
        "wind_profile": str(mc.wind_profile),
        "wind_ref_alt_m": float(mc.wind_ref_alt_m),
        "wind_u_stats": _summarize_metric(wind_u0_used, p_lo=mc.p_lo, p_hi=mc.p_hi),
        "wind_v_stats": _summarize_metric(wind_v0_used, p_lo=mc.p_lo, p_hi=mc.p_hi),
        "wind_u_ref_stats": _summarize_metric(wind_u_ref_used, p_lo=mc.p_lo, p_hi=mc.p_hi),
        "wind_v_ref_stats": _summarize_metric(wind_v_ref_used, p_lo=mc.p_lo, p_hi=mc.p_hi),
        "wind_gust_x_stats": _summarize_metric(gust_x_used, p_lo=mc.p_lo, p_hi=mc.p_hi),
        "wind_gust_y_stats": _summarize_metric(gust_y_used, p_lo=mc.p_lo, p_hi=mc.p_hi),
        "apogee_stats": _summarize_metric(arr("apogee"), p_lo=mc.p_lo, p_hi=mc.p_hi),
        "stability_min_over_flight_stats": _summarize_metric(arr("stability_min_over_flight"), p_lo=mc.p_lo, p_hi=mc.p_hi),
        "stability_at_max_q_stats": _summarize_metric(arr("stability_at_max_q"), p_lo=mc.p_lo, p_hi=mc.p_hi),
        "max_mach_stats": _summarize_metric(arr("max_mach"), p_lo=mc.p_lo, p_hi=mc.p_hi),
        "max_acceleration_stats": _summarize_metric(arr("max_acceleration"), p_lo=mc.p_lo, p_hi=mc.p_hi),
        "max_dynamic_pressure_stats": _summarize_metric(arr("max_dynamic_pressure"), p_lo=mc.p_lo, p_hi=mc.p_hi),
        "rail_exit_speed_stats": _summarize_metric(arr("rail_exit_speed"), p_lo=mc.p_lo, p_hi=mc.p_hi),
    }

    if return_runs:
        summary["runs"] = runs

    return summary


def _estimate_airframe_mass(tube_l: float, nose_l: float, mass_per_length: float, payload_mass: float) -> float:
    return float((tube_l + nose_l) * mass_per_length + payload_mass)


def build_motor(design: dict, cfg: SimConfig) -> tuple[SolidMotor, dict]:
    motor_params = MotorParams(
        prop_density=float(design["prop_density"]),
        isp_eff_s=float(design["isp_eff_s"]),
        eta_thrust=float(design["eta_thrust"]),
        burn_rate=float(design["burn_rate"]),
        curve_alpha=float(design["curve_alpha"]),
        curve_beta=float(design["curve_beta"]),
        throat_r=float(design["throat_r"]),
        expansion_ratio=float(design["expansion_ratio"]),
    )

    derived_motor = derive_motor_from_geometry(
        num_grains=int(design["num_grains"]),
        grain_h=float(design["grain_h"]),
        grain_out_r=float(design["grain_out_r"]),
        grain_in_r=float(design["grain_in_r"]),
        params=motor_params,
    )

    t, thrust = generate_thrust_curve(derived_motor, motor_params)
    thrust_source = np.column_stack([t, thrust])

    max_thrust_n = float(np.max(thrust)) if len(thrust) else float("nan")
    # Early thrust at ~5% of burn time (helps avoid designs that never leave the rail).
    if len(thrust) >= 5:
        idx_05 = int(round(0.05 * (len(thrust) - 1)))
        thrust_tau05_n = float(thrust[idx_05])
    else:
        thrust_tau05_n = float("nan")

    # Simple dry mass estimate proportional to prop mass.
    dry_mass = float(0.35 * derived_motor.prop_mass_kg + 0.5)

    # Simple inertia estimate (cylinder-like)
    r = float(design["grain_out_r"]) + 0.01
    grain_sep = 0.005
    end_margin = 0.02
    n_grains = int(design["num_grains"])
    grain_h = float(design["grain_h"])

    # Geometry along motor axis (measured from nozzle plane).
    stack_len = float(n_grains * grain_h + max(0, n_grains - 1) * grain_sep)
    casing_len = float(stack_len + 2.0 * end_margin)

    # Place the grain stack after a small end margin so grains don't start at the nozzle plane.
    # RocketPy uses these positions along the motor axis.
    com = float(end_margin + stack_len / 2.0)

    # Use casing length for a slightly more realistic inertia length-scale.
    L = casing_len
    Ixx = 0.5 * dry_mass * (r**2)
    Izz = (1.0 / 12.0) * dry_mass * (3 * r**2 + L**2)
    dry_inertia = (Ixx, Ixx, Izz)

    motor = SolidMotor(
        thrust_source=thrust_source,
        burn_time=float(derived_motor.burn_time_s),
        dry_mass=dry_mass,
        dry_inertia=dry_inertia,
        nozzle_radius=float(derived_motor.nozzle_exit_r),
        grain_number=int(design["num_grains"]),
        grain_density=float(design["prop_density"]),
        grain_outer_radius=float(design["grain_out_r"]),
        grain_initial_inner_radius=float(design["grain_in_r"]),
        grain_initial_height=float(design["grain_h"]),
        grain_separation=float(grain_sep),
        grains_center_of_mass_position=com,
        center_of_dry_mass_position=com,
        throat_radius=float(design["throat_r"]),
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    derived = {
        "burn_time_s": float(derived_motor.burn_time_s),
        "prop_mass_kg": float(derived_motor.prop_mass_kg),
        "total_impulse_ns": float(derived_motor.total_impulse_ns),
        "nozzle_exit_r": float(derived_motor.nozzle_exit_r),
        "motor_dry_mass_kg": float(dry_mass),
        "max_thrust_n": float(max_thrust_n),
        "thrust_tau05_n": float(thrust_tau05_n),
    }

    return motor, derived


def build_rocket(design: dict, motor: SolidMotor, motor_derived: dict, cfg: SimConfig) -> tuple[Rocket, dict]:
    tube_d = derive_tube_d(
        float(design["grain_out_r"]),
        clearance_m=cfg.tube_clearance_m,
        wall_thickness_m=cfg.tube_wall_thickness_m,
    )
    tube_l = float(design["tube_l"])
    nose_l = float(design["nose_l"])

    # Keep a simple mass model for now.
    airframe_mass = _estimate_airframe_mass(
        tube_l=tube_l,
        nose_l=nose_l,
        mass_per_length=cfg.airframe_mass_per_length_kg_m,
        payload_mass=cfg.payload_mass_kg,
    )

    # RocketPy mass is dry mass without motor prop? We keep it simple and let motor handle propellant.
    mass = float(airframe_mass)

    # Rough inertia values; can be refined later.
    Ixx = max(1e-6, 0.25 * mass * (tube_d / 2.0) ** 2)
    Izz = max(1e-6, (1.0 / 12.0) * mass * (3 * (tube_d / 2.0) ** 2 + (tube_l + nose_l) ** 2))

    rocket = Rocket(
        radius=tube_d / 2.0,
        mass=mass,
        inertia=(Izz, Izz, Ixx),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=tube_l / 2.0,
        coordinate_system_orientation="tail_to_nose",
    )

    rocket.add_nose(length=nose_l, kind=str(design["nose_type"]), position=tube_l)
    rocket.add_trapezoidal_fins(
        n=int(design["fin_n"]),
        root_chord=float(design["fin_root"]),
        tip_chord=float(design["fin_tip"]),
        span=float(design["fin_span"]),
        sweep_length=float(design["fin_sweep"]),
        position=0.1,
    )
    rocket.add_motor(motor, position=0)

    motor_casing_length = derive_motor_casing_length(
        grain_h=float(design["grain_h"]),
        num_grains=int(design["num_grains"]),
        grain_sep=0.005,
    )

    derived = {
        **motor_derived,
        "tube_d": float(tube_d),
        "tube_r": float(tube_d / 2.0),
        "rocket_mass_est_kg": float(mass),
        "motor_casing_length": float(motor_casing_length),
    }

    return rocket, derived


def simulate_metrics(design: dict, env: Environment, cfg: SimConfig) -> dict:
    motor, motor_derived = build_motor(design, cfg)
    rocket, derived = build_rocket(design, motor, motor_derived, cfg)

    c = check_constraints(design, derived)
    if not c.ok:
        return {
            "ok": False,
            "fail_reason": ";".join(c.reasons),
            **design,
            **derived,
        }

    flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=float(cfg.rail_length_m),
        ode_solver=str(cfg.ode_solver),
        terminate_on_apogee=bool(cfg.terminate_on_apogee),
        max_time=float(cfg.max_time_s),
    )

    apogee = float(getattr(flight, "apogee", float("nan")))
    rail_exit_v = float(getattr(flight, "out_of_rail_velocity", float("nan")))

    # Treat non-liftoff / non-rail-exit as a failed design even if RocketPy returns a Flight object.
    if (not np.isfinite(apogee)) or (apogee <= 1.0) or (not np.isfinite(rail_exit_v)) or (rail_exit_v <= 0.1):
        return {
            "ok": False,
            "fail_reason": "no_liftoff_or_no_rail_exit",
            **design,
            **derived,
            "apogee": apogee,
            "rail_exit_speed": rail_exit_v,
        }

    # Robust stability metrics using Flight's built-ins.
    min_stab = float(getattr(flight, "min_stability_margin", np.nan))
    max_q_t = float(getattr(flight, "max_dynamic_pressure_time", np.nan))
    sm_at_max_q = float(rocket.static_margin(max_q_t)) if math.isfinite(max_q_t) else float("nan")

    # Use flight.static_margin function when available.
    try:
        sm_min = float(getattr(flight, "min_stability_margin", np.nan))
    except Exception:
        sm_min = float("nan")

    return {
        "ok": True,
        "fail_reason": "",
        **design,
        **derived,
        "apogee": apogee,
        "max_mach": float(getattr(flight, "max_mach_number", np.nan)),
        "max_acceleration": float(getattr(flight, "max_acceleration", np.nan)),
        "max_dynamic_pressure": float(getattr(flight, "max_dynamic_pressure", np.nan)),
        "rail_exit_speed": rail_exit_v,
        "stability_initial": float(rocket.static_margin(0)),
        "stability_min_over_flight": float(sm_min),
        "stability_at_max_q": float(sm_at_max_q),
        "min_stability_margin": float(min_stab),
    }
