from __future__ import annotations

from .constraints import check_constraints
from .derive import derive_tube_d, derive_motor_casing_length
import numpy as np

from .motor_param import MotorParams, derive_motor_from_geometry, beta_shape


def derive_fast(
    design: dict,
    *,
    tube_clearance_m: float,
    tube_wall_thickness_m: float,
    airframe_mass_per_length_kg_m: float,
    payload_mass_kg: float,
) -> dict:
    tube_d = derive_tube_d(float(design["grain_out_r"]), clearance_m=tube_clearance_m, wall_thickness_m=tube_wall_thickness_m)

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

    md = derive_motor_from_geometry(
        num_grains=int(design["num_grains"]),
        grain_h=float(design["grain_h"]),
        grain_out_r=float(design["grain_out_r"]),
        grain_in_r=float(design["grain_in_r"]),
        params=motor_params,
    )

    motor_casing_length = derive_motor_casing_length(
        grain_h=float(design["grain_h"]),
        num_grains=int(design["num_grains"]),
        grain_sep=0.005,
    )

    return {
        "tube_d": float(tube_d),
        "tube_r": float(tube_d / 2.0),
        "burn_time_s": float(md.burn_time_s),
        "prop_mass_kg": float(md.prop_mass_kg),
        "total_impulse_ns": float(md.total_impulse_ns),
        "nozzle_exit_r": float(md.nozzle_exit_r),
        "motor_casing_length": float(motor_casing_length),
        # crude mass model (matches simulate.py)
        "rocket_mass_est_kg": float((float(design["tube_l"]) + float(design["nose_l"])) * airframe_mass_per_length_kg_m + payload_mass_kg),
        "motor_dry_mass_kg": float(0.35 * md.prop_mass_kg + 0.5),
        # peak thrust estimate: avg_thrust * max(shape)
        # Include the same baseline used in motor_param.generate_thrust_curve (default 0.05).
        "peak_thrust_est_n": float(
            (md.total_impulse_ns / max(md.burn_time_s, 1e-9))
            * float(
                np.max(
                    (beta_shape(np.linspace(0.0, 1.0, 200), float(design["curve_alpha"]), float(design["curve_beta"])) + 0.05) / 1.05
                )
            )
        ),
        # early thrust estimate at tau=0.05
        "thrust_tau05_est_n": float(
            (md.total_impulse_ns / max(md.burn_time_s, 1e-9))
            * float(
                ((beta_shape(np.array([0.05]), float(design["curve_alpha"]), float(design["curve_beta"]))[0] + 0.05) / 1.05)
            )
        ),
    }


def is_feasible_fast(design: dict, derived: dict) -> tuple[bool, str]:
    c = check_constraints(design, derived)
    return c.ok, ";".join(c.reasons)
