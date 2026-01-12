from __future__ import annotations

import math


def derive_tube_d(grain_out_r: float, *, clearance_m: float, wall_thickness_m: float) -> float:
    # tube_d is always derived to prevent inconsistencies.
    return 2.0 * (grain_out_r + clearance_m + wall_thickness_m)


def grain_propellant_volume(
    grain_out_r: float,
    grain_in_r: float,
    grain_h: float,
    num_grains: int,
) -> float:
    # Volume of cylindrical annulus per grain.
    annulus_area = math.pi * (grain_out_r**2 - grain_in_r**2)
    return annulus_area * grain_h * float(num_grains)


def derive_nozzle_exit_r(throat_r: float, expansion_ratio: float) -> float:
    # Expansion ratio eps = Ae/At = (re/rt)^2
    return float(throat_r * math.sqrt(max(expansion_ratio, 1.0)))


def derive_motor_casing_length(grain_h: float, num_grains: int, grain_sep: float, end_margin: float = 0.02) -> float:
    if num_grains <= 0:
        return 0.0
    return float(num_grains * grain_h + max(0, num_grains - 1) * grain_sep + 2.0 * end_margin)
