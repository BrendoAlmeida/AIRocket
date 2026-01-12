from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DesignConstants:
    # Tube diameter is derived from motor outer radius + clearance + wall thickness.
    tube_clearance_m: float = 0.004  # radial clearance
    tube_wall_thickness_m: float = 0.002
    grain_separation_m: float = 0.005

    # Rough mass model (kept simple; can be upgraded later).
    airframe_areal_density_kg_m2: float = 2.5
    payload_mass_kg: float = 0.5


@dataclass(frozen=True)
class Bounds:
    # Grains
    num_grains_min: int = 2
    num_grains_max: int = 8
    grain_h_min: float = 0.05
    grain_h_max: float = 0.15
    grain_out_r_min: float = 0.02
    grain_out_r_max: float = 0.05
    grain_in_r_min: float = 0.008
    grain_in_r_max: float = 0.015

    # Nozzle
    throat_r_min: float = 0.005
    throat_r_max: float = 0.012
    expansion_ratio_min: float = 2.0
    expansion_ratio_max: float = 10.0

    # Airframe
    tube_l_min: float = 0.8
    tube_l_max: float = 3.0
    nose_l_min: float = 0.2
    nose_l_max: float = 0.6

    # Fins
    fin_root_min: float = 0.08
    fin_root_max: float = 0.30
    fin_tip_min: float = 0.02
    fin_span_min: float = 0.05
    fin_span_max: float = 0.20
    fin_sweep_min: float = 0.0
    fin_sweep_max: float = 0.15

    # Motor performance (parametric)
    prop_density_min: float = 1500.0
    prop_density_max: float = 1850.0
    isp_eff_min: float = 160.0
    isp_eff_max: float = 240.0
    eta_thrust_min: float = 0.85
    eta_thrust_max: float = 1.05

    burn_rate_min: float = 0.002
    burn_rate_max: float = 0.012

    curve_alpha_min: float = 1.5
    curve_alpha_max: float = 6.0
    curve_beta_min: float = 1.5
    curve_beta_max: float = 6.0


NOSE_KINDS = ("conical", "ogive", "vonKarman")
FIN_COUNTS = (3, 4)


def sample_design(rng: np.random.Generator, bounds: Bounds) -> dict:
    num_grains = int(rng.integers(bounds.num_grains_min, bounds.num_grains_max + 1))

    grain_h = float(rng.uniform(bounds.grain_h_min, bounds.grain_h_max))
    grain_out_r = float(rng.uniform(bounds.grain_out_r_min, bounds.grain_out_r_max))

    grain_in_r = float(rng.uniform(bounds.grain_in_r_min, min(bounds.grain_in_r_max, grain_out_r * 0.95)))

    throat_r = float(rng.uniform(bounds.throat_r_min, bounds.throat_r_max))
    expansion_ratio = float(rng.uniform(bounds.expansion_ratio_min, bounds.expansion_ratio_max))

    tube_l = float(rng.uniform(bounds.tube_l_min, bounds.tube_l_max))
    nose_l = float(rng.uniform(bounds.nose_l_min, bounds.nose_l_max))
    nose_type = str(rng.choice(NOSE_KINDS))

    fin_n = int(rng.choice(FIN_COUNTS))
    fin_root = float(rng.uniform(bounds.fin_root_min, bounds.fin_root_max))
    fin_tip = float(rng.uniform(bounds.fin_tip_min, min(fin_root, fin_root)))
    fin_span = float(rng.uniform(bounds.fin_span_min, bounds.fin_span_max))
    fin_sweep = float(rng.uniform(bounds.fin_sweep_min, bounds.fin_sweep_max))

    prop_density = float(rng.uniform(bounds.prop_density_min, bounds.prop_density_max))
    isp_eff_s = float(rng.uniform(bounds.isp_eff_min, bounds.isp_eff_max))
    eta_thrust = float(rng.uniform(bounds.eta_thrust_min, bounds.eta_thrust_max))
    burn_rate = float(rng.uniform(bounds.burn_rate_min, bounds.burn_rate_max))

    curve_alpha = float(rng.uniform(bounds.curve_alpha_min, bounds.curve_alpha_max))
    curve_beta = float(rng.uniform(bounds.curve_beta_min, bounds.curve_beta_max))

    return {
        "num_grains": num_grains,
        "grain_h": grain_h,
        "grain_out_r": grain_out_r,
        "grain_in_r": grain_in_r,
        "throat_r": throat_r,
        "expansion_ratio": expansion_ratio,
        "tube_l": tube_l,
        "nose_l": nose_l,
        "nose_type": nose_type,
        "fin_n": fin_n,
        "fin_root": fin_root,
        "fin_tip": fin_tip,
        "fin_span": fin_span,
        "fin_sweep": fin_sweep,
        "prop_density": prop_density,
        "isp_eff_s": isp_eff_s,
        "eta_thrust": eta_thrust,
        "burn_rate": burn_rate,
        "curve_alpha": curve_alpha,
        "curve_beta": curve_beta,
    }
