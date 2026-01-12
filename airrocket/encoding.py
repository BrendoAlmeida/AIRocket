from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .design_space import NOSE_KINDS, FIN_COUNTS


# Canonical order for GA vectors (no derived fields like tube_d).
DESIGN_KEYS = (
    "num_grains",
    "grain_h",
    "grain_out_r",
    "grain_in_r",
    "throat_r",
    "expansion_ratio",
    "tube_l",
    "nose_l",
    "nose_type",
    "fin_n",
    "fin_root",
    "fin_tip",
    "fin_span",
    "fin_sweep",
    "prop_density",
    "isp_eff_s",
    "eta_thrust",
    "burn_rate",
    "curve_alpha",
    "curve_beta",
)


@dataclass(frozen=True)
class DiscreteSpec:
    nose_types: tuple[str, ...] = NOSE_KINDS
    fin_counts: tuple[int, ...] = FIN_COUNTS


def encode_design(design: dict, spec: DiscreteSpec = DiscreteSpec()) -> np.ndarray:
    nose_idx = spec.nose_types.index(str(design["nose_type"]))
    fin_idx = spec.fin_counts.index(int(design["fin_n"]))

    v = np.array(
        [
            int(design["num_grains"]),
            float(design["grain_h"]),
            float(design["grain_out_r"]),
            float(design["grain_in_r"]),
            float(design["throat_r"]),
            float(design["expansion_ratio"]),
            float(design["tube_l"]),
            float(design["nose_l"]),
            float(nose_idx),
            float(fin_idx),
            float(design["fin_root"]),
            float(design["fin_tip"]),
            float(design["fin_span"]),
            float(design["fin_sweep"]),
            float(design["prop_density"]),
            float(design["isp_eff_s"]),
            float(design["eta_thrust"]),
            float(design["burn_rate"]),
            float(design["curve_alpha"]),
            float(design["curve_beta"]),
        ],
        dtype=float,
    )
    return v


def decode_design(v: np.ndarray, spec: DiscreteSpec = DiscreteSpec()) -> dict:
    v = np.asarray(v, dtype=float)
    nose_idx = int(np.clip(round(v[8]), 0, len(spec.nose_types) - 1))
    fin_idx = int(np.clip(round(v[9]), 0, len(spec.fin_counts) - 1))

    return {
        "num_grains": int(round(v[0])),
        "grain_h": float(v[1]),
        "grain_out_r": float(v[2]),
        "grain_in_r": float(v[3]),
        "throat_r": float(v[4]),
        "expansion_ratio": float(v[5]),
        "tube_l": float(v[6]),
        "nose_l": float(v[7]),
        "nose_type": spec.nose_types[nose_idx],
        "fin_n": spec.fin_counts[fin_idx],
        "fin_root": float(v[10]),
        "fin_tip": float(v[11]),
        "fin_span": float(v[12]),
        "fin_sweep": float(v[13]),
        "prop_density": float(v[14]),
        "isp_eff_s": float(v[15]),
        "eta_thrust": float(v[16]),
        "burn_rate": float(v[17]),
        "curve_alpha": float(v[18]),
        "curve_beta": float(v[19]),
    }
