from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintResult:
    ok: bool
    reasons: list[str]


def check_constraints(design: dict, derived: dict) -> ConstraintResult:
    reasons: list[str] = []

    # Basic geometry
    if design["grain_in_r"] <= 0:
        reasons.append("grain_in_r<=0")
    if design["grain_out_r"] <= 0:
        reasons.append("grain_out_r<=0")
    if design["grain_in_r"] >= design["grain_out_r"]:
        reasons.append("grain_in_r>=grain_out_r")

    if design["grain_h"] <= 0:
        reasons.append("grain_h<=0")
    if int(design["num_grains"]) < 1:
        reasons.append("num_grains<1")

    if design["throat_r"] <= 0:
        reasons.append("throat_r<=0")

    # Fin consistency
    if design["fin_tip"] <= 0 or design["fin_root"] <= 0:
        reasons.append("fin_chords<=0")
    if design["fin_tip"] > design["fin_root"]:
        reasons.append("fin_tip>fin_root")
    if design["fin_span"] <= 0:
        reasons.append("fin_span<=0")
    if design["fin_sweep"] < 0:
        reasons.append("fin_sweep<0")

    # Derived tube diameter consistency
    tube_d = float(derived.get("tube_d", 0.0))
    if tube_d <= 0:
        reasons.append("tube_d<=0")

    tube_r = tube_d / 2.0
    if tube_r <= design["grain_out_r"]:
        reasons.append("tube_radius<=grain_out_r")

    # Motor casing should fit in body tube length (simple check)
    motor_len = float(derived.get("motor_casing_length", 0.0))
    if motor_len <= 0:
        reasons.append("motor_casing_length<=0")
    if motor_len > float(design["tube_l"]) * 0.95:
        reasons.append("motor_casing_length>0.95*tube_l")

    # Nozzle should not exceed tube radius
    nozzle_exit_r = float(derived.get("nozzle_exit_r", 0.0))
    if nozzle_exit_r <= 0:
        reasons.append("nozzle_exit_r<=0")
    if nozzle_exit_r > tube_r:
        reasons.append("nozzle_exit_r>tube_radius")

    # Impulse and burn time sanity
    if float(derived.get("burn_time_s", 0.0)) <= 0:
        reasons.append("burn_time_s<=0")
    if float(derived.get("total_impulse_ns", 0.0)) <= 0:
        reasons.append("total_impulse<=0")
    if float(derived.get("prop_mass_kg", 0.0)) <= 0:
        reasons.append("prop_mass<=0")

    # Thrust-to-weight sanity (helps avoid "apogee = 0" designs).
    rocket_mass = derived.get("rocket_mass_est_kg", None)
    motor_dry_mass = derived.get("motor_dry_mass_kg", None)
    prop_mass = derived.get("prop_mass_kg", None)
    if rocket_mass is not None and motor_dry_mass is not None and prop_mass is not None:
        try:
            total_mass = float(rocket_mass) + float(motor_dry_mass) + float(prop_mass)
            weight_n = total_mass * 9.80665
            thrust_n = derived.get("max_thrust_n", derived.get("peak_thrust_est_n", None))
            if thrust_n is not None:
                if float(thrust_n) <= 1.2 * weight_n:
                    reasons.append("insufficient_thrust_to_weight")

            # Early thrust check (~5% burn): avoids designs with very delayed thrust rise.
            thrust_tau05 = derived.get("thrust_tau05_n", derived.get("thrust_tau05_est_n", None))
            if thrust_tau05 is not None:
                if float(thrust_tau05) <= 1.05 * weight_n:
                    reasons.append("insufficient_early_thrust")
        except Exception:
            # If something is malformed, let other constraints catch it.
            pass

    return ConstraintResult(ok=(len(reasons) == 0), reasons=reasons)
