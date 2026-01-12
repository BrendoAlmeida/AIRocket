from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .design_space import Bounds, sample_design
from .env import EnvConfig, get_process_env, init_process_env
from .encoding import decode_design, encode_design, DiscreteSpec
from .fast_eval import derive_fast, is_feasible_fast
from .simulate import SimConfig, simulate_metrics, simulate_monte_carlo, MonteCarloConfig


@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 300
    generations: int = 40
    elite_size: int = 40
    mutation_sigma: float = 0.02
    mutation_rate: float = 0.25
    crossover_rate: float = 0.7
    topk_validate: int = 10  # used in Fast mode
    n_jobs_validate: int = 1  # used in Fast mode (Top-K validation)
    random_state: int = 42


@dataclass(frozen=True)
class Objective:
    target_apogee: float
    min_stability: float

    # weights
    w_apogee: float = 1.0
    w_stability: float = 1000.0
    w_mach: float = 50.0
    w_accel: float = 1.0
    w_ok_rate: float = 2000.0


def _apply_fixed(design: dict, fixed: dict | None) -> dict:
    if not fixed:
        return design
    out = dict(design)
    for k, v in fixed.items():
        out[k] = v
    return out


def _clip_and_fix(vec: np.ndarray, bounds: Bounds, spec: DiscreteSpec, fixed: dict | None) -> np.ndarray:
    # Clip continuous values to bounds; snap discrete indices.
    v = vec.copy()

    # num_grains
    v[0] = float(int(np.clip(round(v[0]), bounds.num_grains_min, bounds.num_grains_max)))

    # grain_h/out/in
    v[1] = float(np.clip(v[1], bounds.grain_h_min, bounds.grain_h_max))
    v[2] = float(np.clip(v[2], bounds.grain_out_r_min, bounds.grain_out_r_max))
    v[3] = float(np.clip(v[3], bounds.grain_in_r_min, min(bounds.grain_in_r_max, v[2] * 0.95)))

    # throat + expansion
    v[4] = float(np.clip(v[4], bounds.throat_r_min, bounds.throat_r_max))
    v[5] = float(np.clip(v[5], bounds.expansion_ratio_min, bounds.expansion_ratio_max))

    # tube_l, nose_l
    v[6] = float(np.clip(v[6], bounds.tube_l_min, bounds.tube_l_max))
    v[7] = float(np.clip(v[7], bounds.nose_l_min, bounds.nose_l_max))

    # nose_type index
    v[8] = float(int(np.clip(round(v[8]), 0, len(spec.nose_types) - 1)))

    # fin_n index
    v[9] = float(int(np.clip(round(v[9]), 0, len(spec.fin_counts) - 1)))

    # fins
    v[10] = float(np.clip(v[10], bounds.fin_root_min, bounds.fin_root_max))
    v[11] = float(np.clip(v[11], bounds.fin_tip_min, v[10]))
    v[12] = float(np.clip(v[12], bounds.fin_span_min, bounds.fin_span_max))
    v[13] = float(np.clip(v[13], bounds.fin_sweep_min, bounds.fin_sweep_max))

    # motor perf
    v[14] = float(np.clip(v[14], bounds.prop_density_min, bounds.prop_density_max))
    v[15] = float(np.clip(v[15], bounds.isp_eff_min, bounds.isp_eff_max))
    v[16] = float(np.clip(v[16], bounds.eta_thrust_min, bounds.eta_thrust_max))
    v[17] = float(np.clip(v[17], bounds.burn_rate_min, bounds.burn_rate_max))
    v[18] = float(np.clip(v[18], bounds.curve_alpha_min, bounds.curve_alpha_max))
    v[19] = float(np.clip(v[19], bounds.curve_beta_min, bounds.curve_beta_max))

    # Apply fixed parameters by decoding->override->encoding to keep categorical handling consistent.
    if fixed:
        d = decode_design(v, spec)
        d = _apply_fixed(d, fixed)
        v = encode_design(d, spec)

    return v


def _fitness_from_metrics(m: dict, obj: Objective) -> float:
    # Lower is better.
    if not m.get("ok", True):
        return 1e12

    apogee = float(m.get("apogee", np.nan))
    stab = float(m.get("stability_min_over_flight", m.get("min_stability_margin", np.nan)))
    max_mach = float(m.get("max_mach", np.nan))
    max_acc = float(m.get("max_acceleration", np.nan))
    ok_rate = float(m.get("ok_rate", 1.0))

    if not np.isfinite(apogee) or not np.isfinite(stab):
        return 1e12

    ok_rate_pen = 0.0
    if np.isfinite(ok_rate):
        ok_rate_pen = max(0.0, 1.0 - ok_rate)

    err_ap = abs(apogee - obj.target_apogee)
    # Penalize stability below minimum heavily; above minimum, small penalty.
    if stab < obj.min_stability:
        err_stab = (obj.min_stability - stab)
    else:
        err_stab = 0.0

    mach_pen = max(0.0, max_mach - 1.5) if np.isfinite(max_mach) else 0.0
    acc_pen = max(0.0, max_acc - 150.0) if np.isfinite(max_acc) else 0.0

    return (
        obj.w_apogee * err_ap
        + obj.w_stability * err_stab
        + obj.w_mach * mach_pen
        + obj.w_accel * acc_pen
        + obj.w_ok_rate * ok_rate_pen
    )


def _mc_to_metrics(mc: dict, *, p_apogee: str = "p50", p_stab: str = "p5") -> dict:
    """Convert Monte Carlo summary -> metrics dict consumed by _fitness_from_metrics.

    Defaults:
    - apogee uses median (p50)
    - stability uses conservative low percentile (p5)
    """
    ap_stats = mc.get("apogee_stats", {}) or {}
    stab_stats = mc.get("stability_min_over_flight_stats", {}) or {}
    stab_q_stats = mc.get("stability_at_max_q_stats", {}) or {}
    mach_stats = mc.get("max_mach_stats", {}) or {}
    acc_stats = mc.get("max_acceleration_stats", {}) or {}

    ok_rate = float(mc.get("mc_ok_rate", np.nan))
    # Consider MC invalid if no successful runs.
    if not np.isfinite(ok_rate) or ok_rate <= 0.0:
        return {"ok": False, "fail_reason": "mc_no_success"}

    return {
        "ok": True,
        "ok_rate": ok_rate,
        "apogee": float(ap_stats.get(p_apogee, np.nan)),
        "stability_min_over_flight": float(stab_stats.get(p_stab, np.nan)),
        "stability_at_max_q": float(stab_q_stats.get(p_stab, np.nan)),
        # Keep these as conservative high percentiles.
        "max_mach": float(mach_stats.get("p95", mach_stats.get("p50", np.nan))),
        "max_acceleration": float(acc_stats.get("p95", acc_stats.get("p50", np.nan))),
    }


def _fmt_design_short(d: dict) -> str:
    # Compact, stable-ordered subset so logs stay readable.
    keys = [
        "num_grains",
        "grain_h",
        "throat_r",
        "expansion_ratio",
        "tube_l",
        "nose_type",
        "fin_n",
        "fin_root",
        "fin_span",
    ]
    parts = []
    for k in keys:
        if k not in d:
            continue
        v = d[k]
        if isinstance(v, (int, np.integer)):
            parts.append(f"{k}={int(v)}")
        elif isinstance(v, (float, np.floating)):
            parts.append(f"{k}={float(v):.4g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _fmt_metrics_short(m: dict) -> str:
    ap = float(m.get("apogee", np.nan))
    stab = float(m.get("stability_min_over_flight", m.get("min_stability_margin", np.nan)))
    okr = m.get("ok_rate", None)
    okr_s = f" ok_rate={float(okr):.2f}" if okr is not None and np.isfinite(float(okr)) else ""
    return f"apogee={ap:.1f}m stab_min={stab:.2f}cal{okr_s}" if np.isfinite(ap) and np.isfinite(stab) else f"apogee={ap} stab_min={stab}{okr_s}"


def _true_metrics_for_design_process(
    design: dict,
    sim_cfg: SimConfig,
    mc_cfg: MonteCarloConfig | None,
    env_cfg: EnvConfig,
) -> dict:
    """Compute RocketPy-based metrics in a subprocess.

    Uses per-process cached Environment (see airrocket.env) to avoid pickling RocketPy objects.
    """
    env = get_process_env(env_cfg)
    if mc_cfg is None:
        return simulate_metrics(design, env, sim_cfg)

    mc = simulate_monte_carlo(design, env, sim_cfg, mc_cfg)
    return _mc_to_metrics(mc)


def optimize_hard(
    *,
    env,
    objective: Objective,
    ga: GAConfig,
    bounds: Bounds | None = None,
    fixed: dict | None = None,
    sim_cfg: SimConfig | None = None,
    mc_cfg: MonteCarloConfig | None = None,
    env_cfg: EnvConfig | None = None,
) -> tuple[dict, dict]:
    bounds = bounds or Bounds()
    spec = DiscreteSpec()
    sim_cfg = sim_cfg or SimConfig()

    rng = np.random.default_rng(ga.random_state)

    pop = []
    for _ in range(ga.pop_size):
        d = sample_design(rng, bounds)
        d = _apply_fixed(d, fixed)
        pop.append(encode_design(d, spec))
    pop = np.vstack(pop)

    best_design = None
    best_metrics = None
    best_fit = float("inf")

    for _gen in range(ga.generations):
        metrics_list = []
        fitness = np.zeros(ga.pop_size)

        for i in range(ga.pop_size):
            v = _clip_and_fix(pop[i], bounds, spec, fixed)
            d = decode_design(v, spec)
            if mc_cfg is None:
                m = simulate_metrics(d, env, sim_cfg)
            else:
                mc = simulate_monte_carlo(d, env, sim_cfg, mc_cfg)
                m = _mc_to_metrics(mc)
            metrics_list.append(m)
            fitness[i] = _fitness_from_metrics(m, objective)

        idx = int(np.argmin(fitness))
        if float(fitness[idx]) < best_fit:
            best_fit = float(fitness[idx])
            best_design = decode_design(_clip_and_fix(pop[idx], bounds, spec, fixed), spec)
            best_metrics = metrics_list[idx]

        if (_gen == 0) or ((_gen + 1) % 10 == 0) or (_gen == ga.generations - 1):
            pop_ok = float(np.mean([bool(m.get("ok", False)) for m in metrics_list])) if metrics_list else float("nan")
            ap = float(best_metrics.get("apogee", np.nan)) if best_metrics else float("nan")
            stab = float(best_metrics.get("stability_min_over_flight", np.nan)) if best_metrics else float("nan")
            okr = best_metrics.get("ok_rate", None) if best_metrics else None
            okr_s = f", ok_rate={float(okr):.2f}" if okr is not None and np.isfinite(float(okr)) else ""
            print(
                f"[GA hard] gen {_gen+1}/{ga.generations} best_fit={best_fit:.4g} apogee={ap:.1f}m stab_min={stab:.2f}cal"
                f"{okr_s} pop_ok={pop_ok*100:.1f}%"
            )

            topn = int(min(3, len(fitness)))
            if topn > 0:
                top_idx = np.argsort(fitness)[:topn]
                print(f"[GA hard] top{topn} (gen {_gen+1})")
                for rank, i in enumerate(top_idx, start=1):
                    d_i = decode_design(_clip_and_fix(pop[int(i)], bounds, spec, fixed), spec)
                    m_i = metrics_list[int(i)]
                    fit_i = float(fitness[int(i)])
                    print(f"  #{rank} fit={fit_i:.4g} {_fmt_metrics_short(m_i)} | {_fmt_design_short(d_i)}")

        elite_idx = np.argsort(fitness)[: ga.elite_size]
        elites = pop[elite_idx]

        # Next generation
        new_pop = [elites[i % len(elites)].copy() for i in range(ga.pop_size)]
        new_pop = np.vstack(new_pop)

        for i in range(ga.elite_size, ga.pop_size):
            if rng.random() < ga.crossover_rate:
                p1 = elites[rng.integers(0, len(elites))]
                p2 = elites[rng.integers(0, len(elites))]
                mask = rng.random(p1.shape[0]) > 0.5
                child = np.where(mask, p1, p2)
            else:
                child = elites[rng.integers(0, len(elites))].copy()

            if rng.random() < ga.mutation_rate:
                child = child + rng.normal(0.0, ga.mutation_sigma, size=child.shape[0])
                # stronger discrete mutation sometimes
                if rng.random() < 0.15:
                    child[8] = float(rng.integers(0, len(spec.nose_types)))
                if rng.random() < 0.15:
                    child[9] = float(rng.integers(0, len(spec.fin_counts)))

            new_pop[i] = _clip_and_fix(child, bounds, spec, fixed)

        pop = new_pop

    assert best_design is not None and best_metrics is not None
    return best_design, best_metrics


def optimize_fast(
    *,
    env,
    objective: Objective,
    ga: GAConfig,
    surrogate_bundle: dict,
    bounds: Bounds | None = None,
    fixed: dict | None = None,
    sim_cfg: SimConfig | None = None,
    mc_cfg: MonteCarloConfig | None = None,
    env_cfg: EnvConfig | None = None,
) -> tuple[dict, dict]:
    """Fast GA: evaluates whole population with surrogate; validates Top-K with RocketPy each generation."""
    bounds = bounds or Bounds()
    spec = DiscreteSpec()
    sim_cfg = sim_cfg or SimConfig()

    model = surrogate_bundle["model"]
    spec_s = surrogate_bundle["spec"]

    rng = np.random.default_rng(ga.random_state)

    pop = []
    for _ in range(ga.pop_size):
        d = sample_design(rng, bounds)
        d = _apply_fixed(d, fixed)
        pop.append(encode_design(d, spec))
    pop = np.vstack(pop)

    best_design = None
    best_metrics = None
    best_fit = float("inf")

    # Cache of validated designs
    validated: dict[str, dict] = {}

    use_parallel_validate = bool(env_cfg is not None and int(ga.n_jobs_validate) > 1)
    if int(ga.n_jobs_validate) > 1 and env_cfg is None:
        print(
            "[GA fast] Aviso: n_jobs_validate>1 mas env_cfg=None; "
            "não dá pra paralelizar no Windows sem recriar o Environment no subprocesso. "
            "Passe env_cfg=EnvConfig(...) na chamada."
        )

    pool_ctx = (
        ProcessPoolExecutor(
            max_workers=int(ga.n_jobs_validate),
            initializer=init_process_env,
            initargs=(env_cfg,),
        )
        if use_parallel_validate
        else nullcontext()
    )

    with pool_ctx as ex:
        for _gen in range(ga.generations):
            # Decode + feasibility gate
            designs = []
            ok_mask = np.ones(ga.pop_size, dtype=bool)
            fail_reason = [""] * ga.pop_size

            for i in range(ga.pop_size):
                v = _clip_and_fix(pop[i], bounds, spec, fixed)
                d = decode_design(v, spec)
                der = derive_fast(
                    d,
                    tube_clearance_m=sim_cfg.tube_clearance_m,
                    tube_wall_thickness_m=sim_cfg.tube_wall_thickness_m,
                    airframe_mass_per_length_kg_m=sim_cfg.airframe_mass_per_length_kg_m,
                    payload_mass_kg=sim_cfg.payload_mass_kg,
                )
                ok, reason = is_feasible_fast(d, der)
                ok_mask[i] = ok
                fail_reason[i] = reason
                designs.append(d)

            X = pd.DataFrame(designs)
            X = X.loc[:, list(spec_s.feature_cols)]

            preds = model.predict(X)
            pred_metrics = []
            for i in range(ga.pop_size):
                m = {
                    "ok": bool(ok_mask[i]),
                    "fail_reason": fail_reason[i],
                }
                if ok_mask[i]:
                    for j, t in enumerate(spec_s.target_cols):
                        m[t] = float(preds[i, j])
                pred_metrics.append(m)

            fitness = np.array([_fitness_from_metrics(m, objective) for m in pred_metrics], dtype=float)

            # Validate Top-K with RocketPy (optionally with Monte Carlo)
            k = int(min(max(1, ga.topk_validate), ga.pop_size))
            top_idx = np.argsort(fitness)[:k]

            pending: list[tuple[int, str, dict]] = []
            for i in top_idx:
                ii = int(i)
                d = designs[ii]
                key = str(sorted(d.items()))
                if key in validated:
                    m_true = validated[key]
                    pred_metrics[ii] = m_true
                    fitness[ii] = _fitness_from_metrics(m_true, objective)
                else:
                    pending.append((ii, key, d))

            if pending:
                if use_parallel_validate and ex is not None:
                    futures = {
                        ex.submit(_true_metrics_for_design_process, d, sim_cfg, mc_cfg, env_cfg): (ii, key)
                        for (ii, key, d) in pending
                    }
                    for fut in as_completed(futures):
                        ii, key = futures[fut]
                        m_true = fut.result()
                        validated[key] = m_true
                        pred_metrics[ii] = m_true
                        fitness[ii] = _fitness_from_metrics(m_true, objective)
                else:
                    for (ii, key, d) in pending:
                        if mc_cfg is None:
                            m_true = simulate_metrics(d, env, sim_cfg)
                        else:
                            mc = simulate_monte_carlo(d, env, sim_cfg, mc_cfg)
                            m_true = _mc_to_metrics(mc)
                        validated[key] = m_true
                        pred_metrics[ii] = m_true
                        fitness[ii] = _fitness_from_metrics(m_true, objective)

            idx = int(np.argmin(fitness))
            if float(fitness[idx]) < best_fit:
                best_fit = float(fitness[idx])
                best_design = designs[idx]
                best_metrics = pred_metrics[idx]

            if (_gen == 0) or ((_gen + 1) % 10 == 0) or (_gen == ga.generations - 1):
                pop_ok = float(np.mean(ok_mask)) if ok_mask is not None else float("nan")
                ap = float(best_metrics.get("apogee", np.nan)) if best_metrics else float("nan")
                stab = float(best_metrics.get("stability_min_over_flight", np.nan)) if best_metrics else float("nan")
                okr = best_metrics.get("ok_rate", None) if best_metrics else None
                okr_s = f", ok_rate={float(okr):.2f}" if okr is not None and np.isfinite(float(okr)) else ""
                print(
                    f"[GA fast] gen {_gen+1}/{ga.generations} best_fit={best_fit:.4g} apogee={ap:.1f}m stab_min={stab:.2f}cal"
                    f"{okr_s} pop_ok={pop_ok*100:.1f}%"
                )

                # In Fast mode, only rank within the validated subset.
                topn = int(min(3, len(top_idx)))
                if topn > 0:
                    best_validated = sorted([int(i) for i in top_idx], key=lambda i: float(fitness[i]))[:topn]
                    print(f"[GA fast] top{topn} validated (gen {_gen+1})")
                    for rank, i in enumerate(best_validated, start=1):
                        d_i = designs[int(i)]
                        m_i = pred_metrics[int(i)]
                        fit_i = float(fitness[int(i)])
                        print(f"  #{rank} fit={fit_i:.4g} {_fmt_metrics_short(m_i)} | {_fmt_design_short(d_i)}")

            elite_idx = np.argsort(fitness)[: ga.elite_size]
            elites = pop[elite_idx]

            new_pop = [elites[i % len(elites)].copy() for i in range(ga.pop_size)]
            new_pop = np.vstack(new_pop)

            for i in range(ga.elite_size, ga.pop_size):
                if rng.random() < ga.crossover_rate:
                    p1 = elites[rng.integers(0, len(elites))]
                    p2 = elites[rng.integers(0, len(elites))]
                    mask = rng.random(p1.shape[0]) > 0.5
                    child = np.where(mask, p1, p2)
                else:
                    child = elites[rng.integers(0, len(elites))].copy()

                if rng.random() < ga.mutation_rate:
                    child = child + rng.normal(0.0, ga.mutation_sigma, size=child.shape[0])
                    if rng.random() < 0.15:
                        child[8] = float(rng.integers(0, len(spec.nose_types)))
                    if rng.random() < 0.15:
                        child[9] = float(rng.integers(0, len(spec.fin_counts)))

                new_pop[i] = _clip_and_fix(child, bounds, spec, fixed)

            pop = new_pop

    assert best_design is not None and best_metrics is not None
    return best_design, best_metrics
