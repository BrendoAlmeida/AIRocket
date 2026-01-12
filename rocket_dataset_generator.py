import os
import multiprocessing as mp
from pathlib import Path
import time
from collections import deque
import json
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd

from airrocket.dataset import make_rng, sample_design_row
from airrocket.design_space import Bounds
from airrocket.env import EnvConfig, get_thread_env, get_process_env, init_process_env
from airrocket.simulate import SimConfig, simulate_metrics, simulate_monte_carlo, MonteCarloConfig


def _flatten_stats(prefix: str, stats: dict) -> dict:
    out: dict[str, float] = {}
    for k, v in (stats or {}).items():
        out[f"{prefix}_{k}"] = v
    return out


def simulate_one_design(
    i: int,
    verbose: bool = False,
    ode_solver: str = "LSODA",
    env_mode: str = "thread",  # "thread" | "process"
    flight_terminate_on_apogee: bool = True,
    flight_max_time: float = 300.0,
    seed: int | None = None,
    env_cfg: EnvConfig | None = None,
    bounds: Bounds | None = None,
    mc_enabled: bool = False,
    mc_n_runs: int = 20,
    mc_random_state_base: int = 123,
    mc_wind_profile: str = "linear",
    mc_wind_ref_alt_m: float = 2500.0,
    # wind at ground
    mc_wind_u_sigma_m_s: float = 2.0,
    mc_wind_v_sigma_m_s: float = 2.0,
    # wind at ref altitude
    mc_wind_u_ref_sigma_m_s: float = 6.0,
    mc_wind_v_ref_sigma_m_s: float = 6.0,
    mc_wind_gust_sigma_m_s: float = 0.0,
    mc_only_if_ok: bool = True,
):
    if env_cfg is None:
        env_cfg = EnvConfig()
    if bounds is None:
        bounds = Bounds()

    # Make sampling deterministic per index when seed is provided.
    rng = make_rng(None if seed is None else (int(seed) + int(i)))
    row = sample_design_row(rng, bounds)

    env = get_thread_env(env_cfg) if env_mode == "thread" else get_process_env(env_cfg)
    sim_cfg = SimConfig(
        ode_solver=ode_solver,
        terminate_on_apogee=flight_terminate_on_apogee,
        max_time_s=float(flight_max_time),
    )

    try:
        out = simulate_metrics(row, env, sim_cfg)

        # If MC is enabled, pre-populate stable schema columns (NaNs) even when MC is skipped.
        if mc_enabled:
            plo = 5
            phi = 95
            try:
                # Match defaults from MonteCarloConfig.
                from airrocket.simulate import MonteCarloConfig as _MCC

                _tmp = _MCC()
                plo = int(_tmp.p_lo)
                phi = int(_tmp.p_hi)
            except Exception:
                pass

            def _nan_stats(prefix: str) -> None:
                for k in ("mean", "std", "min", "max", f"p{plo}", "p50", f"p{phi}"):
                    out.setdefault(f"{prefix}_{k}", float("nan"))

            out.setdefault("mc_n", int(mc_n_runs))
            out.setdefault("mc_ok", float("nan"))
            out.setdefault("mc_fail", float("nan"))
            out.setdefault("mc_ok_rate", float("nan"))
            out.setdefault("mc_wind_profile", str(mc_wind_profile))
            out.setdefault("mc_wind_ref_alt_m", float(mc_wind_ref_alt_m))
            out.setdefault("mc_fail_reason_counts", "{}")

            _nan_stats("mc_wind_u0")
            _nan_stats("mc_wind_v0")
            _nan_stats("mc_wind_u_ref")
            _nan_stats("mc_wind_v_ref")

            _nan_stats("mc_apogee")
            _nan_stats("mc_stability_min")
            _nan_stats("mc_stability_at_max_q")
            _nan_stats("mc_max_mach")
            _nan_stats("mc_max_acceleration")
            _nan_stats("mc_max_dynamic_pressure")
            _nan_stats("mc_rail_exit_speed")

        if mc_enabled and ((not mc_only_if_ok) or bool(out.get("ok", False))):
            mc_cfg = MonteCarloConfig(
                n_runs=int(mc_n_runs),
                random_state=int(mc_random_state_base) + int(i),
                wind_profile=str(mc_wind_profile),
                wind_ref_alt_m=float(mc_wind_ref_alt_m),
                wind_u_sigma_m_s=float(mc_wind_u_sigma_m_s),
                wind_v_sigma_m_s=float(mc_wind_v_sigma_m_s),
                wind_u_ref_sigma_m_s=float(mc_wind_u_ref_sigma_m_s),
                wind_v_ref_sigma_m_s=float(mc_wind_v_ref_sigma_m_s),
                wind_gust_sigma_m_s=float(mc_wind_gust_sigma_m_s),
            )
            mc = simulate_monte_carlo(row, env, sim_cfg, mc_cfg)

            # High-level MC metadata
            out["mc_n"] = mc.get("mc_n")
            out["mc_ok"] = mc.get("mc_ok")
            out["mc_fail"] = mc.get("mc_fail")
            out["mc_ok_rate"] = mc.get("mc_ok_rate")
            out["mc_wind_profile"] = mc.get("wind_profile")
            out["mc_wind_ref_alt_m"] = mc.get("wind_ref_alt_m")
            out["mc_fail_reason_counts"] = json.dumps(mc.get("mc_fail_reason_counts", {}), ensure_ascii=False)

            # Wind stats (ground + ref altitude)
            out.update(_flatten_stats("mc_wind_u0", mc.get("wind_u_stats", {})))
            out.update(_flatten_stats("mc_wind_v0", mc.get("wind_v_stats", {})))
            out.update(_flatten_stats("mc_wind_u_ref", mc.get("wind_u_ref_stats", {})))
            out.update(_flatten_stats("mc_wind_v_ref", mc.get("wind_v_ref_stats", {})))

            # Output stats
            out.update(_flatten_stats("mc_apogee", mc.get("apogee_stats", {})))
            out.update(_flatten_stats("mc_stability_min", mc.get("stability_min_over_flight_stats", {})))
            out.update(_flatten_stats("mc_stability_at_max_q", mc.get("stability_at_max_q_stats", {})))
            out.update(_flatten_stats("mc_max_mach", mc.get("max_mach_stats", {})))
            out.update(_flatten_stats("mc_max_acceleration", mc.get("max_acceleration_stats", {})))
            out.update(_flatten_stats("mc_max_dynamic_pressure", mc.get("max_dynamic_pressure_stats", {})))
            out.update(_flatten_stats("mc_rail_exit_speed", mc.get("rail_exit_speed_stats", {})))

        return out
    except Exception as e:
        if verbose:
            print(f"Design {i} falhou na simulação: {e}")
        return None


def generate_full_dna_dataset(
    iterations: int = 1000,
    n_workers: int | None = None,
    verbose: bool = False,
    ode_solver: str | None = None,
    backend: str = "process",  # "thread" | "process"
    progress_every: int = 1,
    max_pending: int | None = None,
    checkpoint_path: str | os.PathLike | None = None,
    checkpoint_every: int | None = None,
    flight_terminate_on_apogee: bool = True,
    flight_max_time: float = 300.0,
    poll_interval_s: float = 5.0,
    stall_timeout_s: float = 120.0,
    restart_on_stall: bool = True,
    max_restarts: int = 10,
    task_timeout_s: float | None = None,
    max_task_retries: int = 1,
    seed: int | None = None,
    env_cfg: EnvConfig | None = None,
    bounds: Bounds | None = None,
    # Monte Carlo per-design (robust but expensive)
    mc_enabled: bool = False,
    mc_n_runs: int = 20,
    mc_random_state_base: int = 123,
    mc_wind_profile: str = "linear",
    mc_wind_ref_alt_m: float = 2500.0,
    mc_wind_u_sigma_m_s: float = 2.0,
    mc_wind_v_sigma_m_s: float = 2.0,
    mc_wind_u_ref_sigma_m_s: float = 6.0,
    mc_wind_v_ref_sigma_m_s: float = 6.0,
    mc_wind_gust_sigma_m_s: float = 0.0,
    mc_only_if_ok: bool = True,
):
    if n_workers is None:
        cpu = os.cpu_count() or 1
        n_workers = max(1, min(cpu, 8))
    else:
        n_workers = max(1, int(n_workers))

    backend = (backend or "").strip().lower()
    if backend not in {"thread", "process"}:
        raise ValueError("backend deve ser 'thread' ou 'process'")

    progress_every = max(1, int(progress_every))

    if ode_solver is None:
        if backend == "thread" and n_workers > 1:
            ode_solver = "DOP853"
        else:
            ode_solver = "LSODA"

    if max_pending is None:
        # Evita enfileirar milhares de jobs de uma vez (economiza RAM e overhead).
        # Em processos, cada job pode consumir bastante memória (RocketPy/SciPy),
        # então mantemos poucos pendentes.
        max_pending = n_workers * (2 if backend == "process" else 20)
    max_pending = max(1, int(max_pending))
    max_pending = min(max_pending, int(iterations))

    if checkpoint_every is not None:
        checkpoint_every = max(1, int(checkpoint_every))

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_checkpoint():
        if checkpoint_path is None:
            return
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        pd.DataFrame(results).to_csv(tmp_path, index=False)
        os.replace(tmp_path, checkpoint_path)

    results: list[dict] = []
    failed = 0
    success = 0
    aborted = 0

    poll_interval_s = max(0.1, float(poll_interval_s))
    stall_timeout_s = max(poll_interval_s, float(stall_timeout_s))
    max_restarts = max(0, int(max_restarts))

    if task_timeout_s is not None:
        task_timeout_s = max(0.1, float(task_timeout_s))
    else:
        # O Flight(max_time=flight_max_time) deveria limitar a simulação, mas
        # na prática podem ocorrer travamentos em integração/IO. Damos uma folga.
        task_timeout_s = max(float(flight_max_time) + 30.0, float(flight_max_time) * 1.5)

    max_task_retries = max(0, int(max_task_retries))

    if backend == "process":
        # Usamos multiprocessing.Pool para poder terminar/reiniciar workers em caso de travamento.
        ctx = mp.get_context("spawn")
        restarts = 0
        next_i = 0
        last_progress = time.monotonic()
        pending: dict[int, tuple["mp.pool.ApplyResult", float, int]] = {}
        # pending[i] = (ApplyResult, submit_time_monotonic, attempts)
        retry_queue: deque[tuple[int, int]] = deque()

        def _terminate_pool(p):
            try:
                p.terminate()
            finally:
                try:
                    p.join()
                except Exception:
                    pass

        def _submit(pool, i: int, attempts: int = 0):
            pending[i] = (
                pool.apply_async(
                simulate_one_design,
                args=(
                    i,
                    verbose,
                    ode_solver,
                    "process",
                    flight_terminate_on_apogee,
                    flight_max_time,
                    seed,
                    env_cfg,
                    bounds,
                    mc_enabled,
                    mc_n_runs,
                    mc_random_state_base,
                    mc_wind_profile,
                    mc_wind_ref_alt_m,
                    mc_wind_u_sigma_m_s,
                    mc_wind_v_sigma_m_s,
                    mc_wind_u_ref_sigma_m_s,
                    mc_wind_v_ref_sigma_m_s,
                    mc_wind_gust_sigma_m_s,
                    mc_only_if_ok,
                ),
                ),
                time.monotonic(),
                int(attempts),
            )

        while next_i < iterations or pending or retry_queue:
            # (Re)cria pool
            pool = ctx.Pool(
                processes=n_workers,
                initializer=init_process_env,
                initargs=(env_cfg or EnvConfig(),),
                maxtasksperchild=1,
            )
            try:
                # Submete primeiro os retries, depois novas iterações, até max_pending
                while retry_queue and len(pending) < max_pending:
                    i, attempts = retry_queue.popleft()
                    _submit(pool, i, attempts)

                while next_i < iterations and len(pending) < max_pending:
                    _submit(pool, next_i, 0)
                    next_i += 1

                while pending:
                    progressed = False
                    finished_indices = []
                    timed_out_indices: list[int] = []

                    # Varre o que já ficou pronto sem bloquear
                    now = time.monotonic()
                    for i, (ar, submitted_at, attempts) in list(pending.items()):
                        if ar.ready():
                            try:
                                row = ar.get(timeout=0)
                            except Exception as e:
                                if verbose:
                                    print(f"Design {i} falhou na simulação: {e}")
                                row = None

                            finished_indices.append(i)
                            progressed = True

                            if row is None:
                                failed += 1
                            else:
                                results.append(row)
                                success += 1
                                last_progress = time.monotonic()
                                if verbose and (success % progress_every == 0):
                                    print(f"Linha gerada: {success}/{iterations} | id={row['id']}")
                                if checkpoint_every is not None and (success % checkpoint_every == 0):
                                    _write_checkpoint()

                    for i in finished_indices:
                        pending.pop(i, None)

                    # Timeout por job: se algum job ficou tempo demais, reiniciamos o pool
                    # (isso efetivamente cancela os processos travados).
                    for i, (ar, submitted_at, attempts) in list(pending.items()):
                        if (now - submitted_at) >= task_timeout_s:
                            timed_out_indices.append(i)

                    if timed_out_indices:
                        if verbose:
                            oldest = max(now - pending[i][1] for i in timed_out_indices)
                            print(
                                f"Timeout em {len(timed_out_indices)} job(s) (mais antigo ~{oldest:.0f}s). "
                                "Terminando pool para cancelar travados e reiniciando..."
                            )

                        # Decide quais índices vão ser re-tentados vs abortados
                        to_abort = 0
                        for i, (_ar, _submitted_at, attempts) in list(pending.items()):
                            if attempts < max_task_retries:
                                retry_queue.append((i, attempts + 1))
                            else:
                                to_abort += 1

                        aborted += to_abort
                        pending.clear()

                        # Mata os processos (cancela travados)
                        _terminate_pool(pool)

                        # Recria pool no próximo ciclo, e re-submete o que precisa
                        restarts += 1
                        if restarts > max_restarts and max_restarts > 0:
                            if verbose:
                                print(
                                    "Muitos reinícios por timeout; retornando parcial. "
                                    "(Dica: reduza n_workers/max_pending ou aumente max_restarts.)"
                                )
                            next_i = iterations
                            break
                        break

                    while next_i < iterations and len(pending) < max_pending:
                        _submit(pool, next_i, 0)
                        next_i += 1

                    if not progressed:
                        elapsed = time.monotonic() - last_progress
                        if verbose:
                            print(
                                f"Sem novas linhas por {elapsed:.1f}s | pendentes={len(pending)} | "
                                f"progresso={success}/{iterations}"
                            )
                        if checkpoint_path is not None:
                            _write_checkpoint()

                        if elapsed >= stall_timeout_s:
                            if restart_on_stall and restarts < max_restarts:
                                if verbose:
                                    print(
                                        "Stall detectado; terminando pool e reiniciando. "
                                        "(Alguns jobs pendentes serão descartados.)"
                                    )
                                aborted += len(pending)
                                pending.clear()
                                restarts += 1
                                _terminate_pool(pool)
                                break
                            else:
                                if verbose:
                                    print(
                                        "Stall detectado; encerrando e retornando parcial. "
                                        "(Dica: reduza n_workers/max_pending, flight_max_time ou aumente max_restarts.)"
                                    )
                                _terminate_pool(pool)
                                pending.clear()
                                next_i = iterations
                                break

                        time.sleep(poll_interval_s)

                # Saiu do loop de pending sem break por stall
                try:
                    pool.close()
                finally:
                    pool.join()
            finally:
                # Garantia de limpeza caso exceção
                try:
                    pool.close()
                    pool.join()
                except Exception:
                    pass

        if checkpoint_path is not None:
            _write_checkpoint()

    else:
        # Thread backend
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            pending_futures = set()
            next_i = 0
            last_progress = time.monotonic()

            def _submit(i: int):
                return executor.submit(
                    simulate_one_design,
                    i,
                    verbose,
                    ode_solver,
                    "thread",
                    flight_terminate_on_apogee,
                    flight_max_time,
                    seed,
                    env_cfg,
                    bounds,
                    mc_enabled,
                    mc_n_runs,
                    mc_random_state_base,
                    mc_wind_profile,
                    mc_wind_ref_alt_m,
                    mc_wind_u_sigma_m_s,
                    mc_wind_v_sigma_m_s,
                    mc_wind_u_ref_sigma_m_s,
                    mc_wind_v_ref_sigma_m_s,
                    mc_wind_gust_sigma_m_s,
                    mc_only_if_ok,
                )

            while next_i < iterations and len(pending_futures) < max_pending:
                pending_futures.add(_submit(next_i))
                next_i += 1

            while pending_futures:
                done, pending_futures = wait(
                    pending_futures,
                    timeout=poll_interval_s,
                    return_when=FIRST_COMPLETED,
                )

                if not done:
                    elapsed = time.monotonic() - last_progress
                    if verbose:
                        print(
                            f"Sem novas linhas por {elapsed:.0f}s | pendentes={len(pending_futures)} | "
                            f"progresso={success}/{iterations}"
                        )
                    if checkpoint_path is not None:
                        _write_checkpoint()
                    if elapsed >= stall_timeout_s:
                        if verbose:
                            print(
                                "Stall detectado; encerrando e retornando parcial. "
                                "(Dica: reduza n_workers/max_pending ou flight_max_time.)"
                            )
                        break
                    continue

                for future in done:
                    row = future.result()
                    if row is None:
                        failed += 1
                    else:
                        results.append(row)
                        success += 1
                        last_progress = time.monotonic()
                        if verbose and (success % progress_every == 0):
                            print(f"Linha gerada: {success}/{iterations} | id={row['id']}")
                        if checkpoint_every is not None and (success % checkpoint_every == 0):
                            _write_checkpoint()

                    if next_i < iterations:
                        pending_futures.add(_submit(next_i))
                        next_i += 1

        if checkpoint_path is not None:
            _write_checkpoint()

    if verbose:
        print(
            f"Concluído. Sucessos: {len(results)} | Falhas: {failed} | "
            f"Abortados: {aborted} | Workers: {n_workers} | Backend: {backend} | ODE: {ode_solver}"
        )

    return pd.DataFrame(results)
