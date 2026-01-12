from __future__ import annotations

import threading
from dataclasses import dataclass

from rocketpy import Environment

_thread_local = threading.local()
_process_env: Environment | None = None


@dataclass(frozen=True)
class EnvConfig:
    latitude: float = 0.0
    longitude: float = 0.0
    elevation: float = 0.0
    date: tuple[int, int, int, int] | None = (2025, 1, 1, 12)
    atmospheric_model: str = "standard_atmosphere"


def make_env(cfg: EnvConfig) -> Environment:
    env = Environment(latitude=cfg.latitude, longitude=cfg.longitude, elevation=cfg.elevation)
    if cfg.date is not None:
        env.set_date(cfg.date)
    env.set_atmospheric_model(type=cfg.atmospheric_model)
    return env


def get_thread_env(cfg: EnvConfig) -> Environment:
    env = getattr(_thread_local, "env", None)
    if env is None:
        env = make_env(cfg)
        _thread_local.env = env
    return env


def init_process_env(cfg: EnvConfig) -> None:
    global _process_env
    _process_env = make_env(cfg)


def get_process_env(cfg: EnvConfig) -> Environment:
    global _process_env
    if _process_env is None:
        _process_env = make_env(cfg)
    return _process_env
