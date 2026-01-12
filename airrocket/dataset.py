from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from .design_space import Bounds, sample_design


@dataclass(frozen=True)
class DatasetConfig:
    seed: int | None = None


def make_rng(seed: int | None) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def sample_design_row(rng: np.random.Generator, bounds: Bounds) -> dict:
    design = sample_design(rng, bounds)
    return {"id": str(uuid.uuid4())[:8], **design}
