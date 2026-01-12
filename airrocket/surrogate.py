from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb


@dataclass(frozen=True)
class SurrogateSpec:
    feature_cols: tuple[str, ...]
    categorical_cols: tuple[str, ...]
    target_cols: tuple[str, ...]


def default_surrogate_spec() -> SurrogateSpec:
    # Inputs are the *free design variables* (no derived tube_d).
    feature_cols = (
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
    categorical_cols = ("nose_type",)
    # Robust targets for Fast mode.
    target_cols = (
        "apogee",
        "stability_min_over_flight",
        "stability_at_max_q",
        "max_mach",
        "max_acceleration",
    )
    return SurrogateSpec(feature_cols=feature_cols, categorical_cols=categorical_cols, target_cols=target_cols)


def make_model(random_state: int = 42) -> Pipeline:
    # ColumnTransformer: one-hot encode categoricals, passthrough others.
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["nose_type"]),
        ],
        remainder="passthrough",
    )

    base = xgb.XGBRegressor(
        n_estimators=6000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=0,
    )

    # MultiOutputRegressor trains one model per target.
    model = Pipeline(
        steps=[
            ("pre", pre),
            ("model", MultiOutputRegressor(base)),
        ]
    )
    return model


def fit_surrogate(df: pd.DataFrame, spec: SurrogateSpec, *, random_state: int = 42) -> Pipeline:
    clean = df.copy()
    # Ensure numeric types for numeric cols.
    for c in spec.feature_cols:
        if c not in spec.categorical_cols:
            clean[c] = pd.to_numeric(clean[c], errors="coerce")

    X = clean.loc[:, list(spec.feature_cols)]
    y = clean.loc[:, list(spec.target_cols)]

    model = make_model(random_state=random_state)
    model.fit(X, y)
    return model


def predict_surrogate(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def save_bundle(path: str, *, model: Pipeline, spec: SurrogateSpec) -> None:
    joblib.dump({"model": model, "spec": spec}, path)


def load_bundle(path: str) -> dict:
    return joblib.load(path)
