from __future__ import annotations

import numpy as np
import pandas as pd


BANDS = ["u", "g", "r", "i", "z"]
CAT_COLS = ["spectral_type", "galaxy_population"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for left, right in zip(BANDS[:-1], BANDS[1:]):
        out[f"{left}-{right}"] = out[left] - out[right]

    out["u-r"] = out["u"] - out["r"]
    out["u-i"] = out["u"] - out["i"]
    out["u-z"] = out["u"] - out["z"]
    out["g-i"] = out["g"] - out["i"]
    out["g-z"] = out["g"] - out["z"]
    out["r-z"] = out["r"] - out["z"]

    band_values = out[BANDS]
    out["mag_mean"] = band_values.mean(axis=1)
    out["mag_std"] = band_values.std(axis=1)
    out["mag_min"] = band_values.min(axis=1)
    out["mag_max"] = band_values.max(axis=1)
    out["mag_range"] = out["mag_max"] - out["mag_min"]

    out["redshift_abs"] = out["redshift"].abs()
    out["redshift_log1p"] = np.log1p(out["redshift"].clip(lower=0))
    out["redshift_x_u-r"] = out["redshift"] * out["u-r"]
    out["redshift_x_g-i"] = out["redshift"] * out["g-i"]

    out["alpha_sin"] = np.sin(np.deg2rad(out["alpha"]))
    out["alpha_cos"] = np.cos(np.deg2rad(out["alpha"]))
    out["delta_sin"] = np.sin(np.deg2rad(out["delta"]))
    out["delta_cos"] = np.cos(np.deg2rad(out["delta"]))

    return out


def encode_categories(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()
    for col in CAT_COLS:
        categories = sorted(set(train_out[col].astype(str)) | set(test_out[col].astype(str)))
        mapping = {value: idx for idx, value in enumerate(categories)}
        train_out[col] = train_out[col].astype(str).map(mapping).astype("int16")
        test_out[col] = test_out[col].astype(str).map(mapping).astype("int16")
    return train_out, test_out


def make_xy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str = "class",
    id_col: str = "id",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    train_fe = add_features(train)
    test_fe = add_features(test)
    train_fe, test_fe = encode_categories(train_fe, test_fe)

    drop_cols = [id_col, target]
    features = [col for col in train_fe.columns if col not in drop_cols]
    return train_fe[features], train_fe[target], test_fe[features], features
