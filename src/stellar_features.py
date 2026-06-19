from __future__ import annotations

import numpy as np
import pandas as pd


BANDS = ["u", "g", "r", "i", "z"]
CAT_COLS = ["spectral_type", "galaxy_population"]
ADVANCED_CAT_COLS = ["spectral_population"]
RAW_NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
REALMLP_FLOOR_CAT_COLS = [f"{col}_floor_cat" for col in RAW_NUM_COLS]
REALMLP_BIN_CAT_COLS = ["delta_qbin_100", "delta_qbin_500"]
REALMLP_COMBO_CAT_COLS = ["alpha_floor_x_delta_floor", "u_floor_x_z_floor"]
REALMLP_CAT_COLS = [*REALMLP_FLOOR_CAT_COLS, *REALMLP_BIN_CAT_COLS, *REALMLP_COMBO_CAT_COLS]


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


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_features(df)

    band_values = out[BANDS]
    out["mag_median"] = band_values.median(axis=1)
    out["griz_mean"] = out[["g", "r", "i", "z"]].mean(axis=1)
    out["optical_center"] = out[["g", "r", "i"]].mean(axis=1)

    out["curve_ugr"] = out["u-g"] - out["g-r"]
    out["curve_gri"] = out["g-r"] - out["r-i"]
    out["curve_riz"] = out["r-i"] - out["i-z"]
    out["curve_ugri"] = out["curve_ugr"] - out["curve_gri"]
    out["curve_griz"] = out["curve_gri"] - out["curve_riz"]

    redshift_nonneg = out["redshift"].clip(lower=0)
    out["redshift_sqrt"] = np.sqrt(redshift_nonneg)
    out["redshift_sq"] = out["redshift"] ** 2

    color_cols = [
        "u-g",
        "g-r",
        "r-i",
        "i-z",
        "u-r",
        "g-i",
        "r-z",
        "curve_ugr",
        "curve_gri",
        "curve_riz",
    ]
    for col in color_cols:
        out[f"{col}_x_redshift"] = out[col] * out["redshift"]
        out[f"{col}_x_redshift_log1p"] = out[col] * out["redshift_log1p"]

    denom = out["mag_range"].abs() + 1e-6
    for col in ["u-g", "g-r", "r-i", "i-z", "u-r", "g-i", "r-z"]:
        out[f"{col}_norm_range"] = out[col] / denom

    out["u_g_ratio"] = out["u"] / (out["g"].abs() + 1e-6)
    out["g_r_ratio"] = out["g"] / (out["r"].abs() + 1e-6)
    out["r_i_ratio"] = out["r"] / (out["i"].abs() + 1e-6)
    out["i_z_ratio"] = out["i"] / (out["z"].abs() + 1e-6)

    alpha = np.deg2rad(out["alpha"])
    delta = np.deg2rad(out["delta"])
    out["sky_x"] = np.cos(delta) * np.cos(alpha)
    out["sky_y"] = np.cos(delta) * np.sin(alpha)
    out["sky_z"] = np.sin(delta)

    if set(CAT_COLS).issubset(out.columns):
        out["spectral_population"] = (
            out["spectral_type"].astype(str) + "_" + out["galaxy_population"].astype(str)
        )

    return out


def _safe_divide_by_redshift(values: pd.Series, redshift: pd.Series) -> pd.Series:
    sign = np.where(redshift.to_numpy() < 0, -1.0, 1.0)
    denom = np.where(redshift.abs().to_numpy() > 1e-5, redshift.to_numpy(), sign * 1e-5)
    divided = values.to_numpy() / denom
    return pd.Series(np.clip(divided, -1_000_000.0, 1_000_000.0), index=values.index)


def _fit_quantile_edges(values: pd.Series, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values.to_numpy(dtype=np.float64), quantiles)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _apply_quantile_bin(values: pd.Series, edges: np.ndarray) -> pd.Series:
    return pd.cut(values, bins=edges, labels=False, include_lowest=True).fillna(-1).astype("int16").astype(str)


def add_realmlp_style_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature set inspired by the public RealMLP v5 single-model recipe.

    Target encoding from that notebook must stay fold-local, so this function only
    creates non-target-derived numerical, binned, and interaction features.
    """
    train_out = add_advanced_features(train)
    test_out = add_advanced_features(test)

    for out in (train_out, test_out):
        out["g_div_redshift"] = _safe_divide_by_redshift(out["g"], out["redshift"])
        out["i_div_redshift"] = _safe_divide_by_redshift(out["i"], out["redshift"])
        out["realmlp_mag_mean"] = out[BANDS].mean(axis=1)
        out["realmlp_mag_range"] = out[BANDS].max(axis=1) - out[BANDS].min(axis=1)
        shifted_redshift = out["redshift"] - min(float(train_out["redshift"].min()), float(test_out["redshift"].min()), 0.0)
        out["realmlp_log1p_shifted_redshift"] = np.log1p(shifted_redshift.clip(lower=0))

        for col in RAW_NUM_COLS:
            out[f"{col}_floor_cat"] = np.floor(out[col]).clip(-10000, 10000).astype("int16").astype(str)

        out["alpha_floor_x_delta_floor"] = out["alpha_floor_cat"] + "_" + out["delta_floor_cat"]
        out["u_floor_x_z_floor"] = out["u_floor_cat"] + "_" + out["z_floor_cat"]

    for n_bins in (100, 500):
        edges = _fit_quantile_edges(train_out["delta"], n_bins)
        train_out[f"delta_qbin_{n_bins}"] = _apply_quantile_bin(train_out["delta"], edges)
        test_out[f"delta_qbin_{n_bins}"] = _apply_quantile_bin(test_out["delta"], edges)

    return train_out, test_out


def categorical_columns_for_feature_set(feature_set: str) -> list[str]:
    cols = [*CAT_COLS]
    if feature_set in {"advanced", "realmlp"}:
        cols.extend(ADVANCED_CAT_COLS)
    if feature_set == "realmlp":
        cols.extend(REALMLP_CAT_COLS)
    return list(dict.fromkeys(cols))


def encode_categories(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str = "base",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()
    for col in categorical_columns_for_feature_set(feature_set):
        if col not in train_out.columns or col not in test_out.columns:
            continue
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
    feature_set: str = "base",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    if feature_set == "base":
        train_fe = add_features(train)
        test_fe = add_features(test)
    elif feature_set == "advanced":
        train_fe = add_advanced_features(train)
        test_fe = add_advanced_features(test)
    elif feature_set == "realmlp":
        train_fe, test_fe = add_realmlp_style_features(train, test)
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    train_fe, test_fe = encode_categories(train_fe, test_fe, feature_set=feature_set)

    drop_cols = [id_col, target]
    features = [col for col in train_fe.columns if col not in drop_cols]
    return train_fe[features], train_fe[target], test_fe[features], features
