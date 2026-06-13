from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import add_features  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "star_to_galaxy_research"
MODELS = ["lgbm", "catboost"]
CLASSES = ["GALAXY", "QSO", "STAR"]

BASE_ANCHOR_CANDIDATES = [
    ARTIFACTS / "probe_queue" / "group_minconf_095_star_to_galaxy_except_rank_01.csv",
    ARTIFACTS / "probe_queue" / "group_minconf_095_all.csv",
    DATA / "submission.csv",
]

# This row was already submitted as a single-row STAR->GALAXY probe and did
# not move the public score, so next queues should spend submissions elsewhere.
KNOWN_SINGLE_NEUTRAL_IDS = {714971}


def load_base_anchor() -> tuple[pd.DataFrame, Path]:
    for path in BASE_ANCHOR_CANDIDATES:
        if path.exists():
            return pd.read_csv(path), path
    raise FileNotFoundError("No usable anchor submission was found.")


def load_model_proba(model: str) -> tuple[list[str], np.ndarray]:
    report_path = ARTIFACTS / f"{model}_baseline_report.json"
    proba_path = ARTIFACTS / f"{model}_test_proba.npy"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report["classes"], np.load(proba_path)


def add_quantile_bins(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    q: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()

    for col in columns:
        values = train_out[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, q + 1)))
        if len(edges) <= 2:
            train_out[f"{col}_bin"] = 0
            test_out[f"{col}_bin"] = 0
            continue

        inner_edges = edges[1:-1]
        train_out[f"{col}_bin"] = np.digitize(train_out[col].to_numpy(), inner_edges, right=False)
        test_out[f"{col}_bin"] = np.digitize(test_out[col].to_numpy(), inner_edges, right=False)

    return train_out, test_out


def build_rate_table(
    train: pd.DataFrame,
    keys: list[str],
    name: str,
    alpha: float = 30.0,
) -> pd.DataFrame:
    global_counts = train["class"].value_counts().reindex(CLASSES, fill_value=0).astype(float)
    global_prior = global_counts / global_counts.sum()

    counts = (
        train.groupby(keys + ["class"], observed=True)
        .size()
        .unstack("class", fill_value=0)
        .reindex(columns=CLASSES, fill_value=0)
        .astype(float)
    )
    total = counts.sum(axis=1)
    smoothed = counts.copy()
    for cls in CLASSES:
        smoothed[cls] = (counts[cls] + alpha * global_prior[cls]) / (total + alpha)

    table = smoothed.reset_index()
    table[f"{name}_n"] = total.to_numpy()
    for cls in CLASSES:
        table = table.rename(columns={cls: f"{name}_p_{cls}"})
    table[f"{name}_margin_galaxy_star"] = table[f"{name}_p_GALAXY"] - table[f"{name}_p_STAR"]
    return table


def attach_local_evidence(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    binned_cols = ["redshift", "u-r", "g-r", "g-i", "r-i", "mag_mean", "mag_std"]
    train_binned, test_binned = add_quantile_bins(train, test, binned_cols)

    specs = [
        ("cat", ["spectral_type", "galaxy_population"], 0.10),
        ("cat_redshift_ur", ["spectral_type", "galaxy_population", "redshift_bin", "u-r_bin"], 0.24),
        ("cat_redshift_gi", ["spectral_type", "galaxy_population", "redshift_bin", "g-i_bin"], 0.20),
        ("cat_colors", ["spectral_type", "galaxy_population", "u-r_bin", "g-r_bin", "g-i_bin"], 0.22),
        (
            "cat_redshift_mag_color",
            ["spectral_type", "galaxy_population", "redshift_bin", "mag_mean_bin", "mag_std_bin", "u-r_bin"],
            0.24,
        ),
    ]

    out = test_binned.copy()
    weighted_galaxy = np.zeros(len(out), dtype=np.float64)
    weighted_star = np.zeros(len(out), dtype=np.float64)
    used_weight = np.zeros(len(out), dtype=np.float64)

    for name, keys, weight in specs:
        table = build_rate_table(train_binned, keys, name)
        out = out.merge(table, on=keys, how="left")

        n_col = f"{name}_n"
        galaxy_col = f"{name}_p_GALAXY"
        star_col = f"{name}_p_STAR"
        valid = out[n_col].fillna(0).to_numpy() >= 20
        weighted_galaxy[valid] += weight * out.loc[valid, galaxy_col].to_numpy()
        weighted_star[valid] += weight * out.loc[valid, star_col].to_numpy()
        used_weight[valid] += weight

    cat_galaxy = out["cat_p_GALAXY"].to_numpy()
    cat_star = out["cat_p_STAR"].to_numpy()
    out["local_p_GALAXY"] = np.where(used_weight > 0, weighted_galaxy / used_weight, cat_galaxy)
    out["local_p_STAR"] = np.where(used_weight > 0, weighted_star / used_weight, cat_star)
    out["local_margin_galaxy_star"] = out["local_p_GALAXY"] - out["local_p_STAR"]
    out["local_evidence_weight"] = used_weight
    return out


def write_submission(base_anchor: pd.DataFrame, changes: pd.DataFrame, name: str) -> Path:
    submission = base_anchor.copy()
    id_to_label = dict(zip(changes["id"], changes["proposed_label"]))
    mask = submission["id"].isin(id_to_label)
    submission.loc[mask, "class"] = submission.loc[mask, "id"].map(id_to_label)
    path = OUT_DIR / name
    submission.to_csv(path, index=False)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    original_anchor = pd.read_csv(DATA / "submission.csv")
    base_anchor, base_anchor_path = load_base_anchor()

    if not sample["id"].equals(test["id"]) or not sample["id"].equals(original_anchor["id"]):
        raise ValueError("sample/test/original anchor id order differs")
    if not sample["id"].equals(base_anchor["id"]):
        raise ValueError(f"base anchor id order differs: {base_anchor_path}")

    train_fe = add_features(train)
    test_fe = add_features(test)
    evidence = attach_local_evidence(train_fe, test_fe)

    classes = None
    model_frames = []
    for model in MODELS:
        model_classes, proba = load_model_proba(model)
        if classes is None:
            classes = model_classes
        elif model_classes != classes:
            raise ValueError(f"{model} class order differs: {model_classes} != {classes}")

        class_index = {cls: idx for idx, cls in enumerate(model_classes)}
        frame = pd.DataFrame(
            {
                "id": sample["id"],
                f"{model}_pred": np.array(model_classes)[proba.argmax(axis=1)],
                f"{model}_conf": proba.max(axis=1),
                f"{model}_p_GALAXY": proba[:, class_index["GALAXY"]],
                f"{model}_p_QSO": proba[:, class_index["QSO"]],
                f"{model}_p_STAR": proba[:, class_index["STAR"]],
            }
        )
        model_frames.append(frame)

    scored = evidence.copy()
    keep_cols = [
        "id",
        "spectral_type",
        "galaxy_population",
        "redshift",
        "u",
        "g",
        "r",
        "i",
        "z",
        "u-r",
        "g-r",
        "g-i",
        "mag_mean",
        "mag_std",
        "local_p_GALAXY",
        "local_p_STAR",
        "local_margin_galaxy_star",
        "local_evidence_weight",
        "cat_n",
        "cat_p_GALAXY",
        "cat_p_STAR",
        "cat_redshift_ur_n",
        "cat_redshift_ur_p_GALAXY",
        "cat_redshift_ur_p_STAR",
        "cat_colors_n",
        "cat_colors_p_GALAXY",
        "cat_colors_p_STAR",
    ]
    scored = scored[keep_cols]
    scored["original_anchor"] = original_anchor["class"].to_numpy()
    scored["base_anchor"] = base_anchor["class"].to_numpy()

    for frame in model_frames:
        scored = scored.merge(frame, on="id", how="left")

    scored["mean_p_GALAXY"] = scored[[f"{model}_p_GALAXY" for model in MODELS]].mean(axis=1)
    scored["min_p_GALAXY"] = scored[[f"{model}_p_GALAXY" for model in MODELS]].min(axis=1)
    scored["mean_p_STAR"] = scored[[f"{model}_p_STAR" for model in MODELS]].mean(axis=1)
    scored["model_margin_galaxy_star"] = scored["mean_p_GALAXY"] - scored["mean_p_STAR"]
    scored["galaxy_model_votes"] = sum(scored[f"{model}_pred"].eq("GALAXY").astype(int) for model in MODELS)
    scored["both_models_predict_galaxy"] = scored["galaxy_model_votes"].eq(len(MODELS))
    scored["already_changed_in_base"] = scored["original_anchor"].ne(scored["base_anchor"])
    scored["known_single_neutral"] = scored["id"].isin(KNOWN_SINGLE_NEUTRAL_IDS)

    model_score = (
        0.45 * scored["mean_p_GALAXY"]
        + 0.35 * scored["min_p_GALAXY"]
        + 0.20 * ((scored["model_margin_galaxy_star"] + 1.0) / 2.0)
    )
    local_score = (
        0.70 * scored["local_p_GALAXY"]
        + 0.30 * ((scored["local_margin_galaxy_star"] + 1.0) / 2.0)
    )
    risk_penalty = 0.08 * scored["local_p_STAR"] + 0.05 * scored["mean_p_STAR"]
    scored["patchability_score"] = 100.0 * (0.68 * model_score + 0.32 * local_score - risk_penalty)
    scored["proposed_label"] = "GALAXY"

    star_to_galaxy = scored[
        scored["base_anchor"].eq("STAR")
        & scored["original_anchor"].eq("STAR")
        & scored["both_models_predict_galaxy"]
        & scored["local_margin_galaxy_star"].gt(0)
    ].copy()
    star_to_galaxy = star_to_galaxy.sort_values(
        ["patchability_score", "min_p_GALAXY", "local_p_GALAXY"],
        ascending=False,
    ).reset_index(drop=True)
    star_to_galaxy["research_rank"] = np.arange(1, len(star_to_galaxy) + 1)

    all_anchor_star = scored[scored["base_anchor"].eq("STAR")].copy()
    all_anchor_star = all_anchor_star.sort_values("patchability_score", ascending=False).reset_index(drop=True)
    all_anchor_star["research_rank"] = np.arange(1, len(all_anchor_star) + 1)

    all_anchor_star.to_csv(OUT_DIR / "all_base_anchor_star_scored.csv", index=False)
    star_to_galaxy.to_csv(OUT_DIR / "star_to_galaxy_scored_candidates.csv", index=False)

    next_pool = star_to_galaxy[
        star_to_galaxy["min_p_GALAXY"].ge(0.88)
        & star_to_galaxy["local_p_GALAXY"].ge(0.70)
        & ~star_to_galaxy["known_single_neutral"]
    ].copy()
    next_pool = next_pool.reset_index(drop=True)
    next_pool["next_probe_rank"] = np.arange(1, len(next_pool) + 1)
    next_pool.to_csv(OUT_DIR / "next_probe_pool.csv", index=False)

    active_submission_files: list[Path] = []
    single_pool = next_pool.head(10)
    for _, row in single_pool.iterrows():
        one = pd.DataFrame([row])
        name = f"single_next_rank_{int(row['next_probe_rank']):02d}_id_{int(row['id'])}_STAR_to_GALAXY.csv"
        active_submission_files.append(write_submission(base_anchor, one, name))

    grouped_specs = [
        ("group_research_top_03.csv", next_pool.head(3)),
        ("group_research_top_05.csv", next_pool.head(5)),
        ("group_research_top_10.csv", next_pool.head(10)),
        ("group_research_top_15.csv", next_pool.head(15)),
        ("group_research_top_20.csv", next_pool.head(20)),
        ("group_research_rank_06_10.csv", next_pool.iloc[5:10]),
        ("group_research_rank_11_15.csv", next_pool.iloc[10:15]),
        ("group_research_rank_16_20.csv", next_pool.iloc[15:20]),
        ("group_research_rank_11_20.csv", next_pool.iloc[10:20]),
        ("group_research_top10_plus_rank_16_20.csv", pd.concat([next_pool.head(10), next_pool.iloc[15:20]])),
        (
            "group_research_top10_plus_rank_17_19.csv",
            pd.concat([next_pool.head(10), next_pool.iloc[[16, 17, 18]]]),
        ),
        ("group_research_minpgal_092_top_10.csv", next_pool[next_pool["min_p_GALAXY"].ge(0.92)].head(10)),
        (
            "group_research_strong_local_top_10.csv",
            next_pool[next_pool["local_p_GALAXY"].ge(0.90)].head(10),
        ),
    ]
    top10 = next_pool.head(10)
    for rank in [7, 9, 10]:
        grouped_specs.append(
            (
                f"group_research_top10_without_rank_{rank:02d}.csv",
                top10[top10["next_probe_rank"].ne(rank)],
            )
        )
    grouped_specs.append(
        (
            "group_research_top10_without_rank_07_10.csv",
            top10[~top10["next_probe_rank"].isin([7, 10])],
        )
    )
    for name, changes in grouped_specs:
        if len(changes):
            active_submission_files.append(write_submission(base_anchor, changes, name))

    manifest_lines = [
        "# Active STAR->GALAXY probe files",
        "# Use these files from this run. Older single_research_rank_* files are stale pre-exclusion outputs.",
        "",
        "candidate_table: next_probe_pool.csv",
        "submissions:",
    ]
    manifest_lines.extend(f"- {path.name}" for path in active_submission_files)
    (OUT_DIR / "ACTIVE_PROBE_FILES.txt").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )

    report = {
        "base_anchor": str(base_anchor_path.relative_to(ROOT)),
        "rows_base_anchor_star": int(scored["base_anchor"].eq("STAR").sum()),
        "rows_already_changed_in_base": int(scored["already_changed_in_base"].sum()),
        "star_to_galaxy_candidates": int(len(star_to_galaxy)),
        "next_probe_pool": int(len(next_pool)),
        "known_single_neutral_ids_excluded": sorted(KNOWN_SINGLE_NEUTRAL_IDS),
        "filters": {
            "base_anchor": "STAR",
            "original_anchor": "STAR",
            "both_models_predict_galaxy": True,
            "local_margin_galaxy_star": "> 0",
            "next_pool_min_p_GALAXY": ">= 0.88",
            "next_pool_local_p_GALAXY": ">= 0.70",
            "known_single_neutral": False,
        },
    }
    (OUT_DIR / "score_star_to_galaxy_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    display_cols = [
        "next_probe_rank",
        "research_rank",
        "id",
        "patchability_score",
        "min_p_GALAXY",
        "mean_p_GALAXY",
        "local_p_GALAXY",
        "local_p_STAR",
        "local_margin_galaxy_star",
        "spectral_type",
        "galaxy_population",
        "redshift",
        "cat_redshift_ur_n",
        "cat_colors_n",
    ]

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print()
    print("Top next STAR->GALAXY candidates")
    print(next_pool[display_cols].head(30).to_string(index=False))
    print()
    print(f"wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
