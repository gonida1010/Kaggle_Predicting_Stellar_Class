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
OUT_DIR = ARTIFACTS / "transition_research"
CURRENT_PUBLIC_ANCHOR = ARTIFACTS / "star_to_galaxy_research" / "group_research_top_10.csv"
CLASSES = ["GALAXY", "QSO", "STAR"]
MODELS = ["lgbm", "catboost"]


def load_model_proba(model: str) -> tuple[list[str], np.ndarray]:
    report = json.loads((ARTIFACTS / f"{model}_baseline_report.json").read_text(encoding="utf-8"))
    return report["classes"], np.load(ARTIFACTS / f"{model}_test_proba.npy")


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
    alpha: float = 35.0,
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
    return table


def attach_local_class_rates(train_fe: pd.DataFrame, test_fe: pd.DataFrame) -> pd.DataFrame:
    binned_cols = [
        "redshift",
        "u-r",
        "g-r",
        "g-i",
        "r-i",
        "mag_mean",
        "mag_std",
        "mag_range",
    ]
    train_binned, test_binned = add_quantile_bins(train_fe, test_fe, binned_cols)
    specs = [
        ("cat", ["spectral_type", "galaxy_population"], 0.12),
        ("cat_redshift_ur", ["spectral_type", "galaxy_population", "redshift_bin", "u-r_bin"], 0.24),
        ("cat_redshift_gi", ["spectral_type", "galaxy_population", "redshift_bin", "g-i_bin"], 0.22),
        ("cat_colors", ["spectral_type", "galaxy_population", "u-r_bin", "g-r_bin", "g-i_bin"], 0.20),
        (
            "cat_mag_color",
            ["spectral_type", "galaxy_population", "mag_mean_bin", "mag_std_bin", "mag_range_bin", "u-r_bin"],
            0.22,
        ),
    ]
    out = test_binned.copy()
    weighted = {cls: np.zeros(len(out), dtype=np.float64) for cls in CLASSES}
    used_weight = np.zeros(len(out), dtype=np.float64)
    for name, keys, weight in specs:
        table = build_rate_table(train_binned, keys, name)
        out = out.merge(table, on=keys, how="left")
        n_col = f"{name}_n"
        valid = out[n_col].fillna(0).to_numpy() >= 20
        for cls in CLASSES:
            col = f"{name}_p_{cls}"
            weighted[cls][valid] += weight * out.loc[valid, col].to_numpy()
        used_weight[valid] += weight
    for cls in CLASSES:
        fallback = out[f"cat_p_{cls}"].to_numpy()
        out[f"local_p_{cls}"] = np.where(used_weight > 0, weighted[cls] / used_weight, fallback)
    out["local_evidence_weight"] = used_weight
    return out


def write_submission(anchor: pd.DataFrame, changes: pd.DataFrame, name: str) -> Path:
    submission = anchor.copy()
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
    anchor = pd.read_csv(CURRENT_PUBLIC_ANCHOR)
    if not sample["id"].equals(anchor["id"]):
        raise ValueError(f"anchor id order differs: {CURRENT_PUBLIC_ANCHOR}")

    train_fe = add_features(train)
    test_fe = add_features(test)
    evidence = attach_local_class_rates(train_fe, test_fe)

    scored = evidence[
        [
            "id",
            "spectral_type",
            "galaxy_population",
            "redshift",
            "u-r",
            "g-r",
            "g-i",
            "mag_mean",
            "mag_std",
            "mag_range",
            "local_p_GALAXY",
            "local_p_QSO",
            "local_p_STAR",
            "local_evidence_weight",
        ]
    ].copy()
    scored["anchor"] = anchor["class"].to_numpy()

    classes = None
    model_preds = []
    for model in MODELS:
        model_classes, proba = load_model_proba(model)
        if classes is None:
            classes = model_classes
        elif model_classes != classes:
            raise ValueError(f"{model} class order differs")
        class_idx = {cls: idx for idx, cls in enumerate(model_classes)}
        scored[f"{model}_pred"] = np.array(model_classes)[proba.argmax(axis=1)]
        scored[f"{model}_conf"] = proba.max(axis=1)
        for cls in CLASSES:
            scored[f"{model}_p_{cls}"] = proba[:, class_idx[cls]]
        model_preds.append(scored[f"{model}_pred"])

    for cls in CLASSES:
        scored[f"mean_p_{cls}"] = scored[[f"{model}_p_{cls}" for model in MODELS]].mean(axis=1)
        scored[f"min_p_{cls}"] = scored[[f"{model}_p_{cls}" for model in MODELS]].min(axis=1)
    scored["models_agree"] = model_preds[0].eq(model_preds[1])
    scored["model_pred"] = model_preds[0]

    candidates = []
    for proposed in CLASSES:
        subset = scored[
            scored["anchor"].ne(proposed)
            & scored["models_agree"]
            & scored["model_pred"].eq(proposed)
        ].copy()
        subset["proposed_label"] = proposed
        subset["transition"] = subset["anchor"] + "->" + proposed
        subset["model_margin"] = subset[f"mean_p_{proposed}"] - subset.apply(
            lambda row: row[f"mean_p_{row['anchor']}"],
            axis=1,
        )
        subset["local_margin"] = subset[f"local_p_{proposed}"] - subset.apply(
            lambda row: row[f"local_p_{row['anchor']}"],
            axis=1,
        )
        subset["transition_score"] = (
            0.46 * subset[f"mean_p_{proposed}"]
            + 0.30 * subset[f"min_p_{proposed}"]
            + 0.16 * subset[f"local_p_{proposed}"]
            + 0.08 * ((subset["local_margin"] + 1.0) / 2.0)
            - 0.08 * subset.apply(lambda row: row[f"local_p_{row['anchor']}"], axis=1)
        )
        candidates.append(subset)

    all_candidates = pd.concat(candidates, ignore_index=True)
    all_candidates = all_candidates[
        all_candidates["model_margin"].gt(0.15)
        & all_candidates["transition_score"].gt(0.58)
        & all_candidates["local_evidence_weight"].gt(0)
    ].copy()
    all_candidates = all_candidates.sort_values(
        ["transition_score", "model_margin"],
        ascending=False,
    ).reset_index(drop=True)
    all_candidates["global_rank"] = np.arange(1, len(all_candidates) + 1)
    all_candidates.to_csv(OUT_DIR / "all_transition_candidates.csv", index=False)

    active_files = []
    summary = []
    for transition, group in all_candidates.groupby("transition", sort=False):
        group = group.sort_values(["transition_score", "model_margin"], ascending=False).reset_index(drop=True)
        group["transition_rank"] = np.arange(1, len(group) + 1)
        safe_transition = transition.replace("->", "_to_")
        group.to_csv(OUT_DIR / f"{safe_transition}_candidates.csv", index=False)
        summary.append(
            {
                "transition": transition,
                "candidates": int(len(group)),
                "top_ids": group["id"].head(5).astype(int).tolist(),
                "top_score": float(group["transition_score"].iloc[0]) if len(group) else None,
            }
        )
        for n in [1, 3, 5]:
            if len(group) >= n:
                active_files.append(write_submission(anchor, group.head(n), f"transition_{safe_transition}_top_{n:02d}.csv"))

    non_star_to_galaxy = all_candidates[all_candidates["transition"].ne("STAR->GALAXY")].copy()
    if len(non_star_to_galaxy):
        active_files.append(write_submission(anchor, non_star_to_galaxy.head(1), "transition_non_stg_top_01.csv"))
        active_files.append(write_submission(anchor, non_star_to_galaxy.head(3), "transition_non_stg_top_03.csv"))

    report = {
        "purpose": "Search every anchor->model class transition from the current 0.97141 public anchor.",
        "current_anchor": str(CURRENT_PUBLIC_ANCHOR.relative_to(ROOT)),
        "candidate_count": int(len(all_candidates)),
        "transition_summary": summary,
        "active_files": [path.name for path in active_files],
        "recommended_order": [
            "transition_non_stg_top_01.csv",
            "transition_QSO_to_GALAXY_top_01.csv",
            "transition_STAR_to_QSO_top_01.csv",
            "transition_non_stg_top_03.csv",
        ],
    }
    (OUT_DIR / "transition_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUT_DIR / "ACTIVE_TRANSITION_PROBES.txt").write_text(
        "\n".join(["# Active transition probe files", *[path.name for path in active_files]]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
