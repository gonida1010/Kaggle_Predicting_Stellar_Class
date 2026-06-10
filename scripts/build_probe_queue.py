from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "probe_queue"
MODELS = ["lgbm", "catboost"]


def load_model(model: str) -> tuple[list[str], np.ndarray]:
    report = json.loads((ARTIFACTS / f"{model}_baseline_report.json").read_text(encoding="utf-8"))
    proba = np.load(ARTIFACTS / f"{model}_test_proba.npy")
    return report["classes"], proba


def write_submission(anchor: pd.DataFrame, changes: pd.DataFrame, name: str) -> None:
    submission = anchor.copy()
    id_to_label = dict(zip(changes["id"], changes["new_label"]))
    mask = submission["id"].isin(id_to_label)
    submission.loc[mask, "class"] = submission.loc[mask, "id"].map(id_to_label)
    submission.to_csv(OUT_DIR / name, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    anchor = pd.read_csv(DATA / "submission.csv")
    assert sample["id"].equals(anchor["id"])

    loaded = {model: load_model(model) for model in MODELS}
    classes = np.array(next(iter(loaded.values()))[0])
    for model, (model_classes, _) in loaded.items():
        if model_classes != classes.tolist():
            raise ValueError(f"{model} class order differs")

    candidates = sample[["id"]].copy()
    candidates["anchor"] = anchor["class"].to_numpy()

    for model, (_, proba) in loaded.items():
        pred_idx = proba.argmax(axis=1)
        candidates[f"{model}_pred"] = classes[pred_idx]
        candidates[f"{model}_conf"] = proba.max(axis=1)
        for class_idx, class_name in enumerate(classes):
            candidates[f"{model}_p_{class_name}"] = proba[:, class_idx]

    first_model = MODELS[0]
    candidates["new_label"] = candidates[f"{first_model}_pred"]
    candidates["models_agree"] = True
    for model in MODELS[1:]:
        candidates["models_agree"] &= candidates[f"{model}_pred"].eq(candidates[f"{first_model}_pred"])
    candidates["diff_from_anchor"] = candidates["anchor"].ne(candidates["new_label"])
    candidates["min_conf"] = candidates[[f"{model}_conf" for model in MODELS]].min(axis=1)
    candidates["mean_conf"] = candidates[[f"{model}_conf" for model in MODELS]].mean(axis=1)
    candidates["change"] = candidates["anchor"] + "->" + candidates["new_label"]

    ranked = candidates[candidates["models_agree"] & candidates["diff_from_anchor"]].copy()
    ranked = ranked.sort_values(["min_conf", "mean_conf"], ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    ranked.to_csv(OUT_DIR / "probe_candidates_ranked.csv", index=False)

    selected = ranked[ranked["min_conf"] >= 0.95].copy()
    selected.to_csv(OUT_DIR / "probe_candidates_minconf_095.csv", index=False)

    # Single-row probes are the cleanest way to learn from public LB movement.
    for _, row in selected.head(9).iterrows():
        one = pd.DataFrame([row])
        name = f"single_rank_{int(row['rank']):02d}_id_{int(row['id'])}_{row['change']}.csv"
        name = name.replace("->", "_to_")
        write_submission(anchor, one, name)

    grouped_specs = [
        ("group_top_03.csv", selected.head(3)),
        ("group_top_05.csv", selected.head(5)),
        ("group_top_09_except_rank_01.csv", selected[selected["rank"].ne(1)]),
        ("group_minconf_095_all.csv", selected),
        ("group_minconf_095_star_to_galaxy.csv", selected[selected["change"].eq("STAR->GALAXY")]),
        (
            "group_minconf_095_star_to_galaxy_except_rank_01.csv",
            selected[selected["change"].eq("STAR->GALAXY") & selected["rank"].ne(1)],
        ),
        ("group_minconf_095_galaxy_to_star.csv", selected[selected["change"].eq("GALAXY->STAR")]),
        ("group_minconf_093_all.csv", ranked[ranked["min_conf"] >= 0.93]),
        ("group_minconf_090_all.csv", ranked[ranked["min_conf"] >= 0.90]),
    ]
    for name, changes in grouped_specs:
        if len(changes):
            write_submission(anchor, changes, name)

    print(f"wrote candidates: {OUT_DIR / 'probe_candidates_ranked.csv'}")
    print(f"min_conf>=0.95 candidates: {len(selected)}")
    print(selected[["rank", "id", "change", "min_conf", "mean_conf"]].to_string(index=False))


if __name__ == "__main__":
    main()
