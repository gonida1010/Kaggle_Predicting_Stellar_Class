from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


def load_model_probs(model_name: str) -> tuple[list[str], np.ndarray] | None:
    report_path = ARTIFACTS / f"{model_name}_baseline_report.json"
    proba_path = ARTIFACTS / f"{model_name}_test_proba.npy"
    if not report_path.exists() or not proba_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report["classes"], np.load(proba_path)


def main() -> None:
    sample = pd.read_csv(DATA / "sample_submission.csv")
    anchor = pd.read_csv(DATA / "submission.csv")

    loaded = {
        name: value
        for name in ["lgbm", "catboost"]
        if (value := load_model_probs(name)) is not None
    }
    if not loaded:
        raise FileNotFoundError("No model probability artifacts found.")

    classes = list(next(iter(loaded.values()))[0])
    for name, (model_classes, _) in loaded.items():
        if model_classes != classes:
            raise ValueError(f"{name} class order differs: {model_classes} != {classes}")

    probas = np.stack([value[1] for value in loaded.values()], axis=0)
    mean_proba = probas.mean(axis=0)
    pred = np.array(classes)[mean_proba.argmax(axis=1)]
    conf = mean_proba.max(axis=1)

    model_preds = {}
    model_confs = {}
    for name, (_, proba) in loaded.items():
        model_preds[name] = np.array(classes)[proba.argmax(axis=1)]
        model_confs[name] = proba.max(axis=1)

    out = sample[["id"]].copy()
    out["anchor"] = anchor["class"].to_numpy()
    out["ensemble"] = pred
    out["ensemble_conf"] = conf
    for cls_idx, cls in enumerate(classes):
        out[f"p_{cls}"] = mean_proba[:, cls_idx]
    for name in loaded:
        out[f"{name}_pred"] = model_preds[name]
        out[f"{name}_conf"] = model_confs[name]

    out["diff"] = out["anchor"].ne(out["ensemble"])
    if len(loaded) > 1:
        first_name = next(iter(loaded))
        out["models_agree"] = True
        for name in loaded:
            out["models_agree"] &= out[f"{name}_pred"].eq(out[f"{first_name}_pred"])
    else:
        out["models_agree"] = True

    diff = out[out["diff"]].copy()
    print("loaded models:", ", ".join(loaded))
    print("anchor/ensemble disagreements:", len(diff))
    print()
    print(pd.crosstab(diff["anchor"], diff["ensemble"]))
    print()

    diff.sort_values("ensemble_conf", ascending=False).to_csv(
        ARTIFACTS / "anchor_model_ensemble_disagreements.csv",
        index=False,
    )

    for threshold in [0.98, 0.99, 0.995, 0.999]:
        mask = out["diff"] & out["models_agree"] & (out["ensemble_conf"] >= threshold)
        candidates = out[mask].sort_values("ensemble_conf", ascending=False)
        print(f"agreeing patch candidates >= {threshold}: {len(candidates)}")
        if len(candidates):
            print(pd.crosstab(candidates["anchor"], candidates["ensemble"]))
            print(candidates.head(30).to_string(index=False))
            print()

        patched = anchor.copy()
        patched.loc[mask, "class"] = out.loc[mask, "ensemble"].to_numpy()
        patched.to_csv(
            ARTIFACTS / f"anchor_model_ensemble_patch_conf_{str(threshold).replace('.', '')}.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
