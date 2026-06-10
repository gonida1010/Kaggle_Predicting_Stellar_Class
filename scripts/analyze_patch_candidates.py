from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.stellar_features import make_xy  # noqa: E402


DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    anchor = pd.read_csv(DATA / "submission.csv")
    report = json.loads((ARTIFACTS / "lgbm_baseline_report.json").read_text(encoding="utf-8"))
    test_proba = np.load(ARTIFACTS / "lgbm_test_proba.npy")

    _, _, x_test, features = make_xy(train, test)
    classes = np.array(report["classes"])
    model_pred = classes[test_proba.argmax(axis=1)]
    model_conf = test_proba.max(axis=1)

    out = test[["id"]].copy()
    out["anchor"] = anchor["class"].to_numpy()
    out["model"] = model_pred
    out["model_conf"] = model_conf
    for idx, cls in enumerate(classes):
        out[f"p_{cls}"] = test_proba[:, idx]
    out["diff"] = out["anchor"].ne(out["model"])

    diff = out[out["diff"]].copy()
    print("features:", len(features))
    print("rows:", len(out))
    print("anchor/model disagreements:", len(diff))
    print()
    print("model prediction share")
    print(out["model"].value_counts(normalize=True).sort_index().round(6))
    print()
    print("anchor -> model disagreement table")
    print(pd.crosstab(diff["anchor"], diff["model"]))
    print()

    for threshold in [0.98, 0.99, 0.995, 0.999]:
        high = diff[diff["model_conf"] >= threshold]
        print(f"diff rows with model_conf >= {threshold}: {len(high)}")
        if len(high):
            print(pd.crosstab(high["anchor"], high["model"]))
            print(high.sort_values("model_conf", ascending=False).head(20).to_string(index=False))
            print()

    diff.sort_values("model_conf", ascending=False).to_csv(
        ARTIFACTS / "anchor_lgbm_disagreements.csv",
        index=False,
    )

    for threshold in [0.99, 0.995, 0.999]:
        patched = anchor.copy()
        mask = out["diff"] & (out["model_conf"] >= threshold)
        patched.loc[mask, "class"] = out.loc[mask, "model"].to_numpy()
        path = ARTIFACTS / f"anchor_lgbm_patch_conf_{str(threshold).replace('.', '')}.csv"
        patched.to_csv(path, index=False)
        print(f"wrote {path.name}: changed_rows={int(mask.sum())}")


if __name__ == "__main__":
    main()
