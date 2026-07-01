from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle_share_20260701" / "27th-place-robust-oof-selection.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    cells = [
        markdown(
            """# 27th Place: Robust OOF Selection and 0.5% Class-wise Blending

Final private leaderboard: **0.97029**, rank **27 / 2,824**.

This notebook reproduces the final selection stage of my solution. It does not pretend
to retrain every level-1 model from scratch. It starts from precomputed, strictly
out-of-fold train probabilities and matching test probabilities, applies the same
class-wise micro-blend used for the final submission, verifies it over 50 meta-folds,
and writes the submission.

The main lesson was simple: the highest public leaderboard file was not my best private
file. A public-oriented reference scored 0.97254 public and 0.97005 private, while the
OOF-selected file finished at 0.97029 private."""
        ),
        markdown(
            """## Method

- Metric: balanced accuracy.
- Starting candidate: OOF 0.9706563117.
- Auxiliary source: one-vs-rest CatBoost with non-leaky RealMLP-style features.
- Final change: mix only the auxiliary GALAXY probability at weight 0.005.
- Candidate search: weights 0.005 through 0.055.
- Stability check: 10 seeds x 5 stratified meta-folds.
- Final OOF: 0.9706589608.
- Prediction changes versus the starting candidate: 3 train rows and 1 test row.

The source model itself had lower standalone OOF. It was useful because its GALAXY
probability contained complementary boundary information. This is class-wise
micro-blending, not a large unconditional average."""
        ),
        code(
            """from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

CLASSES = np.array(["GALAXY", "QSO", "STAR"])
TARGET_MAP = {label: idx for idx, label in enumerate(CLASSES)}
ALPHA = 0.005

def find_file(filename):
    roots = [Path("/kaggle/input"), Path("kaggle_share_20260701/input_dataset"), Path(".")]
    matches = []
    for root in roots:
        if root.exists():
            matches.extend(root.rglob(filename))
    unique = sorted({path.resolve() for path in matches})
    if not unique:
        raise FileNotFoundError(f"Could not find {filename}")
    return unique[0]

def competition_dir():
    candidates = [Path("/kaggle/input/playground-series-s6e6"), Path("data")]
    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "sample_submission.csv").exists():
            return candidate
    raise FileNotFoundError("Competition train.csv and sample_submission.csv were not found")

COMP_DIR = competition_dir()
OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("kaggle_share_20260701/local_run")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(COMP_DIR / "train.csv")
sample = pd.read_csv(COMP_DIR / "sample_submission.csv")
y = train["class"].map(TARGET_MAP).to_numpy()

print("competition directory:", COMP_DIR)
print("train:", train.shape, "sample:", sample.shape)
print("output directory:", OUTPUT_DIR)"""
        ),
        code(
            """def normalize_probabilities(values):
    values = np.asarray(values, dtype=np.float64)
    row_sum = values.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("Probability rows must have a positive sum")
    return values / row_sum

def load_probabilities(filename, rows):
    values = normalize_probabilities(np.load(find_file(filename)))
    expected = (rows, len(CLASSES))
    if values.shape != expected:
        raise ValueError(f"{filename}: expected {expected}, got {values.shape}")
    return values

reference_oof = load_probabilities("reference_56_oof.npy", len(train))
start_oof = load_probabilities("start_181_oof.npy", len(train))
start_test = load_probabilities("start_181_test.npy", len(sample))
ovr_oof = load_probabilities("ovr_catboost_oof.npy", len(train))
ovr_test = load_probabilities("ovr_catboost_test.npy", len(sample))

print("loaded all OOF/test probability arrays")"""
        ),
        code(
            """def class_column_blend(base, source, class_index, alpha):
    blended = np.asarray(base, dtype=np.float64).copy()
    blended[:, class_index] = (
        (1.0 - alpha) * blended[:, class_index]
        + alpha * np.asarray(source, dtype=np.float64)[:, class_index]
    )
    return normalize_probabilities(blended)

final_oof = class_column_blend(start_oof, ovr_oof, class_index=0, alpha=ALPHA)
final_test = class_column_blend(start_test, ovr_test, class_index=0, alpha=ALPHA)

reference_pred = reference_oof.argmax(axis=1)
start_pred = start_oof.argmax(axis=1)
final_pred = final_oof.argmax(axis=1)
start_test_pred = start_test.argmax(axis=1)
final_test_pred = final_test.argmax(axis=1)

scores = pd.DataFrame(
    {
        "candidate": ["reference_56", "start_181", "final_193"],
        "oof_balanced_accuracy": [
            balanced_accuracy_score(y, reference_pred),
            balanced_accuracy_score(y, start_pred),
            balanced_accuracy_score(y, final_pred),
        ],
    }
)
display(scores)

assert abs(scores.iloc[-1, 1] - 0.9706589608099554) < 1e-12
print("changed train rows vs start:", int((start_pred != final_pred).sum()))
print("changed test rows vs start:", int((start_test_pred != final_test_pred).sum()))"""
        ),
        markdown(
            """## Why 0.005?

Larger weights imported more of the auxiliary model's standalone errors. The smallest
weight produced the best OOF score and changed only one test prediction."""
        ),
        code(
            """rows = []
for alpha in np.linspace(0.005, 0.055, 11):
    candidate_oof = class_column_blend(start_oof, ovr_oof, 0, float(alpha))
    candidate_test = class_column_blend(start_test, ovr_test, 0, float(alpha))
    rows.append(
        {
            "alpha": alpha,
            "oof_balanced_accuracy": balanced_accuracy_score(y, candidate_oof.argmax(axis=1)),
            "changed_train_rows": int((start_pred != candidate_oof.argmax(axis=1)).sum()),
            "changed_test_rows": int((start_test_pred != candidate_test.argmax(axis=1)).sum()),
        }
    )

alpha_scan = pd.DataFrame(rows)
display(alpha_scan)

fig, axis = plt.subplots(figsize=(10, 5))
axis.plot(alpha_scan["alpha"], alpha_scan["oof_balanced_accuracy"], marker="o", linewidth=2)
axis.axvline(ALPHA, color="#c62828", linestyle="--", label="selected alpha = 0.005")
axis.set_title("GALAXY class-wise blend weight search")
axis.set_xlabel("Auxiliary GALAXY probability weight")
axis.set_ylabel("OOF balanced accuracy")
axis.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
axis.grid(alpha=0.25)
axis.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "alpha_scan.png", dpi=180)
plt.show()"""
        ),
        markdown(
            """## Repeated meta-fold stability

These are not new model folds. They are repeated slices of the fixed OOF predictions,
used as a guard against selecting a gain concentrated in one part of the training set.
The final candidate improved over the earlier reference in all 50 slices."""
        ),
        code(
            """meta_rows = []
for seed in range(20260623, 20260633):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, valid_index) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        reference_score = balanced_accuracy_score(y[valid_index], reference_pred[valid_index])
        final_score = balanced_accuracy_score(y[valid_index], final_pred[valid_index])
        meta_rows.append(
            {
                "seed": seed,
                "fold": fold,
                "reference_score": reference_score,
                "final_score": final_score,
                "delta": final_score - reference_score,
            }
        )

meta = pd.DataFrame(meta_rows)
display(meta.describe())
print("positive fold rate:", float((meta["delta"] > 0).mean()))
print("minimum fold delta:", float(meta["delta"].min()))

fig, axis = plt.subplots(figsize=(11, 5))
axis.bar(np.arange(len(meta)), meta["delta"], color="#2e7d32")
axis.axhline(0, color="black", linewidth=1)
axis.set_title("Final candidate delta over reference across 50 meta-folds")
axis.set_xlabel("Meta-fold")
axis.set_ylabel("Balanced accuracy delta")
axis.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "meta_fold_delta.png", dpi=180)
plt.show()"""
        ),
        code(
            """report = pd.DataFrame(
    classification_report(
        y,
        final_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
).T
display(report)

matrix = confusion_matrix(y, final_pred)
fig, axis = plt.subplots(figsize=(7, 6))
image = axis.imshow(matrix, cmap="Blues")
for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
        axis.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center")
axis.set_xticks(range(3), CLASSES)
axis.set_yticks(range(3), CLASSES)
axis.set_xlabel("Predicted")
axis.set_ylabel("True")
axis.set_title("Final OOF confusion matrix")
fig.colorbar(image, ax=axis, fraction=0.046)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "oof_confusion_matrix.png", dpi=180)
plt.show()"""
        ),
        code(
            """submission = sample.copy()
submission["class"] = CLASSES[final_test_pred]
submission_path = OUTPUT_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)

assert submission["id"].equals(sample["id"])
assert submission["class"].notna().all()
assert set(submission["class"]) <= set(CLASSES)

digest = hashlib.sha256(submission_path.read_bytes()).hexdigest()
print("wrote:", submission_path)
print("rows:", len(submission))
print("class counts:")
print(submission["class"].value_counts())
print("sha256:", digest)
submission.head()"""
        ),
        markdown(
            """## Result and caveats

- Final OOF balanced accuracy: 0.9706589608.
- Final private leaderboard score: 0.97029.
- Final rank: 27 / 2,824.

The final-stage OOF/test arrays inherit earlier model-bank work. Public OOF sources were
used only where matching OOF and test probabilities were available; public leaderboard
scores were not used by this final selector. My local additions included non-leaky
feature variants, one-vs-rest CatBoost, class-wise probability blending, subset guards,
and repeated meta-fold audits.

Important public references included Chris Deotte's GPU Logistic Regression Stacker and
RealMLP notebooks, plus the one-vs-rest work shared by kirill0212. Please review and
credit the original notebooks when reusing those model sources.

Code in this notebook is released under the MIT License. The probability dataset is
published for reproducibility of this final stage, not as hidden ground truth."""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "none",
                "dataSources": [],
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
