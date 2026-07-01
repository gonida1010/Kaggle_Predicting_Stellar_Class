from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
ARTIFACTS = ROOT / "artifacts"
CLASSES = ["GALAXY", "QSO", "STAR"]
TARGET_MAP = {label: idx for idx, label in enumerate(CLASSES)}

SOURCE_CSV = OUTPUTS / "193_PRIVATE_CV_oof970659.csv"
SOURCE_DIR = ARTIFACTS / "robust_private_cv_after181_20260624"
SOURCE_STEM = (
    "193_PRIVATE_CV_after181_classblend_"
    "our-ovr-catboost-realmlp-features_GALAXY_a0p0050_oof0970659"
)
SOURCE_OOF = SOURCE_DIR / f"{SOURCE_STEM}_oof.npy"
SOURCE_TEST = SOURCE_DIR / f"{SOURCE_STEM}_test.npy"

FINAL_NAME = "FINAL_PRIVATE_CV_OOF970659.csv"
FINAL_CSV = OUTPUTS / FINAL_NAME
REPORT_DIR = ARTIFACTS / "final_private_submission_20260630"
REPORT_CSV = REPORT_DIR / FINAL_NAME
REPORT_JSON = REPORT_DIR / "final_private_submission_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_submission(path: Path, sample: pd.DataFrame) -> dict:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "class"]:
        raise ValueError(f"{path} columns must be ['id', 'class']")
    if len(frame) != len(sample):
        raise ValueError(f"{path} rows={len(frame)}, expected={len(sample)}")
    if not frame["id"].equals(sample["id"]):
        raise ValueError(f"{path} id values or order differ from sample_submission.csv")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicated ids")
    if frame["class"].isna().any():
        raise ValueError(f"{path} contains missing predictions")
    invalid = sorted(set(frame["class"]) - set(CLASSES))
    if invalid:
        raise ValueError(f"{path} contains invalid labels: {invalid}")
    return {
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
        "ids_match_sample": True,
        "duplicated_ids": 0,
        "missing_predictions": 0,
        "class_counts": {
            label: int((frame["class"] == label).sum())
            for label in CLASSES
        },
        "class_share": {
            label: float((frame["class"] == label).mean())
            for label in CLASSES
        },
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for required in (SOURCE_CSV, SOURCE_OOF, SOURCE_TEST):
        if not required.exists():
            raise FileNotFoundError(required)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    train = pd.read_csv(DATA / "train.csv")
    source_validation = validate_submission(SOURCE_CSV, sample)

    shutil.copyfile(SOURCE_CSV, FINAL_CSV)
    shutil.copyfile(SOURCE_CSV, REPORT_CSV)
    final_validation = validate_submission(FINAL_CSV, sample)
    report_copy_validation = validate_submission(REPORT_CSV, sample)

    source_hash = sha256(SOURCE_CSV)
    final_hash = sha256(FINAL_CSV)
    report_copy_hash = sha256(REPORT_CSV)
    if len({source_hash, final_hash, report_copy_hash}) != 1:
        raise RuntimeError("Final CSV is not byte-identical to the selected source.")

    y = train["class"].map(TARGET_MAP).to_numpy()
    oof = np.load(SOURCE_OOF)
    test = np.load(SOURCE_TEST)
    if oof.shape != (len(train), len(CLASSES)):
        raise ValueError(f"OOF shape {oof.shape} is invalid")
    if test.shape != (len(sample), len(CLASSES)):
        raise ValueError(f"test probability shape {test.shape} is invalid")
    oof_pred = oof.argmax(axis=1)
    recalls = recall_score(y, oof_pred, labels=[0, 1, 2], average=None)
    oof_score = float(balanced_accuracy_score(y, oof_pred))

    final_frame = pd.read_csv(FINAL_CSV)
    probability_labels = np.asarray(CLASSES)[test.argmax(axis=1)]
    probability_submission_matches = bool(
        np.array_equal(final_frame["class"].to_numpy(), probability_labels)
    )
    if not probability_submission_matches:
        raise RuntimeError("Final CSV labels differ from the selected test probability argmax.")

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "track": "private_generalization",
        "selection": {
            "source_csv": str(SOURCE_CSV.relative_to(ROOT)),
            "source_oof": str(SOURCE_OOF.relative_to(ROOT)),
            "source_test": str(SOURCE_TEST.relative_to(ROOT)),
            "reason": (
                "Highest retained robust OOF/CV candidate after paired-fold, subset, "
                "nested-blend, SDSS augmentation, and stacker ablations."
            ),
            "public_leaderboard_was_selection_metric": False,
        },
        "oof_validation": {
            "balanced_accuracy": oof_score,
            "class_recalls": {
                label: float(value)
                for label, value in zip(CLASSES, recalls)
            },
            "confusion_matrix": confusion_matrix(y, oof_pred, labels=[0, 1, 2]).tolist(),
            "rows": int(len(y)),
        },
        "final_submission": {
            "path": str(FINAL_CSV.relative_to(ROOT)),
            "archived_path": str(REPORT_CSV.relative_to(ROOT)),
            "sha256": final_hash,
            "byte_identical_to_source": True,
            "labels_match_test_probability_argmax": probability_submission_matches,
            **final_validation,
        },
        "source_validation": source_validation,
        "archived_copy_validation": report_copy_validation,
    }
    REPORT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"FINAL PRIVATE SUBMISSION: {FINAL_CSV}")


if __name__ == "__main__":
    main()
