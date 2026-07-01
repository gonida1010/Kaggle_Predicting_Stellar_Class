from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARE_DIR = ROOT / "kaggle_share_20260701"
DATASET_DIR = SHARE_DIR / "input_dataset"

FILES = {
    "reference_56_oof.npy": (
        ROOT
        / "artifacts"
        / "te_disagreement_patch_classwise37"
        / "56_high_gi_low_rz_base_galaxy_to_star_c0_55_m0_15_b0_60_oof.npy"
    ),
    "start_181_oof.npy": (
        ROOT
        / "artifacts"
        / "robust_private_cv_after171_20260624"
        / "181_PRIVATE_CV_after171_classblend_our-catboost-realmlp-features_QSO_a0p0050_oof0970656_oof.npy"
    ),
    "start_181_test.npy": (
        ROOT
        / "artifacts"
        / "robust_private_cv_after171_20260624"
        / "181_PRIVATE_CV_after171_classblend_our-catboost-realmlp-features_QSO_a0p0050_oof0970656_test.npy"
    ),
    "ovr_catboost_oof.npy": (
        ROOT / "artifacts" / "ovr_catboost_realmlp_features" / "ovr_catboost_oof_proba.npy"
    ),
    "ovr_catboost_test.npy": (
        ROOT / "artifacts" / "ovr_catboost_realmlp_features" / "ovr_catboost_test_proba.npy"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "title": "Stellar S6E6 robust OOF final-stage inputs",
        "created_at": "2026-07-01",
        "class_order": ["GALAXY", "QSO", "STAR"],
        "description": (
            "Precomputed OOF/test probabilities used to reproduce the final robust "
            "selection stage of the 27th-place solution. These are not labels."
        ),
        "files": {},
    }

    for output_name, source in FILES.items():
        if not source.exists():
            raise FileNotFoundError(source)
        destination = DATASET_DIR / output_name
        shutil.copy2(source, destination)
        manifest["files"][output_name] = {
            "source": str(source.relative_to(ROOT)),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
        print(f"copied {output_name}: {destination.stat().st_size:,} bytes", flush=True)

    metadata = {
        "title": "Stellar S6E6 Robust OOF Final Stage",
        "id": "YOUR_KAGGLE_USERNAME/stellar-s6e6-robust-oof-final-stage",
        "licenses": [{"name": "CC-BY-4.0"}],
    }
    (DATASET_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (DATASET_DIR / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {DATASET_DIR / 'manifest.json'}", flush=True)
    print("Edit dataset-metadata.json and replace YOUR_KAGGLE_USERNAME before upload.", flush=True)


if __name__ == "__main__":
    main()
