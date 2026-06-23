from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from build_available_prediction_stacker import (
    ARTIFACTS,
    CLASSES,
    DATA,
    ROOT,
    TARGET_MAP,
    archive4_pairs,
    local_file_pairs,
    normalize_probs,
    own_model_pairs,
)


DEFAULT_OUT = ARTIFACTS / "research_material_stack"
TRAIN_ROWS = 577347
TEST_ROWS = 247435


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stack every valid OOF/test probability material we have produced or collected. "
            "This is the missing step between algorithm-level CSV/proba outputs and final private-CV candidates."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--base-oof",
        type=Path,
        default=ARTIFACTS
        / "te_disagreement_patch_classwise37"
        / "56_high_gi_low_rz_base_galaxy_to_star_c0_55_m0_15_b0_60_oof.npy",
    )
    parser.add_argument(
        "--base-test",
        type=Path,
        default=ARTIFACTS
        / "te_disagreement_patch_classwise37"
        / "56_high_gi_low_rz_base_galaxy_to_star_c0_55_m0_15_b0_60_test.npy",
    )
    parser.add_argument("--base-name", default="56_te_disagreement_oof0970573")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--max-added-weight", type=float, default=0.20)
    parser.add_argument("--weight-steps", type=int, default=40)
    parser.add_argument("--bias-low", type=float, default=0.94)
    parser.add_argument("--bias-high", type=float, default=1.06)
    parser.add_argument("--bias-steps", type=int, default=13)
    parser.add_argument("--min-gain", type=float, default=1e-8)
    parser.add_argument("--max-candidates", type=int, default=35)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--include-external-bank", action="store_true", default=True)
    parser.add_argument("--no-external-bank", dest="include_external_bank", action="store_false")
    parser.add_argument("--include-artifact-bank", action="store_true", default=True)
    parser.add_argument("--no-artifact-bank", dest="include_artifact_bank", action="store_false")
    parser.add_argument("--exclude-smoke", action="store_true", default=True)
    parser.add_argument("--top-source-report", type=int, default=80)
    parser.add_argument("--output-prefix", default="68_PRIVATE_CV_research_material_stack")
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def as_abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_proba(path: Path, rows: int) -> np.ndarray | None:
    try:
        arr = np.load(path)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    if arr.shape != (rows, len(CLASSES)):
        return None
    return normalize_probs(arr.astype(np.float32))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for class_idx in range(len(CLASSES)):
        mask = y_true == class_idx
        if mask.any():
            recalls.append(float((y_pred[mask] == class_idx).mean()))
    return float(np.mean(recalls))


def class_recalls(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {}
    for class_idx, label in enumerate(CLASSES):
        mask = y_true == class_idx
        out[label] = float((y_pred[mask] == class_idx).mean()) if mask.any() else float("nan")
    return out


def apply_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    adjusted = normalize_probs(proba * bias.reshape(1, -1))
    return adjusted


def score_proba(y: np.ndarray, proba: np.ndarray, bias: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    adjusted = apply_bias(proba, bias) if bias is not None else normalize_probs(proba)
    pred = adjusted.argmax(axis=1)
    return balanced_accuracy(y, pred), pred


def optimize_bias(
    y: np.ndarray,
    proba: np.ndarray,
    start_bias: np.ndarray,
    low: float,
    high: float,
    steps: int,
    passes: int = 2,
) -> tuple[np.ndarray, float, np.ndarray, list[dict]]:
    bias = start_bias.astype(np.float64).copy()
    best_score, best_pred = score_proba(y, proba, bias)
    rows = []
    for pass_idx in range(passes):
        improved = False
        for class_idx, label in enumerate(CLASSES):
            for multiplier in np.linspace(low, high, steps):
                trial = bias.copy()
                trial[class_idx] *= float(multiplier)
                trial = trial / trial.mean()
                score, pred = score_proba(y, proba, trial)
                rows.append(
                    {
                        "pass": pass_idx + 1,
                        "class": label,
                        "multiplier": float(multiplier),
                        "score": float(score),
                        "bias": dict(zip(CLASSES, trial.tolist())),
                    }
                )
                if score > best_score:
                    bias = trial
                    best_score = float(score)
                    best_pred = pred
                    improved = True
        if not improved:
            break
    return bias, best_score, best_pred, rows


def blend(base: np.ndarray, candidate: np.ndarray, weight: float) -> np.ndarray:
    return normalize_probs((1.0 - weight) * base + weight * candidate)


def pair_test_path(oof_path: Path) -> Path | None:
    name = oof_path.name
    replacements = [
        ("_oof_proba.npy", "_test_proba.npy"),
        ("_oof.npy", "_test.npy"),
        ("_oof_raw.npy", "_test_raw.npy"),
        ("oof_proba.npy", "test_proba.npy"),
        ("oof.npy", "test.npy"),
    ]
    for old, new in replacements:
        if name.endswith(old):
            return oof_path.with_name(name[: -len(old)] + new)
    return None


def should_skip(path: Path, exclude_smoke: bool) -> bool:
    text = str(path)
    if "partial" in text:
        return True
    if exclude_smoke and ("smoke" in text or "screen" in text):
        return True
    if "boundary_pair_experiments" in text:
        return True
    return False


def discover_artifact_pairs(exclude_smoke: bool) -> list[dict]:
    records = []
    for oof_path in sorted(ARTIFACTS.rglob("*oof*.npy")):
        if should_skip(oof_path, exclude_smoke):
            continue
        test_path = pair_test_path(oof_path)
        if test_path is None or not test_path.exists() or should_skip(test_path, exclude_smoke):
            continue
        oof = load_proba(oof_path, TRAIN_ROWS)
        test = load_proba(test_path, TEST_ROWS)
        if oof is None or test is None:
            continue
        rel = oof_path.relative_to(ROOT)
        name = str(rel).replace("artifacts/", "").replace("/", "::").replace("_oof_proba.npy", "")
        name = name.replace("_oof.npy", "").replace(".npy", "")
        records.append({"name": name, "oof": oof, "test": test, "source": str(rel)})
    return records


def load_all_records(args: argparse.Namespace) -> list[dict]:
    records = []
    if args.include_external_bank:
        train = pd.read_csv(DATA / "train.csv", usecols=["class"])
        sample = pd.read_csv(DATA / "sample_submission.csv")
        records.extend(archive4_pairs(len(train), len(sample)))
        records.extend(local_file_pairs(len(train), len(sample)))
        records.extend(own_model_pairs(len(train), len(sample)))
    if args.include_artifact_bank:
        records.extend(discover_artifact_pairs(args.exclude_smoke))

    seen = {}
    unique = []
    for record in records:
        key = (record["oof"].shape, record["test"].shape, record["name"])
        if key in seen:
            continue
        seen[key] = True
        unique.append(record)
    return unique


def write_submission(path: Path, sample: pd.DataFrame, proba: np.ndarray) -> None:
    submission = sample.copy()
    submission["class"] = np.array(CLASSES)[normalize_probs(proba).argmax(axis=1)]
    submission.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = as_abs(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)

    train = pd.read_csv(DATA / "train.csv", usecols=["class"])
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["class"].map(TARGET_MAP).to_numpy()

    base_oof = load_proba(as_abs(args.base_oof), len(train))
    base_test = load_proba(as_abs(args.base_test), len(sample))
    if base_oof is None or base_test is None:
        raise FileNotFoundError("Base OOF/test probability pair is missing or has an invalid shape.")

    records = load_all_records(args)
    base_score, base_pred = score_proba(y, base_oof)
    unit_bias = np.ones(len(CLASSES), dtype=np.float64)
    best_bias, best_score, best_pred, bias_rows = optimize_bias(
        y,
        base_oof,
        unit_bias,
        args.bias_low,
        args.bias_high,
        args.bias_steps,
    )
    best_oof = base_oof.copy()
    best_test = base_test.copy()

    progress(f"base={args.base_name} raw OOF BAC={base_score:.9f}")
    progress(f"base bias OOF BAC={best_score:.9f}; delta={best_score - base_score:+.9f}")
    progress(f"loaded candidate sources={len(records)}")

    source_rows = []
    for record in records:
        single_score, single_pred = score_proba(y, record["oof"])
        agree = float((single_pred == best_pred).mean())
        source_rows.append(
            {
                "name": record["name"],
                "source": record.get("source", ""),
                "single_oof": float(single_score),
                "agreement_vs_current": agree,
                **{f"recall_{k}": v for k, v in class_recalls(y, single_pred).items()},
            }
        )
    source_rows_sorted = sorted(
        source_rows,
        key=lambda row: (row["single_oof"], -row["agreement_vs_current"]),
        reverse=True,
    )
    selected_names = {row["name"] for row in source_rows_sorted[: max(1, int(args.max_candidates))]}
    records = [record for record in records if record["name"] in selected_names]
    progress(f"selected candidate sources={len(records)} from source_count={len(source_rows)}")

    stage_rows = [
        {
            "stage": "base_raw",
            "model": args.base_name,
            "weight": 0.0,
            "score": float(base_score),
            "delta_vs_raw_base": 0.0,
            "bias": dict(zip(CLASSES, unit_bias.tolist())),
        },
        {
            "stage": "base_bias",
            "model": args.base_name,
            "weight": 0.0,
            "score": float(best_score),
            "delta_vs_raw_base": float(best_score - base_score),
            "bias": dict(zip(CLASSES, best_bias.tolist())),
        },
    ]
    search_rows = []
    candidate_names_used: set[str] = set()
    weights = np.linspace(0.0, args.max_added_weight, args.weight_steps + 1)[1:]

    for round_idx in range(args.rounds):
        progress(f"research material blend round {round_idx + 1}/{args.rounds}")
        round_best = None
        for source_idx, record in enumerate(records, start=1):
            if record["name"] in candidate_names_used:
                continue
            if args.progress_every > 0 and (source_idx == 1 or source_idx % args.progress_every == 0):
                progress(f"round {round_idx + 1}: scoring source {source_idx}/{len(records)} {record['name']}")
            for weight in weights:
                blended_oof = blend(best_oof, record["oof"], float(weight))
                trial_bias = best_bias
                score, pred = score_proba(y, blended_oof, best_bias)
                row = {
                    "round": round_idx + 1,
                    "candidate": record["name"],
                    "weight": float(weight),
                    "score": float(score),
                    "delta_vs_current": float(score - best_score),
                    "bias": dict(zip(CLASSES, trial_bias.tolist())),
                }
                search_rows.append(row)
                if score > best_score + args.min_gain and (round_best is None or score > round_best["score"]):
                    round_best = {
                        **row,
                        "record": record,
                        "oof": blended_oof,
                        "test": blend(best_test, record["test"], float(weight)),
                        "bias_array": trial_bias,
                        "pred": pred,
                    }
        if round_best is None:
            progress("no OOF-improving source found; stopping")
            break
        tuned_bias, tuned_score, tuned_pred, _ = optimize_bias(
            y,
            round_best["oof"],
            round_best["bias_array"],
            args.bias_low,
            args.bias_high,
            args.bias_steps,
            passes=2,
        )
        if tuned_score >= round_best["score"]:
            round_best["score"] = float(tuned_score)
            round_best["bias_array"] = tuned_bias
            round_best["pred"] = tuned_pred
        best_score = float(round_best["score"])
        best_oof = round_best["oof"]
        best_test = round_best["test"]
        best_bias = round_best["bias_array"]
        best_pred = round_best["pred"]
        candidate_names_used.add(round_best["candidate"])
        stage_rows.append(
            {
                "stage": f"blend_round_{round_idx + 1}",
                "model": round_best["candidate"],
                "weight": round_best["weight"],
                "score": best_score,
                "delta_vs_raw_base": float(best_score - base_score),
                "bias": dict(zip(CLASSES, best_bias.tolist())),
            }
        )
        progress(
            f"accepted {round_best['candidate']} weight={round_best['weight']:.4f}; "
            f"OOF BAC={best_score:.9f}"
        )

    final_oof = apply_bias(best_oof, best_bias).astype(np.float32)
    final_test = apply_bias(best_test, best_bias).astype(np.float32)
    score_tag = f"{best_score:.6f}".replace(".", "")
    artifact_submission = output_dir / "research_material_stack_submission.csv"
    output_submission = ROOT / "outputs" / f"{args.output_prefix}_oof{score_tag}.csv"

    write_submission(artifact_submission, sample, final_test)
    write_submission(output_submission, sample, final_test)
    np.save(output_dir / "research_material_stack_oof.npy", final_oof)
    np.save(output_dir / "research_material_stack_test.npy", final_test)
    pd.DataFrame(source_rows).sort_values("single_oof", ascending=False).head(args.top_source_report).to_csv(
        output_dir / "source_summary_top.csv",
        index=False,
    )
    pd.DataFrame(search_rows).sort_values("score", ascending=False).to_csv(output_dir / "blend_search.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(output_dir / "accepted_stages.csv", index=False)
    pd.DataFrame(bias_rows).sort_values("score", ascending=False).to_csv(output_dir / "base_bias_search.csv", index=False)

    report = {
        "purpose": "Stack available algorithm OOF/test sources plus generated research-material OOF/test candidates.",
        "base_name": args.base_name,
        "raw_base_oof_balanced_accuracy": float(base_score),
        "best_oof_balanced_accuracy": float(best_score),
        "delta_vs_raw_base": float(best_score - base_score),
        "best_bias": dict(zip(CLASSES, best_bias.tolist())),
        "accepted_stages": stage_rows,
        "class_recalls": class_recalls(y, best_pred),
        "source_count": len(records),
        "artifact_submission": str(artifact_submission.relative_to(ROOT)),
        "output_submission": str(output_submission.relative_to(ROOT)),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
