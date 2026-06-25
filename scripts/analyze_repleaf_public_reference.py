from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"

ID = "id"
TARGET = "class"
LABELS = ["GALAXY", "QSO", "STAR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze newly downloaded public/reference submissions and build small "
            "PUBLIC-LB exploration candidates. These candidates are not CV/private "
            "candidates because the downloaded files contain hard labels only."
        )
    )
    parser.add_argument("--public-anchor", type=Path, default=Path("/Users/parkyeonggon/Downloads/submission (10).csv"))
    parser.add_argument("--repleaf-submission", type=Path, default=Path("/Users/parkyeonggon/Downloads/submission (11).csv"))
    parser.add_argument("--our-public", type=Path, default=OUTPUTS / "151_PUBLIC_HYBRID_sg_only_rank_top21_vs097227.csv")
    parser.add_argument("--public-base", type=Path, default=OUTPUTS / "21_PUBLIC_097227_ridge_consensus_direct.csv")
    parser.add_argument("--private-main", type=Path, default=OUTPUTS / "193_PRIVATE_CV_oof970659.csv")
    parser.add_argument("--private-alt", type=Path, nargs="*", default=[
        OUTPUTS / "90_PRIVATE_CV_subset_guard_68_plus_84_good_union_oof0970627.csv",
        OUTPUTS / "84_PRIVATE_CV_classwise_research_blend_oof0970621.csv",
    ])
    parser.add_argument(
        "--public-sg-rank",
        type=Path,
        default=ARTIFACTS / "public_archive9_explore_20260623" / "star_to_galaxy_public_private_rank_after_121.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "repleaf_public_reference_20260625")
    parser.add_argument("--output-rank-start", type=int, default=205)
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def load_submission(path: Path, sample_ids: np.ndarray) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=[ID, TARGET])
    if not np.array_equal(df[ID].to_numpy(), sample_ids):
        df = df.set_index(ID).reindex(sample_ids).reset_index()
    if df[TARGET].isna().any():
        raise ValueError(f"{path} has missing labels after id alignment")
    invalid = sorted(set(df[TARGET].unique()) - set(LABELS))
    if invalid:
        raise ValueError(f"{path} has invalid labels: {invalid}")
    return pd.Series(df[TARGET].to_numpy(dtype=object), index=sample_ids, name=path.name)


def transition_counts(before: pd.Series, after: pd.Series) -> dict[str, int]:
    changed = before.ne(after)
    transitions = (before[changed] + "->" + after[changed]).tolist()
    return dict(Counter(transitions))


def class_counts(labels: pd.Series) -> dict[str, int]:
    return {label: int(labels.eq(label).sum()) for label in LABELS}


def add_diagnostic_features(test: pd.DataFrame) -> pd.DataFrame:
    out = test[[ID, "spectral_type", "galaxy_population", "redshift", "u", "g", "r", "i", "z"]].copy()
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["g_i"] = out["g"] - out["i"]
    bands = out[["u", "g", "r", "i", "z"]]
    out["mag_range"] = bands.max(axis=1) - bands.min(axis=1)
    return out.set_index(ID)


def load_public_rank(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[ID, "public_sg_rank", "public_sg_robust_score", "public_sg_pos_rate", "public_sg_margin"])
    rank = pd.read_csv(path)
    rank = rank.reset_index(names="public_sg_rank")
    rank["public_sg_rank"] += 1
    keep = rank[[ID, "public_sg_rank", "robust_score", "pos_rate", "proba_margin"]].copy()
    return keep.rename(
        columns={
            "robust_score": "public_sg_robust_score",
            "pos_rate": "public_sg_pos_rate",
            "proba_margin": "public_sg_margin",
        }
    )


def write_submission(sample_ids: np.ndarray, labels: pd.Series, path: Path) -> None:
    pd.DataFrame({ID: sample_ids, TARGET: labels.reindex(sample_ids).to_numpy(dtype=object)}).to_csv(path, index=False)


def build_candidate(
    sample_ids: np.ndarray,
    anchor: pd.Series,
    rows: pd.DataFrame,
    output_path: Path,
) -> dict:
    labels = anchor.copy()
    labels.loc[rows[ID].to_numpy()] = rows["repleaf"].to_numpy(dtype=object)
    write_submission(sample_ids, labels, output_path)
    changed = labels.ne(anchor)
    return {
        "file": output_path.name,
        "path": str(output_path.relative_to(ROOT)),
        "changed_rows": int(changed.sum()),
        "transition_counts": transition_counts(anchor, labels),
        "class_counts": class_counts(labels),
    }


def summarize_segment(rows: pd.DataFrame, name: str) -> dict:
    if rows.empty:
        return {"segment": name, "rows": 0}
    return {
        "segment": name,
        "rows": int(len(rows)),
        "transition_counts": dict(Counter(rows["transition"])),
        "spectral_type_counts": rows["spectral_type"].value_counts().head(10).to_dict(),
        "galaxy_population_counts": rows["galaxy_population"].value_counts().head(10).to_dict(),
        "redshift_median": float(rows["redshift"].median()),
        "g_i_median": float(rows["g_i"].median()),
        "mag_range_median": float(rows["mag_range"].median()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    progress("Loading sample/test and submissions")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sample_ids = sample[ID].to_numpy()
    test_features = add_diagnostic_features(pd.read_csv(DATA / "test.csv"))

    submissions = {
        "public_anchor_097233": load_submission(args.public_anchor, sample_ids),
        "repleaf_single": load_submission(args.repleaf_submission, sample_ids),
        "our_public_151": load_submission(args.our_public, sample_ids),
        "public_base_097227": load_submission(args.public_base, sample_ids),
        "private_main_193": load_submission(args.private_main, sample_ids),
    }
    for idx, path in enumerate(args.private_alt, start=1):
        if path.exists():
            submissions[f"private_alt_{idx}_{path.stem[:24]}"] = load_submission(path, sample_ids)

    anchor = submissions["public_anchor_097233"]
    repleaf = submissions["repleaf_single"]
    private_main = submissions["private_main_193"]

    progress("Building disagreement table")
    rows = pd.DataFrame(
        {
            ID: sample_ids,
            "anchor": anchor.to_numpy(dtype=object),
            "repleaf": repleaf.to_numpy(dtype=object),
            "private_main": private_main.to_numpy(dtype=object),
            "our_public_151": submissions["our_public_151"].to_numpy(dtype=object),
            "public_base_097227": submissions["public_base_097227"].to_numpy(dtype=object),
        }
    )
    for name, labels in submissions.items():
        if name.startswith("private_alt"):
            rows[name] = labels.to_numpy(dtype=object)

    rows["anchor_ne_repleaf"] = rows["anchor"].ne(rows["repleaf"])
    rows["repleaf_eq_private_main"] = rows["repleaf"].eq(rows["private_main"])
    private_cols = [col for col in rows.columns if col.startswith("private_")]
    rows["private_agree_count"] = sum(rows["repleaf"].eq(rows[col]) for col in private_cols)
    rows["transition"] = rows["anchor"] + "->" + rows["repleaf"]
    rows = rows.join(test_features, on=ID)
    rows = rows.merge(load_public_rank(args.public_sg_rank), on=ID, how="left")
    rows["public_sg_robust_score"] = rows["public_sg_robust_score"].fillna(0.0)
    rows["public_sg_rank"] = rows["public_sg_rank"].fillna(999999).astype(int)
    rows["public_sg_pos_rate"] = rows["public_sg_pos_rate"].fillna(0.0)
    rows["public_sg_margin"] = rows["public_sg_margin"].fillna(0.0)

    pool = rows[rows["anchor_ne_repleaf"] & rows["repleaf_eq_private_main"]].copy()
    pool["is_sg"] = pool["transition"].eq("STAR->GALAXY")
    pool["is_gs"] = pool["transition"].eq("GALAXY->STAR")
    pool["rank_score"] = (
        pool["is_sg"].astype(float) * 10.0
        + pool["public_sg_robust_score"] * 100000.0
        + pool["private_agree_count"].astype(float) * 0.2
        + pool["public_sg_pos_rate"].astype(float) * 0.05
        + pool["public_sg_margin"].astype(float) * 0.05
    )
    pool = pool.sort_values(
        ["is_sg", "rank_score", "private_agree_count", "public_sg_robust_score", ID],
        ascending=[False, False, False, False, True],
    )

    sg_pool = pool[pool["is_sg"]].copy()
    gs_pool = pool[pool["is_gs"]].copy()
    all_pool = pool.copy()

    progress("Writing analysis tables")
    pair_rows = []
    names = list(submissions)
    for left in names:
        for right in names:
            if left >= right:
                continue
            pair_rows.append(
                {
                    "left": left,
                    "right": right,
                    "diff_rows": int(submissions[left].ne(submissions[right]).sum()),
                    "transition_counts_left_to_right": transition_counts(submissions[left], submissions[right]),
                }
            )
    pd.DataFrame(pair_rows).to_csv(args.output_dir / "pairwise_submission_diffs.csv", index=False)
    pool.to_csv(args.output_dir / "repleaf_private_agree_patch_pool.csv", index=False)
    sg_pool.to_csv(args.output_dir / "repleaf_private_agree_sg_pool.csv", index=False)
    all_pool.head(120).to_csv(args.output_dir / "top120_patch_candidates.csv", index=False)

    progress("Writing public exploration candidates")
    candidate_specs: list[tuple[str, pd.DataFrame]] = [
        ("PUBLIC_REF10_repleaf_priv_sg_top08", sg_pool.head(8)),
        ("PUBLIC_REF10_repleaf_priv_sg_top16", sg_pool.head(16)),
        ("PUBLIC_REF10_repleaf_priv_sg_top24", sg_pool.head(24)),
        ("PUBLIC_REF10_repleaf_priv_sg_top32", sg_pool.head(32)),
        ("PUBLIC_REF10_repleaf_priv_sg_gs_16_08", pd.concat([sg_pool.head(16), gs_pool.head(8)], ignore_index=True)),
        ("PUBLIC_REF10_repleaf_priv_all_top32", all_pool.head(32)),
    ]

    manifest = []
    seen_hashes: set[str] = set()
    output_rank = int(args.output_rank_start)
    for label, selected in candidate_specs:
        if selected.empty:
            continue
        labels = anchor.copy()
        labels.loc[selected[ID].to_numpy()] = selected["repleaf"].to_numpy(dtype=object)
        candidate_hash = hashlib.md5(labels.to_numpy(dtype="U8").tobytes()).hexdigest()
        if candidate_hash in seen_hashes:
            progress(f"Skipping duplicate candidate: {label}")
            continue
        seen_hashes.add(candidate_hash)
        output_path = OUTPUTS / f"{output_rank}_{label}.csv"
        manifest.append(build_candidate(sample_ids, anchor, selected, output_path))
        manifest[-1]["candidate"] = label
        manifest[-1]["selected_ids"] = selected[ID].astype(int).tolist()
        output_rank += 1

    pd.DataFrame(manifest).to_csv(args.output_dir / "candidate_manifest.csv", index=False)

    report = {
        "purpose": "PUBLIC-LB exploration from downloaded hard-label submissions; not a CV/private validation result.",
        "inputs": {
            "public_anchor": str(args.public_anchor),
            "repleaf_submission": str(args.repleaf_submission),
            "our_public": str(args.our_public.relative_to(ROOT)) if args.our_public.is_relative_to(ROOT) else str(args.our_public),
            "private_main": str(args.private_main.relative_to(ROOT)) if args.private_main.is_relative_to(ROOT) else str(args.private_main),
        },
        "class_counts": {name: class_counts(labels) for name, labels in submissions.items()},
        "key_diffs": {
            "public_anchor_vs_repleaf": {
                "diff_rows": int(anchor.ne(repleaf).sum()),
                "transition_counts": transition_counts(anchor, repleaf),
            },
            "public_anchor_vs_private_main": {
                "diff_rows": int(anchor.ne(private_main).sum()),
                "transition_counts": transition_counts(anchor, private_main),
            },
            "public_anchor_vs_our_public_151": {
                "diff_rows": int(anchor.ne(submissions["our_public_151"]).sum()),
                "transition_counts": transition_counts(anchor, submissions["our_public_151"]),
            },
        },
        "segments": [
            summarize_segment(pool, "anchor != repleaf == private_main"),
            summarize_segment(sg_pool, "STAR->GALAXY only"),
            summarize_segment(gs_pool, "GALAXY->STAR only"),
        ],
        "candidate_manifest": manifest,
        "next_cv_step": (
            "Install repleafgbm/repleafgbm-native and run scripts/train_repleafgbm_cv.py "
            "to create OOF/test probabilities. Hard-label submission files cannot honestly "
            "raise OOF/CV because they do not contain fold-level probabilities."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# Repleaf/Public Reference Analysis",
        "",
        "이 폴더는 2026-06-25에 받은 `submission (10).csv`, `submission (11).csv`를 분석한 기록입니다.",
        "",
        "- `submission (10).csv`: 퍼블릭 0.97233 계열 hard-label 앵커로 취급했습니다.",
        "- `submission (11).csv`: RepLeafGBM 단일모델 hard-label 결과로 취급했습니다.",
        "- 이 분석에서 만든 CSV는 퍼블릭 탐색용입니다. OOF/CV 후보가 아닙니다.",
        "- RepLeaf를 진짜 일반화 재료로 쓰려면 `scripts/train_repleafgbm_cv.py`로 OOF/test probability를 만들어야 합니다.",
        "",
        "## 생성 파일",
        "",
        "- `pairwise_submission_diffs.csv`: 받은 파일과 기존 후보 간 row 차이",
        "- `repleaf_private_agree_patch_pool.csv`: RepLeaf와 우리 private 후보가 동시에 앵커와 다르게 보는 row",
        "- `candidate_manifest.csv`: 생성된 퍼블릭 탐색 후보 목록",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    progress(f"Wrote report to {args.output_dir}")
    for item in manifest:
        print(f"- {item['path']} changed_rows={item['changed_rows']} transitions={item['transition_counts']}", flush=True)


if __name__ == "__main__":
    main()
