from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    submission = pd.read_csv(DATA / "submission.csv")

    print("train shape:", train.shape)
    print("sample shape:", sample.shape)
    print("submission shape:", submission.shape)
    print("submission columns:", submission.columns.tolist())
    print("ids equal sample:", sample["id"].equals(submission["id"]))
    print("missing predictions:", int(submission["class"].isna().sum()))
    print("valid labels:", sorted(submission["class"].dropna().unique().tolist()))
    print()
    print("train class share")
    print(train["class"].value_counts(normalize=True).sort_index().round(6))
    print()
    print("anchor submission class share")
    print(submission["class"].value_counts(normalize=True).sort_index().round(6))


if __name__ == "__main__":
    main()
