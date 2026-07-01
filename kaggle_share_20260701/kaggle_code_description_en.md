# 27th Place - Robust OOF Selection and Class-wise Micro-Blending

This notebook reconstructs the final selection stage of my 27th-place solution.

It loads matching OOF/test probabilities, applies a 0.5% GALAXY-only blend from an
one-vs-rest CatBoost source, evaluates the result with balanced accuracy, checks the
gain across 50 repeated meta-folds, reports class metrics, and writes the submission.

The final selector did not use public leaderboard scores:

- OOF balanced accuracy: 0.9706589608
- Private leaderboard: 0.97029
- Final rank: 27 / 2,824

This is a reproducible final-stage notebook, not a claim that every level-1 model is
retrained inside one short run. The required precomputed prediction dataset is attached.
