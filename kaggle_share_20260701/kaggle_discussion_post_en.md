# [27th Place] Robust OOF Selection Beat My Public-LB File

I finished 27th out of 2,824 teams with a private score of 0.97029.

My highest public-oriented file scored 0.97254 on the public leaderboard, but only
0.97005 on the private leaderboard. The final file was selected without using public
leaderboard feedback. Its OOF balanced accuracy was 0.9706589608, and it finished
higher on the hidden test set.

The final candidate was built in several stages:

1. I collected matching OOF and test probabilities from local models and public model
   sources.
2. I built guarded candidates with class-wise probability blending, subset rollback,
   and selective transfer.
3. I evaluated 1,199 candidates and audited the top 120.
4. I checked class recall, weak subsets, prediction-change counts, and 10 seeds x
   5 meta-folds instead of selecting only the maximum full-OOF score.
5. The last step mixed only the GALAXY probability from my one-vs-rest CatBoost source
   at weight 0.005.

The final step changed only 3 OOF predictions and 1 test prediction relative to the
previous stable candidate:

```text
new_GALAXY = 0.995 * current_GALAXY
           + 0.005 * OVR_CatBoost_GALAXY
```

After renormalization, OOF increased from 0.9706563117 to 0.9706589608. The auxiliary
model had lower standalone OOF, but its GALAXY probability contained complementary
boundary information.

I also had a candidate with slightly higher raw OOF, but it changed substantially more
rows and was less robust across subsets. I kept the smaller and more stable correction.

The most important result for me was the public/private comparison:

- Public-oriented file: public 0.97254, private 0.97005
- Robust OOF-selected file: private 0.97029, final rank 27

Small public gains were noisy. Repeated OOF checks and conservative changes were more
useful for final selection.

I published a notebook that reproduces the final selection stage from the attached
OOF/test probability dataset. It includes the blend-weight scan, 50 meta-fold checks,
class report, confusion matrix, and submission generation. It does not claim to retrain
all level-1 models in one short notebook.

Public model references that influenced the model bank include
[Chris Deotte's GPU Logistic Regression Stacker](https://www.kaggle.com/code/cdeotte/gpu-logistic-regression-stacker),
[RealMLP v5](https://www.kaggle.com/code/cdeotte/realmlp-v5-for-s6e6), and
[kirill0212's one-vs-rest XGBoost](https://www.kaggle.com/code/kirill0212/ps6e6-one-vs-rest-xgb).
Thank you to everyone who shared reproducible OOF/test predictions and analysis during
the competition.
