# Kaggle sharing package

This folder contains the public, post-competition reconstruction of the final robust
OOF selection stage.

## Contents

- `27th-place-robust-oof-selection.ipynb`: self-contained Kaggle notebook.
- `input_dataset/`: precomputed OOF/test probability files and Kaggle dataset metadata.
- `kaggle_discussion_post_en.md`: English discussion post.
- `kaggle_code_description_en.md`: short code-page description.

## Prepare

```bash
.venv/bin/python scripts/package_kaggle_share_20260701.py
.venv/bin/python kaggle_share_20260701/build_notebook.py
```

Before uploading the dataset, edit `input_dataset/dataset-metadata.json` and replace
`YOUR_KAGGLE_USERNAME`.

The notebook expects the competition data and the uploaded probability dataset as
Kaggle inputs. It also runs locally from the repository root for verification.

## Reproducibility scope

The notebook reproduces the final `181 -> 193` decision, repeated meta-fold validation,
class metrics, plots, and submission generation. It does not retrain all 24 level-1
sources. That distinction is stated in the notebook and discussion post.

## License

Notebook and code: MIT.

Probability dataset: CC BY 4.0, with attribution to the public source notebooks listed
in the notebook. Locally trained probability sources remain identified in the manifest.
