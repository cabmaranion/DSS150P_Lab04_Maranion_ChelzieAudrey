# Baseline

## Source
- File: `drebin_dataset.csv` (a subset of DREBIN, Mendeley Data, DOI 10.17632/bpkksc9v5s.7)
- Not stored in the repo. It must be downloaded from the source.
- Shape: 15,036 rows, 216 columns (215 binary features plus `class`)
- Classes: 9,476 benign (B), 5,560 malware (S), about 63/37
- SHA-256 of the file I downloaded: see `docs/source_manifest.yml`

## Task
- Binary classification. Malware = 1 (S), benign = 0 (B)

## Preprocessing (in order)
1. In `TelephonyManager.getSimCountryIso`, replace `?` with 0 and cast to int
2. Map `class` to `is_malware` (S=1, B=0), then drop `class`
3. No missing values and no other non-binary columns were found
4. No oversampling or undersampling

## Split and seeds
- `SEED = 246112165`, also passed to `np.random.seed`
- Stratified 80/20 split with `random_state=SEED`: train 12,028 rows, test 3,008 rows (1,896 benign, 1,112 malware)
- Tuning CV: RepeatedStratifiedKFold, 5 folds, 1 repeat, seeded, scored by ROC-AUC

## Model selection
- Six models compared. XGBoost won on tuning CV ROC-AUC (0.9987, versus 0.9983 for Random Forest and 0.9980 for MLP)
- Best XGBoost parameters: n_estimators 276, max_depth 7, learning_rate 0.1978, subsample 0.8526, colsample_bytree 0.7013, scale_pos_weight 1.0 (from a 50-iteration randomized search)

## Declared baseline (XGBoost, test set, threshold 0.5)
| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Log-loss |
|---|---|---|---|---|---|---|
| 0.9900 | 0.9874 | 0.9856 | 0.9865 | 0.9991 | 0.9985 | 0.0340 |

Confusion matrix: [[1882, 14], [16, 1096]]
The other five models are in the project README results table.

## Environment
- Python 3.12.10
- numpy 2.4.6, pandas 3.0.4, scikit-learn 1.9.0, xgboost 3.3.0, shap 0.52.0, scipy 1.18.0, joblib 1.5.3

## Known reproducibility risks

1. Dataset not in the repo: The dataset's SHA-256 hash is recorded in `source_manifest.yml` and the pipeline will check it at load time, but the original project never recorded a hash, so this protects future runs and cannot prove the file matches the original.
2. Newer Python and library versions: Newer releases can change defaults, solver behavior, or random-number handling, so metrics may shift slightly or a call may fail, which is why the original versions will be pinned.
3. `n_jobs=-1` parallelism: Parallel workers can change floating-point order and seed splitting across machines, so I will run with `n_jobs=1` and `n_jobs=-1` (same seeds) and compare metrics and confusion matrix.

## What counts as "consistent with the baseline"

A rerun is consistent if the XGBoost test F1 and ROC-AUC are within an absolute difference of 0.001 of 0.9865 and 0.9991, and the confusion matrix is [[1882, 14], [16, 1096]]. If the library versions match the original, I expect an exact confusion matrix match and metrics agreeing to four decimals; anything outside tolerance is a problem to investigate (seeds, `n_jobs`, data hash, preprocessing), not accepted drift. If they don't match, I will record the new versions and results beside the baseline and check that XGBoost still ranks first on test F1 and ROC-AUC because a version change can legitimately flip a few predictions, and the conclusion (XGBoost is the best model) is what matters.