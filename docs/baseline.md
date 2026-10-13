# Baseline

## Source
- File: `drebin_dataset.csv` (a subset of DREBIN, Mendeley Data, DOI 10.17632/bpkksc9v5s.7)
- Not stored in the repo. It must be downloaded from the source.
- Shape: 15,036 rows, 216 columns (215 binary features plus `class`)
- Classes: 9,476 benign (B), 5,560 malware (S), about 63/37
- SHA-256 of the file I downloaded: see `metadata/source_manifest.yml`

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
4. Duplicate rows: 7,775 of 15,036 rows are exact duplicates. Because the split is random, copies of the same row can land in both train and test, like a test question that was already on your practice sheet, so the model gets credit for remembering rather than learning. This can inflate the test score. Also, 3 feature patterns appear with both labels (7,778 minus 7,775), so the same feature vector is labeled both benign and malware, which is label noise that no model can get fully right, and it is a reason to keep the F1 and ROC-AUC claims modest. The original baseline used the same random split on the same data, so it carries the same inflation. I am not changing the split, because the goal is to reproduce the baseline, so I report this as a limitation instead.

## What counts as "consistent with the baseline"

A rerun is consistent if the XGBoost test F1 and ROC-AUC are within an absolute difference of 0.001 of 0.9865 and 0.9991, and the confusion matrix is [[1882, 14], [16, 1096]]. If the library versions match the original, I expect an exact confusion matrix match and metrics agreeing to four decimals; anything outside tolerance is a problem to investigate (seeds, `n_jobs`, data hash, preprocessing), not accepted drift. If they don't match, I will record the new versions and results beside the baseline and check that XGBoost still ranks first on test F1 and ROC-AUC because a version change can legitimately flip a few predictions, and the conclusion (XGBoost is the best model) is what matters.

### Rule revision, 2026-10-13

**Original rule (committed in 4f81661, written before any rerun):** XGBoost test F1 and
ROC-AUC within 0.001 of 0.9865 and 0.9991, and the confusion matrix [[1882, 14], [16, 1096]],
expected to match exactly when library versions match. An earlier draft also allowed each cell
to be within 2 counts when versions differ, but that clause is not in the committed version.

**What changed it:** The rerun did not match exactly, although the library versions, data
hash, seed and parameters were the same. The logistic regression control matched the original
exactly in two run I made (accuracy 0.9764, F1 0.9683, ROC-AUC 0.9961, [[1853, 43], [28, 1084]]).
In a sweep over n_jobs = 1, 2, 4, 8 and -1 (same seed, versions, data and parameters), F1 ranged
from 0.9852 to 0.9883, ROC-AUC from 0.9991 to 0.9993, and confusion-matrix cells differed from
the baseline by up to 3 counts. So thread count changes XGBoost's predictions, and exact XGBoost
counts are not something this pipeline can guarantee. The original rule is kept above, not deleted.

**Layer 1, exact (data side, any mismatch is a failure to investigate):**
- SHA-256 of the dataset equals the value in source_manifest.yml
- Shapes: 15,036 x 216 raw; train (12028, 215), test (3008, 215)
- Test class counts: 1896 benign, 1112 malware
- Logistic regression control reproduces [[1853, 43], [28, 1084]]

**Layer 2, tolerance (XGBoost only):**
- F1 and ROC-AUC each within an absolute difference of 0.001 of 0.9865 and 0.9991
- Every confusion-matrix cell within 2 counts of the baseline

**How the numbers were chosen:** 0.001 is unchanged from the original rule, which I wrote before
any rerun. It is equivalent to about 2 net errors on the 3,008 test rows (one extra error moves
F1 by about 0.00045). The cell tolerance of 2 comes from an earlier draft written before the
n_jobs sweep, where it applied only if library versions differed. It was not in the committed
rule, so it is not a fully pre-set number. I now apply it to runs with matching versions as well,
because the sweep showed that thread count alone moves the results. The two checks are not the
same size: a run can have every cell within 2 and still fail the F1 check (the n_jobs=8 run
does), and, in principle, a run can pass F1 and fail the cell check. F1, ROC-AUC and the cell
check must all pass. The cell check is kept because F1 can hide a trade between false positives
and false negatives, which matters for malware detection. The rule was applied unchanged to all
five runs, and every run is reported below.

**Results of the n_jobs sweep (same seed, versions, data and parameters):**

| n_jobs | F1 | F1 gap | ROC-AUC check | Cells within 2? | Layer 2 |
|---|---|---|---|---|---|
| 1 (default) | 0.9874 | 0.0009 | pass | yes | pass |
| 2 | 0.9852 | 0.0013 | pass | no (off by 3) | fail (F1 and cells) |
| 4 | 0.9874 | 0.0009 | pass | yes | pass |
| 8 | 0.9883 | 0.0018 | pass | yes | fail (F1) |
| -1 | 0.9883 | 0.0018 | pass | yes | fail (F1) |

Two of five runs pass, including the default.

**Notes on the results:**
- The failing runs are better than the baseline, not worse. The n_jobs=8 and -1 runs made 26
  mistakes against the baseline's 30. The rule is two-sided because the aim is to reproduce the
  baseline, not beat it.
- The tolerance is narrower than the thread noise. The 2-thread and 8-thread runs differ by 0.0031
  in F1 and by 5 in one cell, so a legitimate run can fail. This is reported, not retuned.
- n_jobs=-1 is the setting the original notebook used. It fails on this machine, which has 8
  logical cores. I do not know the core count of the original machine.

**Default setting:** n_jobs = 1, because its result does not depend on how many cores a machine has.

**If Layer 1 fails:** stop, the comparison is invalid.
**If Layer 1 passes and Layer 2 fails:** investigate (seeds, n_jobs, versions) before reporting.


