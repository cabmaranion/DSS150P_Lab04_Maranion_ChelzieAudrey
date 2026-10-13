# Reproduction notes

## Findings from the n_jobs sweep

Setup: same data hash, seed, parameters and library versions in every run. Only `n_jobs`
changed (1, 2, 4, 8 and -1). Full results are in `docs/baseline.md` and in
`outputs/metrics_njobs*.json`.

- **The failing runs are better than the baseline, not worse.** The n_jobs=8 and -1 runs
  made 26 mistakes against the baseline's 30. The rule is two-sided on purpose: the goal is
  to reproduce the baseline, not to beat it, so a better result still counts as a mismatch.
  A mismatch is not an error. It only means the baseline was not reproduced. No good result
  was discarded.
- **The rule is tighter than the thread noise.** The n_jobs=2 and n_jobs=8 runs differ only
  in thread count, yet one cell differs by 5 and F1 differs by 0.0031. Both are larger than
  the tolerances, so thread count alone can push a legitimate run outside the rule, and the
  same code could fail on another machine. This is reported as a finding, not fixed by
  loosening the rule. Any future change to the thresholds will be a new dated revision with
  its own reason.
- **The original setting fails here.** n_jobs=-1, which the old notebook used, fails on this
  machine (8 logical cores). I can't say whether the original machine had a different core
  count.
- **Default.** n_jobs=1, because its result does not depend on how many cores a machine has.
  It passes the rule on this machine.

  
## Exit codes of `python -m src.run_all`

| Exit code | Meaning | What to do |
|---|---|---|
| 0 | Layer 1 and Layer 2 both passed | Report the run as consistent with the baseline |
| 1 | The script stopped before the checks finished: the data hash does not match the manifest, or something crashed (missing file, import error, and so on) | Read the message or traceback, fix the cause, and rerun. No comparison was made |
| 2 | Layer 1 failed (shapes, class counts, or the logistic regression control) | The comparison is invalid, even if Layer 2 printed PASS |
| 3 | Layer 1 passed, Layer 2 failed | The data side is sound, so investigate the XGBoost result (seeds, `n_jobs`, versions) before reporting |

The hash mismatch shares code 1 with crashes on purpose. No comparison is made in either case,
and the printed message tells them apart. A separate code can be added if something starts
reading the exit code automatically.

A Layer 1 failure gets its own code because it makes the comparison invalid. The data side is
wrong, so a passing XGBoost result would be meaningless.

## Design decision: checkpoint is written last

The checkpoint promises that the raw data is safely saved. It is written only after the raw
file is saved. If the script crashes in between, no checkpoint exists, so the next run
ingests again instead of trusting data that was never saved. The cost is repeated work, which
is safe. If the raw file is deleted later, the checkpoint check also notices (`batch_ok`)
and re-ingests. Tested on 2026-10-13 by deleting the raw file and rerunning.

## Evidence: hash check stops a wrong dataset

Tested 2026-10-13. I changed one character of the `sha256:` line in
`metadata/source_manifest.yml`. The run stopped with "Data hash does not match the manifest.
Stop." and exit code 1. After restoring the line, the run passed with exit code 0.

## Why the curated file keeps the original row order

`train_test_split` picks rows by position using the seed. With the same seed and a different
row order, different rows would land in the test set and the metrics would change.

