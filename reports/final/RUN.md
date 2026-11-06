# Final evaluation run (prereg-v1.2)

Pre-registration: docs/PREREG.md v1.2 (main tag `prereg-v1.2`, commit 47fb264), merged into
branch `feat/w4-pipeline` before this run.

Code commit used for every command below: `9d51446207bd8d803cd4e3c2a3708f8276752dba`
(feat(pipeline): add H5 secondary comparison (item-CF vs flat popularity)).

Environment: Python 3.12.16, numpy 1.26.4, pandas 2.2.3 (recorded in each JSON `meta`).
Data: `NBR_DATA_DIR=C:\dev_ML\data\processed`; profile `holdout` (Instacart holdout; Complete Journey uses
the `medium` data via `complete_journey_profile: medium`).

## Commands (fixed order, each executed exactly once)

| # | Command | Start (UTC) | End (UTC) | Wall time | Exit |
|---|---|---|---|---|---|
| 1 | `python -m nbr tune --profile holdout --dataset instacart` | 2026-11-05T10:14:04Z | 10:19:21Z | 317.1 s | 0 |
| 2 | `python -m nbr evaluate --profile holdout --dataset instacart --final` | 10:19:21Z | 10:26:48Z | 445.4 s | 0 |
| 3 | `python -m nbr tune --profile holdout --dataset complete_journey` | 10:26:48Z | 10:28:09Z | 81.0 s | 0 |
| 4 | `python -m nbr evaluate --profile holdout --dataset complete_journey` | 10:28:09Z | 10:31:14Z | 184.0 s | 0 |
| 5 | `python -m nbr evaluate --profile holdout --dataset complete_journey --transfer-from instacart` | 10:31:14Z | 10:34:18Z | 183.3 s | 0 |

Total wall time: 20 min 14 s. The commands were chained so that a failure would have stopped the chain.

## Re-run of command 2

None. Command 2 finished with exit code 0 on its first execution; the Instacart holdout eval split
(`holdout/instacart/eval_history.parquet`, `eval_target.parquet`) was read exactly once, by command 2.

## Complete Journey reproducibility check

Commands 3 and 4 use the same `medium` Complete Journey data and the same code paths as the development run
(`reports/metrics/medium_complete_journey_*`, branch commit 7baa6a1), so their outputs were compared:

- Tuning (`holdout_complete_journey_tuning.json` vs `medium_complete_journey_tuning.json`): identical in every
  field except `meta`, `timing_seconds` and `profile`. Selections gamma=1.0, gamma_c=0.9, alpha=10, r=10 in both.
- Evaluation (`holdout_complete_journey_eval.json` vs `medium_complete_journey_eval.json`): identical in every
  field except `meta`, `timing_seconds`, `profile`, `selections.source` and the `H5` key, which exists only in
  the final run (added in commit 9d51446 before the run).
- Transfer evaluation differs by design: it applies the Instacart *holdout* selections
  (gamma=0.8, gamma_c=0.8, alpha=100, r=10), whereas the development transfer run applied the Instacart
  *medium* selections (gamma=0.8, gamma_c=0.9, alpha=100, r=10).

## Files

`holdout_*.json` are byte-identical copies of `reports/metrics/holdout_*.json` written by the commands above;
`SHA256SUMS` lists their SHA-256 digests (`sha256sum -c SHA256SUMS` from this directory).
