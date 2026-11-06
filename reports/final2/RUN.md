# Final evaluation run on holdout2 (prereg-v1.3)

Pre-registration: docs/PREREG.md v1.3 (main tag `prereg-v1.3`, commit 7bf3262). Base of this branch
(`feat/w10b-final`): main at `8cace67` (holdout2 data and selection report, W10a). The sections
"Pre-specified output keys", "Pre-checks" and "Selections" were committed before the eval split was read
(commits `dc849d7`, `a0f130c`, `e7ad59d`); the remaining sections were written after command 5.

Commits before any run (this branch, all before command 1):

| Commit | Content |
|---|---|
| `f7cfb0a` | data/SOURCE.md: note on the W10a post-hoc read of the v1.2 holdout eval-target user ids |
| `535b490` | src/nbr/pipeline.py: `final_lr_ucf_vs_final_recall` comparison (C3 vs C0, Recall@10) and decision wording on the two C3-vs-C0 entries; reporting only, no list/selection change; tests (181 passed). Sample smoke run (tune + evaluate --skip-l3, both datasets) compared with the W10a smoke outputs: content identical except the added key, the added decision field and worktree paths. |
| `dc849d7` | this file, draft with the pre-specified key table |
| `a0f130c` | pre-check record |

Code commit used for the commands: `src/` and `configs/` were last changed in `535b490`; HEAD was `a0f130c` for
command 1 and `e7ad59d` (selections fixed in this file) for commands 2-5; the working tree was clean for every
command (runner log `logs/commands.json`).

Environment: Python 3.12.16, numpy 1.26.4, pandas 2.2.3 (recorded in each JSON `meta`), Windows 11.
Data: `NBR_DATA_DIR=C:\dev_ML\data\processed`; profile `holdout2` (Instacart holdout2, 20,000 users disjoint from
sample, medium and holdout; Complete Journey uses the `medium` data via `complete_journey_profile: medium`).

## Pre-specified output keys (PREREG v1.3 -> JSON keys, read from the code before any run)

Files: `holdout2_instacart_eval.json` (Instacart, confirmatory), `holdout2_complete_journey_eval.json` (CJ refit,
secondary), `holdout2_complete_journey_eval_transfer.json` (CJ transfer, secondary). Every entry written by
`nbr.pipeline._compare` carries `n_users`, `comparator_mean`, `model_mean`, `diff` (= model - comparator) and
`ci` at both levels `0.95` and `0.99375` (configs/common.yaml `bootstrap.levels`); entries marked "decision"
also carry `decision` at both levels (PREREG rule: lower bound > 0 "improvement observed", upper bound < 0
"lower", otherwise "no difference established").

Primary (Instacart holdout2, family of 3; the 99.375 % interval is the multiplicity-adjusted one):

| PREREG v1.3 | JSON key | comparator -> model | metric, users | decision |
|---|---|---|---|---|
| H6: R2 vs R1 | `lr_repeat.primary.H6_repeat_recall` | `repeat_star` (R1) -> `repeat_lr` (R2) | repeat Recall@10, users with >= 1 repeat target item | yes |
| H7': C3 vs C0, NDCG@10 | `integration.secondary.final_lr_ucf_vs_final_ndcg` | `final` (C0) -> `final_lr_ucf` (C3) | NDCG@10, all users | yes (added in 535b490) |
| H7': C3 vs C0, Recall@10 (reported) | `integration.secondary.final_lr_ucf_vs_final_recall` | `final` (C0) -> `final_lr_ucf` (C3) | Recall@10, all users | yes (key and decision added in 535b490) |
| H8: E2 vs flat popularity | `integration.primary.H8_new_recall` | `flat_explore` -> `usercf` (E2) | new-item Recall@10, users with >= 1 new target item | yes |

Secondary (no confirmatory claim):

| PREREG v1.3 | JSON key | comparator -> model | metric | decision field |
|---|---|---|---|---|
| H9: C3 vs C1 | `integration.primary.H9_ndcg`, `integration.primary.H9_recall` | `final_lr` (C1) -> `final_lr_ucf` (C3) | NDCG@10, Recall@10, all users | written by the code (W8 layout); secondary under v1.3 |
| H1 replication | `primary.H1_repeat_recall` | `repeat_pf` -> `repeat_star` | repeat Recall@10 | yes |
| H2 replication | `primary.H2_new_recall` | `flat_explore` -> `explore_star` | new-item Recall@10 | yes |
| H3 replication | `primary.H3_ndcg`, `primary.H3_recall` | `personal_frequency` -> `final` (C0) | NDCG@10, Recall@10 | yes |
| H4 replication | `primary.H4_category_hit` | `population_category_ranking` -> `category_ranking` | category hit@10 | yes |
| H5 replication | `H5` (top level) | `flat_explore` -> `itemcf_flat` | new-item Recall@10 | 95 % CI only (as in v1.2) |
| C3 vs personal frequency | `integration.secondary.final_lr_ucf_vs_personal_frequency_ndcg` | `personal_frequency` -> `final_lr_ucf` | NDCG@10 | CI only |
| E2 vs flat item-CF | `integration.secondary.usercf_vs_itemcf_flat_new_recall` | `itemcf_flat` -> `usercf` | new-item Recall@10 | CI only |
| H7 (C1 vs C0, v1.2 layout) | `lr_repeat.primary.H7_ndcg`, `lr_repeat.primary.H7_recall` | `final` -> `final_lr` | NDCG@10, Recall@10 | written by the code; secondary under v1.3 |
| other CI-only comparisons | `integration.secondary.final_star_ucf_vs_final_ndcg` (C3b vs C0), `integration.secondary.final_lr_r_vs_final_lr_ndcg` (C2 vs C1), `lr_repeat.secondary.*`, `secondary.*` (final_cf, final_r10, explore_cf, itemcf_flat vs explore_star, final vs popularity) | as named | as named | CI only |
| history-length bins | `by_history_length` (H1-H3), `lr_repeat.by_history_length` (H6, H7), `integration.by_history_length` (H8, H9) | as named | 95 % CI only | none |
| CJ refit / transfer | same keys in the two Complete Journey files | | | |

Not computed: `explore_L3` (`--skip-l3`, PREREG v1.3 item 6); R3 (not carried forward).

Selections to be fixed after command 1 (from `holdout2_instacart_tuning.json`): `selections.{gamma, gamma_c, alpha, r}`
and `at_grid_edge.{gamma, gamma_c, alpha, r}`; `integration.selections.{k, weighting, C, r.{final_lr_ucf, final_star_ucf, final_lr_r}}`
and `integration.at_grid_edge.{k, C, r.*}`; `lr_repeat.C` (grid of `integration.lr.C_grid`).

## Pre-checks (2026-11-06, commit `dc849d7`, working tree clean, `NBR_DATA_DIR=C:\dev_ML\data\processed`)

Code commit used for the commands: HEAD of this branch at command time (recorded per command in the
commands table); `src/` and `configs/` were last changed in `535b490`.

1. `python -m nbr check --profile holdout2` (exit 0, 2.6 s; reads the fit split only):

```
profile=holdout2 role=final data_dir=C:\dev_ML\data\processed
instacart/fit: users=20000 history_rows=2950640 target_rows=207902 users_with_repeat_target=18600 users_with_new_target=16344
instacart/eval: skipped (holdout guard)
```

2. Holdout guard on the holdout2 config, direct call (`load_split(load_config('holdout2'), 'instacart', 'eval')`
   without `allow_holdout_eval`), exit 1 before any Parquet file is opened:

```
role final data_dir C:\dev_ML\data\processed
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\dev_ML-wt-w10b\src\nbr\data.py", line 49, in load_split
    raise PermissionError(
PermissionError: profile role is 'final': the Instacart eval split is read once, after all selections are fixed; pass allow_holdout_eval=True explicitly
```

3. `python -m nbr evaluate --profile holdout2 --dataset instacart` WITHOUT `--final`, run before command 1
   (exit 1): the command stops in `load_selections` with
   `FileNotFoundError: ... reports\metrics\holdout2_instacart_tuning.json` because `run_evaluate` loads the
   tuning selections before `load_split`; the guard itself is therefore reached only once the tuning file
   exists. The same command is repeated after command 1 and before command 2 (see "Guard check after
   command 1" below); the eval split is never opened on this path.

## Selections (written after command 1, before command 2)

Source: `reports/metrics/holdout2_instacart_tuning.json` written by command 1 (commit `a0f130c`, fit split of
holdout2 Instacart: 20,000 users, 18,600 with a repeat target item, 16,344 with a new target item). Values as
written by the code; the Instacart eval split has not been read at this point.

| Selection | Value | Grid | At grid edge | Selection score (fit split) |
|---|---|---|---|---|
| gamma* (R1 decay) | 0.8 | {0.5, 0.6, 0.7, 0.8, 0.9, 1.0} | no | repeat Recall@10 0.581628 (0.5: 0.551498, 0.6: 0.561797, 0.7: 0.574499, 0.9: 0.580986, 1.0: 0.554819) |
| gamma_c*, alpha* (E1) | 0.8, 100 | {0.8, 0.9, 1.0} x {1, 10, 100} | gamma_c yes (lower), alpha yes (upper) | new-item Recall@10 0.042091 (0.9/100: 0.041960, 1.0/100: 0.041700) |
| r* (C0) | 10 | {5, ..., 10} | yes (upper) | NDCG@10 0.419145 (r = 9: 0.408743) |
| LR C* (R2) | 0.1 | {0.001, 0.01, 0.1, 1, 10} | no | selected on 4,000 validation users (16,000 training users), refit on all fit users |
| user-CF k*, weighting* (E2) | 100, binary | {10, 20, 50, 100, 200, 400} x {binary, log1p} | k no | new-item Recall@10 0.060198 |
| r (C3 = final_lr_ucf) | 10 | {0, ..., 10} | yes (upper) | NDCG@10 0.422704 on the 4,000 LR validation users (train-only model); r = 9: 0.412607 |
| r (C3b = final_star_ucf) | 10 | {0, ..., 10} | yes (upper) | NDCG@10 0.419587 on all 20,000 fit users; r = 9: 0.410039 |
| r (C2 = final_lr_r) | 10 | {0, ..., 10} | yes (upper) | NDCG@10 0.422474 on the 4,000 LR validation users; r = 9: 0.411587 |

User-CF grid (fit new-item Recall@10, 16,344 users): binary k = 10: 0.042806, 20: 0.051979, 50: 0.058278,
100: 0.060198, 200: 0.059618, 400: 0.058162; log1p k = 10: 0.041379, 20: 0.050005, 50: 0.057426, 100: 0.058534,
200: 0.058330, 400: 0.056573. `at_grid_edge` flags as written: three-stage {gamma: false, gamma_c: true,
alpha: true, r: true}; integration {k: false, C: false, r: {final_lr_ucf: true, final_star_ucf: true, final_lr_r: true}}.

Command 1 record (runner log): start 2026-11-06T08:02:07.899Z, end 08:23:13.619Z, wall 1265.7 s, exit 0, peak
working set 2551.7 MiB (process tree sampled every 0.5 s; the JSON `peak_memory_mib.integration` is 2551.7 MiB).

## Guard check after command 1 (before command 2)

`python -m nbr evaluate --profile holdout2 --dataset instacart` WITHOUT `--final`, run after the selections
commit `e7ad59d` and before command 2 (`logs/guard_after_command1.log`): the tuning selections, the LR
artifact and the integration selections load, then `nbr.data.load_split` raises
`PermissionError: profile role is 'final': the Instacart eval split is read once, after all selections are
fixed; pass allow_holdout_eval=True explicitly` before any eval Parquet file is opened (exit 1). Command 2 was
started only after this message was found in the log.

## Commands (fixed order, each executed exactly once)

Runner: `NBR_DATA_DIR=C:\dev_ML\data\processed`, `PYTHONPATH=src`, working directory `C:\dev_ML-wt-w10b`,
Python `C:\dev_ML\.venv\Scripts\python.exe` (3.12.16). Peak memory = largest per-process peak working set in the
launched process tree, sampled every 0.5 s (the venv `python.exe` is a launcher; the interpreter is its child).
The in-JSON figure is the largest `peak_memory_mib` value the pipeline recorded in-process.

| # | Command | Start (UTC) | End (UTC) | Wall time | Exit | Peak working set (runner / in-JSON) | HEAD |
|---|---|---|---|---|---|---|---|
| 1 | `python -m nbr tune --profile holdout2 --dataset instacart` | 2026-11-06T08:02:07.899Z | 08:23:13.619Z | 1265.7 s | 0 | 2551.7 / 2551.7 MiB | `a0f130c` |
| - | selections written to this file and committed (`e7ad59d`); guard check (above) | 08:23:13 | 08:24:29 | - | - | - | - |
| 2 | `python -m nbr evaluate --profile holdout2 --dataset instacart --final --skip-l3` | 08:24:29.566Z | 08:29:16.378Z | 286.8 s | 0 | 4036.9 / 4036.9 MiB | `e7ad59d` |
| 3 | `python -m nbr tune --profile holdout2 --dataset complete_journey` | 08:29:16.378Z | 08:40:52.705Z | 696.3 s | 0 | 2553.8 / 2553.8 MiB | `e7ad59d` |
| 4 | `python -m nbr evaluate --profile holdout2 --dataset complete_journey --skip-l3` | 08:40:52.706Z | 08:46:03.110Z | 310.4 s | 0 | 3304.7 / 3304.7 MiB | `e7ad59d` |
| 5 | `python -m nbr evaluate --profile holdout2 --dataset complete_journey --transfer-from instacart --skip-l3` | 08:46:03.111Z | 08:52:09.393Z | 366.3 s | 0 | 3301.9 / 3301.9 MiB | `e7ad59d` |

Total wall time of the five commands: 2925.5 s (48 min 46 s); commands 2-5 were chained so that a failure would
have stopped the chain, and the chain was started only after the guard check passed.

## Re-run of command 2

None. Command 2 finished with exit code 0 on its first execution; the Instacart holdout2 eval split
(`holdout2/instacart/eval_history.parquet`, `eval_target.parquet`) was read exactly once, by command 2.
No command was re-run for any reason.

## Complete Journey reproducibility check

Commands 3 and 4 use the same `medium` Complete Journey data as the W8b development run
(`C:\dev_ML-wt-w8\reports\metrics\medium_complete_journey_{tuning,eval}.json`, commit 6936873, user-CF k grid
20-400, `--skip-l3`), so every key of the two documents was compared (no field excluded; user-CF grid rows
aligned by (k, weighting)).

- Tuning (`holdout2_complete_journey_tuning.json` vs W8b): 48 differing paths, all of them `meta`
  (git_commit, profile, utc_time), `profile`, the two `label` fields ("development (medium)" -> "final (holdout2)"),
  timing fields (`timing_seconds`, `integration.timing_seconds`, `integration.usercf.timing_seconds`,
  `integration.usercf.grid[*].seconds`, `lr_repeat.timing_seconds`, `lr_repeat.validation[*].fit_seconds`),
  peak-memory fields (`peak_memory_mib`, `integration.process_peak_working_set_mib`), the `lr_repeat.artifact`
  path, the two added user-CF rows k = 10 (binary 0.011893, log1p 0.013973, fit new-item Recall@10) and the
  flags `integration.at_grid_edge.k` and `integration.usercf.at_grid_edge.k` (true -> false, because k* = 20 is no
  longer the lower edge of the grid). Every selection and every selection score is identical:
  gamma = 1.0, gamma_c = 0.9, alpha = 10, r = 10; LR C = 0.01; user-CF k* = 20, weighting binary
  (fit new-item Recall@10 0.014327; the k = 10 rows score lower); r = 10 for final_lr_ucf, final_star_ucf and
  final_lr_r. k* did not change.
- Evaluation (`holdout2_complete_journey_eval.json` vs W8b): 55 differing paths, all of them `meta`, `profile`,
  the three `label` fields, `selections.source` and `integration.selections.source` (paths), `lr_repeat.artifact`
  (path), `timing_seconds` (23 keys), `peak_memory_mib` (19 keys) and `integration.process_peak_working_set_mib`,
  plus the two reporting additions of commit 535b490: `integration.secondary.final_lr_ucf_vs_final_ndcg.decision`
  and the new entry `integration.secondary.final_lr_ucf_vs_final_recall`. Every metric, interval and decision that
  exists in both documents is identical.
- Transfer evaluation (command 5) applies the Instacart holdout2 selections (gamma = 0.8, gamma_c = 0.8,
  alpha = 100, r = 10; LR C = 0.1; user-CF k = 100 binary; r = 10 for the three combinations) to the CJ data;
  the W8b development transfer run is not part of this comparison.

## Primary results as written by the code (Instacart holdout2, `holdout2_instacart_eval.json`)

20,000 users; 18,686 with a repeat target item; 16,270 with a new target item. diff = model - comparator; the
99.375 % interval is the multiplicity-adjusted one (PREREG v1.3 item 4).

| PREREG v1.3 | key | n_users | comparator mean | model mean | diff | 95 % CI | 99.375 % CI | decision 95 % / 99.375 % |
|---|---|---|---|---|---|---|---|---|
| H6 (R2 vs R1, repeat Recall@10) | `lr_repeat.primary.H6_repeat_recall` | 18,686 | 0.560556 | 0.572423 | +0.011866 | [0.009726, 0.013917] | [0.008803, 0.014879] | improvement observed / improvement observed |
| H7' (C3 vs C0, NDCG@10) | `integration.secondary.final_lr_ucf_vs_final_ndcg` | 20,000 | 0.419213 | 0.425539 | +0.006327 | [0.005332, 0.007240] | [0.005066, 0.007628] | improvement observed / improvement observed |
| H7' (C3 vs C0, Recall@10, reported) | `integration.secondary.final_lr_ucf_vs_final_recall` | 20,000 | 0.346106 | 0.351740 | +0.005634 | [0.004283, 0.006861] | [0.003921, 0.007366] | improvement observed / improvement observed |
| H8 (E2 vs flat popularity, new-item Recall@10) | `integration.primary.H8_new_recall` | 16,270 | 0.041299 | 0.060222 | +0.018922 | [0.017126, 0.020901] | [0.016496, 0.021701] | improvement observed / improvement observed |

Secondary entries (H9, H1-H5 replication, C3 vs personal frequency, E2 vs flat item-CF, CJ refit and transfer)
are read from the JSON files listed below; they carry no confirmatory claim.

## Files

`holdout2_*.json` are byte-identical copies of `reports/metrics/holdout2_*.json` written by the commands above
(verified with `cmp`); `SHA256SUMS` lists their SHA-256 digests (`sha256sum -c SHA256SUMS` from this directory).
`logs/` holds the runner records: `commands.json` (commands 1-5: UTC start/end, wall time, exit code, peak
memory), `cmd1.log` ... `cmd5.log` (stdout+stderr of each command), `prechecks.json`,
`guard_after_command1.json` and `guard_after_command1.log`.
