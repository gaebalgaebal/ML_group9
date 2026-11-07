# Confirmatory run of the exploration policy on holdout3 (prereg-v1.4)

Pre-registration: docs/PREREG.md v1.4 (main tag `prereg-v1.4`, commit 5427b0a; the v1.4 amendment is commit 614209f,
the base of this branch `feat/w17-policy`). The sections "Pre-specified output keys", "Pre-checks" and "Selections"
were committed before the holdout3 Instacart eval split was read (commits `271da53`, `fececaa`, `4a88bc0`); the sections from "Guard check after command 1" on were written after command 5.
Same procedure and failure policy as reports/final2/RUN.md: five commands, each executed exactly once, chained;
a failure in command 2 stops everything (no partial output is opened, no re-run); other commands admit technical
fixes only, recorded here.

Commits of this branch before any holdout3 run (all before command 1):

| Commit | Content |
|---|---|
| `0c7aa89` | src/nbr/policy.py (P_T, S, new-item hit share, H11-H13 block), `evaluate --policy` wiring, tests (265 collected, all passed), scripts/check_policy_dev.py |
| (P1.2) | reports/posthoc/policy_dev_medium.{json,md}: development check of the policy code on medium (not confirmatory) |
| (P2) | configs/holdout3.yaml; holdout3 data (data/SOURCE.md, data/MANIFEST.sha256, reports/holdout_prep/holdout3_selection.json) |
| (this file) | pre-specified key table and pre-check record |

Fixed by PREREG v1.4 and not re-selected: T = 4 (history_len = distinct history baskets of the evaluated split),
delta = 0.003, r_short = 9; CI levels 0.95 and 0.9833 (= 1 - 0.05/3, family of 3); paired bootstrap over users,
1,000 resamples, seed 42 (configs/common.yaml `bootstrap.n`, `bootstrap.seed`; the policy levels are constants of
nbr.policy). Systems: C3 = `final_lr_ucf` (R2 + E2, r* re-selected on the fit split, PREREG v1.3); P_T = C3 with
r_u = 9 for users with history_len <= 4 and r_u = r* otherwise (`nbr.policy.combine_policy`); S = `usercf` (E2 top 10
on its own); comparator of H13 = `flat_explore` (flat popularity excluding history items).

Environment: Python 3.12.16, numpy 1.26.4, pandas 2.2.3 (recorded in each JSON `meta`), Windows 11.
Data: `NBR_DATA_DIR=C:\dev_ML\data\processed`; profile `holdout3` (Instacart holdout3, 20,000 users disjoint from
sample, medium, holdout and holdout2; Complete Journey uses the `medium` data via `complete_journey_profile: medium`).

## Pre-specified output keys (PREREG v1.4 -> JSON keys, read from the code before any run)

Files: `holdout3_instacart_eval.json` (Instacart, confirmatory), `holdout3_complete_journey_eval.json` (CJ refit with
the same policy, secondary), `holdout3_complete_journey_eval_transfer.json` (CJ transfer, secondary). Every entry of the
`policy` block (written by `nbr.policy.evaluate_policy`) carries `n_users`, `comparator_mean`, `model_mean`, `diff`
(= model - comparator) and `ci` at both levels `0.95` and `0.9833`; the three primary entries also carry `decision` at
both levels and the `rule` text. Entries of the other blocks (`primary`, `H5`, `secondary`, `lr_repeat`, `integration`)
are written exactly as in the holdout2 run (CI levels `0.95` and `0.99375`).

Primary (Instacart holdout3, family of 3; the 98.33 % interval `ci.0.9833` is the pre-registered one):

| PREREG v1.4 | JSON key | comparator -> model | metric, users | decision rule (`decision.0.9833`) |
|---|---|---|---|---|
| H11 (non-inferiority): P_T vs C3 | `policy.primary.H11_ndcg` | `final_lr_ucf` (C3) -> `P_T` | NDCG@10, all users | "non-inferior" if the lower bound > -0.003, else "non-inferiority not established" |
| H12: P_T vs C3 | `policy.primary.H12_new_hit_share` | `final_lr_ucf` (C3) -> `P_T` | share of users with >= 1 new-item hit in the top 10, all users (users without a new target item count 0) | "improvement observed" if the lower bound > 0; "lower" if the upper bound < 0; else "no difference established" |
| H13: S vs flat popularity | `policy.primary.H13_new_hit_share` | `flat_explore` -> `usercf` (S = E2) | share of users with >= 1 new-item hit in the 10-item exploration list, users with >= 1 new target item | same rule as H12 |

Secondary (no confirmatory claim; CI only unless stated):

| PREREG v1.4 item 5 | JSON key | comparator -> model | metric, users |
|---|---|---|---|
| Recall@10 of P_T vs C3 | `policy.secondary.P_T_vs_final_lr_ucf_recall` | `final_lr_ucf` -> `P_T` | Recall@10, all users |
| new-item Recall@10 of P_T vs C3 | `policy.secondary.P_T_vs_final_lr_ucf_new_recall` | `final_lr_ucf` -> `P_T` | new-item Recall@10, users with new target |
| repeat Recall@10 of P_T vs C3 | `policy.secondary.P_T_vs_final_lr_ucf_repeat_recall` | `final_lr_ucf` -> `P_T` | repeat Recall@10, users with repeat target |
| new-item hit share over new-target users | `policy.secondary.P_T_vs_final_lr_ucf_new_hit_share_new_users` | `final_lr_ucf` -> `P_T` | new-item hit share@10, users with new target |
| S vs flat popularity, new-item Recall@10 | `policy.secondary.usercf_vs_flat_explore_new_recall` (same numbers as H8) | `flat_explore` -> `usercf` | new-item Recall@10, users with new target |
| history-length bins | `policy.by_history_length[*]` (keys H11_ndcg, H12_new_hit_share, H13_new_hit_share and the four P_T-vs-C3 secondary keys; 95 % CI per bin; bins (0,4], (4,6], (6,10], (10,20], (20,inf); CJ (0,20], (20,50], (50,100], (100,inf)) | as named | as named |
| r_short = 8 variant (T = 4) | `policy.variants.P_T4_r8.comparisons.{ndcg, new_hit_share, recall, new_recall, repeat_recall, new_hit_share_new_users}` | `final_lr_ucf` -> `P_T4_r8` | as named |
| T = 6 sensitivity | `policy.variants.P_T6_r9.comparisons.*` | `final_lr_ucf` -> `P_T6_r9` | as named |
| T = 10 sensitivity | `policy.variants.P_T10_r9.comparisons.*` | `final_lr_ucf` -> `P_T10_r9` | as named |
| list means, user counts | `policy.list_metrics.{final_lr_ucf, P_T, usercf, flat_explore, P_T4_r8, P_T6_r9, P_T10_r9}`, `policy.n_users` (all, new_target, repeat_target, short_history, short_history_by_T) | | |
| H6 replication (R2 vs R1) | `lr_repeat.primary.H6_repeat_recall` | `repeat_star` -> `repeat_lr` | repeat Recall@10; decision at 0.95 / 0.99375 as written by the code |
| H7' replication (C3 vs C0) | `integration.secondary.final_lr_ucf_vs_final_ndcg`, `integration.secondary.final_lr_ucf_vs_final_recall` | `final` -> `final_lr_ucf` | NDCG@10, Recall@10; decision as written by the code |
| H8 replication (E2 vs flat popularity) | `integration.primary.H8_new_recall` | `flat_explore` -> `usercf` | new-item Recall@10; decision as written by the code |
| H1-H5, H7, H9 replication and the other CI-only comparisons | `primary.*`, `H5`, `secondary.*`, `lr_repeat.*`, `integration.*`, `by_history_length` (as in reports/final2/RUN.md) | as named | as named |
| CJ refit / transfer | the same keys in the two Complete Journey files (the policy block included) | | |

Not computed: `explore_L3` (`--skip-l3`, PREREG v1.3 item 6); R3 (not carried forward).

Selections to be fixed after command 1 (from `holdout3_instacart_tuning.json`): `selections.{gamma, gamma_c, alpha, r}`
and `at_grid_edge.{gamma, gamma_c, alpha, r}`; `integration.selections.{k, weighting, C, r.{final_lr_ucf, final_star_ucf, final_lr_r}}`
and `integration.at_grid_edge.{k, C, r.*}`; `lr_repeat.C` (grid of `integration.lr.C_grid`). The policy changes no
selection (PREREG v1.4 item 3): r* of `final_lr_ucf` is the `r_default` of P_T.

## Pre-checks (2026-11-07, commit `df49c96`, working tree clean, `NBR_DATA_DIR=C:\dev_ML\data\processed`)

Code commit used for the commands: HEAD of this branch at command time (recorded per command in the commands
table); `src/` was last changed in `0c7aa89` (src/nbr/policy.py, pipeline.py, __main__.py) and `configs/` in
`583675d` (configs/holdout3.yaml). Runner records: `logs/prechecks.json` (UTC start/end, wall time, exit code,
peak working set of the process tree sampled every 0.5 s, HEAD, dirty flag), `logs/cmdcheck.log`,
`logs/cmdguard_direct.log`, `logs/cmdguard.log`.

1. `python -m nbr check --profile holdout3` (exit 0, 2.6 s, peak working set 1302.0 MiB; reads the fit split only):

```
profile=holdout3 role=final data_dir=C:\dev_ML\data\processed
instacart/fit: users=20000 history_rows=2971602 target_rows=206517 users_with_repeat_target=18619 users_with_new_target=16304
instacart/eval: skipped (holdout guard)
```

2. Holdout guard on the holdout3 config, direct call (`load_split(load_config('holdout3'), 'instacart', 'eval')`
   without `allow_holdout_eval`), exit 1 before any Parquet file is opened:

```
role final data_dir C:\dev_ML\data\processed
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\dev_ML-wt-w17\src\nbr\data.py", line 49, in load_split
    raise PermissionError(
PermissionError: profile role is 'final': the Instacart eval split is read once, after all selections are fixed; pass allow_holdout_eval=True explicitly
```

3. `python -m nbr evaluate --profile holdout3 --dataset instacart --skip-l3 --policy` WITHOUT `--final`, run before
   command 1 (exit 1): the command stops in `load_selections` with
   `FileNotFoundError: ... reports\metrics\holdout3_instacart_tuning.json` because `run_evaluate` loads the
   tuning selections before `load_split`; the guard itself is therefore reached only once the tuning file
   exists (as in the holdout2 run, PREREG v1.3 run note 2). The same command is repeated after command 1 and
   before command 2 (see "Guard check after command 1" below); the eval split is never opened on this path.

## Selections (written after command 1, before command 2)

Source: `reports/metrics/holdout3_instacart_tuning.json` written by command 1 (commit `fececaa`, fit split of
holdout3 Instacart: 20,000 users, 18,619 with a repeat target item, 16,304 with a new target item). Values as
written by the code and rendered from the JSON (no number typed by hand); the Instacart eval split has not been
read at this point. LR C: the validation scores of C = 0.1, 1 and 10 are equal at six decimals and the tie rule
of the code (smaller C) selects 0.1.

| Selection | Value | Grid | At grid edge | Selection score (fit split) |
|---|---|---|---|---|
| gamma* (R1 decay) | 0.8 | {0.5, 0.6, 0.7, 0.8, 0.9, 1} | no | repeat Recall@10 0.585560 (0.5: 0.554945, 0.6: 0.565323, 0.7: 0.577621, 0.9: 0.584469, 1: 0.562341) |
| gamma_c*, alpha* (E1) | 0.9, 100 | {0.8, 0.9, 1} x {1, 10, 100} | gamma_c no, alpha yes | new-item Recall@10 0.042159 (0.8/100: 0.041914, 1/100: 0.041603) |
| r* (C0) | 10 | {5, ..., 10} | yes | NDCG@10 0.422763 (r = 9: 0.412196) |
| LR C* (R2) | 0.1 | {0.001, 0.01, 0.1, 1, 10} | no | selected on 4,000 validation users (16,000 training users), refit on all fit users; validation repeat Recall@10 by C: 0.001: 0.593747, 0.01: 0.594519, 0.1: 0.594644, 1: 0.594644, 10: 0.594644 |
| user-CF k*, weighting* (E2) | 100, binary | {10, 20, 50, 100, 200, 400} x {binary, log1p} | k no | new-item Recall@10 0.060824 |
| r (C3 = final_lr_ucf) | 10 | {0, ..., 10} | yes | NDCG@10 0.424872 on the 4,000 LR validation users (train-only model); r = 9: 0.415275 |
| r (C3b = final_star_ucf) | 10 | {0, ..., 10} | yes | NDCG@10 0.423138 on the 20,000 all fit users; r = 9: 0.413489 |
| r (C2 = final_lr_r) | 10 | {0, ..., 10} | yes | NDCG@10 0.424612 on the 4,000 LR validation users (train-only model); r = 9: 0.414013 |

User-CF grid (fit new-item Recall@10, 16,304 users): binary k = 10: 0.045079, k = 20: 0.053142, k = 50: 0.059080, k = 100: 0.060824, k = 200: 0.060388, k = 400: 0.059359; log1p k = 10: 0.043251, k = 20: 0.052196, k = 50: 0.059104, k = 100: 0.060768, k = 200: 0.060687, k = 400: 0.060090.
`at_grid_edge` flags as written: three-stage {"gamma": false, "gamma_c": false, "alpha": true, "r": true}; integration {"k": false, "C": false, "r": {"final_lr_ucf": true, "final_star_ucf": true, "final_lr_r": true}}.
n_users (fit): all 20,000, repeat_target 18,619, new_target 16,304.
peak_memory_mib (in-JSON): {"stage1_repeat": 1588.8, "stage2_explore": 2218.5, "stage3_combine": 2224.3, "lr_repeat": 2477.9, "integration": 2490.8}; timing_seconds: {"prepare": 3.9, "stage1_repeat": 24.3, "stage2_explore": 156.3, "stage3_combine": 25.9, "lr_repeat": 12.4, "integration": 604.7}.

The policy (PREREG v1.4 item 3) changes no selection: P_T uses r_default = r (C3 = final_lr_ucf) = 10 and
r_short = 9 for users with history_len <= 4; S is the usercf list with k = 100, binary weighting.

Command 1 record (runner log `logs/commands.json`): start 2026-11-07T07:18:02.960Z, end 07:31:52.102Z, wall
829.1 s, exit 0, peak working set 2490.8 MiB (process tree sampled every 0.5 s; the JSON `peak_memory_mib.integration`
is 2490.8 MiB), HEAD `fececaa`, working tree clean.

## Guard check after command 1 (before command 2)

`python -m nbr evaluate --profile holdout3 --dataset instacart --skip-l3 --policy` WITHOUT `--final`, run after the selections
commit `4a88bc0` and before command 2 (`logs/guard_after_command1.log`; start 2026-11-07T07:32:29.661Z, end 2026-11-07T07:32:31.219Z,
1.6 s, exit 1, working tree clean): the tuning selections, the LR artifact and the integration
selections load, then `nbr.data.load_split` raises `PermissionError: profile role is 'final': the Instacart eval split is read
once, after all selections are fixed; pass allow_holdout_eval=True explicitly` before any eval Parquet file is opened.
Command 2 was started only after this message was found in the log (the launch step checked for it).

## Commands (fixed order, each executed exactly once)

Runner: `NBR_DATA_DIR=C:\dev_ML\data\processed`, `PYTHONPATH=src`, working directory `C:\dev_ML-wt-w17`,
Python `C:\dev_ML\.venv\Scripts\python.exe` (3.12.16). Peak memory = largest per-process peak working set in the
launched process tree, sampled every 0.5 s (the venv `python.exe` is a launcher; the interpreter is its child).
The in-JSON figure is the largest `peak_memory_mib` value the pipeline recorded in-process.

| # | Command | Start (UTC) | End (UTC) | Wall time | Exit | Peak working set (runner / in-JSON) | HEAD |
|---|---|---|---|---|---|---|---|
| 1 | `python -m nbr tune --profile holdout3 --dataset instacart` | 2026-11-07T07:18:02.960Z | 07:31:52.102Z | 829.1 s | 0 | 2490.8 / 2490.8 MiB | `fececaa` |
| - | selections written to this file and committed (`4a88bc0`); guard check (above) | 07:31:52 | 07:32:41 | - | - | - | - |
| 2 | `python -m nbr evaluate --profile holdout3 --dataset instacart --final --skip-l3 --policy` | 2026-11-07T07:32:41.958Z | 07:37:46.155Z | 304.2 s | 0 | 3986.8 / 3986.8 MiB | `4a88bc0` |
| 3 | `python -m nbr tune --profile holdout3 --dataset complete_journey` | 2026-11-07T07:37:46.186Z | 07:49:08.288Z | 680.2 s | 0 | 2551.9 / 2551.9 MiB | `4a88bc0` |
| 4 | `python -m nbr evaluate --profile holdout3 --dataset complete_journey --skip-l3 --policy` | 2026-11-07T07:49:08.321Z | 07:52:32.058Z | 203.7 s | 0 | 3297.1 / 3297.1 MiB | `4a88bc0` |
| 5 | `python -m nbr evaluate --profile holdout3 --dataset complete_journey --transfer-from instacart --skip-l3 --policy` | 2026-11-07T07:52:32.087Z | 07:56:34.729Z | 242.6 s | 0 | 3302.3 / 3302.3 MiB | `4a88bc0` |

Total wall time: 2259.8 s (37 min 40 s).

Total wall time of the five commands: 2259.8 s (37 min 40 s); commands 2-5 were chained so that a
failure would have stopped the chain (`--stop-on-failure`), and the chain was started only after the guard check passed.

## Re-run of command 2

None. Command 2 finished with exit code 0 on its first execution; the Instacart holdout3 eval split
(`holdout3/instacart/eval_history.parquet`, `eval_target.parquet`) was read exactly once, by command 2.
No command was re-run for any reason, and no technical fix was needed for commands 1, 3, 4 or 5.

## Complete Journey reproducibility check

Commands 3 and 4 use the same `medium` Complete Journey data as the W8b development run and the holdout2 run, so every key
of the two documents was compared with (a) the W8b medium files (`C:\dev_ML-wt-w8\reports\metrics\medium_complete_journey_{tuning,eval}.json`,
commit 6936873, user-CF k grid 20-400, `--skip-l3`; the comparison of reports/final2/RUN.md) and (b) the holdout2 run files
(`reports/final2/holdout2_complete_journey_{tuning,eval}.json`, same k grid as this run, no policy block). Differing paths are
listed by category: volatile = `meta`, `label`, `profile`, timing, peak memory, artifact and source paths; content = everything else.

- tuning vs W8b medium (`medium_complete_journey_tuning.json` -> `holdout3_complete_journey_tuning.json`): 37 differing paths, of which 31 volatile (meta/timing/memory/paths), 2 label/profile, 4 other content; 0 removed; 0 added.
    - `integration.at_grid_edge.k`: True -> False
    - `integration.label`: 'development (medium)' -> 'final (holdout3)'
    - `integration.usercf.at_grid_edge.k`: True -> False
    - `integration.usercf.grid`: 'len 10' -> 'len 12'
- evaluation vs W8b medium (`medium_complete_journey_eval.json` -> `holdout3_complete_journey_eval.json`): 53 differing paths, of which 49 volatile (meta/timing/memory/paths), 2 label/profile, 2 other content; 0 removed; 5 added (integration.secondary.final_lr_ucf_vs_final_ndcg.decision, integration.secondary.final_lr_ucf_vs_final_recall, peak_memory_mib.policy, policy, timing_seconds.policy_metrics).
    - `integration.label`: 'development (medium)' -> 'final (holdout3)'
    - `lr_repeat.label`: 'development (medium)' -> 'final (holdout3)'
- tuning vs holdout2 (reports/final2) (`holdout2_complete_journey_tuning.json` -> `holdout3_complete_journey_tuning.json`): 46 differing paths, of which 43 volatile (meta/timing/memory/paths), 2 label/profile, 1 other content; 0 removed; 0 added.
    - `integration.label`: 'final (holdout2)' -> 'final (holdout3)'
- evaluation vs holdout2 (reports/final2) (`holdout2_complete_journey_eval.json` -> `holdout3_complete_journey_eval.json`): 53 differing paths, of which 49 volatile (meta/timing/memory/paths), 2 label/profile, 2 other content; 0 removed; 3 added (peak_memory_mib.policy, policy, timing_seconds.policy_metrics).
    - `integration.label`: 'final (holdout2)' -> 'final (holdout3)'
    - `lr_repeat.label`: 'final (holdout2)' -> 'final (holdout3)'

- Transfer evaluation (command 5) applies the Instacart holdout3 selections (gamma = 0.8, gamma_c = 0.9,
  alpha = 100, r = 10; LR C = 0.1; user-CF k = 100 binary; r = 10, 10, 10 for the three combinations) and the same policy to the CJ data.

## Primary results as written by the code (Instacart holdout3, `holdout3_instacart_eval.json`)

20,000 users; 18,738 with a repeat target item; 16,340 with a new target item; 4,252 users with history_len <= 4 (T = 6: 7,155; T = 10: 10,837). diff = model - comparator; the 98.33 % interval is the pre-registered one (PREREG v1.4 item 4); r* = 10, r_short = 9, delta = 0.003.

| PREREG v1.4 | key | n_users | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision 95 % / 98.33 % |
|---|---|---|---|---|---|---|---|---|
| H11 (P_T vs C3, NDCG@10; non-inferior if the 98.33 % lower bound > -0.003) | `policy.primary.H11_ndcg` | 20,000 | 0.427448 | 0.426121 | -0.001327 | [-0.001519, -0.001141] | [-0.001552, -0.001100] | non-inferior / non-inferior |
| H12 (P_T vs C3, new-item hit share@10, all users) | `policy.primary.H12_new_hit_share` | 20,000 | 0.004150 | 0.014250 | +0.010100 | [0.008749, 0.011500] | [0.008450, 0.011783] | improvement observed / improvement observed |
| H13 (S = E2 vs flat popularity, new-item hit share@10, users with a new target item) | `policy.primary.H13_new_hit_share` | 16,340 | 0.168115 | 0.215116 | +0.047001 | [0.042043, 0.052020] | [0.041331, 0.052794] | improvement observed / improvement observed |

Secondary entries (P_T-vs-C3 Recall@10 / new-item Recall@10 / repeat Recall@10, history-length bins, the r_short = 8 variant,
T = 6 / 10, the H6, H7', H8 and H1-H5 replication, CJ refit and transfer) are read from the JSON files listed below; they carry
no confirmatory claim. The consolidated tables (`reports/results/`, T1 column `tier`) classify every row by the explicit
PREREG v1.4 table in scripts/results_sources.yaml.

## Files

`holdout3_*.json` are byte-identical copies of `reports/metrics/holdout3_*.json` written by the commands above
(verified with `cmp`); `SHA256SUMS` lists their SHA-256 digests (`sha256sum -c SHA256SUMS` from this directory).
`logs/` holds the runner records: `commands.json` (commands 1-5: UTC start/end, wall time, exit code, peak
memory), `cmd1.log` ... `cmd5.log` (stdout+stderr of each command), `prechecks.json`, `cmdcheck.log`,
`cmdguard_direct.log`, `cmdguard.log` (pre-checks), `guard_after_command1.json` and `guard_after_command1.log`.

