# Confirmatory run of the two-CF final system on holdout4 (prereg-v1.5)

Pre-registration: docs/PREREG.md v1.5 (main tag `prereg-v1.5`, commit bb826a6; the v1.5 amendment is commit 04deff4,
the base of this branch `feat/w19-icf-final`) and v1.5.1 (main tag `prereg-v1.5.1`, commit 6c3e6bd, merged into this branch
as `27d64f6`; secondary replication of the v1.4 exploration policy with C4 as the base). Chronology: v1.5.1 was committed
after the holdout4 data were created (`d7d23a0`, make_holdout.py run at HEAD 655bb32) and before any holdout4 command was run,
i.e. before the holdout4 Instacart eval split was read. The v1.5.1 text was drafted as `C:\dev_ML\Claude outputs\prereg_v1.5.1.md`
(mtime 2026-11-07T12:09:21Z, 1,330 bytes, LF line endings, SHA-256 70da79c658fec8989cf95ad913630b33bacd077b180fd6358000a6ae90d00710); its
bytes equal the five lines added to docs/PREREG.md in 6c3e6bd (compared after LF normalisation; the committed file ends with them).
C4 (PREREG v1.5) = R2' + E2 (`final_lri_ucf`); the label C4 had denoted a planned, never evaluated combination E3 + R2 in the
consolidated tables before v1.5 (T4 placeholder rows "not run", T6 note "E3 + R2 not implemented"); that placeholder was never run.
The sections "Pre-specified output keys", "Pre-checks" and "Selections"
were committed before the holdout4 Instacart eval split was read (three commits, listed in the "Commit record" section at the end of this
file, which was written after command 5); the sections from "Guard check after command 1" on were written after command 5.
Same procedure and failure policy as reports/final3/RUN.md: five commands, each executed exactly once, chained;
a failure in command 2 stops everything (no partial output is opened, no re-run); other commands admit technical
fixes only, recorded here.

Commits of this branch before any holdout4 run (all before command 1):

| Commit | Content |
|---|---|
| `655bb32` | src/nbr/features_icf.py (icf feature), src/nbr/lr_repeat.py (optional icf column, R2' bundles), src/nbr/v15.py (H14, H15 block), src/nbr/pipeline.py (R2' fit, C4 slot selection, lists repeat_lri / final_lri_ucf, "v15" blocks), configs/common.yaml (`lr_repeat.itemcf_feature: false`), scripts/check_icf_dev.py, tests (301 collected, all passed), sample smoke check (every existing key unchanged in content; only the v15 blocks and the new list timings added) |
| `d4000e7` | reports/posthoc/icf_dev_medium.{json,md}: development check of the PREREG v1.5 code on medium (not confirmatory) |
| `d7d23a0` | configs/holdout4.yaml; holdout4 data (data/SOURCE.md, data/MANIFEST.sha256, reports/holdout_prep/holdout4_selection.json, README make_holdout line) |
| `27d64f6` | merge of main (docs/PREREG.md v1.5.1 only) |
| (this file) | pre-specified key table and pre-check record |

Fixed by PREREG v1.5 and not re-selected: the icf definition (icf(u, i) = log(1 + sum over j in I(H_u) of sim(j, i)), sim from the
top-20 cosine item-neighbour lists of the binary user x item matrix of the scored history, self excluded; `explore.itemcf.neighbors` = 20),
delta = 0.003 (the v1.4 margin), CI levels 0.95 and 0.975 (= 1 - 0.05/2, family of 2; `configs/holdout4.yaml` `bootstrap.levels`, so every block
of this run carries the 97.5 % level; the v15 block levels are constants of nbr.v15); paired bootstrap over users, 1,000 resamples, seed 42
(`bootstrap.n`, `bootstrap.seed`). Systems: R2 = `repeat_lr` (PREREG v1.3); R2' = `repeat_lri` (R2 with the icf feature appended, 13 features;
neighbours computed on the history being scored; C selected separately from R2 on the same validation users); E2 = `usercf`; C3 =
`final_lr_ucf` (R2 + E2, r re-selected on the fit split, PREREG v1.3); C4 = `final_lri_ucf` = combine(R2', E2, r) with r re-selected the same way
as C3's (train-only R2' bundle, LR validation users, NDCG@10); comparator of H15 = `personal_frequency` (padded with popularity).

Environment: Python 3.12.16, numpy 1.26.4, pandas 2.2.3 (recorded in each JSON `meta`), Windows 11.
Data: `NBR_DATA_DIR=C:\dev_ML\data\processed`; profile `holdout4` (Instacart holdout4, 20,000 users disjoint from
sample, medium, holdout, holdout2 and holdout3; Complete Journey uses the `medium` data via `complete_journey_profile: medium`).

## Pre-specified output keys (PREREG v1.5 -> JSON keys, read from the code before any run)

Files: `holdout4_instacart_eval.json` (Instacart, confirmatory), `holdout4_complete_journey_eval.json` (CJ refit, secondary),
`holdout4_complete_journey_eval_transfer.json` (CJ transfer, secondary). Every entry of the `v15` block (written by
`nbr.v15.evaluate_v15`) carries `n_users`, `comparator_mean`, `model_mean`, `diff` (= model - comparator) and `ci` at both levels
`0.95` and `0.975`; the two primary entries also carry `decision` at both levels, the `rule` text and (H14) `delta`. Entries of the
other blocks (`primary`, `H5`, `secondary`, `lr_repeat`, `integration`) are written exactly as in the holdout3 run except that the
profile's bootstrap levels are `0.95` and `0.975` (instead of `0.99375`); no `policy` block is written (`--policy` is not passed).

Primary (Instacart holdout4, family of 2; the 97.5 % interval `ci.0.975` is the pre-registered one):

| PREREG v1.5 | JSON key | comparator -> model | metric, users | decision rule (`decision.0.975`) |
|---|---|---|---|---|
| H14 (non-inferiority): C4 vs C3 | `v15.primary.H14_ndcg` | `final_lr_ucf` (C3) -> `final_lri_ucf` (C4) | NDCG@10, all users | "non-inferior" if the lower bound > -0.003, else "non-inferiority not established" |
| H15: C4 vs personal frequency | `v15.primary.H15_ndcg` | `personal_frequency` -> `final_lri_ucf` (C4) | NDCG@10, all users | "improvement observed" if the lower bound > 0; "lower" if the upper bound < 0; else "no difference established" |

Secondary (no confirmatory claim; CI only unless stated):

| PREREG v1.5 item 7 | JSON key | comparator -> model | metric, users |
|---|---|---|---|
| repeat Recall@10 of R2' vs R2 | `v15.secondary.repeat_lri_vs_repeat_lr_repeat_recall` | `repeat_lr` -> `repeat_lri` | repeat Recall@10, users with >= 1 repeat target item |
| repeat Recall@10 of R2' vs R1 | `v15.secondary.repeat_lri_vs_repeat_star_repeat_recall` | `repeat_star` -> `repeat_lri` | repeat Recall@10, users with >= 1 repeat target item |
| Recall@10 of C4 vs C3 | `v15.secondary.final_lri_ucf_vs_final_lr_ucf_recall` | `final_lr_ucf` -> `final_lri_ucf` | Recall@10, all users |
| R2' C, standardized coefficients, intercept | `v15.lr_repeat_icf.{C, coefficients, intercept, features, icf_neighbors, artifact, sha256}` (eval); tuning `v15.lr_repeat_icf.*`, `v15.selections.C` | | 13 features: the 12 of R2 and `icf` |
| history-length bins of H14 | `v15.by_history_length[*].H14_ndcg` (95 % CI per bin; bins (0,4], (4,6], (6,10], (10,20], (20,inf); CJ (0,20], (20,50], (50,100], (100,inf)) | `final_lr_ucf` -> `final_lri_ucf` | NDCG@10 |
| list means, user counts | `v15.list_metrics.{final_lr_ucf, final_lri_ucf, personal_frequency, repeat_star, repeat_lr, repeat_lri}`, `v15.n_users`; `list_metrics.{repeat_lri, final_lri_ucf}` of the main document | | |
| H6 replication (R2 vs R1) | `lr_repeat.primary.H6_repeat_recall` | `repeat_star` -> `repeat_lr` | repeat Recall@10; decision at 0.95 / 0.975 as written by the code |
| H7' replication (C3 vs C0) | `integration.secondary.final_lr_ucf_vs_final_ndcg`, `integration.secondary.final_lr_ucf_vs_final_recall` | `final` -> `final_lr_ucf` | NDCG@10, Recall@10; decision as written by the code |
| H8 replication (E2 vs flat popularity) | `integration.primary.H8_new_recall` | `flat_explore` -> `usercf` | new-item Recall@10; decision as written by the code |
| H1-H5, H7, H9 replication and the other CI-only comparisons | `primary.*`, `H5`, `secondary.*`, `lr_repeat.*`, `integration.*`, `by_history_length` (as in reports/final3/RUN.md, levels 0.95 / 0.975) | as named | as named |
| CJ refit / transfer | the same keys in the two Complete Journey files (the v15 block included) | | |

Secondary replication of the v1.4 exploration policy (PREREG v1.5.1; `--policy` on commands 2, 4 and 5; written by
`nbr.policy.evaluate_policy` with base "C3" into the block `policy` and with base "C4" into the block `policy_c4`, same keys; both blocks carry
`label` = "secondary replication on holdout4 (PREREG v1.5.1)"; CI levels 0.95 and 0.9833 and the v1.4 decision wording as written by the
code; T = 4, delta = 0.003, r_short = 9; no confirmatory claim, the confirmatory policy result remains H11-H13 on holdout3):

| PREREG v1.5.1 | JSON key | comparator -> model | metric, users |
|---|---|---|---|
| NDCG@10 of P_T(C4) minus C4 (as H11) | `policy_c4.primary.H11_ndcg` | `final_lri_ucf` (C4) -> `P_T` (= combine_policy(repeat_lri, usercf, T=4, r_short=9, r_default=r of C4)) | NDCG@10, all users; wording "non-inferior" if the 98.33 % lower bound > -0.003 |
| new-item hit share of P_T(C4) minus C4 (as H12) | `policy_c4.primary.H12_new_hit_share` | `final_lri_ucf` -> `P_T` | share of users with >= 1 new-item hit in the top 10, all users |
| new-item hit share of S minus flat popularity (as H13) | `policy_c4.primary.H13_new_hit_share` (identical to `policy.primary.H13_new_hit_share`: S and the comparator do not depend on the base) | `flat_explore` -> `usercf` | users with >= 1 new target item |
| P_T(C4) vs C4 secondary metrics | `policy_c4.secondary.{P_T_vs_final_lri_ucf_recall, P_T_vs_final_lri_ucf_new_recall, P_T_vs_final_lri_ucf_repeat_recall, P_T_vs_final_lri_ucf_new_hit_share_new_users, usercf_vs_flat_explore_new_recall}` | `final_lri_ucf` -> `P_T` | as named |
| variants on C4 | `policy_c4.variants.{P_T4_r8, P_T6_r9, P_T10_r9}.comparisons.*` | `final_lri_ucf` -> variant | as named |
| bins, list means, user counts | `policy_c4.by_history_length[*]`, `policy_c4.list_metrics`, `policy_c4.n_users` | | |
| reference: the v1.4 policy on C3 | `policy.primary.{H11_ndcg, H12_new_hit_share, H13_new_hit_share}`, `policy.secondary.*`, `policy.variants.*`, `policy.by_history_length` (keys as in reports/final3/RUN.md) | `final_lr_ucf` (C3) -> `P_T` | as in v1.4 |
| CJ refit / transfer | the same two blocks in the two Complete Journey files | | |

Not computed: `explore_L3` (`--skip-l3`, PREREG v1.3 item 6); R3.

Selections to be fixed after command 1 (from `holdout4_instacart_tuning.json`): `selections.{gamma, gamma_c, alpha, r}` and
`at_grid_edge.{gamma, gamma_c, alpha, r}`; `integration.selections.{k, weighting, C, r.{final_lr_ucf, final_star_ucf, final_lr_r}}` and
`integration.at_grid_edge.{k, C, r.*}`; `lr_repeat.C` (R2, grid of `integration.lr.C_grid`); `v15.selections.{C, r.final_lri_ucf}` (R2' and C4)
and `v15.at_grid_edge.{C, r.final_lri_ucf}`; `v15.lr.same_validation_users_as_R2` must be true.

## Pre-checks (2026-11-07, commit `b9695db`, working tree clean, `NBR_DATA_DIR=C:\dev_ML\data\processed`)

Code commit used for the commands: HEAD of this branch at command time (recorded per command in the commands
table); `src/` was last changed in `655bb32` (features_icf.py, lr_repeat.py, v15.py, pipeline.py, policy.py, __main__.py) and `configs/`
in `d7d23a0` (configs/holdout4.yaml; common.yaml gained `lr_repeat.itemcf_feature: false` in `655bb32`). Runner records:
`logs/prechecks.json` (UTC start/end, wall time, exit code, peak working set of the process tree sampled every 0.5 s, HEAD, dirty
flag), `logs/cmdcheck.log`, `logs/cmdguard_direct.log`, `logs/cmdguard.log`.

1. `python -m nbr check --profile holdout4` (exit 0, 2.6 s, peak working set 1279.1 MiB; reads the fit split only):

```
profile=holdout4 role=final data_dir=C:\dev_ML\data\processed
instacart/fit: users=20000 history_rows=2921464 target_rows=208185 users_with_repeat_target=18594 users_with_new_target=16351
instacart/eval: skipped (holdout guard)
```

2. Holdout guard on the holdout4 config, direct call (`load_split(load_config('holdout4'), 'instacart', 'eval')`
   without `allow_holdout_eval`), exit 1 before any Parquet file is opened:

```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\dev_ML-wt-w19\src\nbr\data.py", line 49, in load_split
    raise PermissionError(
PermissionError: profile role is 'final': the Instacart eval split is read once, after all selections are fixed; pass allow_holdout_eval=True explicitly
role final data_dir C:\dev_ML\data\processed
```

3. `python -m nbr evaluate --profile holdout4 --dataset instacart --skip-l3 --policy` WITHOUT `--final`, run before
   command 1 (exit 1): the command stops in `load_selections` with
   `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\dev_ML-wt-w19\\reports\\metrics\\holdout4_instacart_tuning.json'`
   because `run_evaluate` loads the tuning selections before `load_split`; the guard itself is therefore reached only once the
   tuning file exists (as in the holdout2 and holdout3 runs). The same command is repeated after command 1 and before command 2
   (see "Guard check after command 1" below); the eval split is never opened on this path.

## Selections (written after command 1, before command 2)

Source: `reports/metrics/holdout4_instacart_tuning.json` written by command 1 (commit `ba0f387`, fit split of
holdout4 Instacart: 20,000 users, 18,594 with a repeat target item, 16,351 with a new target item). Values as
written by the code and rendered from the JSON (no number typed by hand); the Instacart eval split has not been
read at this point. LR C of R2: the validation scores of C = 0.01, 0.1, 1 and 10 are equal at six decimals and the tie rule
of the code (smaller C) selects 0.01; LR C of R2' is selected separately on the same validation users (0.01, the largest
validation score). r of C4 = 10 (grid edge, as for C3).

| Selection | Value | Grid | At grid edge | Selection score (fit split) |
|---|---|---|---|---|
| gamma* (R1 decay) | 0.8 | {0.5, 0.6, 0.7, 0.8, 0.9, 1} | no | repeat Recall@10 0.584655 (0.5: 0.554325, 0.6: 0.564993, 0.7: 0.577908, 0.9: 0.581982, 1: 0.559566) |
| gamma_c*, alpha* (E1) | 0.8, 100 | {0.8, 0.9, 1} x {1, 10, 100} | gamma_c yes, alpha yes | new-item Recall@10 0.042603 (0.9/100: 0.041975, 1/100: 0.041681) |
| r* (C0) | 10 | {5, ..., 10} | yes | NDCG@10 0.419983 (r = 9: 0.409567) |
| LR C* (R2) | 0.01 | {0.001, 0.01, 0.1, 1, 10} | no | selected on 4,000 validation users (16,000 training users), refit on all fit users; validation repeat Recall@10 by C: 0.001: 0.595092, 0.01: 0.595750, 0.1: 0.595750, 1: 0.595750, 10: 0.595750 |
| user-CF k*, weighting* (E2) | 100, log1p | {10, 20, 50, 100, 200, 400} x {binary, log1p} | k no | new-item Recall@10 0.059911 |
| r (C3 = final_lr_ucf) | 10 | {0, ..., 10} | yes | NDCG@10 0.432412 on the 4,000 LR validation users (train-only model); r = 9: 0.422475 |
| r (C3b = final_star_ucf) | 10 | {0, ..., 10} | yes | NDCG@10 0.420410 on the 20,000 all fit users; r = 9: 0.411121 |
| r (C2 = final_lr_r) | 10 | {0, ..., 10} | yes | NDCG@10 0.432010 on the 4,000 LR validation users (train-only model); r = 9: 0.420744 |
| LR C* (R2' = repeat_lri) | 0.01 | {0.001, 0.01, 0.1, 1, 10} | no | selected on the same 4,000 validation users (16,000 training users; same_validation_users_as_R2 = True), refit on all fit users; validation repeat Recall@10 by C: 0.001: 0.595192, 0.01: 0.595711, 0.1: 0.595690, 1: 0.595690, 10: 0.595690 |
| r (C4 = final_lri_ucf) | 10 | {0, ..., 10} | yes | NDCG@10 0.432427 on the 4,000 LR validation users (train-only model); r = 9: 0.422627 |

User-CF grid (fit new-item Recall@10, 16,351 users): binary k = 10: 0.043325, k = 20: 0.051553, k = 50: 0.059305, k = 100: 0.059661, k = 200: 0.059722, k = 400: 0.058322; log1p k = 10: 0.041530, k = 20: 0.051530, k = 50: 0.059292, k = 100: 0.059911, k = 200: 0.059478, k = 400: 0.058537.
`at_grid_edge` flags as written: three-stage {"gamma": false, "gamma_c": true, "alpha": true, "r": true}; integration {"k": false, "C": false, "r": {"final_lr_ucf": true, "final_star_ucf": true, "final_lr_r": true}}.
n_users (fit): all 20,000, repeat_target 18,594, new_target 16,351.
peak_memory_mib (in-JSON): {"stage1_repeat": 1659.6, "stage2_explore": 2293.9, "stage3_combine": 2299.2, "lr_repeat": 2539.8, "lr_repeat_icf": 2542.7, "integration": 2554.1, "v15": 2561.5}; timing_seconds: {"prepare": 5.1, "stage1_repeat": 43.4, "stage2_explore": 266.8, "stage3_combine": 48.6, "lr_repeat": 24.1, "lr_repeat_icf": 59.0, "integration": 968.9, "v15": 52.0}.
v15 `at_grid_edge` flags as written: {"C": false, "r": {"final_lri_ucf": true}}; icf_neighbors = 20; R2' features: ['dec_g0.5', 'dec_g0.8', 'dec_g1.0', 'recency', 'mean_gap', 'once', 'overdue', 'item_repeat_rate', 'log_pop', 'user_repeat_rate', 'log_hist_len', 'share', 'icf'].
R2' standardized coefficients (fit split refit, C = 0.01, intercept -2.851930): dec_g0.5 +0.1648, dec_g0.8 +0.3005, dec_g1.0 +0.0719, recency -0.3052, mean_gap -0.2163, once -0.2029, overdue -0.4241, item_repeat_rate +0.2396, log_pop +0.0782, user_repeat_rate +0.2064, log_hist_len -0.2276, share +0.1032, icf -0.0023.
R2' artifact: `instacart_lr_repeat_icf.joblib` sha256 24433c2d1281d80cc517cb70b5a3488b16d74da212c48b605ae171fb71ef1e74; R2 artifact: `instacart_lr_repeat.joblib` sha256 e438e1647c0b1d8e856e758dd9f42deb8e44ed5b6ae1069a36795c2a9ea6bdbb.

The PREREG v1.5.1 policy replication changes no selection: P_T (base C3) uses r_default = r (C3 = final_lr_ucf) = 10
and P_T(C4) (base C4) uses r_default = r (C4 = final_lri_ucf) = 10; r_short = 9 for users with history_len <= 4;
S is the usercf list with k = 100, log1p weighting.

Command 1 record (runner log `logs/commands.json`): start 2026-11-07T13:20:32.200Z, end 2026-11-07T13:45:01.768Z, wall
1469.6 s, exit 0, peak working set 2561.5 MiB (process tree sampled every 0.5 s; the JSON
`peak_memory_mib.v15` is 2561.5 MiB), HEAD `ba0f387`, working tree clean.
