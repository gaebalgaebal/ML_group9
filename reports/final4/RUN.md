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

## Guard check after command 1 (before command 2)

`python -m nbr evaluate --profile holdout4 --dataset instacart --skip-l3 --policy` WITHOUT `--final`, run after the selections
commit `6fba95a` and before command 2 (`logs/guard_after_command1.log`; start 2026-11-07T13:45:44.971Z, end 2026-11-07T13:45:51.172Z,
6.2 s, exit 1, working tree clean): the tuning selections, the R2 and R2' artifacts, the integration and
v15 selections load, then `nbr.data.load_split` raises `PermissionError: profile role is 'final': the Instacart eval split is read
once, after all selections are fixed; pass allow_holdout_eval=True explicitly` before any eval Parquet file is opened.
Commands 2-5 (chained, `--stop-on-failure`) were started only after this message was found in the log.

## Incident: command 2 interrupted by the host (2026-11-07; written before any decision on a continuation)

- Command 2, `python -m nbr evaluate --profile holdout4 --dataset instacart --final --skip-l3 --policy`, started at
  2026-11-07T13:45:51.399Z at HEAD `6fba95a` with a clean working tree (`logs/cmd2_interrupted.log`, header lines only).
- At about 13:48:59Z (the time the harness wrote its kill marker; about 188 s after the start) the process and its runner were
  killed by the Claude Code harness because the machine was low on memory ("stopped because the system is running low on
  memory"); this was a host-level action, not an error of the command (free physical memory measured right afterwards: 6.8 GB of
  32 GB; the largest other process on the machine was Notion with 7.1 GB; command 2 of the holdout3 run had needed 3.99 GB
  peak). No exit code was recorded (`logs/commands.json` holds the command 1 record only); no
  `reports/metrics/holdout4_instacart_eval.json` exists, so there is no partial output and nothing was opened. Commands 3-5
  were never started.
- load_split was reached: NTFS last-access updates are enabled on the data volume, and the holdout4 eval files show
  `eval_history.parquet` accessed at 13:45:52.66Z and `eval_target.parquet` at 13:45:53.35Z, 1.3-2.0 s after the start of
  command 2. The holdout4 Instacart eval split was therefore read once into the memory of the command-2 process, which was
  then killed without writing any result.
- Per the failure policy of this run (same as reports/final3/RUN.md: a failure in command 2 stops everything, no partial output
  is opened, no re-run), the run stops here. A continuation would be a second read of the holdout4 Instacart eval split and is
  not executed on the executor's own initiative; it requires a decision recorded as a PREREG deviation before it is run.

## Deviation PREREG v1.5.2: one re-run of command 2 (written before the re-run)

- docs/PREREG.md v1.5.2 (main commit `0708061`, tag `prereg-v1.5.2`, merged into this branch as `5e24d2f`; `git diff --stat 6fba95a HEAD`
  lists only docs/PREREG.md and files under reports/final4/) authorises exactly one re-run of command 2 and the first runs of commands 3
  to 5, unchanged (code, configs, selections `6fba95a`, seeds, hypotheses, delta and CI levels), outside the host tool. If the re-run of
  command 2 fails before `reports/metrics/holdout4_instacart_eval.json` is written, no further run is made and H14, H15 and policy_c4
  are reported as not obtained. Every report states that the holdout4 Instacart eval split was loaded twice and that the first load
  produced no output.
- Evidence that the first command 2 produced no output: `logs/cmd2_interrupted.log` has 2 lines (the two runner header lines,
  no program output); `reports/metrics/` holds ['holdout4_instacart_tuning.json'] and no file matching `holdout4_*eval*` (0 eval outputs);
  `artifacts/holdout4/` holds only the two Instacart artifacts of command 1; `logs/commands.json` holds the command 1 record only.
  No holdout4 eval JSON was ever opened by the executor.
- Re-run plan: commands 2-5 of the pre-specified list, unchanged, chained with `--stop-on-failure` by the same runner
  (`run_cmds.py`, copied with `cmd2to5.json` to the git-ignored folder `artifacts/w19_runner/` so that the working tree stays clean),
  started by the project owner in a standalone PowerShell window (not a Claude Code shell), recording UTC start/end, wall time, exit
  code, peak working set and HEAD into `logs/commands.json` (appended after the command 1 record) and `logs/cmd2.log` ... `logs/cmd5.log`.
  HEAD at the re-run: the commit of this section (clean working tree). Expected outputs: `reports/metrics/holdout4_instacart_eval.json`,
  `holdout4_complete_journey_tuning.json`, `holdout4_complete_journey_eval.json`, `holdout4_complete_journey_eval_transfer.json`,
  `artifacts/holdout4/complete_journey_lr_repeat.joblib` and `complete_journey_lr_repeat_icf.joblib`.

## Guard check after command 1 (before command 2)

`python -m nbr evaluate --profile holdout4 --dataset instacart --skip-l3 --policy` WITHOUT `--final`, run after the selections
commit `6fba95a` and before command 2 (`logs/guard_after_command1.log`; start 2026-11-07T13:45:44.971Z, end 2026-11-07T13:45:51.172Z,
6.2 s, exit 1, working tree clean): the tuning selections, the R2 and R2' artifacts, the integration
and v15 selections load, then `nbr.data.load_split` raises `PermissionError: profile role is 'final': the Instacart eval split is read
once, after all selections are fixed; pass allow_holdout_eval=True explicitly` before any eval Parquet file is opened.
Command 2 was started only after this message was found in the log (the launch step checked for it).

## Commands (fixed order, each executed exactly once)

Runner: `NBR_DATA_DIR=C:\dev_ML\data\processed`, `PYTHONPATH=src`, working directory `C:\dev_ML-wt-w19`,
Python `C:\dev_ML\.venv\Scripts\python.exe` (3.12.16). Peak memory = largest per-process peak working set in the
launched process tree, sampled every 0.5 s (the venv `python.exe` is a launcher; the interpreter is its child).
The in-JSON figure is the largest `peak_memory_mib` value the pipeline recorded in-process.

| # | Command | Start (UTC) | End (UTC) | Wall time | Exit | Peak working set (runner / in-JSON) | HEAD |
|---|---|---|---|---|---|---|---|
| 1 | `python -m nbr tune --profile holdout4 --dataset instacart` | 2026-11-07T13:20:32.200Z | 13:45:01.768Z | 1469.6 s | 0 | 2561.5 / 2561.5 MiB | `ba0f387` |
| - | selections written to this file and committed (`6fba95a`); guard check (above) | 13:45:01 | 15:17:54 | - | - | - | - |
| 2 | `python -m nbr evaluate --profile holdout4 --dataset instacart --final --skip-l3 --policy` | 2026-11-07T15:17:54.019Z | 15:25:54.948Z | 480.9 s | 0 | 3635.6 / 3635.6 MiB | `5c40aab` |
| 3 | `python -m nbr tune --profile holdout4 --dataset complete_journey` | 2026-11-07T15:25:54.998Z | 15:40:50.486Z | 895.5 s | 0 | 2553.4 / 2553.4 MiB | `5c40aab` |
| 4 | `python -m nbr evaluate --profile holdout4 --dataset complete_journey --skip-l3 --policy` | 2026-11-07T15:40:50.543Z | 15:44:31.402Z | 220.9 s | 0 | 3300.0 / 3300.0 MiB | `5c40aab` |
| 5 | `python -m nbr evaluate --profile holdout4 --dataset complete_journey --transfer-from instacart --skip-l3 --policy` | 2026-11-07T15:44:31.446Z | 15:49:05.179Z | 273.7 s | 0 | 3300.0 / 3300.0 MiB | `5c40aab` |

Total wall time: 3340.6 s (55 min 41 s).

Total wall time of the five commands: 3340.6 s (55 min 41 s); commands 2-5 were chained so that a
failure would have stopped the chain (`--stop-on-failure`); the chain was started by the project owner in a standalone PowerShell
window (outside the host tool, PREREG v1.5.2) after the guard check and the deviation record above.

## Re-run of command 2

One, authorised by PREREG v1.5.2 (see the deviation section above). The first execution of command 2 (2026-11-07T13:45:51Z,
HEAD `6fba95a`) was terminated by the host tool after `load_split` had read the holdout4 Instacart eval split; it produced no
output and no result was seen. The re-run (command 2 of the table above) is the only execution that produced a result.
The holdout4 Instacart eval split (`holdout4/instacart/eval_history.parquet`, `eval_target.parquet`) was therefore loaded
twice and the first load produced no output. Commands 3, 4 and 5 ran once, and no technical fix was needed for them.

## Execution environment of commands 2-5 (PREREG v1.5.2 re-run) and the runner record

- Commands 2-5 were started by the project owner in a standalone PowerShell window outside Claude Code (not a Claude Code
  background or foreground shell), with the venv Python given explicitly to the runner, after the deviation commit `5c40aab`.
  Free physical memory reported by Win32_OperatingSystem just before the start was about 2 GB; other applications were closed
  during command 2 to free memory. Command 2 (the authorised re-run) finished with exit 0 (480.9 s, peak working set 3635.6 MiB).
- Runner: `logs/runner/run_cmds.py` (SHA-256 663cc537ecef4ea83ff9e543044fb837a6a9232ad0c13f06f50f15307859033f) with the command list
  `logs/runner/cmd2to5.json` (SHA-256 c6dfd16bcc67d953056fbf6f3291c76ba2307341c544a6a56a912514c4f524f3), byte-identical copies of
  `artifacts/w19_runner/run_cmds.py` and `cmd2to5.json` used for the chain (the same runner, with the command list of commands 1
  and of the pre-checks, ran command 1 and the pre-checks from this session). Command line:
  `python artifacts/w19_runner/run_cmds.py --cwd C:\dev_ML-wt-w19 --out-dir C:\dev_ML-wt-w19eportsinal4\logs --python
  C:\dev_ML\.venv\Scripts\python.exe --commands artifacts/w19_runner/cmd2to5.json --stop-on-failure`.
- `worktree_dirty` in the records: false for commands 1 and 2, true for commands 3-5. The runner writes its records into the
  tracked folder `reports/final4/logs/` (`commands.json` is tracked and gains a record after every command; `cmd<id>.log` files are
  new, untracked files), and the flag is `git status --porcelain` being non-empty at the start of a command, which includes
  untracked files; from the end of command 2 on it is therefore true by construction. Verified after the chain: `git status --short`
  listed only `reports/final4/logs/commands.json` (modified) and `cmd2.log` ... `cmd5.log` (untracked), and
  `git diff --stat 5c40aab -- src configs scripts data tests` was empty, i.e. no code, config, data or test file changed during
  the chain; HEAD was `5c40aab` for commands 2-5 and `ba0f387` for command 1.

## Complete Journey reproducibility check

Commands 3 and 4 use the same `medium` Complete Journey data as the W8b development run, the holdout3 run and the P1.4 development check of
this branch, so every key of the two documents was compared with (a) the W8b medium files (`C:\dev_ML-wt-w8\reports\metrics\medium_complete_journey_{tuning,eval}.json`,
commit 6936873, user-CF k grid 20-400, `--skip-l3`, CI levels 0.95 / 0.99375, no v15 or policy blocks), (b) the holdout3 run files
(`reports/final3/holdout3_complete_journey_{tuning,eval}.json`, same k grid, CI levels 0.95 / 0.99375, policy block, no v15 / policy_c4) and
(c) the P1.4 medium check of this branch (`reports/metrics/medium_complete_journey_{tuning,eval}.json` of the icf_dev run, commit of
`reports/posthoc/icf_dev_medium.json`; same code and the same feature flag, CI levels 0.95 / 0.99375 from common.yaml, `--policy`).
Differing paths are listed by category: volatile = `meta`, `label`, `profile`, timing, peak memory, artifact and source paths; content =
everything else. CI-level differences (`0.99375` -> `0.975`) are expected from `configs/holdout4.yaml` and are listed as such.

- tuning vs W8b medium (`medium_complete_journey_tuning.json` -> `holdout4_complete_journey_tuning.json`): 37 differing paths, of which 31 volatile (meta/timing/memory/paths), 3 label/profile, 0 CI-level (0.99375 -> 0.975), 3 other content; 0 removed; 5 added (peak_memory_mib.lr_repeat_icf, peak_memory_mib.v15, timing_seconds.lr_repeat_icf, timing_seconds.v15, v15).
    - `integration.at_grid_edge.k`: True -> False
    - `integration.usercf.at_grid_edge.k`: True -> False
    - `integration.usercf.grid`: 'len 10' -> 'len 12'
- evaluation vs W8b medium (`medium_complete_journey_eval.json` -> `holdout4_complete_journey_eval.json`): 53 differing paths, of which 49 volatile (meta/timing/memory/paths), 4 label/profile, 0 CI-level (0.99375 -> 0.975), 0 other content; 37 removed (integration.primary.H8_new_recall.ci.0.99375, integration.primary.H8_new_recall.decision.0.99375, integration.primary.H9_ndcg.ci.0.99375, integration.primary.H9_ndcg.decision.0.99375, integration.primary.H9_recall.ci.0.99375, integration.primary.H9_recall.decision.0.99375, integration.secondary.final_lr_r_vs_final_lr_ndcg.ci.0.99375, integration.secondary.final_lr_ucf_vs_final_ndcg.ci.0.99375, integration.secondary.final_lr_ucf_vs_personal_frequency_ndcg.ci.0.99375, integration.secondary.final_star_ucf_vs_final_ndcg.ci.0.99375, integration.secondary.usercf_vs_itemcf_flat_new_recall.ci.0.99375, lr_repeat.primary.H6_repeat_recall.ci.0.99375, lr_repeat.primary.H6_repeat_recall.decision.0.99375, lr_repeat.primary.H7_ndcg.ci.0.99375, lr_repeat.primary.H7_ndcg.decision.0.99375, lr_repeat.primary.H7_recall.ci.0.99375, lr_repeat.primary.H7_recall.decision.0.99375, lr_repeat.secondary.final_lr_vs_personal_frequency_ndcg.ci.0.99375, lr_repeat.secondary.repeat_lr_vs_repeat_pf_repeat_recall.ci.0.99375, primary.H1_repeat_recall.ci.0.99375, primary.H1_repeat_recall.decision.0.99375, primary.H2_new_recall.ci.0.99375, primary.H2_new_recall.decision.0.99375, primary.H3_ndcg.ci.0.99375, primary.H3_ndcg.decision.0.99375, primary.H3_recall.ci.0.99375, primary.H3_recall.decision.0.99375, primary.H4_category_hit.ci.0.99375, primary.H4_category_hit.decision.0.99375, secondary.explore_cf_vs_explore_star_new_recall.ci.0.99375, secondary.final_cf_vs_final_ndcg.ci.0.99375, secondary.final_cf_vs_final_recall.ci.0.99375, secondary.final_r10_vs_final_ndcg.ci.0.99375, secondary.final_r10_vs_final_recall.ci.0.99375, secondary.final_vs_popularity_ndcg.ci.0.99375, secondary.final_vs_popularity_recall.ci.0.99375, secondary.itemcf_flat_vs_explore_star_new_recall.ci.0.99375); 54 added (integration.primary.H8_new_recall.ci.0.975, integration.primary.H8_new_recall.decision.0.975, integration.primary.H9_ndcg.ci.0.975, integration.primary.H9_ndcg.decision.0.975, integration.primary.H9_recall.ci.0.975, integration.primary.H9_recall.decision.0.975, integration.secondary.final_lr_r_vs_final_lr_ndcg.ci.0.975, integration.secondary.final_lr_ucf_vs_final_ndcg.ci.0.975, integration.secondary.final_lr_ucf_vs_final_ndcg.decision, integration.secondary.final_lr_ucf_vs_final_recall, integration.secondary.final_lr_ucf_vs_personal_frequency_ndcg.ci.0.975, integration.secondary.final_star_ucf_vs_final_ndcg.ci.0.975, integration.secondary.usercf_vs_itemcf_flat_new_recall.ci.0.975, list_metrics.final_lri_ucf, list_metrics.repeat_lri, lr_repeat.primary.H6_repeat_recall.ci.0.975, lr_repeat.primary.H6_repeat_recall.decision.0.975, lr_repeat.primary.H7_ndcg.ci.0.975, lr_repeat.primary.H7_ndcg.decision.0.975, lr_repeat.primary.H7_recall.ci.0.975, lr_repeat.primary.H7_recall.decision.0.975, lr_repeat.secondary.final_lr_vs_personal_frequency_ndcg.ci.0.975, lr_repeat.secondary.repeat_lr_vs_repeat_pf_repeat_recall.ci.0.975, peak_memory_mib.final_lri_ucf, peak_memory_mib.policy, peak_memory_mib.policy_c4, peak_memory_mib.repeat_lri, peak_memory_mib.v15, policy, policy_c4, primary.H1_repeat_recall.ci.0.975, primary.H1_repeat_recall.decision.0.975, primary.H2_new_recall.ci.0.975, primary.H2_new_recall.decision.0.975, primary.H3_ndcg.ci.0.975, primary.H3_ndcg.decision.0.975, primary.H3_recall.ci.0.975, primary.H3_recall.decision.0.975, primary.H4_category_hit.ci.0.975, primary.H4_category_hit.decision.0.975, secondary.explore_cf_vs_explore_star_new_recall.ci.0.975, secondary.final_cf_vs_final_ndcg.ci.0.975, secondary.final_cf_vs_final_recall.ci.0.975, secondary.final_r10_vs_final_ndcg.ci.0.975, secondary.final_r10_vs_final_recall.ci.0.975, secondary.final_vs_popularity_ndcg.ci.0.975, secondary.final_vs_popularity_recall.ci.0.975, secondary.itemcf_flat_vs_explore_star_new_recall.ci.0.975, timing_seconds.list_final_lri_ucf, timing_seconds.list_repeat_lri, timing_seconds.policy_c4_metrics, timing_seconds.policy_metrics, timing_seconds.v15_metrics, v15).
- tuning vs holdout3 (reports/final3) (`holdout3_complete_journey_tuning.json` -> `holdout4_complete_journey_tuning.json`): 46 differing paths, of which 43 volatile (meta/timing/memory/paths), 3 label/profile, 0 CI-level (0.99375 -> 0.975), 0 other content; 0 removed; 5 added (peak_memory_mib.lr_repeat_icf, peak_memory_mib.v15, timing_seconds.lr_repeat_icf, timing_seconds.v15, v15).
- evaluation vs holdout3 (reports/final3) (`holdout3_complete_journey_eval.json` -> `holdout4_complete_journey_eval.json`): 60 differing paths, of which 55 volatile (meta/timing/memory/paths), 5 label/profile, 0 CI-level (0.99375 -> 0.975), 0 other content; 40 removed (integration.primary.H8_new_recall.ci.0.99375, integration.primary.H8_new_recall.decision.0.99375, integration.primary.H9_ndcg.ci.0.99375, integration.primary.H9_ndcg.decision.0.99375, integration.primary.H9_recall.ci.0.99375, integration.primary.H9_recall.decision.0.99375, integration.secondary.final_lr_r_vs_final_lr_ndcg.ci.0.99375, integration.secondary.final_lr_ucf_vs_final_ndcg.ci.0.99375, integration.secondary.final_lr_ucf_vs_final_ndcg.decision.0.99375, integration.secondary.final_lr_ucf_vs_final_recall.ci.0.99375, integration.secondary.final_lr_ucf_vs_final_recall.decision.0.99375, integration.secondary.final_lr_ucf_vs_personal_frequency_ndcg.ci.0.99375, integration.secondary.final_star_ucf_vs_final_ndcg.ci.0.99375, integration.secondary.usercf_vs_itemcf_flat_new_recall.ci.0.99375, lr_repeat.primary.H6_repeat_recall.ci.0.99375, lr_repeat.primary.H6_repeat_recall.decision.0.99375, lr_repeat.primary.H7_ndcg.ci.0.99375, lr_repeat.primary.H7_ndcg.decision.0.99375, lr_repeat.primary.H7_recall.ci.0.99375, lr_repeat.primary.H7_recall.decision.0.99375, lr_repeat.secondary.final_lr_vs_personal_frequency_ndcg.ci.0.99375, lr_repeat.secondary.repeat_lr_vs_repeat_pf_repeat_recall.ci.0.99375, primary.H1_repeat_recall.ci.0.99375, primary.H1_repeat_recall.decision.0.99375, primary.H2_new_recall.ci.0.99375, primary.H2_new_recall.decision.0.99375, primary.H3_ndcg.ci.0.99375, primary.H3_ndcg.decision.0.99375, primary.H3_recall.ci.0.99375, primary.H3_recall.decision.0.99375, primary.H4_category_hit.ci.0.99375, primary.H4_category_hit.decision.0.99375, secondary.explore_cf_vs_explore_star_new_recall.ci.0.99375, secondary.final_cf_vs_final_ndcg.ci.0.99375, secondary.final_cf_vs_final_recall.ci.0.99375, secondary.final_r10_vs_final_ndcg.ci.0.99375, secondary.final_r10_vs_final_recall.ci.0.99375, secondary.final_vs_popularity_ndcg.ci.0.99375, secondary.final_vs_popularity_recall.ci.0.99375, secondary.itemcf_flat_vs_explore_star_new_recall.ci.0.99375); 52 added (integration.primary.H8_new_recall.ci.0.975, integration.primary.H8_new_recall.decision.0.975, integration.primary.H9_ndcg.ci.0.975, integration.primary.H9_ndcg.decision.0.975, integration.primary.H9_recall.ci.0.975, integration.primary.H9_recall.decision.0.975, integration.secondary.final_lr_r_vs_final_lr_ndcg.ci.0.975, integration.secondary.final_lr_ucf_vs_final_ndcg.ci.0.975, integration.secondary.final_lr_ucf_vs_final_ndcg.decision.0.975, integration.secondary.final_lr_ucf_vs_final_recall.ci.0.975, integration.secondary.final_lr_ucf_vs_final_recall.decision.0.975, integration.secondary.final_lr_ucf_vs_personal_frequency_ndcg.ci.0.975, integration.secondary.final_star_ucf_vs_final_ndcg.ci.0.975, integration.secondary.usercf_vs_itemcf_flat_new_recall.ci.0.975, list_metrics.final_lri_ucf, list_metrics.repeat_lri, lr_repeat.primary.H6_repeat_recall.ci.0.975, lr_repeat.primary.H6_repeat_recall.decision.0.975, lr_repeat.primary.H7_ndcg.ci.0.975, lr_repeat.primary.H7_ndcg.decision.0.975, lr_repeat.primary.H7_recall.ci.0.975, lr_repeat.primary.H7_recall.decision.0.975, lr_repeat.secondary.final_lr_vs_personal_frequency_ndcg.ci.0.975, lr_repeat.secondary.repeat_lr_vs_repeat_pf_repeat_recall.ci.0.975, peak_memory_mib.final_lri_ucf, peak_memory_mib.policy_c4, peak_memory_mib.repeat_lri, peak_memory_mib.v15, policy_c4, primary.H1_repeat_recall.ci.0.975, primary.H1_repeat_recall.decision.0.975, primary.H2_new_recall.ci.0.975, primary.H2_new_recall.decision.0.975, primary.H3_ndcg.ci.0.975, primary.H3_ndcg.decision.0.975, primary.H3_recall.ci.0.975, primary.H3_recall.decision.0.975, primary.H4_category_hit.ci.0.975, primary.H4_category_hit.decision.0.975, secondary.explore_cf_vs_explore_star_new_recall.ci.0.975, secondary.final_cf_vs_final_ndcg.ci.0.975, secondary.final_cf_vs_final_recall.ci.0.975, secondary.final_r10_vs_final_ndcg.ci.0.975, secondary.final_r10_vs_final_recall.ci.0.975, secondary.final_vs_popularity_ndcg.ci.0.975, secondary.final_vs_popularity_recall.ci.0.975, secondary.itemcf_flat_vs_explore_star_new_recall.ci.0.975, timing_seconds.list_final_lri_ucf, timing_seconds.list_repeat_lri, timing_seconds.policy_c4_metrics, timing_seconds.v15_metrics, v15).
- tuning vs P1.4 medium icf check (this branch) (`medium_complete_journey_tuning.json` -> `holdout4_complete_journey_tuning.json`): 63 differing paths, of which 59 volatile (meta/timing/memory/paths), 4 label/profile, 0 CI-level (0.99375 -> 0.975), 0 other content; 0 removed; 0 added.
- evaluation vs P1.4 medium icf check (this branch) (`medium_complete_journey_eval.json` -> `holdout4_complete_journey_eval.json`): 79 differing paths, of which 72 volatile (meta/timing/memory/paths), 7 label/profile, 0 CI-level (0.99375 -> 0.975), 0 other content; 40 removed (integration.primary.H8_new_recall.ci.0.99375, integration.primary.H8_new_recall.decision.0.99375, integration.primary.H9_ndcg.ci.0.99375, integration.primary.H9_ndcg.decision.0.99375, integration.primary.H9_recall.ci.0.99375, integration.primary.H9_recall.decision.0.99375, integration.secondary.final_lr_r_vs_final_lr_ndcg.ci.0.99375, integration.secondary.final_lr_ucf_vs_final_ndcg.ci.0.99375, integration.secondary.final_lr_ucf_vs_final_ndcg.decision.0.99375, integration.secondary.final_lr_ucf_vs_final_recall.ci.0.99375, integration.secondary.final_lr_ucf_vs_final_recall.decision.0.99375, integration.secondary.final_lr_ucf_vs_personal_frequency_ndcg.ci.0.99375, integration.secondary.final_star_ucf_vs_final_ndcg.ci.0.99375, integration.secondary.usercf_vs_itemcf_flat_new_recall.ci.0.99375, lr_repeat.primary.H6_repeat_recall.ci.0.99375, lr_repeat.primary.H6_repeat_recall.decision.0.99375, lr_repeat.primary.H7_ndcg.ci.0.99375, lr_repeat.primary.H7_ndcg.decision.0.99375, lr_repeat.primary.H7_recall.ci.0.99375, lr_repeat.primary.H7_recall.decision.0.99375, lr_repeat.secondary.final_lr_vs_personal_frequency_ndcg.ci.0.99375, lr_repeat.secondary.repeat_lr_vs_repeat_pf_repeat_recall.ci.0.99375, primary.H1_repeat_recall.ci.0.99375, primary.H1_repeat_recall.decision.0.99375, primary.H2_new_recall.ci.0.99375, primary.H2_new_recall.decision.0.99375, primary.H3_ndcg.ci.0.99375, primary.H3_ndcg.decision.0.99375, primary.H3_recall.ci.0.99375, primary.H3_recall.decision.0.99375, primary.H4_category_hit.ci.0.99375, primary.H4_category_hit.decision.0.99375, secondary.explore_cf_vs_explore_star_new_recall.ci.0.99375, secondary.final_cf_vs_final_ndcg.ci.0.99375, secondary.final_cf_vs_final_recall.ci.0.99375, secondary.final_r10_vs_final_ndcg.ci.0.99375, secondary.final_r10_vs_final_recall.ci.0.99375, secondary.final_vs_popularity_ndcg.ci.0.99375, secondary.final_vs_popularity_recall.ci.0.99375, secondary.itemcf_flat_vs_explore_star_new_recall.ci.0.99375); 40 added (integration.primary.H8_new_recall.ci.0.975, integration.primary.H8_new_recall.decision.0.975, integration.primary.H9_ndcg.ci.0.975, integration.primary.H9_ndcg.decision.0.975, integration.primary.H9_recall.ci.0.975, integration.primary.H9_recall.decision.0.975, integration.secondary.final_lr_r_vs_final_lr_ndcg.ci.0.975, integration.secondary.final_lr_ucf_vs_final_ndcg.ci.0.975, integration.secondary.final_lr_ucf_vs_final_ndcg.decision.0.975, integration.secondary.final_lr_ucf_vs_final_recall.ci.0.975, integration.secondary.final_lr_ucf_vs_final_recall.decision.0.975, integration.secondary.final_lr_ucf_vs_personal_frequency_ndcg.ci.0.975, integration.secondary.final_star_ucf_vs_final_ndcg.ci.0.975, integration.secondary.usercf_vs_itemcf_flat_new_recall.ci.0.975, lr_repeat.primary.H6_repeat_recall.ci.0.975, lr_repeat.primary.H6_repeat_recall.decision.0.975, lr_repeat.primary.H7_ndcg.ci.0.975, lr_repeat.primary.H7_ndcg.decision.0.975, lr_repeat.primary.H7_recall.ci.0.975, lr_repeat.primary.H7_recall.decision.0.975, lr_repeat.secondary.final_lr_vs_personal_frequency_ndcg.ci.0.975, lr_repeat.secondary.repeat_lr_vs_repeat_pf_repeat_recall.ci.0.975, primary.H1_repeat_recall.ci.0.975, primary.H1_repeat_recall.decision.0.975, primary.H2_new_recall.ci.0.975, primary.H2_new_recall.decision.0.975, primary.H3_ndcg.ci.0.975, primary.H3_ndcg.decision.0.975, primary.H3_recall.ci.0.975, primary.H3_recall.decision.0.975, primary.H4_category_hit.ci.0.975, primary.H4_category_hit.decision.0.975, secondary.explore_cf_vs_explore_star_new_recall.ci.0.975, secondary.final_cf_vs_final_ndcg.ci.0.975, secondary.final_cf_vs_final_recall.ci.0.975, secondary.final_r10_vs_final_ndcg.ci.0.975, secondary.final_r10_vs_final_recall.ci.0.975, secondary.final_vs_popularity_ndcg.ci.0.975, secondary.final_vs_popularity_recall.ci.0.975, secondary.itemcf_flat_vs_explore_star_new_recall.ci.0.975).

- Transfer evaluation (command 5) applies the Instacart holdout4 selections (gamma = 0.8, gamma_c = 0.8,
  alpha = 100, r = 10; LR C = 0.01 for R2 and 0.01 for R2'; user-CF k = 100 log1p; r = 10, 10, 10 for the three v1.3 combinations and r = 10 for C4) and the
  Instacart R2 / R2' artifacts to the CJ data (CJ item neighbours for the icf feature of the eval history).

## Primary results as written by the code (Instacart holdout4, `holdout4_instacart_eval.json`)

20,000 users; 18,701 with a repeat target item; 16,335 with a new target item. diff = model - comparator; the 97.5 % interval is the pre-registered one (PREREG v1.5 item 6); r (C4) = 10, C of R2' = 0.01, delta = 0.003.

| PREREG v1.5 | key | n_users | comparator mean | model mean | diff | 95 % CI | 97.5 % CI | decision 95 % / 97.5 % |
|---|---|---|---|---|---|---|---|---|
| H14 (C4 vs C3, NDCG@10; non-inferior if the 97.5 % lower bound > -0.003) | `v15.primary.H14_ndcg` | 20,000 | 0.425787 | 0.425765 | -0.000022 | [-0.000110, 0.000068] | [-0.000127, 0.000083] | non-inferior / non-inferior |
| H15 (C4 vs personal frequency, NDCG@10, all users) | `v15.primary.H15_ndcg` | 20,000 | 0.393402 | 0.425765 | +0.032363 | [0.030749, 0.033998] | [0.030502, 0.034341] | improvement observed / improvement observed |

Secondary entries (R2' vs R2 and R2' vs R1 repeat Recall@10, C4 vs C3 Recall@10, the R2' coefficients, the history-length bins of H14,
the H6, H7', H8 and H1-H5 replication, the PREREG v1.5.1 policy replication (blocks `policy` and `policy_c4`), CJ refit and transfer) are
read from the JSON files listed below; they carry no confirmatory claim. The consolidated tables (`reports/results/`, T1 column `tier`)
classify every row by the explicit PREREG v1.5 table in scripts/results_sources.yaml.

## Secondary results as written by the code (no confirmatory claim)

| dataset | comparison | key | n_users | comparator mean | model mean | diff | 95 % CI | 97.5 % CI | decision |
|---|---|---|---|---|---|---|---|---|---|
| Instacart holdout4 repeat_lri vs repeat_lr: repeat Recall@10, users with repeat target | `v15.secondary.repeat_lri_vs_repeat_lr_repeat_recall` | 18,701 | 0.575034 | 0.574878 | -0.000156 | [-0.000351, 0.000035] | [-0.000380, 0.000063] | CI only |
| Instacart holdout4 repeat_lri vs repeat_star: repeat Recall@10, users with repeat target | `v15.secondary.repeat_lri_vs_repeat_star_repeat_recall` | 18,701 | 0.562791 | 0.574878 | +0.012087 | [0.010177, 0.014098] | [0.009862, 0.014438] | CI only |
| Instacart holdout4 final_lri_ucf vs final_lr_ucf: Recall@10, all users | `v15.secondary.final_lri_ucf_vs_final_lr_ucf_recall` | 20,000 | 0.352219 | 0.352147 | -0.000072 | [-0.000162, 0.000023] | [-0.000182, 0.000032] | CI only |
| Complete Journey refit H14_ndcg | `v15.primary.H14_ndcg` | 2,069 | 0.204637 | 0.204819 | +0.000182 | [-0.000363, 0.000724] | [-0.000453, 0.000795] | non-inferior / non-inferior |
| Complete Journey refit H15_ndcg | `v15.primary.H15_ndcg` | 2,069 | 0.185898 | 0.204819 | +0.018920 | [0.014679, 0.023148] | [0.013789, 0.023483] | improvement observed / improvement observed |
| Complete Journey refit repeat_lri vs repeat_lr: repeat Recall@10, users with repeat target | `v15.secondary.repeat_lri_vs_repeat_lr_repeat_recall` | 1,867 | 0.274010 | 0.273880 | -0.000130 | [-0.001169, 0.000888] | [-0.001295, 0.001011] | CI only |
| Complete Journey refit repeat_lri vs repeat_star: repeat Recall@10, users with repeat target | `v15.secondary.repeat_lri_vs_repeat_star_repeat_recall` | 1,867 | 0.249661 | 0.273880 | +0.024218 | [0.015560, 0.033293] | [0.014592, 0.034720] | CI only |
| Complete Journey refit final_lri_ucf vs final_lr_ucf: Recall@10, all users | `v15.secondary.final_lri_ucf_vs_final_lr_ucf_recall` | 2,069 | 0.131525 | 0.131625 | +0.000100 | [-0.000475, 0.000679] | [-0.000524, 0.000776] | CI only |
| Complete Journey transfer H14_ndcg | `v15.primary.H14_ndcg` | 2,069 | 0.199154 | 0.199280 | +0.000126 | [-0.000313, 0.000568] | [-0.000382, 0.000641] | non-inferior / non-inferior |
| Complete Journey transfer H15_ndcg | `v15.primary.H15_ndcg` | 2,069 | 0.185898 | 0.199280 | +0.013381 | [0.008404, 0.018227] | [0.007790, 0.018932] | improvement observed / improvement observed |
| Complete Journey transfer repeat_lri vs repeat_lr: repeat Recall@10, users with repeat target | `v15.secondary.repeat_lri_vs_repeat_lr_repeat_recall` | 1,867 | 0.268575 | 0.268324 | -0.000251 | [-0.001023, 0.000496] | [-0.001149, 0.000609] | CI only |
| Complete Journey transfer repeat_lri vs repeat_star: repeat Recall@10, users with repeat target | `v15.secondary.repeat_lri_vs_repeat_star_repeat_recall` | 1,867 | 0.219343 | 0.268324 | +0.048981 | [0.039679, 0.058639] | [0.038521, 0.060097] | CI only |
| Complete Journey transfer final_lri_ucf vs final_lr_ucf: Recall@10, all users | `v15.secondary.final_lri_ucf_vs_final_lr_ucf_recall` | 2,069 | 0.128100 | 0.128033 | -0.000067 | [-0.000509, 0.000356] | [-0.000574, 0.000400] | CI only |

R2' (repeat_lri) standardized coefficients and intercept as written (`v15.lr_repeat_icf`):

| dataset | C | intercept | dec_g0.5 | dec_g0.8 | dec_g1.0 | recency | mean_gap | once | overdue | item_repeat_rate | log_pop | user_repeat_rate | log_hist_len | share | icf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Instacart holdout4 | 0.01 | -2.8519 | +0.1648 | +0.3005 | +0.0719 | -0.3052 | -0.2163 | -0.2029 | -0.4241 | +0.2396 | +0.0782 | +0.2064 | -0.2276 | +0.1032 | -0.0023 |
| Complete Journey refit | 1 | -5.0381 | -0.1200 | +0.3105 | +0.0525 | -0.3018 | -0.3150 | -0.4549 | -0.3935 | +0.1669 | +0.1715 | +0.2431 | -0.1956 | +0.0585 | -0.0124 |
| Complete Journey transfer | 0.01 | -2.8519 | +0.1648 | +0.3005 | +0.0719 | -0.3052 | -0.2163 | -0.2029 | -0.4241 | +0.2396 | +0.0782 | +0.2064 | -0.2276 | +0.1032 | -0.0023 |

Replication (CI levels 0.95 / 0.975 as written by the code under the holdout4 profile; secondary under PREREG v1.5):

| dataset | hypothesis | key | n_users | comparator mean | model mean | diff | 95 % CI | 97.5 % CI | decision 95 % / 97.5 % |
|---|---|---|---|---|---|---|---|---|---|
| Instacart holdout4 H6 (R2 vs R1, repeat Recall@10) | `lr_repeat.primary.H6_repeat_recall` | 18,701 | 0.562791 | 0.575034 | +0.012243 | [0.010334, 0.014275] | [0.009955, 0.014563] | improvement observed / improvement observed |
| Instacart holdout4 H7' (C3 vs C0, NDCG@10) | `integration.secondary.final_lr_ucf_vs_final_ndcg` | 20,000 | 0.419702 | 0.425787 | +0.006085 | [0.005090, 0.007017] | [0.004991, 0.007189] | improvement observed / improvement observed |
| Instacart holdout4 H7' (C3 vs C0, Recall@10) | `integration.secondary.final_lr_ucf_vs_final_recall` | 20,000 | 0.346623 | 0.352219 | +0.005596 | [0.004376, 0.006754] | [0.004230, 0.007060] | improvement observed / improvement observed |
| Instacart holdout4 H8 (E2 vs flat popularity, new-item Recall@10) | `integration.primary.H8_new_recall` | 16,335 | 0.040838 | 0.058979 | +0.018140 | [0.016177, 0.019963] | [0.015820, 0.020202] | improvement observed / improvement observed |
| Complete Journey refit H6 (R2 vs R1, repeat Recall@10) | `lr_repeat.primary.H6_repeat_recall` | 1,867 | 0.249661 | 0.274010 | +0.024348 | [0.015796, 0.033308] | [0.015069, 0.034948] | improvement observed / improvement observed |
| Complete Journey refit H7' (C3 vs C0, NDCG@10) | `integration.secondary.final_lr_ucf_vs_final_ndcg` | 2,069 | 0.185898 | 0.204637 | +0.018739 | [0.014574, 0.022798] | [0.013863, 0.023329] | improvement observed / improvement observed |
| Complete Journey refit H7' (C3 vs C0, Recall@10) | `integration.secondary.final_lr_ucf_vs_final_recall` | 2,069 | 0.118726 | 0.131525 | +0.012799 | [0.008970, 0.016749] | [0.008373, 0.017133] | improvement observed / improvement observed |
| Complete Journey refit H8 (E2 vs flat popularity, new-item Recall@10) | `integration.primary.H8_new_recall` | 1,971 | 0.012976 | 0.014472 | +0.001496 | [-0.001410, 0.004407] | [-0.001769, 0.005021] | no difference established / no difference established |
| Complete Journey transfer H6 (R2 vs R1, repeat Recall@10) | `lr_repeat.primary.H6_repeat_recall` | 1,867 | 0.219343 | 0.268575 | +0.049231 | [0.039732, 0.058830] | [0.038370, 0.060474] | improvement observed / improvement observed |
| Complete Journey transfer H7' (C3 vs C0, NDCG@10) | `integration.secondary.final_lr_ucf_vs_final_ndcg` | 2,069 | 0.170418 | 0.199154 | +0.028736 | [0.024248, 0.033099] | [0.023833, 0.033678] | improvement observed / improvement observed |
| Complete Journey transfer H7' (C3 vs C0, Recall@10) | `integration.secondary.final_lr_ucf_vs_final_recall` | 2,069 | 0.106512 | 0.128100 | +0.021588 | [0.017796, 0.025810] | [0.017151, 0.026559] | improvement observed / improvement observed |
| Complete Journey transfer H8 (E2 vs flat popularity, new-item Recall@10) | `integration.primary.H8_new_recall` | 1,971 | 0.012976 | 0.015263 | +0.002287 | [-0.000485, 0.005044] | [-0.000865, 0.005211] | no difference established / no difference established |

History-length bins of H14 (C4 vs C3, NDCG@10; diff [95 % CI] (n)):

| bin | Instacart holdout4 | Complete Journey refit | Complete Journey transfer |
|---|---|---|---|
| 1 | (0,4], 4,316 users: +0.000006 [-0.000171, 0.000194] (4,316) | (0,20], 138 users: +0.000599 [-0.001139, 0.002503] (138) | (0,20], 138 users: -0.000013 [-0.000218, 0.000166] (138) |
| 2 | (4,6], 2,903 users: -0.000253 [-0.000668, 0.000065] (2,903) | (20,50], 547 users: -0.000787 [-0.001820, 0.000206] (547) | (20,50], 547 users: -0.000061 [-0.000935, 0.000923] (547) |
| 3 | (6,10], 3,737 users: -0.000033 [-0.000235, 0.000188] (3,737) | (50,100], 653 users: +0.000097 [-0.000916, 0.001068] (653) | (50,100], 653 users: +0.000074 [-0.000735, 0.000954] (653) |
| 4 | (10,20], 4,401 users: +0.000082 [-0.000078, 0.000241] (4,401) | (100,inf), 731 users: +0.000904 [0.000044, 0.001832] (731) | (100,inf), 731 users: +0.000339 [-0.000304, 0.001105] (731) |
| 5 | (20,inf), 4,643 users: +0.000006 [-0.000144, 0.000173] (4,643) | - | - |

## Exploration policy on the final system (PREREG v1.5.1, secondary replication on holdout4)

Blocks `policy` (base C3, the v1.4 policy repeated as a reference) and `policy_c4` (base C4) of the three eval files; T = 4, delta = 0.003, r_short = 9; CI levels 0.95 and 0.9833 and the v1.4 decision wording as written by the code. The confirmatory result for the policy remains H11-H13 on holdout3 (reports/final3/RUN.md).

| dataset | base | comparison | key | n_users | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision 95 % / 98.33 % |
|---|---|---|---|---|---|---|---|---|---|---|
| Instacart holdout4 | C4 as H11 (P_T vs base, NDCG@10; wording non-inferior if the 98.33 % lower bound > -0.003) | `policy_c4.primary.H11_ndcg` | 20,000 | 0.425765 | 0.424294 | -0.001470 | [-0.001671, -0.001249] | [-0.001712, -0.001184] | non-inferior / non-inferior |
| Instacart holdout4 | C4 as H12 (P_T vs base, new-item hit share@10, all users) | `policy_c4.primary.H12_new_hit_share` | 20,000 | 0.004900 | 0.016350 | +0.011450 | [0.010100, 0.012950] | [0.009750, 0.013200] | improvement observed / improvement observed |
| Instacart holdout4 | C4 as H13 (S = E2 vs flat popularity, new-item hit share@10, users with a new target item) | `policy_c4.primary.H13_new_hit_share` | 16,335 | 0.172635 | 0.224304 | +0.051668 | [0.046708, 0.056506] | [0.045690, 0.057502] | improvement observed / improvement observed |
| Instacart holdout4 | C3 as H11 (P_T vs base, NDCG@10; wording non-inferior if the 98.33 % lower bound > -0.003) | `policy.primary.H11_ndcg` | 20,000 | 0.425787 | 0.424287 | -0.001500 | [-0.001703, -0.001277] | [-0.001742, -0.001216] | non-inferior / non-inferior |
| Instacart holdout4 | C3 as H12 (P_T vs base, new-item hit share@10, all users) | `policy.primary.H12_new_hit_share` | 20,000 | 0.004900 | 0.016350 | +0.011450 | [0.010100, 0.012950] | [0.009750, 0.013200] | improvement observed / improvement observed |
| Instacart holdout4 | C3 as H13 (S = E2 vs flat popularity, new-item hit share@10, users with a new target item) | `policy.primary.H13_new_hit_share` | 16,335 | 0.172635 | 0.224304 | +0.051668 | [0.046708, 0.056506] | [0.045690, 0.057502] | improvement observed / improvement observed |
| Complete Journey refit | C4 as H11 (P_T vs base, NDCG@10; wording non-inferior if the 98.33 % lower bound > -0.003) | `policy_c4.primary.H11_ndcg` | 2,069 | 0.204819 | 0.204819 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | non-inferior / non-inferior |
| Complete Journey refit | C4 as H12 (P_T vs base, new-item hit share@10, all users) | `policy_c4.primary.H12_new_hit_share` | 2,069 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | no difference established / no difference established |
| Complete Journey refit | C4 as H13 (S = E2 vs flat popularity, new-item hit share@10, users with a new target item) | `policy_c4.primary.H13_new_hit_share` | 1,971 | 0.079148 | 0.097920 | +0.018772 | [0.006076, 0.031469] | [0.003725, 0.034500] | improvement observed / improvement observed |
| Complete Journey refit | C3 as H11 (P_T vs base, NDCG@10; wording non-inferior if the 98.33 % lower bound > -0.003) | `policy.primary.H11_ndcg` | 2,069 | 0.204637 | 0.204637 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | non-inferior / non-inferior |
| Complete Journey refit | C3 as H12 (P_T vs base, new-item hit share@10, all users) | `policy.primary.H12_new_hit_share` | 2,069 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | no difference established / no difference established |
| Complete Journey refit | C3 as H13 (S = E2 vs flat popularity, new-item hit share@10, users with a new target item) | `policy.primary.H13_new_hit_share` | 1,971 | 0.079148 | 0.097920 | +0.018772 | [0.006076, 0.031469] | [0.003725, 0.034500] | improvement observed / improvement observed |
| Complete Journey transfer | C4 as H11 (P_T vs base, NDCG@10; wording non-inferior if the 98.33 % lower bound > -0.003) | `policy_c4.primary.H11_ndcg` | 2,069 | 0.199280 | 0.199280 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | non-inferior / non-inferior |
| Complete Journey transfer | C4 as H12 (P_T vs base, new-item hit share@10, all users) | `policy_c4.primary.H12_new_hit_share` | 2,069 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | no difference established / no difference established |
| Complete Journey transfer | C4 as H13 (S = E2 vs flat popularity, new-item hit share@10, users with a new target item) | `policy_c4.primary.H13_new_hit_share` | 1,971 | 0.079148 | 0.105530 | +0.026383 | [0.016743, 0.037037] | [0.014206, 0.040081] | improvement observed / improvement observed |
| Complete Journey transfer | C3 as H11 (P_T vs base, NDCG@10; wording non-inferior if the 98.33 % lower bound > -0.003) | `policy.primary.H11_ndcg` | 2,069 | 0.199154 | 0.199154 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | non-inferior / non-inferior |
| Complete Journey transfer | C3 as H12 (P_T vs base, new-item hit share@10, all users) | `policy.primary.H12_new_hit_share` | 2,069 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | no difference established / no difference established |
| Complete Journey transfer | C3 as H13 (S = E2 vs flat popularity, new-item hit share@10, users with a new target item) | `policy.primary.H13_new_hit_share` | 1,971 | 0.079148 | 0.105530 | +0.026383 | [0.016743, 0.037037] | [0.014206, 0.040081] | improvement observed / improvement observed |

Secondary metrics and variants of the C4 base (CI only):

| dataset | comparison | key | n_users | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision |
|---|---|---|---|---|---|---|---|---|---|
| Instacart holdout4 P_T vs final_lri_ucf: Recall@10, all users | `policy_c4.secondary.P_T_vs_final_lri_ucf_recall` | 20,000 | 0.352147 | 0.350228 | -0.001919 | [-0.002349, -0.001488] | [-0.002442, -0.001385] | CI only |
| Instacart holdout4 P_T vs final_lri_ucf: new-item Recall@10, users with new target | `policy_c4.secondary.P_T_vs_final_lri_ucf_new_recall` | 16,335 | 0.002806 | 0.005391 | +0.002585 | [0.002112, 0.003115] | [0.001986, 0.003290] | CI only |
| Instacart holdout4 P_T vs final_lri_ucf: repeat Recall@10, users with repeat target | `policy_c4.secondary.P_T_vs_final_lri_ucf_repeat_recall` | 18,701 | 0.574878 | 0.568152 | -0.006726 | [-0.007429, -0.005983] | [-0.007605, -0.005824] | CI only |
| Instacart holdout4 P_T vs final_lri_ucf: new-item hit share@10, users with new target | `policy_c4.secondary.P_T_vs_final_lri_ucf_new_hit_share_new_users` | 16,335 | 0.005999 | 0.020018 | +0.014019 | [0.012242, 0.015857] | [0.011815, 0.016589] | CI only |
| Instacart holdout4 usercf vs flat_explore: new-item Recall@10, users with new target | `policy_c4.secondary.usercf_vs_flat_explore_new_recall` | 16,335 | 0.040838 | 0.058979 | +0.018140 | [0.016177, 0.019963] | [0.015745, 0.020360] | CI only |
| Instacart holdout4 P_T4_r8 (T = 4, r_short = 8, 4,316 short users) vs C4: NDCG@10, all users | `policy_c4.variants.P_T4_r8.comparisons.ndcg` | 20,000 | 0.425765 | 0.422575 | -0.003190 | [-0.003511, -0.002876] | [-0.003600, -0.002786] | CI only |
| Instacart holdout4 P_T4_r8 (T = 4, r_short = 8, 4,316 short users) vs C4: new-item hit share@10, all users | `policy_c4.variants.P_T4_r8.comparisons.new_hit_share` | 20,000 | 0.004900 | 0.023650 | +0.018750 | [0.016949, 0.020500] | [0.016450, 0.021033] | CI only |
| Instacart holdout4 P_T4_r8 (T = 4, r_short = 8, 4,316 short users) vs C4: Recall@10, all users | `policy_c4.variants.P_T4_r8.comparisons.recall` | 20,000 | 0.352147 | 0.347964 | -0.004183 | [-0.004777, -0.003635] | [-0.004947, -0.003501] | CI only |
| Instacart holdout4 P_T4_r8 (T = 4, r_short = 8, 4,316 short users) vs C4: new-item Recall@10, users with new target | `policy_c4.variants.P_T4_r8.comparisons.new_recall` | 16,335 | 0.002806 | 0.006862 | +0.004056 | [0.003488, 0.004718] | [0.003352, 0.004819] | CI only |
| Instacart holdout4 P_T4_r8 (T = 4, r_short = 8, 4,316 short users) vs C4: repeat Recall@10, users with repeat target | `policy_c4.variants.P_T4_r8.comparisons.repeat_recall` | 18,701 | 0.574878 | 0.561744 | -0.013134 | [-0.014185, -0.012137] | [-0.014448, -0.011884] | CI only |
| Instacart holdout4 P_T4_r8 (T = 4, r_short = 8, 4,316 short users) vs C4: new-item hit share@10, users with new target | `policy_c4.variants.P_T4_r8.comparisons.new_hit_share_new_users` | 16,335 | 0.005999 | 0.028956 | +0.022957 | [0.020690, 0.025589] | [0.020141, 0.025895] | CI only |
| Instacart holdout4 P_T6_r9 (T = 6, r_short = 9, 7,219 short users) vs C4: NDCG@10, all users | `policy_c4.variants.P_T6_r9.comparisons.ndcg` | 20,000 | 0.425765 | 0.423235 | -0.002530 | [-0.002774, -0.002251] | [-0.002837, -0.002186] | CI only |
| Instacart holdout4 P_T6_r9 (T = 6, r_short = 9, 7,219 short users) vs C4: new-item hit share@10, all users | `policy_c4.variants.P_T6_r9.comparisons.new_hit_share` | 20,000 | 0.004900 | 0.023650 | +0.018750 | [0.016899, 0.020750] | [0.016550, 0.021083] | CI only |
| Instacart holdout4 P_T6_r9 (T = 6, r_short = 9, 7,219 short users) vs C4: Recall@10, all users | `policy_c4.variants.P_T6_r9.comparisons.recall` | 20,000 | 0.352147 | 0.348774 | -0.003373 | [-0.003863, -0.002845] | [-0.003958, -0.002727] | CI only |
| Instacart holdout4 P_T6_r9 (T = 6, r_short = 9, 7,219 short users) vs C4: new-item Recall@10, users with new target | `policy_c4.variants.P_T6_r9.comparisons.new_recall` | 16,335 | 0.002806 | 0.007262 | +0.004456 | [0.003783, 0.005160] | [0.003606, 0.005336] | CI only |
| Instacart holdout4 P_T6_r9 (T = 6, r_short = 9, 7,219 short users) vs C4: repeat Recall@10, users with repeat target | `policy_c4.variants.P_T6_r9.comparisons.repeat_recall` | 18,701 | 0.574878 | 0.563661 | -0.011217 | [-0.012171, -0.010282] | [-0.012310, -0.010110] | CI only |
| Instacart holdout4 P_T6_r9 (T = 6, r_short = 9, 7,219 short users) vs C4: new-item hit share@10, users with new target | `policy_c4.variants.P_T6_r9.comparisons.new_hit_share_new_users` | 16,335 | 0.005999 | 0.028956 | +0.022957 | [0.020629, 0.025589] | [0.020324, 0.026263] | CI only |
| Instacart holdout4 P_T10_r9 (T = 10, r_short = 9, 10,956 short users) vs C4: NDCG@10, all users | `policy_c4.variants.P_T10_r9.comparisons.ndcg` | 20,000 | 0.425765 | 0.421415 | -0.004349 | [-0.004671, -0.004006] | [-0.004754, -0.003948] | CI only |
| Instacart holdout4 P_T10_r9 (T = 10, r_short = 9, 10,956 short users) vs C4: new-item hit share@10, all users | `policy_c4.variants.P_T10_r9.comparisons.new_hit_share` | 20,000 | 0.004900 | 0.031700 | +0.026800 | [0.024700, 0.029050] | [0.024217, 0.029450] | CI only |
| Instacart holdout4 P_T10_r9 (T = 10, r_short = 9, 10,956 short users) vs C4: Recall@10, all users | `policy_c4.variants.P_T10_r9.comparisons.recall` | 20,000 | 0.352147 | 0.346331 | -0.005816 | [-0.006457, -0.005195] | [-0.006573, -0.005073] | CI only |
| Instacart holdout4 P_T10_r9 (T = 10, r_short = 9, 10,956 short users) vs C4: new-item Recall@10, users with new target | `policy_c4.variants.P_T10_r9.comparisons.new_recall` | 16,335 | 0.002806 | 0.009419 | +0.006613 | [0.005817, 0.007466] | [0.005633, 0.007644] | CI only |
| Instacart holdout4 P_T10_r9 (T = 10, r_short = 9, 10,956 short users) vs C4: repeat Recall@10, users with repeat target | `policy_c4.variants.P_T10_r9.comparisons.repeat_recall` | 18,701 | 0.574878 | 0.558108 | -0.016771 | [-0.017829, -0.015667] | [-0.018084, -0.015439] | CI only |
| Instacart holdout4 P_T10_r9 (T = 10, r_short = 9, 10,956 short users) vs C4: new-item hit share@10, users with new target | `policy_c4.variants.P_T10_r9.comparisons.new_hit_share_new_users` | 16,335 | 0.005999 | 0.038812 | +0.032813 | [0.029936, 0.035813] | [0.029589, 0.036547] | CI only |
| Complete Journey refit P_T vs final_lri_ucf: Recall@10, all users | `policy_c4.secondary.P_T_vs_final_lri_ucf_recall` | 2,069 | 0.131625 | 0.131625 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T vs final_lri_ucf: new-item Recall@10, users with new target | `policy_c4.secondary.P_T_vs_final_lri_ucf_new_recall` | 1,971 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T vs final_lri_ucf: repeat Recall@10, users with repeat target | `policy_c4.secondary.P_T_vs_final_lri_ucf_repeat_recall` | 1,867 | 0.273880 | 0.273880 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T vs final_lri_ucf: new-item hit share@10, users with new target | `policy_c4.secondary.P_T_vs_final_lri_ucf_new_hit_share_new_users` | 1,971 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit usercf vs flat_explore: new-item Recall@10, users with new target | `policy_c4.secondary.usercf_vs_flat_explore_new_recall` | 1,971 | 0.012976 | 0.014472 | +0.001496 | [-0.001410, 0.004407] | [-0.002109, 0.005105] | CI only |
| Complete Journey refit P_T4_r8 (T = 4, r_short = 8, 1 short users) vs C4: NDCG@10, all users | `policy_c4.variants.P_T4_r8.comparisons.ndcg` | 2,069 | 0.204819 | 0.204819 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T4_r8 (T = 4, r_short = 8, 1 short users) vs C4: new-item hit share@10, all users | `policy_c4.variants.P_T4_r8.comparisons.new_hit_share` | 2,069 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T4_r8 (T = 4, r_short = 8, 1 short users) vs C4: Recall@10, all users | `policy_c4.variants.P_T4_r8.comparisons.recall` | 2,069 | 0.131625 | 0.131625 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T4_r8 (T = 4, r_short = 8, 1 short users) vs C4: new-item Recall@10, users with new target | `policy_c4.variants.P_T4_r8.comparisons.new_recall` | 1,971 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T4_r8 (T = 4, r_short = 8, 1 short users) vs C4: repeat Recall@10, users with repeat target | `policy_c4.variants.P_T4_r8.comparisons.repeat_recall` | 1,867 | 0.273880 | 0.273880 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T4_r8 (T = 4, r_short = 8, 1 short users) vs C4: new-item hit share@10, users with new target | `policy_c4.variants.P_T4_r8.comparisons.new_hit_share_new_users` | 1,971 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T6_r9 (T = 6, r_short = 9, 3 short users) vs C4: NDCG@10, all users | `policy_c4.variants.P_T6_r9.comparisons.ndcg` | 2,069 | 0.204819 | 0.204819 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T6_r9 (T = 6, r_short = 9, 3 short users) vs C4: new-item hit share@10, all users | `policy_c4.variants.P_T6_r9.comparisons.new_hit_share` | 2,069 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T6_r9 (T = 6, r_short = 9, 3 short users) vs C4: Recall@10, all users | `policy_c4.variants.P_T6_r9.comparisons.recall` | 2,069 | 0.131625 | 0.131625 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T6_r9 (T = 6, r_short = 9, 3 short users) vs C4: new-item Recall@10, users with new target | `policy_c4.variants.P_T6_r9.comparisons.new_recall` | 1,971 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T6_r9 (T = 6, r_short = 9, 3 short users) vs C4: repeat Recall@10, users with repeat target | `policy_c4.variants.P_T6_r9.comparisons.repeat_recall` | 1,867 | 0.273880 | 0.273880 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T6_r9 (T = 6, r_short = 9, 3 short users) vs C4: new-item hit share@10, users with new target | `policy_c4.variants.P_T6_r9.comparisons.new_hit_share_new_users` | 1,971 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey refit P_T10_r9 (T = 10, r_short = 9, 31 short users) vs C4: NDCG@10, all users | `policy_c4.variants.P_T10_r9.comparisons.ndcg` | 2,069 | 0.204819 | 0.204743 | -0.000076 | [-0.000215, 0.000081] | [-0.000277, 0.000114] | CI only |
| Complete Journey refit P_T10_r9 (T = 10, r_short = 9, 31 short users) vs C4: new-item hit share@10, all users | `policy_c4.variants.P_T10_r9.comparisons.new_hit_share` | 2,069 | 0.000000 | 0.000483 | +0.000483 | [0.000000, 0.001933] | [0.000000, 0.001933] | CI only |
| Complete Journey refit P_T10_r9 (T = 10, r_short = 9, 31 short users) vs C4: Recall@10, all users | `policy_c4.variants.P_T10_r9.comparisons.recall` | 2,069 | 0.131625 | 0.131643 | +0.000019 | [-0.000146, 0.000271] | [-0.000177, 0.000348] | CI only |
| Complete Journey refit P_T10_r9 (T = 10, r_short = 9, 31 short users) vs C4: new-item Recall@10, users with new target | `policy_c4.variants.P_T10_r9.comparisons.new_recall` | 1,971 | 0.000000 | 0.000127 | +0.000127 | [0.000000, 0.000381] | [0.000000, 0.000507] | CI only |
| Complete Journey refit P_T10_r9 (T = 10, r_short = 9, 31 short users) vs C4: repeat Recall@10, users with repeat target | `policy_c4.variants.P_T10_r9.comparisons.repeat_recall` | 1,867 | 0.273880 | 0.273630 | -0.000250 | [-0.000540, -0.000045] | [-0.000615, 0.000000] | CI only |
| Complete Journey refit P_T10_r9 (T = 10, r_short = 9, 31 short users) vs C4: new-item hit share@10, users with new target | `policy_c4.variants.P_T10_r9.comparisons.new_hit_share_new_users` | 1,971 | 0.000000 | 0.000507 | +0.000507 | [0.000000, 0.001522] | [0.000000, 0.002029] | CI only |
| Complete Journey transfer P_T vs final_lri_ucf: Recall@10, all users | `policy_c4.secondary.P_T_vs_final_lri_ucf_recall` | 2,069 | 0.128033 | 0.128033 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T vs final_lri_ucf: new-item Recall@10, users with new target | `policy_c4.secondary.P_T_vs_final_lri_ucf_new_recall` | 1,971 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T vs final_lri_ucf: repeat Recall@10, users with repeat target | `policy_c4.secondary.P_T_vs_final_lri_ucf_repeat_recall` | 1,867 | 0.268324 | 0.268324 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T vs final_lri_ucf: new-item hit share@10, users with new target | `policy_c4.secondary.P_T_vs_final_lri_ucf_new_hit_share_new_users` | 1,971 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer usercf vs flat_explore: new-item Recall@10, users with new target | `policy_c4.secondary.usercf_vs_flat_explore_new_recall` | 1,971 | 0.012976 | 0.015263 | +0.002287 | [-0.000485, 0.005044] | [-0.001041, 0.005382] | CI only |
| Complete Journey transfer P_T4_r8 (T = 4, r_short = 8, 1 short users) vs C4: NDCG@10, all users | `policy_c4.variants.P_T4_r8.comparisons.ndcg` | 2,069 | 0.199280 | 0.199280 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T4_r8 (T = 4, r_short = 8, 1 short users) vs C4: new-item hit share@10, all users | `policy_c4.variants.P_T4_r8.comparisons.new_hit_share` | 2,069 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T4_r8 (T = 4, r_short = 8, 1 short users) vs C4: Recall@10, all users | `policy_c4.variants.P_T4_r8.comparisons.recall` | 2,069 | 0.128033 | 0.128033 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T4_r8 (T = 4, r_short = 8, 1 short users) vs C4: new-item Recall@10, users with new target | `policy_c4.variants.P_T4_r8.comparisons.new_recall` | 1,971 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T4_r8 (T = 4, r_short = 8, 1 short users) vs C4: repeat Recall@10, users with repeat target | `policy_c4.variants.P_T4_r8.comparisons.repeat_recall` | 1,867 | 0.268324 | 0.268324 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T4_r8 (T = 4, r_short = 8, 1 short users) vs C4: new-item hit share@10, users with new target | `policy_c4.variants.P_T4_r8.comparisons.new_hit_share_new_users` | 1,971 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T6_r9 (T = 6, r_short = 9, 3 short users) vs C4: NDCG@10, all users | `policy_c4.variants.P_T6_r9.comparisons.ndcg` | 2,069 | 0.199280 | 0.199280 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T6_r9 (T = 6, r_short = 9, 3 short users) vs C4: new-item hit share@10, all users | `policy_c4.variants.P_T6_r9.comparisons.new_hit_share` | 2,069 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T6_r9 (T = 6, r_short = 9, 3 short users) vs C4: Recall@10, all users | `policy_c4.variants.P_T6_r9.comparisons.recall` | 2,069 | 0.128033 | 0.128033 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T6_r9 (T = 6, r_short = 9, 3 short users) vs C4: new-item Recall@10, users with new target | `policy_c4.variants.P_T6_r9.comparisons.new_recall` | 1,971 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T6_r9 (T = 6, r_short = 9, 3 short users) vs C4: repeat Recall@10, users with repeat target | `policy_c4.variants.P_T6_r9.comparisons.repeat_recall` | 1,867 | 0.268324 | 0.268324 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T6_r9 (T = 6, r_short = 9, 3 short users) vs C4: new-item hit share@10, users with new target | `policy_c4.variants.P_T6_r9.comparisons.new_hit_share_new_users` | 1,971 | 0.000000 | 0.000000 | +0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | CI only |
| Complete Journey transfer P_T10_r9 (T = 10, r_short = 9, 31 short users) vs C4: NDCG@10, all users | `policy_c4.variants.P_T10_r9.comparisons.ndcg` | 2,069 | 0.199280 | 0.199190 | -0.000090 | [-0.000215, 0.000035] | [-0.000246, 0.000069] | CI only |
| Complete Journey transfer P_T10_r9 (T = 10, r_short = 9, 31 short users) vs C4: new-item hit share@10, all users | `policy_c4.variants.P_T10_r9.comparisons.new_hit_share` | 2,069 | 0.000000 | 0.000483 | +0.000483 | [0.000000, 0.001450] | [0.000000, 0.001933] | CI only |
| Complete Journey transfer P_T10_r9 (T = 10, r_short = 9, 31 short users) vs C4: Recall@10, all users | `policy_c4.variants.P_T10_r9.comparisons.recall` | 2,069 | 0.128033 | 0.127990 | -0.000043 | [-0.000175, 0.000104] | [-0.000204, 0.000154] | CI only |
| Complete Journey transfer P_T10_r9 (T = 10, r_short = 9, 31 short users) vs C4: new-item Recall@10, users with new target | `policy_c4.variants.P_T10_r9.comparisons.new_recall` | 1,971 | 0.000000 | 0.000072 | +0.000072 | [0.000000, 0.000217] | [0.000000, 0.000290] | CI only |
| Complete Journey transfer P_T10_r9 (T = 10, r_short = 9, 31 short users) vs C4: repeat Recall@10, users with repeat target | `policy_c4.variants.P_T10_r9.comparisons.repeat_recall` | 1,867 | 0.268324 | 0.267985 | -0.000339 | [-0.000763, -0.000049] | [-0.000896, 0.000000] | CI only |
| Complete Journey transfer P_T10_r9 (T = 10, r_short = 9, 31 short users) vs C4: new-item hit share@10, users with new target | `policy_c4.variants.P_T10_r9.comparisons.new_hit_share_new_users` | 1,971 | 0.000000 | 0.000507 | +0.000507 | [0.000000, 0.001522] | [0.000000, 0.002029] | CI only |

policy_c4 (Instacart holdout4): 20,000 users, 4,316 with history_len <= 4 (by T: {'4': 4316, '6': 7219, '10': 10956}); r_default = r (C4) = 10, r_short = 9, delta = 0.003; lists {'P_T': 'combine_policy(repeat_lri, usercf, T=4, r_short=9, r_default=10)', 'C3': 'final_lri_ucf', 'S': 'usercf', 'flat': 'flat_explore'}.

## Commit record

| Commit | Content |
|---|---|
| `655bb32` | feat(v15): item-CF feature icf, R2' repeat ranker and C4 = R2' + E2 with the v15 block (H14, H15); policy base parameter and policy_c4 block (PREREG v1.5 / v1.5.1 code) |
| `d4000e7` | reports(posthoc): development check of the PREREG v1.5 / v1.5.1 code on medium (not confirmatory) |
| `d7d23a0` | data: holdout4 confirmatory profile (PREREG v1.5): config, provenance, manifest, selection report |
| `27d64f6` | Merge main into feat/w19-icf-final: docs/PREREG.md v1.5.1 (secondary replication of the v1.4 exploration policy on C4) |
| `b9695db` | docs(final4): RUN.md draft with the pre-specified PREREG v1.5 / v1.5.1 key table (before any holdout4 run) |
| `ba0f387` | docs(final4): record the pre-checks (check, holdout guard) before command 1 |
| `6fba95a` | docs(final4): fix the holdout4 Instacart selections in git before the eval split is read (after command 1) |
| `2fbfcd3` | docs(final4): guard check after command 1 and the incident record: command 2 killed by the host (low memory) after the eval split was read; run stopped, no re-run |
| `5e24d2f` | Merge main into feat/w19-icf-final: docs/PREREG.md v1.5.2 (deviation: one re-run of command 2 after the host kill) |
| `5c40aab` | docs(final4): PREREG v1.5.2 deviation record before the re-run of command 2 (evidence of no output from the first attempt, re-run plan) |
| (this commit) | post-run sections of this file, the five JSON copies, SHA256SUMS, logs |

## Files

`holdout4_*.json` are byte-identical copies of `reports/metrics/holdout4_*.json` written by the commands above
(verified with `cmp`); `SHA256SUMS` lists their SHA-256 digests (`sha256sum -c SHA256SUMS` from this directory).
`logs/` holds the runner records: `commands.json` (commands 1-5: UTC start/end, wall time, exit code, peak
memory), `cmd1.log` ... `cmd5.log` (stdout+stderr of each command), `prechecks.json`, `cmdcheck.log`,
`cmdguard_direct.log`, `cmdguard.log` (pre-checks), `guard_after_command1.json` and `guard_after_command1.log`.
