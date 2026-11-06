# Final evaluation run on holdout2 (prereg-v1.3) — DRAFT, written before command 1

Pre-registration: docs/PREREG.md v1.3 (main tag `prereg-v1.3`, commit 7bf3262). Base of this branch
(`feat/w10b-final`): main at `8cace67` (holdout2 data and selection report, W10a).

Commits before any run (this branch, all before command 1):

| Commit | Content |
|---|---|
| `f7cfb0a` | data/SOURCE.md: note on the W10a post-hoc read of the v1.2 holdout eval-target user ids |
| `535b490` | src/nbr/pipeline.py: `final_lr_ucf_vs_final_recall` comparison (C3 vs C0, Recall@10) and decision wording on the two C3-vs-C0 entries; reporting only, no list/selection change; tests |
| (this file) | RUN.md draft with the pre-specified key table |

Code commit used for every command: (recorded in "Pre-checks" below; `src/` and `configs/` last changed in `535b490`).

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

(to be filled)

## Commands (fixed order, each executed exactly once)

(to be filled)

## Re-run of command 2

(to be filled)

## Complete Journey reproducibility check

(to be filled)

## Files

(to be filled)
