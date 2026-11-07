# Confirmatory run of the exploration policy on holdout3 (prereg-v1.4)

Pre-registration: docs/PREREG.md v1.4 (main tag `prereg-v1.4`, commit 5427b0a; the v1.4 amendment is commit 614209f,
the base of this branch `feat/w17-policy`). The sections "Pre-specified output keys", "Pre-checks" and "Selections"
are committed before the holdout3 Instacart eval split is read; the remaining sections are written after command 5.
Same procedure and failure policy as reports/final2/RUN.md: five commands, each executed exactly once, chained;
a failure in command 2 stops everything (no partial output is opened, no re-run); other commands admit technical
fixes only, recorded here.

Commits of this branch before any holdout3 run (all before command 1):

| Commit | Content |
|---|---|
| `0c7aa89` | src/nbr/policy.py (P_T, S, new-item hit share, H11-H13 block), `evaluate --policy` wiring, tests (262 passed), scripts/check_policy_dev.py |
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

## Pre-checks

(to be recorded before command 1)

## Selections (written after command 1, before command 2)

(to be recorded after command 1)
