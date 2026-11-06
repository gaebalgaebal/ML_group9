# Module catalogue

Codes used in the report, the slides and `reports/results/` (branch `feat/w11-results`,
`scripts/results_sources.yaml`). Statuses: **adopted** = part of the pre-registered model
evaluated on the holdout (PREREG v1.2); **evaluated-not-adopted** = evaluated on development
data (`medium`) only, adoption decision TBD unless stated; **not-examined** = no implementation
or evaluation in this project; **not run** = planned, not implemented. Hypothesis ids follow
`docs/PREREG.md` (H1-H5) and the development tasks (H6-H10; see section 3). Numbers are in
`reports/results/RESULTS.md`, not here.

## 1. Explore modules (candidates = items the user never bought)

| Code | Definition | Implementation | Selection | Status | Basis |
|---|---|---|---|---|---|
| E1 | hierarchical explore: `s(u,i) = theta_u(c) * pop(i) / pop(c)` with the decayed, smoothed L2-category profile `theta_u(c)`; equals popularity excluding history when `alpha = inf` | `src/nbr/explore.py`: `category_profile`, `explore_list` (list `explore_star`); `category_ranking` for H4 | `(gamma_c, alpha)` by new-item Recall@10 (fit split) | adopted | H2, H4 (holdout) |
| E2 | user-based CF: cosine similarity between user x item vectors (binary or log1p basket counts), `score(u,i) = sum over the k nearest users v of sim(u,v) x_{v,i}` over unseen items, padded with popularity | `src/nbr/usercf.py`: `user_vectors`, `user_neighbors`, `ucf_list` (list `usercf`); stand-alone runner `src/nbr/usercf_eval.py` | `(k, weighting)` by new-item Recall@10 (fit split) | evaluated-not-adopted (proposed as the explore list of the v1.3 draft; decision TBD) | H8, H9 (medium) |
| E3 | flat item-CF: item cosine neighbours (20) from the binary user x item matrix, `score(u,i) = sum over bought j of sim(j,i)` over unseen items | `src/nbr/itemcf.py`: `item_neighbors`, `cf_scores`, `cf_list` (list `itemcf_flat`) | none (20 neighbours fixed, PREREG v1.2) | evaluated-not-adopted | H5 (holdout, secondary confirmatory, generated on development data) |
| E4 | theta x item-CF within category: `theta_u(c) * cf(u,i) / sum_{j in c} cf(u,j)` | `src/nbr/explore.py`: `explore_list_cf` (list `explore_cf`) | shares E1's `(gamma_c, alpha)` | evaluated-not-adopted | secondary `explore_cf_vs_explore_star` (holdout, medium) |
| E5 | matrix factorisation | - | - | not-examined | - |
| E6 | product-name tf-idf similarity | - | - | not-examined | - |
| E7 | health-attribute match | - | - | not-examined (the health-attribute case study of PREREG uses existing ML_group9 results only) | - |
| E8 | recent-basket item-CF | - | - | not-examined | - |
| flat_pop | popularity excluding the user's history items (comparator) | `src/nbr/baselines.py`: `popularity_list(exclude_history=True)` (list `flat_explore`) | - | comparator | H2, H5, H8 |
| explore_L3 | E1 at level L3 instead of L2 (ablation) | `src/nbr/pipeline.py:build_lists` (`explore_list(..., "L3")`) | - | ablation, skipped in W8b (`--skip-l3`) | secondary `explore_L3_vs_explore_star` |

## 2. Repeat modules (candidates = items the user bought before)

| Code | Definition | Implementation | Selection | Status | Basis |
|---|---|---|---|---|---|
| R1 | decayed purchase frequency `s_rep(u,i) = sum over baskets b containing i of gamma^age(b)` | `src/nbr/repeat.py`: `repeat_scores`, `repeat_list` (list `repeat_star`) | `gamma` by repeat Recall@10 (fit split) | adopted | H1 (holdout) |
| R2 | L2-regularised logistic regression over 12 per-pair history features (decayed counts for gamma 0.5/0.8/1.0, recency, mean gap, once, overdue, item repeat rate, log pop, user repeat rate, log history length, share); label = pair occurs in the next basket of the fit split | `src/nbr/features.py:repeat_features`; `src/nbr/lr_repeat.py`: `build_training`, `fit_select`, `lr_repeat_list` (list `repeat_lr`); `pipeline.tune_lr_repeat` | `C` on validation users (20 %) | evaluated-not-adopted (proposed as the repeat ranker of the v1.3 draft; decision TBD) | H6, H7 (medium) |
| R3 | R2 features + binned cycle features (item cadence, cadence ratio, one-hot bins of overdue / recency / cadence ratio; 30 features) | branch `feat/w9-r3-cycle`: `src/nbr/features_cycle.py:cycle_features`, `src/nbr/r3_eval.py` (list `repeat_r3`) | `C` as R2, same user split | evaluated-not-adopted (not carried forward in the v1.3 draft) | H10 (medium) |
| R4 | decision-tree repeat ranker | - | - | not-examined | - |
| R5 | personal purchase frequency (`gamma = 1`); as a full list, padded with popularity outside the history | `src/nbr/repeat.py:repeat_list(gamma=1.0)` (list `repeat_pf`); `src/nbr/baselines.py:personal_frequency_list` (list `personal_frequency`) | - | comparator | H1 (repeat component), H3 (full list) |
| R6 | recent-N-basket candidate set | - | - | not-examined | - |

## 3. Combinations (slot rule `nbr.combine.combine`: first `r` slots repeat, then explore, leftover repeat items fill up)

| Code | Repeat | Explore | `r` | List name | Implementation | Status | Basis |
|---|---|---|---|---|---|---|---|
| C0 | R1 | E1 | `r*` of stage 3 (grid 5-10; 10 in the recorded W4 medium and holdout tuning runs, see PREREG v1.2 and `reports/final/RUN.md`) | `final` | `pipeline.build_lists` | adopted | H3 (holdout) |
| C1 | R2 | E1 | same `r*` as C0 (not re-selected) | `final_lr` | `pipeline.build_lists` (`lr_bundle`) | evaluated-not-adopted | H7 (medium) |
| C2 | R2 | E1 | re-selected on the fit split (LR validation users, grid 0-10) | `final_lr_r` | `pipeline.INTEGRATION_COMBOS`, `tune_integration` | evaluated-not-adopted | secondary `final_lr_r_vs_final_lr` (medium) |
| C3 | R2 | E2 | re-selected (LR validation users) | `final_lr_ucf` | same | evaluated-not-adopted (proposed final system of the v1.3 draft; decision TBD) | H9 = C3 vs C1 (medium) |
| C3b | R1 | E2 | re-selected (all fit users) | `final_star_ucf` (W7 stand-alone: `final_ucf` with the C0 `r*`) | same; `usercf_eval.evaluate` | evaluated-not-adopted | secondary `final_star_ucf_vs_final` (medium) |
| C4 | R2 | E3 | - | - | not implemented | not run | - |
| E4+R1 | R1 | E4 | `r*` of stage 3 | `final_cf` | `pipeline.build_lists` (`include_itemcf`) | evaluated-not-adopted | secondary `final_cf_vs_final` (holdout v1.2 auxiliary) |
| final_r10 | R1 | E1 | 10 (forced) | `final_r10` | `pipeline.build_lists` | diagnostic | secondary `final_r10_vs_final` |

Hypothesis map:

| Id | Comparison (model vs comparator) | Metric | Stage |
|---|---|---|---|
| H1 | R1 vs R5 (repeat component) | repeat Recall@10 | holdout (PREREG v1.2) |
| H2 | E1 vs flat_pop | new-item Recall@10 | holdout |
| H3 | C0 vs R5 padded with popularity | NDCG@10 (Recall@10 reported) | holdout |
| H4 | top-10 L2 by `theta_u` vs top-10 L2 by `p(c)` | share of new target items whose L2 is in the top 10 | holdout |
| H5 | E3 vs flat_pop | new-item Recall@10 | holdout, Instacart only (secondary confirmatory) |
| H6 | R2 vs R1 | repeat Recall@10 | medium (W6, W8, W8b) |
| H7 | C1 vs C0 | NDCG@10 (Recall@10 reported) | medium |
| H8 | E2 vs flat_pop | new-item Recall@10 | medium (W7, W8, W8b) |
| H9 | C3 vs C1 | NDCG@10 (Recall@10 reported) | medium (W8, W8b) |
| H10 | R3 vs R2 | repeat Recall@10 | medium (W9) |

## 4. Filtering-method classification

Classification follows the usual textbook split: content-based (item attributes and the user's
own history), memory-based collaborative filtering (neighbourhoods of users or items computed from
the user x item matrix), model-based / learned ranking, hybrid, or not collaborative filtering.
Where the label depends on interpretation it is stated.

| Code | Classification | Reasoning |
|---|---|---|
| E1 | content-based (category attribute of items x the user's own category profile) combined with non-personalised popularity; interpretation-dependent | `theta_u(c)` uses only the user's history and the category attribute of items; `pop(i)` and `p(c)` aggregate all users of the split but involve no user-user or item-item neighbourhood, so most definitions do not count it as CF. Some authors call any population statistic "collaborative"; under that reading E1 is a content-popularity hybrid. |
| E2 | memory-based CF, user-based | cosine neighbours between users; scores are a weighted sum of the neighbours' purchases |
| E3 | memory-based CF, item-based | cosine neighbours between items from the user x item matrix; scores propagate from the user's purchased items |
| E4 | hybrid: content-based category profile x item-based CF | E3 scores re-weighted within category by `theta_u(c)` |
| E5 | model-based CF (not examined) | latent factors of the user x item matrix |
| E6 | content-based (not examined) | text similarity of product names |
| E7 | content-based (not examined) | item health attributes vs the user's attribute preference |
| E8 | memory-based CF, item-based (not examined) | item neighbours restricted to the most recent baskets |
| flat_pop | not CF (non-personalised popularity); interpretation-dependent as for E1 | global basket counts, personalised only by removing the user's own items |
| R1 | not CF: personal-history heuristic (recency-weighted frequency) | uses the user's own baskets only |
| R2 | learned ranking model (supervised logistic regression); hybrid in the feature sense | the features mix the user's own history (decayed counts, recency, gaps) with item-level statistics over all users (`item_repeat_rate`, `log_pop`); no neighbourhood |
| R3 | learned ranking model, as R2 | R2 features plus binned cycle features |
| R4 | learned ranking model (not examined) | decision tree on the same features |
| R5 | not CF: personal frequency | count of baskets containing the item |
| R6 | not CF: candidate rule (not examined) | restricts repeat candidates to the last N baskets |
| C0-C3b, E4+R1 | hybrid of a repeat module and an explore module by slot combination | the classification of each half is inherited; C3 (R2 + E2) is "learned repeat ranker + user-based CF" |

## 5. Where the codes are used

- `reports/results/tables/T6_module_catalog.csv` (branch `feat/w11-results`) carries the same codes,
  statuses and basis hypotheses; `T4_final_lists` and figure F4 use the C labels.
- PREREG v1.3 (draft, uncommitted in the `main` working tree as of 2026-11-06) proposes R2 + E2 = C3
  as the system for the planned `holdout2` run; the adoption decision and the final wording are TBD.
