# Pre-registration — hierarchical next-basket recommendation (v1, 2026-11-05)

Committed before any model is run on the holdout eval split. Changes after that commit are logged under "Deviations".

## Question
Does separating repurchase from exploration, and ranking exploration through the category hierarchy
(L2 category -> item), improve next-basket Top-10 recommendation over personal purchase frequency?

## Data and roles
| Dataset | Development | Final evaluation |
|---|---|---|
| Instacart | `medium` (20,000 users; eval split seen in earlier experiments) | `holdout` (20,000 unused users; eval read once) |
| Complete Journey (dunnhumby source files) | `medium` (2,069 households) | `medium` refit (no unused households; limitation) |

Hierarchy: Instacart department (L1) -> aisle (L2) -> ingredient type (L3, aisle fallback) -> item;
Complete Journey DEPARTMENT -> commodity (L2) -> sub-commodity (L3) -> item. The main model uses L2 only.

## Model
- Repeat score: s_rep(u,i) = sum over history baskets b of gamma^age(b) * 1[i in b]; age = 0 for the most recent basket. gamma = 1 is personal frequency.
- Explore score (items not in the user's history): theta_u(c) * pop(i) / pop(c), with theta_u(c) = (decayed category count + alpha * p(c)) / (decayed total + alpha) and pop(c) = sum of pop over all items in c (bought or not). Equivalently theta_u(c) / p(c) * pop(i) / sum pop: global popularity reweighted by the user's category lift. With alpha = inf (theta_u = p) it reduces exactly to the H2 comparator (popularity excluding history items).
- pop(i) = number of history baskets (all users of the split) containing i; p(c) = category share of pop.
- Final list: top r repeat items + top (10 - r) explore items; if fewer than r repeat items exist, explore items fill the slots.

## Selection (fit split of the same profile and dataset only)
1. gamma* = argmax repeat-component Recall@10 (tie: gamma closer to 1).
2. (gamma_c*, alpha*) = argmax explore-component new-item Recall@10 with phi = pop (tie: larger alpha, then gamma_c = 1).
3. r* = argmax final-list NDCG@10 given steps 1-2 (tie: larger r).
Transfer run for Complete Journey: Instacart holdout selections applied to CJ (secondary).

## Hypotheses (primary; each on Instacart holdout and CJ refit = 8 tests)
| ID | Comparison | Metric | Users |
|---|---|---|---|
| H1 | s_rep(gamma*) vs personal frequency | repeat-component Recall@10 | users with >= 1 repeat target item |
| H2 | hierarchical explore vs flat popularity excluding history items | new-item component Recall@10 | users with >= 1 new target item |
| H3 | final list vs personal frequency (padded with popularity) | NDCG@10 (Recall@10 reported) | all users |
| H4 | top-10 L2 by theta_u vs top-10 L2 by p(c) | share of new target items whose L2 is in the top 10 | users with >= 1 new target item |

Inference: paired bootstrap over users (1,000 resamples, seed 42); 95% and 99.375% (Bonferroni, 8 tests) percentile CIs of the mean difference.
Decision: lower bound > 0 -> "improvement observed"; CI contains 0 -> "no difference established"; upper bound < 0 -> "lower".

## Secondary and exploratory (no confirmatory claims)
- item-CF variants (flat item-CF; theta x item-CF within category); L3 ablation; CJ transfer; selected values at grid edges are reported.
- Health-attribute case study: existing H*' results on Instacart medium (ML_group9 commits 2a87fc5, c44bbf6, e74412b); no new tests.

## Deviations
- v1.1 (2026-11-05, before any model run on any split): within-category normalization changed from unbought items only (Z_u(c)) to all items in c. Reason: with Z_u(c) the alpha -> inf limit does not equal the H2 comparator (categories where the user bought popular items are inflated), so H2 would mix the user-profile effect with a renormalization effect. alpha = inf (theta_u = p) is allowed in code for the comparator and tests; the selection grid stays {1, 10, 100}.
- v1.2 (2026-11-05, after development runs on `medium` fit/eval, before any read of the holdout eval split). The development results are in `reports/metrics/medium_*` of branch feat/w4-pipeline (commit 7baa6a1) and are not confirmatory.
  1. Added a secondary confirmatory hypothesis generated on medium: H5 = flat item-CF (user x item cosine, 20 neighbors, scores propagated from purchased items' neighbor lists) vs flat popularity excluding history items, new-item component Recall@10, users with >= 1 new target item; Instacart holdout only; reported with its own 95% CI and labelled "hypothesis generated on development data". H1-H4 and their Bonferroni level are unchanged.
  2. Added descriptive reporting by history length (distinct history baskets; Instacart (0,4], (4,6], (6,10], (10,20], (20,inf); CJ (0,20], (20,50], (50,100], (100,inf)); no decisions.
  3. Reporting: besides the single combined list (r*), the explore list is reported as a separate output ("new items for you") with the H2/H5 metrics, because r* = 10 was selected on both medium fit splits.
  4. Limitation made explicit: the final Complete Journey refit run uses the same `medium` data as the development run, so CJ refit results are not independent of development; CJ transfer uses the holdout Instacart selections.
  5. Implementation choices recorded: k = 10 fixed in code; ties in selection judged with math.isclose(rel_tol=1e-12); selection grids unchanged (alpha* = 100 and r* = 10 at grid edges on Instacart medium are reported, not extended).
- v1.3 (2026-11-06, after development runs W6–W9 on `medium`, before holdout2 is created or read). Development results: branches feat/w6-lr-repeat (1363e52), feat/w7-user-cf (3f0be10), feat/w8-integrate (233928b, 63bd002), feat/w9-r3-cycle (4aaa64b); none is confirmatory.
  1. Confirmatory sample: `holdout2` = 20,000 Instacart users with a public final order, drawn with seed 42 from the users not in `sample`, `medium` or `holdout` (89,936 remaining). Its Instacart eval split is read once, after this amendment is committed.
  2. Final system chosen on development results: repeat ranker R2 (L2-regularized logistic regression on the features in src/nbr/features.py; label = item in the next basket of the fit split), explore list E2 (user-based CF: cosine over user x item vectors, score = sum over the k nearest users of sim * x_v,i, items not in the user's history), combination C3 = top r of R2 + top (10 - r) of E2.
  3. Selection (fit split only): gamma, gamma_c, alpha as before; LR C in {0.001, 0.01, 0.1, 1, 10} by repeat Recall@10 on validation users (20 % of fit users, seed 42), then refit on all fit users; user-CF k in {10, 20, 50, 100, 200, 400} x weighting {binary, log1p} by new-item Recall@10; r in {0, ..., 10} by NDCG@10 (for LR combinations on validation users with the model trained on the remaining fit users). Ties as implemented in the code used for the run. K = 10.
  4. Primary hypotheses (Instacart holdout2, family of 3): H6 = R2 vs R1, repeat-component Recall@10 (users with >= 1 repeat target item); H7' = C3 vs C0 (the v1.2 final list), NDCG@10 (Recall@10 reported); H8 = E2 vs flat popularity excluding history items, new-item Recall@10 (users with >= 1 new target item). Inference as before; the existing 99.375 % CI is used as the multiplicity-adjusted interval (conservative for 3 tests).
  5. Secondary (no confirmatory claim): H9 = C3 vs C1 (NDCG@10; any difference can only come from users with fewer than 10 repeat candidates); replication of H1-H5 on holdout2; C3 vs personal frequency; E2 vs flat item-CF; history-length bins; Complete Journey refit of all analyses (same data as development, not independent) and transfer.
  6. Not carried forward: R3 binned cycle features (H10: no difference on medium); explore_L3 is not computed (`--skip-l3`; the L3 exclusion was confirmed under v1.2).
  7. k = 10 is added to the user-CF grid because Complete Journey selected the lower edge (k = 20) on medium.
- v1.3 run notes (2026-11-06, after the holdout2 run; details in reports/final2/RUN.md).
  1. Before command 1, commit 535b490 added a reporting-only comparison (C3 vs C0, Recall@10) and decision wording for the two C3-vs-C0 entries; no list, selection or metric computation changed, and the sample smoke outputs were identical apart from the added keys.
  2. The check that `evaluate --profile holdout2 --dataset instacart` without `--final` stops at the holdout guard could be reached only after command 1, because the command loads the tuning selections first; it was recorded between commands 1 and 2, and no eval file was opened on that path.
  3. The pipeline output still labels H7 (C1 vs C0) and H9 (C3 vs C1) as "primary" and the C3-vs-C0 entries as "integration.secondary"; the consolidated tables classify hypotheses by the explicit v1.3 table in scripts/results_sources.yaml (primary: H6, H7', H8; all others secondary).
  4. The holdout2 Instacart eval split was read exactly once (command 2) and no command was re-run.
- v1.3 run notes (2026-11-06, after the holdout2 run; details in reports/final2/RUN.md).
  1. Before command 1, commit 535b490 added a reporting-only comparison (C3 vs C0, Recall@10) and decision wording for the two C3-vs-C0 entries; no list, selection or metric computation changed, and the sample smoke outputs were identical apart from the added keys.
  2. The check that `evaluate --profile holdout2 --dataset instacart` without `--final` stops at the holdout guard could be reached only after command 1, because the command loads the tuning selections first; it was recorded between commands 1 and 2, and no eval file was opened on that path.
  3. The pipeline output still labels H7 (C1 vs C0) and H9 (C3 vs C1) as "primary" and the C3-vs-C0 entries as "integration.secondary"; the consolidated tables classify hypotheses by the explicit v1.3 table in scripts/results_sources.yaml (primary: H6, H7', H8; all others secondary).
  4. The holdout2 Instacart eval split was read exactly once (command 2) and no command was re-run.
