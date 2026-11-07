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
- v1.4 (2026-11-07, after the post-hoc W15 study on medium, before holdout3 is created or read).
  1. Confirmatory sample: `holdout3` = 20,000 Instacart users with a public final order, drawn with seed 42 from the users not in `sample`, `medium`, `holdout` or `holdout2` (69,936 remaining). Its Instacart eval split is read once, after this amendment is committed.
  2. Systems: C3 = the final system of v1.3 (R2 + E2, r selected on the fit split). P_T = C3 with r_u = 9 for users with at most T distinct history baskets and r_u = r* otherwise (one exploration slot for short histories; explore items fill the last 10 - r_u slots). S = the separate exploration list: the top 10 of E2 evaluated on its own. T = 4 baskets and delta = 0.003 are fixed here from the W15 post-hoc study on medium (development data) and are not re-selected.
  3. Selection (fit split only): as v1.3 (gamma, gamma_c, alpha, LR C, user-CF k and weighting, r). The policy does not change any selection.
  4. Primary hypotheses (Instacart holdout3, family of 3, two-sided 98.33 % CI = 1 - 0.05/3):
     H11 (non-inferiority): NDCG@10 of P_T minus C3 over all users; non-inferior if the CI lower bound > -0.003.
     H12: share of users with >= 1 new-item hit in the top-10, P_T minus C3 over all users; improvement if the CI lower bound > 0.
     H13: share of users with >= 1 new-item hit in a 10-item exploration list, S (E2) minus flat popularity excluding history items, over users with >= 1 new target item; improvement if the CI lower bound > 0.
  5. Secondary (no confirmatory claim): Recall@10, new-item Recall@10 and repeat Recall@10 of P_T vs C3; history-length bins; the r_u = 8 variant for short histories; T in {6, 10} as sensitivity; replication of H6, H7' and H8; Complete Journey refit and transfer with the same policy (same data as development, not independent).
  6. Inference: paired bootstrap over users, 1,000 resamples, seed 42; 95 % CIs reported alongside.
- v1.5 (2026-11-07, after two post-hoc development checks on `medium`, before holdout4 is created or read).
  1. Reason: the course instructor confirmed that the final system must use two collaborative-filtering methods. The v1.3 final system C3 uses one (E2, user-based CF); flat item-CF (H5) was evaluated only as a comparator.
  2. Development checks (medium; files in `Claude outputs/ICF_feature_check` and `Claude outputs/CF_fusion_check`; not confirmatory):
     a. Score fusion in the explore list, s = w * s_E2 / max_u + (1 - w) * s_itemcf / max_u, w in {0, 0.25, 0.5, 0.75, 1} selected by new-item Recall@10 on the fit split: w* = 1 (item-CF weight 0) on both datasets; every w < 1 scored lower on the fit split. Not carried forward.
     b. Item-CF feature in R2: icf(u, i) = log(1 + sum over j in I(H_u) of sim(j, i)), i in I(H_u), where sim is taken from the top-20 cosine item-neighbour lists of the binary user x item matrix of the same history (the H5 neighbours; self excluded). Repeat Recall@10, R2 -> R2 + icf: Instacart +0.0002 [-0.0001, 0.0005], Complete Journey -0.0001 [-0.0012, 0.0009]; standardized icf coefficient 0.007 and -0.012. Carried forward as R2'.
  3. Confirmatory sample: `holdout4` = 20,000 Instacart users with a public final order, drawn with seed 42 from the users not in `sample`, `medium`, `holdout`, `holdout2` or `holdout3` (49,936 remaining). Its Instacart eval split is read once, after this amendment is committed.
  4. Systems: R2' = R2 with the feature icf added (13 features; neighbours computed on the history being scored: the fit history for training rows, the eval history for lists; 20 item neighbours, fixed). C4 = top r of R2' + top (10 - r) of E2. C3 (v1.3) is the comparator.
  5. Selection (fit split only): as v1.3; the LR C is selected separately for R2 and R2', and r separately for C3 and C4. No icf variant is selected.
  6. Primary hypotheses (Instacart holdout4, family of 2, two-sided 97.5 % CI = 1 - 0.05/2):
     H14 (non-inferiority): NDCG@10 of C4 minus C3 over all users; non-inferior if the CI lower bound > -0.003 (the v1.4 margin).
     H15: NDCG@10 of C4 minus personal frequency over all users; improvement if the CI lower bound > 0.
  7. Secondary (no confirmatory claim): repeat Recall@10 of R2' vs R2 and R2' vs R1 (users with >= 1 repeat target item); Recall@10 of C4 vs C3; standardized coefficients of R2'; replication of H6, H7' and H8; history-length bins; Complete Journey refit and transfer (same data as development, not independent).
  8. Inference: paired bootstrap over users, 1,000 resamples, seed 42; 95 % CIs reported alongside.
- v1.5.1 (2026-11-07, after v1.5 and before the holdout4 eval split is read; no new result has been read since v1.5).
  1. Added secondary analyses (no confirmatory claim): the v1.4 exploration policy is repeated on holdout4 with C4 as the base. P_T(C4) = C4 with r_u = 9 for users with at most T distinct history baskets and r_u = r* of C4 otherwise; T = 4 and delta = 0.003 as in v1.4 (not re-selected). Comparisons as H11, H12 and H13 of v1.4: NDCG@10 of P_T(C4) minus C4 over all users; share of users with >= 1 new-item hit in the top-10, P_T(C4) minus C4, over all users; share of users with >= 1 new-item hit in the separate 10-item list S (E2) minus flat popularity excluding history items, over users with >= 1 new target item. The v1.4 variants (r_u = 8; T in {6, 10}) and the v1.4 policy on C3 are reported as references.
  2. Reported with 95 % and 98.33 % CIs and the v1.4 decision wording, labelled "secondary replication on holdout4". The confirmatory result for the policy remains H11-H13 on holdout3 (base C3). H14, H15 and their 97.5 % family are unchanged.
  3. Complete Journey: the same policy on C4, refit and transfer (same data as development, not independent).
  4. Reason: v1.5 changed the final system from C3 to C4; the policy is re-run so that the reported policy analysis refers to the final system.
- v1.5.2 (2026-11-08) Deviation, committed before any holdout4 result was produced or seen.
  Command 2 of reports/final4/RUN.md (evaluate --profile holdout4 --dataset instacart --final --skip-l3 --policy),
  started 2026-11-07T13:45:51Z at 6fba95a, was terminated at about 13:48:59Z by the host tool
  (Claude Code background-shell reaping under system memory pressure); no error was raised by the pipeline.
  load_split had read the holdout4 Instacart eval split (eval_history, eval_target); no metric, output file
  or result log line was produced, and no result was seen.
  Authorised: exactly one re-run of command 2 and the first runs of commands 3 to 5, with code, configs,
  selections (6fba95a), seeds, hypotheses, delta and CI levels unchanged; only the execution environment
  changes (the chain runs outside the host tool).
  If the re-run of command 2 fails before reports/metrics/holdout4_instacart_eval.json is written,
  no further run is made and H14, H15 and policy_c4 are reported as not obtained.
  All reports state that the holdout4 Instacart eval split was loaded twice and that the first load produced no output.
