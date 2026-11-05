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
