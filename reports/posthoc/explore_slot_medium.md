# New-item purchase rates and the value of an exploration slot (medium)

**post-hoc, development data, not confirmatory.** A: for every user and every basket from the 2nd basket on, prior(b) = items of the user's earlier baskets of the same split (history + target); new_share(b) = |b minus prior(b)| / |b| over distinct items, any_new(b) = 1 if at least one item is new, n_new(b) = number of new items; basket-weighted = mean over baskets, user-weighted = mean of the per-user means; position = basket_seq; history-length bins = T5 bins of nbr.pipeline.HISTORY_BINS. B: hit probability by rank of the R2 repeat list and the E2 explore list. C: final lists with r in {10, 9, 8} repeat slots per repeat module, paired bootstraps against r = 10, and the fit-split NDCG@10 curves by r. The eval split (already used by earlier development runs) is the main split. Descriptive and exploratory only; no confirmatory claim is made.

- Code commit `8964e01a252242f4029596fabb052a41b06e2854`, 2026-11-07T06:12:25.841588+00:00; profile `medium`; bootstrap n = 1000, seed 42, 95 % CI.
- Position bins 2-4, 5-10, 11-20, 21-50, >50; history-length bins Instacart (0,4], (4,6], (6,10], (10,20], (20,inf); Complete Journey (0,20], (20,50], (50,100], (100,inf).
- Selections (python -m nbr tune, this run): Instacart gamma* 0.8, gamma_c 0.9, alpha 100, r* 10, user-CF k 100 binary, LR C 0.001; Complete Journey gamma* 1, gamma_c 0.9, alpha 10, r* 10, user-CF k 20 binary, LR C 0.01.
- Complete Journey note: its target is the first basket with >= 3 items after the cutoff while history baskets have no size filter, so the A history rows are also given for baskets with >= 3 distinct items; baskets between the last history basket and the target that are in neither frame are not part of prior(b), as in the pipeline.
- Inputs (sha256 in the JSON): instacart: 5 files, complete_journey: 5 files.
- Runtime 644.4 s; peak working set 2713 MiB (in-process).

## Setup check: tuning documents of this run against the W8b reference

- Reference: `C:\dev_ML-wt-w8\reports\metrics` (read only). Result: passed (every selection equal).

Instacart (this run `C:\dev_ML-wt-w15\reports\metrics\medium_instacart_tuning.json`, commit `78ba1fd0dd6b366afb9c84806dba157759ef156b`; reference commit `6936873dd9b541a3a177643e97700182b8d729d8`, 2026-11-06T02:35:19.027921+00:00):

| quantity | this run | reference | equal |
| --- | --- | --- | --- |
| gamma | 0.8 | 0.8 | yes |
| gamma_c | 0.9 | 0.9 | yes |
| alpha | 100.0 | 100.0 | yes |
| r | 10 | 10 | yes |
| lr_C | 0.001 | 0.001 | yes |
| usercf_k | 100 | 100 | yes |
| usercf_weighting | binary | binary | yes |
| r_final_lr_r | 10 | 10 | yes |
| r_final_lr_ucf | 10 | 10 | yes |
| r_final_star_ucf | 10 | 10 | yes |
| user-CF grid rows other than k = 10 (new_recall@10) |  |  | yes |
| stage1_repeat_recall grid |  |  | yes |
| stage2_explore_new_recall grid |  |  | yes |
| stage3_final_ndcg grid |  |  | yes |
| LR validation rows (val_recall by C) |  |  | yes |
| LR artifact sha256 | 0bf51f8852cb | 0bf51f8852cb | yes |
| LR artifact file exists with the recorded sha256 | yes |  |  |

k = 10 rows of this run (absent from the reference grid): binary: new_recall@10 = 0.0440; log1p: new_recall@10 = 0.0414

Complete Journey (this run `C:\dev_ML-wt-w15\reports\metrics\medium_complete_journey_tuning.json`, commit `78ba1fd0dd6b366afb9c84806dba157759ef156b`; reference commit `6936873dd9b541a3a177643e97700182b8d729d8`, 2026-11-06T02:18:38.336788+00:00):

| quantity | this run | reference | equal |
| --- | --- | --- | --- |
| gamma | 1.0 | 1.0 | yes |
| gamma_c | 0.9 | 0.9 | yes |
| alpha | 10.0 | 10.0 | yes |
| r | 10 | 10 | yes |
| lr_C | 0.01 | 0.01 | yes |
| usercf_k | 20 | 20 | yes |
| usercf_weighting | binary | binary | yes |
| r_final_lr_r | 10 | 10 | yes |
| r_final_lr_ucf | 10 | 10 | yes |
| r_final_star_ucf | 10 | 10 | yes |
| user-CF grid rows other than k = 10 (new_recall@10) |  |  | yes |
| stage1_repeat_recall grid |  |  | yes |
| stage2_explore_new_recall grid |  |  | yes |
| stage3_final_ndcg grid |  |  | yes |
| LR validation rows (val_recall by C) |  |  | yes |
| LR artifact sha256 | c71680afc3bb | c71680afc3bb | yes |
| LR artifact file exists with the recorded sha256 | yes |  |  |

k = 10 rows of this run (absent from the reference grid): binary: new_recall@10 = 0.0119; log1p: new_recall@10 = 0.0140

## A. New-item purchase rates

Headline (eval split; bw / uw = basket-weighted / user-weighted):

| dataset | users | target mean new_share | target P(any_new) | target mean n_new | history new_share (bw / uw) | history P(any_new) (bw / uw) | history n_new (bw / uw) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Instacart | 20,000 | 0.400 | 0.816 | 4.26 | 0.362 / 0.487 | 0.771 / 0.846 | 3.74 / 4.95 |
| Complete Journey | 2,069 | 0.545 | 0.953 | 7.29 | 0.542 / 0.624 | 0.851 / 0.907 | 5.52 / 6.84 |

### Eval split (main)

### Instacart, eval split (main)

- Users 20,000; history baskets 313,281; target baskets 20,000; baskets analysed 313,281 (first baskets dropped 20,000). History length (distinct history baskets): min 3, median 9, max 99.
- basket_seq starts at 1 for 100% of the users and is consecutive for 100%; target basket_seq minus last history basket_seq: 1: 20,000 (0 users have baskets between history and target that are in neither frame).
- Share of history baskets with >= 3 distinct items 0.893; smallest target basket 1 items. Users with >= 1 new target item: 16,319 here, 16,319 by nbr.data.split_target (equal).

Overall:

| basket set | baskets | users | weighting | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| history baskets (2nd basket on) | 293,281 | 20,000 | basket-weighted | 0.362 | 0.771 | 3.74 | 10.0 |
| history baskets (2nd basket on) | 293,281 | 20,000 | user-weighted | 0.487 | 0.846 | 4.95 | 9.9 |
| history baskets with >= 3 distinct items (2nd basket on) | 261,984 | 19,575 | basket-weighted | 0.370 | 0.819 | 4.13 | 11.0 |
| history baskets with >= 3 distinct items (2nd basket on) | 261,984 | 19,575 | user-weighted | 0.506 | 0.893 | 5.36 | 10.5 |
| target basket | 20,000 | 20,000 | per user | 0.400 | 0.816 | 4.26 | 10.6 |

Distribution of n_new (basket-weighted shares):

| basket set | n_new = 0 | n_new = 1 | n_new = 2 | n_new >= 3 |
| --- | --- | --- | --- | --- |
| history baskets (2nd basket on) | 0.229 | 0.160 | 0.122 | 0.489 |
| history baskets with >= 3 distinct items (2nd basket on) | 0.181 | 0.145 | 0.126 | 0.547 |
| target basket | 0.184 | 0.146 | 0.126 | 0.545 |

By basket position (position = basket_seq):

History baskets (2nd basket on) (basket-weighted / user-weighted):

| position | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| 2-4 | 57,702 | 20,000 | 0.615 / 0.617 | 0.920 / 0.920 | 6.25 / 6.26 | 9.9 / 9.9 |
| 5-10 | 75,399 | 15,857 | 0.414 / 0.429 | 0.832 / 0.836 | 4.32 / 4.48 | 10.1 / 10.0 |
| 11-20 | 67,467 | 9,083 | 0.297 / 0.314 | 0.747 / 0.761 | 3.14 / 3.34 | 10.2 / 10.3 |
| 21-50 | 71,836 | 4,613 | 0.221 / 0.247 | 0.663 / 0.708 | 2.29 / 2.65 | 10.2 / 10.6 |
| >50 | 20,877 | 1,039 | 0.174 / 0.185 | 0.591 / 0.631 | 1.61 / 1.82 | 9.3 / 10.2 |

History baskets with >= 3 distinct items (2nd basket on) (basket-weighted / user-weighted):

| position | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| 2-4 | 50,757 | 19,210 | 0.631 / 0.639 | 0.965 / 0.964 | 7.00 / 6.80 | 11.0 / 10.6 |
| 5-10 | 66,753 | 15,309 | 0.427 / 0.447 | 0.889 / 0.888 | 4.82 / 4.86 | 11.2 / 10.7 |
| 11-20 | 60,501 | 8,901 | 0.305 / 0.325 | 0.798 / 0.804 | 3.46 / 3.57 | 11.2 / 10.8 |
| 21-50 | 65,197 | 4,552 | 0.225 / 0.254 | 0.705 / 0.742 | 2.49 / 2.81 | 11.0 / 11.1 |
| >50 | 18,776 | 1,030 | 0.178 / 0.190 | 0.636 / 0.661 | 1.77 / 1.93 | 10.2 / 10.6 |

Target basket:

| position | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| 2-4 | 2,298 | 2,298 | 0.587 | 0.896 | 5.72 | 9.5 |
| 5-10 | 7,849 | 7,849 | 0.470 | 0.847 | 5.06 | 10.3 |
| 11-20 | 4,943 | 4,943 | 0.360 | 0.814 | 3.91 | 10.8 |
| 21-50 | 3,814 | 3,814 | 0.257 | 0.752 | 2.89 | 11.5 |
| >50 | 1,096 | 1,096 | 0.179 | 0.660 | 1.82 | 10.6 |

By T5 history-length bin (distinct eval-history baskets of the user):

History baskets (2nd basket on) (basket-weighted / user-weighted):

| history length | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 10,131 | 4,143 | 0.647 / 0.652 | 0.918 / 0.919 | 6.32 / 6.35 | 9.4 / 9.3 |
| (4,6] | 13,084 | 2,933 | 0.580 / 0.581 | 0.892 / 0.892 | 5.81 / 5.83 | 9.6 / 9.6 |
| (6,10] | 28,139 | 3,841 | 0.510 / 0.514 | 0.867 / 0.869 | 5.27 / 5.31 | 9.9 / 9.9 |
| (10,20] | 61,567 | 4,470 | 0.425 / 0.432 | 0.827 / 0.830 | 4.48 / 4.54 | 10.2 / 10.1 |
| (20,inf) | 180,360 | 4,613 | 0.286 / 0.308 | 0.720 / 0.747 | 2.95 / 3.24 | 10.1 / 10.3 |

History baskets with >= 3 distinct items (2nd basket on) (basket-weighted / user-weighted):

| history length | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 8,680 | 3,888 | 0.670 / 0.680 | 0.969 / 0.969 | 7.24 / 7.07 | 10.7 / 10.3 |
| (4,6] | 11,298 | 2,861 | 0.601 / 0.611 | 0.949 / 0.948 | 6.62 / 6.37 | 10.9 / 10.3 |
| (6,10] | 24,554 | 3,789 | 0.529 / 0.539 | 0.924 / 0.922 | 5.95 / 5.75 | 11.1 / 10.5 |
| (10,20] | 54,475 | 4,435 | 0.439 / 0.452 | 0.882 / 0.881 | 5.00 / 4.88 | 11.3 / 10.7 |
| (20,inf) | 162,977 | 4,602 | 0.292 / 0.318 | 0.765 / 0.784 | 3.23 / 3.44 | 11.0 / 10.8 |

Target basket:

| history length | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 4,143 | 4,143 | 0.557 | 0.882 | 5.67 | 9.7 |
| (4,6] | 2,933 | 2,933 | 0.482 | 0.852 | 5.17 | 10.2 |
| (6,10] | 3,841 | 3,841 | 0.419 | 0.832 | 4.54 | 10.5 |
| (10,20] | 4,470 | 4,470 | 0.353 | 0.808 | 3.84 | 10.9 |
| (20,inf) | 4,613 | 4,613 | 0.235 | 0.728 | 2.59 | 11.3 |

### Complete Journey, eval split (main)

- Users 2,069; history baskets 206,682; target baskets 2,069; baskets analysed 206,682 (first baskets dropped 2,069). History length (distinct history baskets): min 4, median 76, max 1167.
- basket_seq starts at 1 for 100% of the users and is consecutive for 100%; target basket_seq minus last history basket_seq: 1: 1,674, 2: 266, 3: 75, 4: 35, 5: 8, 6: 7, 7: 3, 15: 1 (395 users have baskets between history and target that are in neither frame).
- Share of history baskets with >= 3 distinct items 0.727; smallest target basket 3 items. Users with >= 1 new target item: 1,971 here, 1,971 by nbr.data.split_target (equal).

Overall:

| basket set | baskets | users | weighting | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| history baskets (2nd basket on) | 204,613 | 2,069 | basket-weighted | 0.542 | 0.851 | 5.52 | 10.1 |
| history baskets (2nd basket on) | 204,613 | 2,069 | user-weighted | 0.624 | 0.907 | 6.84 | 11.2 |
| history baskets with >= 3 distinct items (2nd basket on) | 148,544 | 2,069 | basket-weighted | 0.550 | 0.945 | 7.31 | 13.4 |
| history baskets with >= 3 distinct items (2nd basket on) | 148,544 | 2,069 | user-weighted | 0.629 | 0.965 | 8.37 | 13.5 |
| target basket | 2,069 | 2,069 | per user | 0.545 | 0.953 | 7.29 | 13.9 |

Distribution of n_new (basket-weighted shares):

| basket set | n_new = 0 | n_new = 1 | n_new = 2 | n_new >= 3 |
| --- | --- | --- | --- | --- |
| history baskets (2nd basket on) | 0.149 | 0.189 | 0.134 | 0.528 |
| history baskets with >= 3 distinct items (2nd basket on) | 0.055 | 0.095 | 0.123 | 0.728 |
| target basket | 0.047 | 0.091 | 0.114 | 0.747 |

By basket position (position = basket_seq):

History baskets (2nd basket on) (basket-weighted / user-weighted):

| position | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| 2-4 | 6,207 | 2,069 | 0.897 / 0.897 | 0.980 / 0.980 | 10.13 / 10.13 | 11.3 / 11.3 |
| 5-10 | 12,355 | 2,068 | 0.784 / 0.784 | 0.957 / 0.957 | 8.50 / 8.50 | 10.8 / 10.8 |
| 11-20 | 19,936 | 2,038 | 0.698 / 0.699 | 0.934 / 0.933 | 7.59 / 7.60 | 11.0 / 11.0 |
| 21-50 | 50,171 | 1,931 | 0.596 / 0.608 | 0.903 / 0.908 | 6.51 / 6.73 | 11.1 / 11.3 |
| >50 | 115,944 | 1,384 | 0.446 / 0.496 | 0.797 / 0.866 | 4.17 / 5.22 | 9.4 / 11.1 |

History baskets with >= 3 distinct items (2nd basket on) (basket-weighted / user-weighted):

| position | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| 2-4 | 4,695 | 1,979 | 0.900 / 0.906 | 0.999 / 0.999 | 12.98 / 12.65 | 14.4 / 14.0 |
| 5-10 | 9,406 | 2,050 | 0.787 / 0.796 | 0.994 / 0.993 | 10.80 / 10.47 | 13.8 / 13.2 |
| 11-20 | 15,077 | 2,030 | 0.700 / 0.711 | 0.986 / 0.984 | 9.70 / 9.35 | 14.1 / 13.4 |
| 21-50 | 38,213 | 1,927 | 0.597 / 0.619 | 0.972 / 0.973 | 8.27 / 8.23 | 14.2 / 13.6 |
| >50 | 81,153 | 1,380 | 0.452 / 0.494 | 0.917 / 0.941 | 5.68 / 6.43 | 12.8 / 13.4 |

Target basket:

| position | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| 2-4 | 0 | 0 | - | - | - | - |
| 5-10 | 22 | 22 | 0.793 | 1.000 | 12.18 | 15.3 |
| 11-20 | 101 | 101 | 0.775 | 0.990 | 11.29 | 14.8 |
| 21-50 | 541 | 541 | 0.667 | 0.989 | 9.45 | 15.0 |
| >50 | 1,405 | 1,405 | 0.477 | 0.935 | 6.09 | 13.4 |

By T5 history-length bin (distinct eval-history baskets of the user):

History baskets (2nd basket on) (basket-weighted / user-weighted):

| history length | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| (0,20] | 1,809 | 138 | 0.826 / 0.835 | 0.963 / 0.965 | 8.75 / 8.72 | 10.5 / 10.4 |
| (20,50] | 19,044 | 547 | 0.723 / 0.730 | 0.947 / 0.948 | 8.26 / 8.35 | 11.7 / 11.7 |
| (50,100] | 47,944 | 653 | 0.608 / 0.612 | 0.917 / 0.918 | 6.97 / 7.05 | 11.8 / 11.9 |
| (100,inf) | 135,816 | 731 | 0.489 / 0.516 | 0.813 / 0.855 | 4.58 / 5.17 | 9.3 / 10.2 |

History baskets with >= 3 distinct items (2nd basket on) (basket-weighted / user-weighted):

| history length | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| (0,20] | 1,367 | 138 | 0.828 / 0.842 | 0.992 / 0.990 | 11.17 / 10.67 | 13.4 / 12.6 |
| (20,50] | 14,772 | 547 | 0.721 / 0.736 | 0.988 / 0.988 | 10.33 / 10.10 | 14.7 / 14.0 |
| (50,100] | 37,680 | 653 | 0.602 / 0.615 | 0.972 / 0.972 | 8.61 / 8.46 | 14.7 / 14.1 |
| (100,inf) | 94,725 | 731 | 0.498 / 0.521 | 0.928 / 0.938 | 6.27 / 6.55 | 12.7 / 12.8 |

Target basket:

| history length | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| (0,20] | 138 | 138 | 0.767 | 0.993 | 10.89 | 14.3 |
| (20,50] | 547 | 547 | 0.666 | 0.989 | 9.40 | 15.0 |
| (50,100] | 653 | 653 | 0.527 | 0.959 | 6.92 | 13.8 |
| (100,inf) | 731 | 731 | 0.428 | 0.912 | 5.35 | 13.0 |

### Fit split (robustness check)

### Instacart, fit split (robustness check)

- Users 20,000; history baskets 293,281; target baskets 20,000; baskets analysed 293,281 (first baskets dropped 20,000). History length (distinct history baskets): min 2, median 8, max 98.
- basket_seq starts at 1 for 100% of the users and is consecutive for 100%; target basket_seq minus last history basket_seq: 1: 20,000 (0 users have baskets between history and target that are in neither frame).
- Share of history baskets with >= 3 distinct items 0.893; smallest target basket 1 items. Users with >= 1 new target item: 16,294 here, 16,294 by nbr.data.split_target (equal).

Overall:

| basket set | baskets | users | weighting | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| history baskets (2nd basket on) | 273,281 | 20,000 | basket-weighted | 0.359 | 0.768 | 3.70 | 10.0 |
| history baskets (2nd basket on) | 273,281 | 20,000 | user-weighted | 0.505 | 0.853 | 5.10 | 9.8 |
| history baskets with >= 3 distinct items (2nd basket on) | 244,289 | 19,324 | basket-weighted | 0.367 | 0.815 | 4.09 | 11.0 |
| history baskets with >= 3 distinct items (2nd basket on) | 244,289 | 19,324 | user-weighted | 0.520 | 0.898 | 5.53 | 10.6 |
| target basket | 20,000 | 20,000 | per user | 0.410 | 0.815 | 4.27 | 10.3 |

Distribution of n_new (basket-weighted shares):

| basket set | n_new = 0 | n_new = 1 | n_new = 2 | n_new >= 3 |
| --- | --- | --- | --- | --- |
| history baskets (2nd basket on) | 0.232 | 0.161 | 0.123 | 0.484 |
| history baskets with >= 3 distinct items (2nd basket on) | 0.185 | 0.147 | 0.127 | 0.542 |
| target basket | 0.185 | 0.147 | 0.117 | 0.550 |

By basket position (position = basket_seq):

History baskets (2nd basket on) (basket-weighted / user-weighted):

| position | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| 2-4 | 53,559 | 20,000 | 0.617 / 0.626 | 0.922 / 0.923 | 6.28 / 6.34 | 9.9 / 9.8 |
| 5-10 | 68,625 | 14,276 | 0.410 / 0.423 | 0.831 / 0.836 | 4.27 / 4.40 | 10.1 / 10.0 |
| 11-20 | 62,997 | 8,422 | 0.294 / 0.309 | 0.743 / 0.755 | 3.10 / 3.27 | 10.2 / 10.2 |
| 21-50 | 68,262 | 4,355 | 0.220 / 0.244 | 0.659 / 0.704 | 2.27 / 2.62 | 10.1 / 10.6 |
| >50 | 19,838 | 993 | 0.174 / 0.185 | 0.589 / 0.628 | 1.61 / 1.82 | 9.3 / 10.1 |

History baskets with >= 3 distinct items (2nd basket on) (basket-weighted / user-weighted):

| position | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| 2-4 | 47,201 | 18,994 | 0.633 / 0.646 | 0.966 / 0.966 | 7.03 / 6.91 | 11.0 / 10.6 |
| 5-10 | 60,845 | 13,812 | 0.423 / 0.440 | 0.888 / 0.887 | 4.76 / 4.76 | 11.1 / 10.7 |
| 11-20 | 56,503 | 8,267 | 0.301 / 0.320 | 0.794 / 0.798 | 3.41 / 3.49 | 11.2 / 10.8 |
| 21-50 | 61,911 | 4,305 | 0.224 / 0.250 | 0.701 / 0.736 | 2.47 / 2.77 | 11.0 / 11.1 |
| >50 | 17,829 | 982 | 0.179 / 0.190 | 0.635 / 0.662 | 1.76 / 1.92 | 10.1 / 10.5 |

Target basket:

| position | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| 2-4 | 4,143 | 4,143 | 0.590 | 0.900 | 5.84 | 9.4 |
| 5-10 | 6,774 | 6,774 | 0.456 | 0.838 | 4.81 | 10.1 |
| 11-20 | 4,470 | 4,470 | 0.349 | 0.798 | 3.82 | 10.8 |
| 21-50 | 3,574 | 3,574 | 0.255 | 0.745 | 2.75 | 11.2 |
| >50 | 1,039 | 1,039 | 0.181 | 0.631 | 1.68 | 10.4 |

By T5 history-length bin (distinct fit-history baskets of the user):

History baskets (2nd basket on) (basket-weighted / user-weighted):

| history length | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 10,731 | 5,724 | 0.658 / 0.677 | 0.921 / 0.927 | 6.45 / 6.59 | 9.4 / 9.4 |
| (4,6] | 11,263 | 2,523 | 0.576 / 0.577 | 0.900 / 0.901 | 5.82 / 5.82 | 9.7 / 9.7 |
| (6,10] | 24,392 | 3,331 | 0.504 / 0.507 | 0.862 / 0.863 | 5.20 / 5.23 | 9.8 / 9.8 |
| (10,20] | 56,050 | 4,067 | 0.421 / 0.428 | 0.825 / 0.828 | 4.44 / 4.49 | 10.1 / 10.1 |
| (20,inf) | 170,845 | 4,355 | 0.285 / 0.306 | 0.718 / 0.744 | 2.93 / 3.22 | 10.1 / 10.3 |

History baskets with >= 3 distinct items (2nd basket on) (basket-weighted / user-weighted):

| history length | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 9,225 | 5,193 | 0.681 / 0.701 | 0.971 / 0.973 | 7.37 / 7.38 | 10.7 / 10.4 |
| (4,6] | 9,811 | 2,467 | 0.592 / 0.600 | 0.951 / 0.950 | 6.58 / 6.31 | 11.0 / 10.4 |
| (6,10] | 21,289 | 3,284 | 0.523 / 0.534 | 0.920 / 0.918 | 5.87 / 5.66 | 11.1 / 10.5 |
| (10,20] | 49,662 | 4,036 | 0.435 / 0.448 | 0.880 / 0.878 | 4.95 / 4.83 | 11.2 / 10.7 |
| (20,inf) | 154,302 | 4,344 | 0.290 / 0.316 | 0.762 / 0.782 | 3.21 / 3.41 | 11.0 / 10.7 |

Target basket:

| history length | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 5,724 | 5,724 | 0.566 | 0.885 | 5.71 | 9.6 |
| (4,6] | 2,523 | 2,523 | 0.471 | 0.842 | 4.98 | 10.2 |
| (6,10] | 3,331 | 3,331 | 0.409 | 0.828 | 4.31 | 10.2 |
| (10,20] | 4,067 | 4,067 | 0.341 | 0.794 | 3.72 | 10.8 |
| (20,inf) | 4,355 | 4,355 | 0.234 | 0.715 | 2.45 | 11.0 |

### Complete Journey, fit split (robustness check)

- Users 2,069; history baskets 180,368; target baskets 2,069; baskets analysed 180,368 (first baskets dropped 2,069). History length (distinct history baskets): min 3, median 66, max 1135.
- basket_seq starts at 1 for 100% of the users and is consecutive for 100%; target basket_seq minus last history basket_seq: 1: 1,673, 2: 280, 3: 77, 4: 17, 5: 12, 6: 4, 7: 1, 8: 1, 9: 1, 10: 1, 13: 1, 18: 1 (396 users have baskets between history and target that are in neither frame).
- Share of history baskets with >= 3 distinct items 0.725; smallest target basket 3 items. Users with >= 1 new target item: 1,965 here, 1,965 by nbr.data.split_target (equal).

Overall:

| basket set | baskets | users | weighting | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| history baskets (2nd basket on) | 178,299 | 2,069 | basket-weighted | 0.552 | 0.855 | 5.62 | 10.1 |
| history baskets (2nd basket on) | 178,299 | 2,069 | user-weighted | 0.638 | 0.911 | 6.97 | 11.1 |
| history baskets with >= 3 distinct items (2nd basket on) | 129,008 | 2,069 | basket-weighted | 0.561 | 0.949 | 7.47 | 13.4 |
| history baskets with >= 3 distinct items (2nd basket on) | 129,008 | 2,069 | user-weighted | 0.644 | 0.968 | 8.52 | 13.4 |
| target basket | 2,069 | 2,069 | per user | 0.555 | 0.950 | 7.92 | 14.8 |

Distribution of n_new (basket-weighted shares):

| basket set | n_new = 0 | n_new = 1 | n_new = 2 | n_new >= 3 |
| --- | --- | --- | --- | --- |
| history baskets (2nd basket on) | 0.145 | 0.188 | 0.134 | 0.533 |
| history baskets with >= 3 distinct items (2nd basket on) | 0.051 | 0.092 | 0.121 | 0.736 |
| target basket | 0.050 | 0.077 | 0.124 | 0.749 |

By basket position (position = basket_seq):

History baskets (2nd basket on) (basket-weighted / user-weighted):

| position | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| 2-4 | 6,205 | 2,069 | 0.897 / 0.897 | 0.980 / 0.980 | 10.13 / 10.13 | 11.3 / 11.3 |
| 5-10 | 12,264 | 2,065 | 0.783 / 0.784 | 0.957 / 0.957 | 8.50 / 8.51 | 10.8 / 10.8 |
| 11-20 | 19,440 | 2,009 | 0.697 / 0.700 | 0.933 / 0.934 | 7.54 / 7.56 | 11.0 / 11.0 |
| 21-50 | 46,811 | 1,863 | 0.592 / 0.606 | 0.901 / 0.907 | 6.44 / 6.63 | 11.1 / 11.2 |
| >50 | 93,579 | 1,272 | 0.449 / 0.496 | 0.794 / 0.865 | 4.13 / 5.21 | 9.2 / 11.0 |

History baskets with >= 3 distinct items (2nd basket on) (basket-weighted / user-weighted):

| position | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| 2-4 | 4,693 | 1,979 | 0.900 / 0.906 | 0.999 / 0.999 | 12.98 / 12.65 | 14.4 / 14.0 |
| 5-10 | 9,332 | 2,046 | 0.786 / 0.797 | 0.994 / 0.994 | 10.81 / 10.48 | 13.8 / 13.2 |
| 11-20 | 14,680 | 1,991 | 0.698 / 0.709 | 0.986 / 0.985 | 9.65 / 9.32 | 14.0 / 13.3 |
| 21-50 | 35,593 | 1,851 | 0.593 / 0.616 | 0.971 / 0.972 | 8.20 / 8.15 | 14.1 / 13.6 |
| >50 | 64,710 | 1,266 | 0.456 / 0.495 | 0.918 / 0.940 | 5.69 / 6.39 | 12.6 / 13.4 |

Target basket:

| position | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| 2-4 | 2 | 2 | 0.742 | 1.000 | 9.00 | 11.5 |
| 5-10 | 47 | 47 | 0.856 | 1.000 | 11.34 | 13.4 |
| 11-20 | 139 | 139 | 0.793 | 0.986 | 13.25 | 17.0 |
| 21-50 | 596 | 596 | 0.666 | 0.978 | 9.39 | 14.8 |
| >50 | 1,285 | 1,285 | 0.466 | 0.931 | 6.53 | 14.7 |

By T5 history-length bin (distinct fit-history baskets of the user):

History baskets (2nd basket on) (basket-weighted / user-weighted):

| history length | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| (0,20] | 2,512 | 206 | 0.819 / 0.832 | 0.965 / 0.967 | 8.87 / 8.79 | 10.8 / 10.5 |
| (20,50] | 19,880 | 591 | 0.716 / 0.726 | 0.944 / 0.946 | 8.17 / 8.23 | 11.7 / 11.6 |
| (50,100] | 50,893 | 699 | 0.602 / 0.606 | 0.915 / 0.917 | 6.87 / 6.96 | 11.9 / 11.9 |
| (100,inf) | 105,014 | 573 | 0.490 / 0.517 | 0.806 / 0.848 | 4.45 / 5.02 | 8.9 / 9.8 |

History baskets with >= 3 distinct items (2nd basket on) (basket-weighted / user-weighted):

| history length | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| (0,20] | 1,902 | 206 | 0.821 / 0.843 | 0.996 / 0.995 | 11.32 / 10.82 | 13.8 / 12.8 |
| (20,50] | 15,403 | 591 | 0.713 / 0.730 | 0.987 / 0.986 | 10.24 / 9.94 | 14.6 / 13.9 |
| (50,100] | 39,990 | 699 | 0.595 / 0.609 | 0.971 / 0.970 | 8.49 / 8.34 | 14.7 / 14.1 |
| (100,inf) | 71,713 | 573 | 0.503 / 0.525 | 0.927 / 0.937 | 6.20 / 6.45 | 12.3 / 12.4 |

Target basket:

| history length | baskets | users | mean new_share | P(any_new) | mean n_new | mean size |
| --- | --- | --- | --- | --- | --- | --- |
| (0,20] | 206 | 206 | 0.802 | 0.990 | 12.39 | 15.7 |
| (20,50] | 591 | 591 | 0.663 | 0.978 | 9.44 | 14.9 |
| (50,100] | 699 | 699 | 0.508 | 0.959 | 7.50 | 15.7 |
| (100,inf) | 573 | 573 | 0.412 | 0.895 | 5.25 | 13.4 |

## B. Hit probability by rank (eval split)

P(hit at rank j) = share of all users whose list has an item at rank j that is in their target basket (users without an item at the rank count as no hit); 'present' = share of users with an item at the rank; 'given present' = hits / users with an item. R2 = LR repeat ranker (genuine repeat candidates only, so short histories run out of items), E2 = user-CF explore list (unseen items, padded with popularity; every E2 hit is a new-item hit).

Instacart (20,000 users); expected hits per user: R2 ranks 1-10 2.964, E2 ranks 1-3 0.096:

| rank | R2 P(hit) | R2 present | R2 P(hit given present) | E2 P(hit) | E2 present | E2 P(hit given present) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.524 | 1.000 | 0.524 | 0.040 | 1.000 | 0.040 |
| 2 | 0.424 | 0.998 | 0.425 | 0.029 | 1.000 | 0.029 |
| 3 | 0.359 | 0.994 | 0.361 | 0.027 | 1.000 | 0.027 |
| 4 | 0.315 | 0.990 | 0.319 | - | - | - |
| 5 | 0.279 | 0.983 | 0.283 | - | - | - |
| 6 | 0.252 | 0.974 | 0.258 | - | - | - |
| 7 | 0.228 | 0.965 | 0.237 | - | - | - |
| 8 | 0.213 | 0.957 | 0.223 | - | - | - |
| 9 | 0.193 | 0.945 | 0.204 | - | - | - |
| 10 | 0.177 | 0.935 | 0.189 | - | - | - |
| 11 | 0.163 | 0.925 | 0.176 | - | - | - |
| 12 | 0.159 | 0.914 | 0.174 | - | - | - |

Complete Journey (2,069 users); expected hits per user: R2 ranks 1-10 1.499, E2 ranks 1-3 0.038:

| rank | R2 P(hit) | R2 present | R2 P(hit given present) | E2 P(hit) | E2 present | E2 P(hit given present) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.290 | 1.000 | 0.290 | 0.014 | 1.000 | 0.014 |
| 2 | 0.232 | 1.000 | 0.232 | 0.014 | 1.000 | 0.014 |
| 3 | 0.168 | 1.000 | 0.168 | 0.011 | 1.000 | 0.011 |
| 4 | 0.164 | 1.000 | 0.164 | - | - | - |
| 5 | 0.132 | 1.000 | 0.132 | - | - | - |
| 6 | 0.119 | 1.000 | 0.119 | - | - | - |
| 7 | 0.117 | 1.000 | 0.117 | - | - | - |
| 8 | 0.100 | 1.000 | 0.100 | - | - | - |
| 9 | 0.080 | 1.000 | 0.080 | - | - | - |
| 10 | 0.098 | 1.000 | 0.098 | - | - | - |
| 11 | 0.069 | 1.000 | 0.069 | - | - | - |
| 12 | 0.081 | 1.000 | 0.081 | - | - | - |

Slot comparisons (paired bootstrap over all users; diff = explore hits minus repeat hits in the slots that change):

| dataset | comparison | repeat hits (mean) | explore hits (mean) | diff [95 % CI] | users |
| --- | --- | --- | --- | --- | --- |
| Instacart | explore rank 1 minus repeat rank 10 (the slot that changes under r = 9) | 0.177 | 0.040 | -0.1366 [-0.1427, -0.1305] | 20,000 |
| Instacart | explore ranks 1 + 2 minus repeat ranks 9 + 10 (the slots that change under r = 8) | 0.369 | 0.069 | -0.3004 [-0.3090, -0.2913] | 20,000 |
| Complete Journey | explore rank 1 minus repeat rank 10 (the slot that changes under r = 9) | 0.098 | 0.014 | -0.0841 [-0.0976, -0.0706] | 2,069 |
| Complete Journey | explore ranks 1 + 2 minus repeat ranks 9 + 10 (the slots that change under r = 8) | 0.178 | 0.028 | -0.1503 [-0.1701, -0.1319] | 2,069 |

By T5 history-length bin (P(hit) at the slot-relevant ranks and the two comparisons):

Instacart:

| history length | users | R2 rank 9 | R2 rank 10 | E2 rank 1 | E2 rank 2 | E2 1 - R2 10 [CI] | E2 1+2 - R2 9+10 [CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 4,143 | 0.150 | 0.139 | 0.055 | 0.038 | -0.0842 [-0.0968, -0.0722] | -0.1967 [-0.2153, -0.1786] |
| (4,6] | 2,933 | 0.162 | 0.164 | 0.047 | 0.041 | -0.1169 [-0.1333, -0.1019] | -0.2376 [-0.2608, -0.2151] |
| (6,10] | 3,841 | 0.186 | 0.170 | 0.045 | 0.032 | -0.1247 [-0.1385, -0.1106] | -0.2783 [-0.2981, -0.2583] |
| (10,20] | 4,470 | 0.209 | 0.185 | 0.036 | 0.023 | -0.1488 [-0.1613, -0.1360] | -0.3340 [-0.3537, -0.3161] |
| (20,inf) | 4,613 | 0.242 | 0.216 | 0.021 | 0.017 | -0.1945 [-0.2070, -0.1810] | -0.4195 [-0.4379, -0.4006] |

Complete Journey:

| history length | users | R2 rank 9 | R2 rank 10 | E2 rank 1 | E2 rank 2 | E2 1 - R2 10 [CI] | E2 1+2 - R2 9+10 [CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (0,20] | 138 | 0.080 | 0.072 | 0.036 | 0.051 | -0.0362 [-0.0942, +0.0147] | -0.0652 [-0.1522, +0.0145] |
| (20,50] | 547 | 0.079 | 0.097 | 0.026 | 0.016 | -0.0713 [-0.0988, -0.0438] | -0.1335 [-0.1755, -0.0951] |
| (50,100] | 653 | 0.074 | 0.096 | 0.009 | 0.012 | -0.0873 [-0.1103, -0.0658] | -0.1485 [-0.1792, -0.1164] |
| (100,inf) | 731 | 0.086 | 0.105 | 0.005 | 0.005 | -0.0999 [-0.1218, -0.0752] | -0.1806 [-0.2134, -0.1477] |

## C. Forced exploration slot (eval split)

Final list = nbr.combine.combine(repeat, E2, r): the first r slots from the repeat module (fewer when the user has fewer repeat candidates; E2 items fill up), the remaining 10 - r slots from E2. Metrics: NDCG@10 and Recall@10 over all users; new-item Recall@10 over users with >= 1 new target item; share of users (all) with >= 1 new-item hit in the top 10; repeat Recall@10 over users with >= 1 repeat target item. 'repeat list' = repeat Recall@10 of the repeat module's own list.

Instacart (20,000 users; new-target users 16,319, repeat-target users 18,643):

| repeat module | r | NDCG@10 | Recall@10 | new-item Recall@10 | share with new hit | repeat Recall@10 | repeat list |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R2 (LR repeat ranker) | 10 | 0.4274 | 0.3515 | 0.0022 | 0.0040 | 0.5757 | 0.5757 |
| R2 (LR repeat ranker) | 9 | 0.4177 | 0.3391 | 0.0121 | 0.0425 | 0.5484 | 0.5757 |
| R2 (LR repeat ranker) | 8 | 0.4056 | 0.3238 | 0.0190 | 0.0675 | 0.5184 | 0.5757 |
| R1 gamma 0.8 (gamma*) | 10 | 0.4210 | 0.3465 | 0.0022 | 0.0040 | 0.5630 | 0.5630 |
| R1 gamma 0.8 (gamma*) | 9 | 0.4115 | 0.3342 | 0.0121 | 0.0425 | 0.5361 | 0.5630 |
| R1 gamma 0.8 (gamma*) | 8 | 0.3998 | 0.3194 | 0.0190 | 0.0675 | 0.5069 | 0.5630 |
| R1 gamma 0.9 | 10 | 0.4177 | 0.3446 | 0.0022 | 0.0040 | 0.5611 | 0.5611 |
| R1 gamma 0.9 | 9 | 0.4081 | 0.3322 | 0.0121 | 0.0425 | 0.5340 | 0.5611 |
| R1 gamma 0.9 | 8 | 0.3964 | 0.3171 | 0.0190 | 0.0675 | 0.5038 | 0.5611 |
| R1 gamma 0.97 | 10 | 0.4062 | 0.3359 | 0.0022 | 0.0040 | 0.5494 | 0.5494 |
| R1 gamma 0.97 | 9 | 0.3970 | 0.3238 | 0.0121 | 0.0425 | 0.5228 | 0.5494 |
| R1 gamma 0.97 | 8 | 0.3856 | 0.3090 | 0.0190 | 0.0675 | 0.4931 | 0.5494 |
| R1 gamma 1 | 10 | 0.3931 | 0.3244 | 0.0022 | 0.0040 | 0.5345 | 0.5345 |
| R1 gamma 1 | 9 | 0.3840 | 0.3119 | 0.0121 | 0.0425 | 0.5077 | 0.5345 |
| R1 gamma 1 | 8 | 0.3729 | 0.2974 | 0.0190 | 0.0675 | 0.4786 | 0.5345 |

Complete Journey (2,069 users; new-target users 1,971, repeat-target users 1,867):

| repeat module | r | NDCG@10 | Recall@10 | new-item Recall@10 | share with new hit | repeat Recall@10 | repeat list |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R2 (LR repeat ranker) | 10 | 0.2046 | 0.1315 | 0.0000 | 0.0000 | 0.2740 | 0.2740 |
| R2 (LR repeat ranker) | 9 | 0.1988 | 0.1254 | 0.0015 | 0.0140 | 0.2600 | 0.2740 |
| R2 (LR repeat ranker) | 8 | 0.1939 | 0.1200 | 0.0033 | 0.0275 | 0.2469 | 0.2740 |
| R1 gamma 0.9 | 10 | 0.1971 | 0.1262 | 0.0000 | 0.0000 | 0.2579 | 0.2579 |
| R1 gamma 0.9 | 9 | 0.1915 | 0.1203 | 0.0015 | 0.0140 | 0.2426 | 0.2579 |
| R1 gamma 0.9 | 8 | 0.1861 | 0.1145 | 0.0033 | 0.0275 | 0.2282 | 0.2579 |
| R1 gamma 0.97 | 10 | 0.2055 | 0.1328 | 0.0000 | 0.0000 | 0.2717 | 0.2717 |
| R1 gamma 0.97 | 9 | 0.2001 | 0.1267 | 0.0015 | 0.0140 | 0.2569 | 0.2717 |
| R1 gamma 0.97 | 8 | 0.1932 | 0.1198 | 0.0033 | 0.0275 | 0.2406 | 0.2717 |
| R1 gamma 1 (gamma*) | 10 | 0.1859 | 0.1187 | 0.0000 | 0.0000 | 0.2497 | 0.2497 |
| R1 gamma 1 (gamma*) | 9 | 0.1812 | 0.1130 | 0.0015 | 0.0140 | 0.2355 | 0.2497 |
| R1 gamma 1 (gamma*) | 8 | 0.1760 | 0.1075 | 0.0033 | 0.0275 | 0.2221 | 0.2497 |

Differences against r = 10 (paired bootstrap, diff [95 % CI]):

Instacart:

| repeat module | comparison | NDCG@10 | Recall@10 | new-item Recall@10 | share with new hit | repeat Recall@10 |
| --- | --- | --- | --- | --- | --- | --- |
| R2 (LR repeat ranker) | r = 9 vs r = 10 | -0.0097 [-0.0101, -0.0093] | -0.0124 [-0.0131, -0.0117] | +0.0099 [+0.0089, +0.0109] | +0.0385 [+0.0358, +0.0411] | -0.0273 [-0.0285, -0.0262] |
| R2 (LR repeat ranker) | r = 8 vs r = 10 | -0.0218 [-0.0224, -0.0211] | -0.0277 [-0.0287, -0.0266] | +0.0168 [+0.0155, +0.0181] | +0.0635 [+0.0600, +0.0669] | -0.0573 [-0.0589, -0.0557] |
| R1 gamma 0.8 (gamma*) | r = 9 vs r = 10 | -0.0096 [-0.0100, -0.0092] | -0.0124 [-0.0132, -0.0116] | +0.0099 [+0.0089, +0.0109] | +0.0385 [+0.0358, +0.0411] | -0.0269 [-0.0282, -0.0257] |
| R1 gamma 0.8 (gamma*) | r = 8 vs r = 10 | -0.0212 [-0.0219, -0.0206] | -0.0271 [-0.0283, -0.0260] | +0.0168 [+0.0155, +0.0181] | +0.0635 [+0.0600, +0.0669] | -0.0562 [-0.0579, -0.0545] |
| R1 gamma 0.9 | r = 9 vs r = 10 | -0.0096 [-0.0100, -0.0092] | -0.0124 [-0.0132, -0.0117] | +0.0099 [+0.0089, +0.0109] | +0.0385 [+0.0358, +0.0411] | -0.0271 [-0.0283, -0.0259] |
| R1 gamma 0.9 | r = 8 vs r = 10 | -0.0214 [-0.0220, -0.0207] | -0.0275 [-0.0287, -0.0264] | +0.0168 [+0.0155, +0.0181] | +0.0635 [+0.0600, +0.0669] | -0.0572 [-0.0591, -0.0555] |
| R1 gamma 0.97 | r = 9 vs r = 10 | -0.0092 [-0.0097, -0.0088] | -0.0121 [-0.0129, -0.0113] | +0.0099 [+0.0089, +0.0109] | +0.0385 [+0.0358, +0.0411] | -0.0266 [-0.0278, -0.0254] |
| R1 gamma 0.97 | r = 8 vs r = 10 | -0.0207 [-0.0213, -0.0200] | -0.0269 [-0.0281, -0.0258] | +0.0168 [+0.0155, +0.0181] | +0.0635 [+0.0600, +0.0669] | -0.0562 [-0.0581, -0.0546] |
| R1 gamma 1 | r = 9 vs r = 10 | -0.0091 [-0.0095, -0.0087] | -0.0124 [-0.0133, -0.0116] | +0.0099 [+0.0089, +0.0109] | +0.0385 [+0.0358, +0.0411] | -0.0269 [-0.0281, -0.0256] |
| R1 gamma 1 | r = 8 vs r = 10 | -0.0203 [-0.0210, -0.0197] | -0.0269 [-0.0282, -0.0258] | +0.0168 [+0.0155, +0.0181] | +0.0635 [+0.0600, +0.0669] | -0.0559 [-0.0575, -0.0541] |

Complete Journey:

| repeat module | comparison | NDCG@10 | Recall@10 | new-item Recall@10 | share with new hit | repeat Recall@10 |
| --- | --- | --- | --- | --- | --- | --- |
| R2 (LR repeat ranker) | r = 9 vs r = 10 | -0.0058 [-0.0068, -0.0049] | -0.0062 [-0.0077, -0.0048] | +0.0015 [+0.0009, +0.0023] | +0.0140 [+0.0092, +0.0198] | -0.0140 [-0.0167, -0.0115] |
| R2 (LR repeat ranker) | r = 8 vs r = 10 | -0.0107 [-0.0122, -0.0094] | -0.0115 [-0.0137, -0.0096] | +0.0033 [+0.0022, +0.0044] | +0.0275 [+0.0203, +0.0348] | -0.0271 [-0.0309, -0.0231] |
| R1 gamma 0.9 | r = 9 vs r = 10 | -0.0055 [-0.0066, -0.0045] | -0.0059 [-0.0074, -0.0045] | +0.0015 [+0.0009, +0.0023] | +0.0140 [+0.0092, +0.0198] | -0.0153 [-0.0188, -0.0121] |
| R1 gamma 0.9 | r = 8 vs r = 10 | -0.0110 [-0.0124, -0.0095] | -0.0118 [-0.0138, -0.0096] | +0.0033 [+0.0022, +0.0044] | +0.0275 [+0.0203, +0.0348] | -0.0298 [-0.0346, -0.0255] |
| R1 gamma 0.97 | r = 9 vs r = 10 | -0.0054 [-0.0064, -0.0045] | -0.0061 [-0.0076, -0.0046] | +0.0015 [+0.0009, +0.0023] | +0.0140 [+0.0092, +0.0198] | -0.0148 [-0.0185, -0.0116] |
| R1 gamma 0.97 | r = 8 vs r = 10 | -0.0122 [-0.0138, -0.0107] | -0.0130 [-0.0152, -0.0108] | +0.0033 [+0.0022, +0.0044] | +0.0275 [+0.0203, +0.0348] | -0.0311 [-0.0361, -0.0266] |
| R1 gamma 1 (gamma*) | r = 9 vs r = 10 | -0.0047 [-0.0057, -0.0038] | -0.0057 [-0.0072, -0.0045] | +0.0015 [+0.0009, +0.0023] | +0.0140 [+0.0092, +0.0198] | -0.0142 [-0.0172, -0.0110] |
| R1 gamma 1 (gamma*) | r = 8 vs r = 10 | -0.0099 [-0.0112, -0.0085] | -0.0113 [-0.0133, -0.0093] | +0.0033 [+0.0022, +0.0044] | +0.0275 [+0.0203, +0.0348] | -0.0275 [-0.0317, -0.0232] |

By T5 history-length bin (diff [95 % CI]; NDCG@10 and the share of users with >= 1 new-item hit):

Instacart, NDCG@10, r = 9 vs r = 10:

| history length | users | R2 (LR repeat ranker) | R1 gamma 0.8 (gamma*) | R1 gamma 0.9 | R1 gamma 0.97 | R1 gamma 1 |
| --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 4,143 | -0.0063 [-0.0072, -0.0055] | -0.0055 [-0.0064, -0.0046] | -0.0055 [-0.0064, -0.0046] | -0.0055 [-0.0064, -0.0046] | -0.0063 [-0.0072, -0.0054] |
| (4,6] | 2,933 | -0.0085 [-0.0098, -0.0074] | -0.0081 [-0.0092, -0.0070] | -0.0080 [-0.0092, -0.0070] | -0.0080 [-0.0092, -0.0070] | -0.0076 [-0.0088, -0.0065] |
| (6,10] | 3,841 | -0.0089 [-0.0099, -0.0078] | -0.0088 [-0.0098, -0.0078] | -0.0087 [-0.0098, -0.0077] | -0.0088 [-0.0099, -0.0078] | -0.0084 [-0.0094, -0.0074] |
| (10,20] | 4,470 | -0.0103 [-0.0112, -0.0093] | -0.0108 [-0.0117, -0.0099] | -0.0101 [-0.0110, -0.0092] | -0.0092 [-0.0101, -0.0083] | -0.0097 [-0.0107, -0.0088] |
| (20,inf) | 4,613 | -0.0136 [-0.0145, -0.0126] | -0.0136 [-0.0145, -0.0127] | -0.0144 [-0.0153, -0.0134] | -0.0136 [-0.0146, -0.0128] | -0.0126 [-0.0135, -0.0117] |

Instacart, NDCG@10, r = 8 vs r = 10:

| history length | users | R2 (LR repeat ranker) | R1 gamma 0.8 (gamma*) | R1 gamma 0.9 | R1 gamma 0.97 | R1 gamma 1 |
| --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 4,143 | -0.0147 [-0.0160, -0.0134] | -0.0138 [-0.0152, -0.0124] | -0.0138 [-0.0152, -0.0124] | -0.0138 [-0.0152, -0.0124] | -0.0138 [-0.0152, -0.0125] |
| (4,6] | 2,933 | -0.0178 [-0.0194, -0.0161] | -0.0179 [-0.0197, -0.0163] | -0.0177 [-0.0195, -0.0159] | -0.0177 [-0.0195, -0.0159] | -0.0179 [-0.0197, -0.0162] |
| (6,10] | 3,841 | -0.0203 [-0.0219, -0.0187] | -0.0203 [-0.0219, -0.0188] | -0.0199 [-0.0214, -0.0184] | -0.0199 [-0.0215, -0.0184] | -0.0192 [-0.0207, -0.0178] |
| (10,20] | 4,470 | -0.0237 [-0.0250, -0.0223] | -0.0232 [-0.0245, -0.0218] | -0.0235 [-0.0248, -0.0220] | -0.0214 [-0.0227, -0.0200] | -0.0219 [-0.0234, -0.0205] |
| (20,inf) | 4,613 | -0.0302 [-0.0315, -0.0287] | -0.0287 [-0.0301, -0.0274] | -0.0297 [-0.0311, -0.0282] | -0.0286 [-0.0300, -0.0272] | -0.0270 [-0.0283, -0.0255] |

Instacart, share with new hit, r = 9 vs r = 10:

| history length | users | R2 (LR repeat ranker) | R1 gamma 0.8 (gamma*) | R1 gamma 0.9 | R1 gamma 0.97 | R1 gamma 1 |
| --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 4,143 | +0.0495 [+0.0430, +0.0562] | +0.0495 [+0.0430, +0.0562] | +0.0495 [+0.0430, +0.0562] | +0.0495 [+0.0430, +0.0562] | +0.0495 [+0.0430, +0.0562] |
| (4,6] | 2,933 | +0.0460 [+0.0389, +0.0535] | +0.0460 [+0.0389, +0.0535] | +0.0460 [+0.0389, +0.0535] | +0.0460 [+0.0389, +0.0535] | +0.0460 [+0.0389, +0.0535] |
| (6,10] | 3,841 | +0.0445 [+0.0378, +0.0508] | +0.0445 [+0.0378, +0.0508] | +0.0445 [+0.0378, +0.0508] | +0.0445 [+0.0378, +0.0508] | +0.0445 [+0.0378, +0.0508] |
| (10,20] | 4,470 | +0.0358 [+0.0306, +0.0414] | +0.0358 [+0.0306, +0.0414] | +0.0358 [+0.0306, +0.0414] | +0.0358 [+0.0306, +0.0414] | +0.0358 [+0.0306, +0.0414] |
| (20,inf) | 4,613 | +0.0215 [+0.0173, +0.0258] | +0.0215 [+0.0173, +0.0258] | +0.0215 [+0.0173, +0.0258] | +0.0215 [+0.0173, +0.0258] | +0.0215 [+0.0173, +0.0258] |

Instacart, share with new hit, r = 8 vs r = 10:

| history length | users | R2 (LR repeat ranker) | R1 gamma 0.8 (gamma*) | R1 gamma 0.9 | R1 gamma 0.97 | R1 gamma 1 |
| --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 4,143 | +0.0784 [+0.0705, +0.0864] | +0.0784 [+0.0705, +0.0864] | +0.0784 [+0.0705, +0.0864] | +0.0784 [+0.0705, +0.0864] | +0.0784 [+0.0705, +0.0864] |
| (4,6] | 2,933 | +0.0801 [+0.0706, +0.0897] | +0.0801 [+0.0706, +0.0897] | +0.0801 [+0.0706, +0.0897] | +0.0801 [+0.0706, +0.0897] | +0.0801 [+0.0706, +0.0897] |
| (6,10] | 3,841 | +0.0737 [+0.0659, +0.0812] | +0.0737 [+0.0659, +0.0812] | +0.0737 [+0.0659, +0.0812] | +0.0737 [+0.0659, +0.0812] | +0.0737 [+0.0659, +0.0812] |
| (10,20] | 4,470 | +0.0568 [+0.0506, +0.0638] | +0.0568 [+0.0506, +0.0638] | +0.0568 [+0.0506, +0.0638] | +0.0568 [+0.0506, +0.0638] | +0.0568 [+0.0506, +0.0638] |
| (20,inf) | 4,613 | +0.0373 [+0.0321, +0.0427] | +0.0373 [+0.0321, +0.0427] | +0.0373 [+0.0321, +0.0427] | +0.0373 [+0.0321, +0.0427] | +0.0373 [+0.0321, +0.0427] |

Complete Journey, NDCG@10, r = 9 vs r = 10:

| history length | users | R2 (LR repeat ranker) | R1 gamma 0.9 | R1 gamma 0.97 | R1 gamma 1 (gamma*) |
| --- | --- | --- | --- | --- | --- |
| (0,20] | 138 | -0.0015 [-0.0055, +0.0028] | -0.0029 [-0.0074, +0.0022] | -0.0010 [-0.0055, +0.0035] | -0.0002 [-0.0038, +0.0039] |
| (20,50] | 547 | -0.0046 [-0.0065, -0.0028] | -0.0052 [-0.0070, -0.0033] | -0.0042 [-0.0062, -0.0023] | -0.0028 [-0.0046, -0.0012] |
| (50,100] | 653 | -0.0063 [-0.0080, -0.0048] | -0.0055 [-0.0071, -0.0040] | -0.0058 [-0.0074, -0.0043] | -0.0056 [-0.0072, -0.0040] |
| (100,inf) | 731 | -0.0071 [-0.0088, -0.0054] | -0.0063 [-0.0077, -0.0047] | -0.0068 [-0.0084, -0.0052] | -0.0063 [-0.0078, -0.0048] |

Complete Journey, NDCG@10, r = 8 vs r = 10:

| history length | users | R2 (LR repeat ranker) | R1 gamma 0.9 | R1 gamma 0.97 | R1 gamma 1 (gamma*) |
| --- | --- | --- | --- | --- | --- |
| (0,20] | 138 | -0.0031 [-0.0092, +0.0024] | -0.0052 [-0.0125, +0.0011] | -0.0020 [-0.0079, +0.0039] | +0.0001 [-0.0049, +0.0055] |
| (20,50] | 547 | -0.0090 [-0.0119, -0.0063] | -0.0093 [-0.0120, -0.0066] | -0.0095 [-0.0126, -0.0067] | -0.0073 [-0.0099, -0.0048] |
| (50,100] | 653 | -0.0109 [-0.0132, -0.0085] | -0.0113 [-0.0137, -0.0089] | -0.0134 [-0.0157, -0.0109] | -0.0109 [-0.0132, -0.0085] |
| (100,inf) | 731 | -0.0133 [-0.0158, -0.0109] | -0.0130 [-0.0154, -0.0107] | -0.0152 [-0.0179, -0.0126] | -0.0128 [-0.0152, -0.0104] |

Complete Journey, share with new hit, r = 9 vs r = 10:

| history length | users | R2 (LR repeat ranker) | R1 gamma 0.9 | R1 gamma 0.97 | R1 gamma 1 (gamma*) |
| --- | --- | --- | --- | --- | --- |
| (0,20] | 138 | +0.0362 [+0.0072, +0.0725] | +0.0362 [+0.0072, +0.0725] | +0.0362 [+0.0072, +0.0725] | +0.0362 [+0.0072, +0.0725] |
| (20,50] | 547 | +0.0256 [+0.0146, +0.0384] | +0.0256 [+0.0146, +0.0384] | +0.0256 [+0.0146, +0.0384] | +0.0256 [+0.0146, +0.0384] |
| (50,100] | 653 | +0.0092 [+0.0031, +0.0168] | +0.0092 [+0.0031, +0.0168] | +0.0092 [+0.0031, +0.0168] | +0.0092 [+0.0031, +0.0168] |
| (100,inf) | 731 | +0.0055 [+0.0014, +0.0109] | +0.0055 [+0.0014, +0.0109] | +0.0055 [+0.0014, +0.0109] | +0.0055 [+0.0014, +0.0109] |

Complete Journey, share with new hit, r = 8 vs r = 10:

| history length | users | R2 (LR repeat ranker) | R1 gamma 0.9 | R1 gamma 0.97 | R1 gamma 1 (gamma*) |
| --- | --- | --- | --- | --- | --- |
| (0,20] | 138 | +0.0870 [+0.0435, +0.1377] | +0.0870 [+0.0435, +0.1377] | +0.0870 [+0.0435, +0.1377] | +0.0870 [+0.0435, +0.1377] |
| (20,50] | 547 | +0.0420 [+0.0274, +0.0603] | +0.0420 [+0.0274, +0.0603] | +0.0420 [+0.0274, +0.0603] | +0.0420 [+0.0274, +0.0603] |
| (50,100] | 653 | +0.0214 [+0.0107, +0.0337] | +0.0214 [+0.0107, +0.0337] | +0.0214 [+0.0107, +0.0337] | +0.0214 [+0.0107, +0.0337] |
| (100,inf) | 731 | +0.0109 [+0.0041, +0.0192] | +0.0109 [+0.0041, +0.0192] | +0.0109 [+0.0041, +0.0192] | +0.0109 [+0.0041, +0.0192] |

## C. Fit-split check: NDCG@10 by r and the selected r per repeat module

r* = pipeline.select_r (argmax over r in 0..10, ties -> larger r). R2 is scored with the C* model fitted on the LR training users and evaluated on the validation users (pipeline.tune_integration); R1 modules on all fit users. 'tuning document' = the same curve recorded by python -m nbr tune (final_lr_ucf for R2, final_star_ucf for gamma*).

Instacart: LR refit C = 0.001 (saved artifact C = 0.001, equal), max |coefficient difference| 0.00e+00; validation users 4,000.

| repeat module | users | n | r=0 | r=1 | r=2 | r=3 | r=4 | r=5 | r=6 | r=7 | r=8 | r=9 | r=10 | r* | r* < 10 | tuning document |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2 (LR repeat ranker) | LR validation users (train-only model) | 4,000 | 0.0292 | 0.1737 | 0.2437 | 0.2898 | 0.3229 | 0.3472 | 0.3672 | 0.3830 | 0.3963 | 0.4071 | 0.4177 | 10 | no | final_lr_ucf: equal |
| R1 gamma 0.8 (gamma*) | all fit users | 20,000 | 0.0301 | 0.1778 | 0.2479 | 0.2929 | 0.3253 | 0.3498 | 0.3695 | 0.3861 | 0.3997 | 0.4112 | 0.4206 | 10 | no | final_star_ucf: equal |
| R1 gamma 0.9 | all fit users | 20,000 | 0.0301 | 0.1782 | 0.2473 | 0.2914 | 0.3234 | 0.3477 | 0.3674 | 0.3838 | 0.3975 | 0.4091 | 0.4188 | 10 | no | - |
| R1 gamma 0.97 | all fit users | 20,000 | 0.0301 | 0.1756 | 0.2423 | 0.2848 | 0.3156 | 0.3392 | 0.3583 | 0.3744 | 0.3878 | 0.3990 | 0.4083 | 10 | no | - |
| R1 gamma 1 | all fit users | 20,000 | 0.0301 | 0.1717 | 0.2365 | 0.2771 | 0.3067 | 0.3296 | 0.3483 | 0.3640 | 0.3768 | 0.3878 | 0.3967 | 10 | no | - |

Complete Journey: LR refit C = 0.01 (saved artifact C = 0.01, equal), max |coefficient difference| 0.00e+00; validation users 414.

| repeat module | users | n | r=0 | r=1 | r=2 | r=3 | r=4 | r=5 | r=6 | r=7 | r=8 | r=9 | r=10 | r* | r* < 10 | tuning document |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2 (LR repeat ranker) | LR validation users (train-only model) | 414 | 0.0092 | 0.0899 | 0.1311 | 0.1519 | 0.1686 | 0.1824 | 0.1925 | 0.2016 | 0.2097 | 0.2156 | 0.2223 | 10 | no | final_lr_ucf: equal |
| R1 gamma 0.9 | all fit users | 2,069 | 0.0123 | 0.0885 | 0.1245 | 0.1460 | 0.1614 | 0.1736 | 0.1834 | 0.1913 | 0.1979 | 0.2031 | 0.2081 | 10 | no | - |
| R1 gamma 0.97 | all fit users | 2,069 | 0.0123 | 0.0903 | 0.1246 | 0.1481 | 0.1646 | 0.1770 | 0.1869 | 0.1953 | 0.2029 | 0.2085 | 0.2139 | 10 | no | - |
| R1 gamma 1 (gamma*) | all fit users | 2,069 | 0.0123 | 0.0830 | 0.1163 | 0.1356 | 0.1515 | 0.1626 | 0.1720 | 0.1795 | 0.1868 | 0.1920 | 0.1973 | 10 | no | final_star_ucf: equal |

## Reproduction checks (eval split, four decimals)

| dataset | quantity | expected | observed | matched |
| --- | --- | --- | --- | --- |
| Instacart | R2_E2_r10_ndcg | 0.4274 | 0.4274 | yes |
| Complete Journey | R2_E2_r10_ndcg | 0.2046 | 0.2046 | yes |
| Complete Journey | R1_g0.97_repeat_list_recall | 0.2717 | 0.2717 | yes |
| Complete Journey | R1_g1_repeat_list_recall | 0.2497 | 0.2497 | yes |

## Figures

- F-A `fig_new_rate_by_position.png` (numbers in `fig_new_rate_by_position.csv`): P(any_new) and mean new_share of the eval history baskets by basket position (basket-weighted), both datasets; dotted lines = target basket.
- F-B `fig_hit_probability_by_rank.png` (`fig_hit_probability_by_rank.csv`): hit probability by rank, R2 ranks 1-12 and E2 ranks 1-3, one panel per dataset.
- F-C `fig_slot_effects.png` (`fig_slot_effects.csv`): change in NDCG@10 and in the share of users with >= 1 new-item hit for r = 9 and r = 8 against r = 10 per repeat module with 95 % CIs, one column per dataset.

## Timing and memory

| dataset | eval load | A eval | E2 list | repeat lists | C lists + metrics | C bootstrap | B | fit load | A fit | C fit curves | peak working set (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Instacart | 5.6 s | 14.3 s | 77.6 s | 38.1 s | 87.3 s | 24.4 s | 1.4 s | 6.4 s | 13.0 s | 219.8 s | 2713 |
| Complete Journey | 5.2 s | 9.2 s | 27.5 s | 22.8 s | 7.2 s | 2.2 s | 0.2 s | 4.1 s | 8.1 s | 67.1 s | 2713 |
