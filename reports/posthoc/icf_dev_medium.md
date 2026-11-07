# Two-CF final system (PREREG v1.5): development check on medium

**development check of the PREREG v1.5 code; not confirmatory.** The W19 code (nbr.features_icf, nbr.v15; `python -m nbr tune` / `evaluate --skip-l3` with `lr_repeat.itemcf_feature: true`) is run on the development profile: R2' = R2 + icf (20 item neighbours, C selected separately on the validation users of R2), C4 = combine(R2', E2, r) with r re-selected like C3's, and the v15 block (H14 non-inferiority with delta = 0.003, H15; CI levels 0.95, 0.975). The existing selections are compared with the W8b reference and C3 must reproduce the W8b NDCG@10. This is a code check, not a selection: nothing is changed on its basis. The holdout4 run (reports/final4) is the confirmatory test.

- Code commit `655bb325aecd1a201c26cee65d92f425438495c5`, 2026-11-07T13:18:42.003694+00:00; profile `medium`; CI levels 0.95, 0.975 (primary level 0.975 = 1 - 0.05/2); paired bootstrap n = 1000, seed 42.
- Pipeline run by this script: **yes**; tuning document commit `655bb325aecd1a201c26cee65d92f425438495c5`; W8b reference `C:\dev_ML-wt-w8\reports\metrics`.
- Expected from the ICF_feature_check development study (approximate, repeat Recall@10 R2 -> R2'): Instacart 0.5757 -> 0.5759; Complete Journey 0.2740 -> 0.2739.
- All checks passed: **yes**.

## Instacart

### Setup check (existing selections vs the W8b reference; LR artifacts)

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
| R2 artifact sha256 (tuning / file / eval) | 0bf51f8852cb | 0bf51f8852cb / 0bf51f8852cb | yes |
| R2' artifact sha256 (tuning / file / eval) | d9da752a481d | d9da752a481d / d9da752a481d | yes |
| R2' validation users = R2 validation users; 13 features (R2 + icf); icf neighbours | 4000 / 13 features / 20 | 4000 / 13 / 20 | yes |

### Reproduction check (C3 = final_lr_ucf NDCG@10)

| expected (4 decimals) | observed | W8b reference | v15 comparator mean | matched |
| --- | --- | --- | --- | --- |
| 0.4274 | 0.427383 | 0.427383 | 0.427383 | yes |

### Selections of the v1.5 systems (fit split; development data)

| quantity | value | grid | at grid edge | selection score |
| --- | --- | --- | --- | --- |
| C (R2) | 0.001 | [0.001, 0.01, 0.1, 1.0, 10.0] | True | validation repeat Recall@10 by C: 0.001: 0.586217, 0.01: 0.585798, 0.1: 0.585815, 1: 0.585815, 10: 0.585815 |
| C (R2') | 0.1 | [0.001, 0.01, 0.1, 1.0, 10.0] | False | validation repeat Recall@10 by C: 0.001: 0.585917, 0.01: 0.585991, 0.1: 0.586055, 1: 0.586055, 10: 0.586055 |
| r (C3 = final_lr_ucf) | 10 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] | True | NDCG@10 on 4,000 validation users: r = 0: 0.029163, r = 1: 0.173668, r = 2: 0.243653, r = 3: 0.289845, r = 4: 0.322858, r = 5: 0.347167, r = 6: 0.367197, r = 7: 0.382953, r = 8: 0.396332, r = 9: 0.407065, r = 10: 0.417675 |
| r (C4 = final_lri_ucf) | 10 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] | True | NDCG@10 on 4,000 validation users: r = 0: 0.029163, r = 1: 0.173904, r = 2: 0.244179, r = 3: 0.290369, r = 4: 0.323133, r = 5: 0.347586, r = 6: 0.367310, r = 7: 0.383244, r = 8: 0.396560, r = 9: 0.407436, r = 10: 0.417796 |

### v15 block (20,000 users; 18,643 with a repeat target item, 16,319 with a new target item)

Primary (decision at 95 % / 97.5 %; H14 rule: non-inferior if the 97.5 % lower bound > -0.003):

| hypothesis | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 97.5 % CI | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H14_ndcg (NDCG@10, all users) | 20,000 | final_lr_ucf -> final_lri_ucf | 0.4274 | 0.4274 | -0.0000 | [-0.0002, +0.0001] | [-0.0002, +0.0002] | non-inferior / non-inferior |
| H15_ndcg (NDCG@10, all users) | 20,000 | personal_frequency -> final_lri_ucf | 0.3929 | 0.4274 | +0.0345 | [+0.0329, +0.0361] | [+0.0326, +0.0363] | improvement observed / improvement observed |

Secondary (CI only):

| comparison | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 97.5 % CI | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| repeat_lri_vs_repeat_lr_repeat_recall (repeat Recall@10, users with repeat target) | 18,643 | repeat_lr -> repeat_lri | 0.5757 | 0.5759 | +0.0002 | [-0.0001, +0.0005] | [-0.0002, +0.0005] | (CI only) |
| repeat_lri_vs_repeat_star_repeat_recall (repeat Recall@10, users with repeat target) | 18,643 | repeat_star -> repeat_lri | 0.5630 | 0.5759 | +0.0128 | [+0.0106, +0.0151] | [+0.0104, +0.0154] | (CI only) |
| final_lri_ucf_vs_final_lr_ucf_recall (Recall@10, all users) | 20,000 | final_lr_ucf -> final_lri_ucf | 0.3515 | 0.3516 | +0.0000 | [-0.0002, +0.0002] | [-0.0002, +0.0003] | (CI only) |

Repeat Recall@10 R2 -> R2': observed 0.5757 -> 0.5759 (expected approx. 0.5757 -> 0.5759 from the ICF_feature_check study).

R2' (C = 0.1, intercept -2.9062; standardized coefficients):

| feature | coefficient |
| --- | --- |
| dec_g0.5 | +0.1422 |
| dec_g0.8 | +0.2988 |
| dec_g1.0 | +0.0910 |
| recency | -0.3496 |
| mean_gap | -0.2282 |
| once | -0.2129 |
| overdue | -0.4542 |
| item_repeat_rate | +0.2476 |
| log_pop | +0.0608 |
| user_repeat_rate | +0.2265 |
| log_hist_len | -0.2544 |
| share | +0.0962 |
| icf | +0.0068 |

By history-length bin (H14: C4 vs C3, NDCG@10; diff [95 % CI]):

| bin | users | H14 NDCG@10 |
| --- | --- | --- |
| (0,4] | 4,143 | +0.0001 [-0.0002, +0.0003] (n = 4,143) |
| (4,6] | 2,933 | -0.0004 [-0.0009, +0.0001] (n = 2,933) |
| (6,10] | 3,841 | +0.0003 [-0.0001, +0.0007] (n = 3,841) |
| (10,20] | 4,470 | -0.0001 [-0.0004, +0.0002] (n = 4,470) |
| (20,inf) | 4,613 | -0.0001 [-0.0004, +0.0002] (n = 4,613) |

### Exploration policy on C4 (PREREG v1.5.1, `policy_c4` block; T = 4, r_short = 9, r_default = r (C4) = 10; 4,143 users with history_len <= 4) next to the policy on C3 (`policy` block; r_default = 10)

Decision wording of PREREG v1.4 at 95 % / 98.33 % (secondary replication; the confirmatory policy result is H11-H13 on holdout3):

| base | hypothesis | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C4 H11_ndcg (NDCG@10, all users) | 20,000 | final_lri_ucf -> P_T | 0.4274 | 0.4261 | -0.0013 | [-0.0015, -0.0011] | [-0.0015, -0.0011] | non-inferior / non-inferior |
| C4 H12_new_hit_share (new-item hit share@10, all users) | 20,000 | final_lri_ucf -> P_T | 0.0040 | 0.0143 | +0.0103 | [+0.0089, +0.0117] | [+0.0087, +0.0119] | improvement observed / improvement observed |
| C4 H13_new_hit_share (new-item hit share@10, users with new target) | 16,319 | flat_explore -> usercf | 0.1704 | 0.2227 | +0.0523 | [+0.0471, +0.0571] | [+0.0461, +0.0580] | improvement observed / improvement observed |
| C3 H11_ndcg (NDCG@10, all users) | 20,000 | final_lr_ucf -> P_T | 0.4274 | 0.4261 | -0.0013 | [-0.0015, -0.0011] | [-0.0015, -0.0011] | non-inferior / non-inferior |
| C3 H12_new_hit_share (new-item hit share@10, all users) | 20,000 | final_lr_ucf -> P_T | 0.0040 | 0.0143 | +0.0103 | [+0.0089, +0.0117] | [+0.0087, +0.0119] | improvement observed / improvement observed |
| C3 H13_new_hit_share (new-item hit share@10, users with new target) | 16,319 | flat_explore -> usercf | 0.1704 | 0.2227 | +0.0523 | [+0.0471, +0.0571] | [+0.0461, +0.0580] | improvement observed / improvement observed |

Variants (CI only):

| base | variant | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C4 P_T4_r8 (T = 4, r_short = 8): NDCG@10, all users | 20,000 | final_lri_ucf -> P_T4_r8 | 0.4274 | 0.4243 | -0.0031 | [-0.0034, -0.0028] | [-0.0034, -0.0027] | (CI only) |
| C4 P_T4_r8 (T = 4, r_short = 8): new-item hit share@10, all users | 20,000 | final_lri_ucf -> P_T4_r8 | 0.0040 | 0.0203 | +0.0163 | [+0.0144, +0.0180] | [+0.0141, +0.0184] | (CI only) |
| C4 P_T6_r9 (T = 6, r_short = 9): NDCG@10, all users | 20,000 | final_lri_ucf -> P_T6_r9 | 0.4274 | 0.4248 | -0.0025 | [-0.0028, -0.0023] | [-0.0028, -0.0022] | (CI only) |
| C4 P_T6_r9 (T = 6, r_short = 9): new-item hit share@10, all users | 20,000 | final_lri_ucf -> P_T6_r9 | 0.0040 | 0.0210 | +0.0170 | [+0.0152, +0.0188] | [+0.0149, +0.0192] | (CI only) |
| C4 P_T10_r9 (T = 10, r_short = 9): NDCG@10, all users | 20,000 | final_lri_ucf -> P_T10_r9 | 0.4274 | 0.4231 | -0.0043 | [-0.0046, -0.0039] | [-0.0047, -0.0039] | (CI only) |
| C4 P_T10_r9 (T = 10, r_short = 9): new-item hit share@10, all users | 20,000 | final_lri_ucf -> P_T10_r9 | 0.0040 | 0.0296 | +0.0255 | [+0.0234, +0.0278] | [+0.0231, +0.0284] | (CI only) |
| C3 P_T4_r8 (T = 4, r_short = 8): NDCG@10, all users | 20,000 | final_lr_ucf -> P_T4_r8 | 0.4274 | 0.4243 | -0.0031 | [-0.0033, -0.0028] | [-0.0034, -0.0027] | (CI only) |
| C3 P_T4_r8 (T = 4, r_short = 8): new-item hit share@10, all users | 20,000 | final_lr_ucf -> P_T4_r8 | 0.0040 | 0.0203 | +0.0163 | [+0.0144, +0.0180] | [+0.0141, +0.0184] | (CI only) |
| C3 P_T6_r9 (T = 6, r_short = 9): NDCG@10, all users | 20,000 | final_lr_ucf -> P_T6_r9 | 0.4274 | 0.4248 | -0.0026 | [-0.0028, -0.0023] | [-0.0029, -0.0023] | (CI only) |
| C3 P_T6_r9 (T = 6, r_short = 9): new-item hit share@10, all users | 20,000 | final_lr_ucf -> P_T6_r9 | 0.0040 | 0.0210 | +0.0170 | [+0.0152, +0.0188] | [+0.0149, +0.0192] | (CI only) |
| C3 P_T10_r9 (T = 10, r_short = 9): NDCG@10, all users | 20,000 | final_lr_ucf -> P_T10_r9 | 0.4274 | 0.4231 | -0.0043 | [-0.0046, -0.0039] | [-0.0047, -0.0039] | (CI only) |
| C3 P_T10_r9 (T = 10, r_short = 9): new-item hit share@10, all users | 20,000 | final_lr_ucf -> P_T10_r9 | 0.0040 | 0.0296 | +0.0255 | [+0.0234, +0.0278] | [+0.0231, +0.0284] | (CI only) |

Expected order of magnitude from W15/W17 (P_T vs C3 on Instacart medium): delta NDCG@10 about -0.0013, delta new-hit share about +0.010; observed with the C4 base: -0.0013 and +0.0103.

## Complete Journey

### Setup check (existing selections vs the W8b reference; LR artifacts)

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
| R2 artifact sha256 (tuning / file / eval) | c71680afc3bb | c71680afc3bb / c71680afc3bb | yes |
| R2' artifact sha256 (tuning / file / eval) | d2966ce0700e | d2966ce0700e / d2966ce0700e | yes |
| R2' validation users = R2 validation users; 13 features (R2 + icf); icf neighbours | 414 / 13 features / 20 | 414 / 13 / 20 | yes |

### Reproduction check (C3 = final_lr_ucf NDCG@10)

| expected (4 decimals) | observed | W8b reference | v15 comparator mean | matched |
| --- | --- | --- | --- | --- |
| 0.2046 | 0.204637 | 0.204637 | 0.204637 | yes |

### Selections of the v1.5 systems (fit split; development data)

| quantity | value | grid | at grid edge | selection score |
| --- | --- | --- | --- | --- |
| C (R2) | 0.01 | [0.001, 0.01, 0.1, 1.0, 10.0] | False | validation repeat Recall@10 by C: 0.001: 0.287599, 0.01: 0.288590, 0.1: 0.285495, 1: 0.285242, 10: 0.285242 |
| C (R2') | 1.0 | [0.001, 0.01, 0.1, 1.0, 10.0] | False | validation repeat Recall@10 by C: 0.001: 0.287621, 0.01: 0.287959, 0.1: 0.288113, 1: 0.288277, 10: 0.288024 |
| r (C3 = final_lr_ucf) | 10 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] | True | NDCG@10 on 414 validation users: r = 0: 0.009230, r = 1: 0.089921, r = 2: 0.131096, r = 3: 0.151934, r = 4: 0.168564, r = 5: 0.182387, r = 6: 0.192492, r = 7: 0.201586, r = 8: 0.209696, r = 9: 0.215599, r = 10: 0.222253 |
| r (C4 = final_lri_ucf) | 10 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] | True | NDCG@10 on 414 validation users: r = 0: 0.009230, r = 1: 0.090585, r = 2: 0.131341, r = 3: 0.152569, r = 4: 0.168897, r = 5: 0.182508, r = 6: 0.192401, r = 7: 0.201699, r = 8: 0.209269, r = 9: 0.216178, r = 10: 0.222788 |

### v15 block (2,069 users; 1,867 with a repeat target item, 1,971 with a new target item)

Primary (decision at 95 % / 97.5 %; H14 rule: non-inferior if the 97.5 % lower bound > -0.003):

| hypothesis | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 97.5 % CI | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H14_ndcg (NDCG@10, all users) | 2,069 | final_lr_ucf -> final_lri_ucf | 0.2046 | 0.2048 | +0.0002 | [-0.0004, +0.0007] | [-0.0005, +0.0008] | non-inferior / non-inferior |
| H15_ndcg (NDCG@10, all users) | 2,069 | personal_frequency -> final_lri_ucf | 0.1859 | 0.2048 | +0.0189 | [+0.0147, +0.0231] | [+0.0138, +0.0235] | improvement observed / improvement observed |

Secondary (CI only):

| comparison | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 97.5 % CI | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| repeat_lri_vs_repeat_lr_repeat_recall (repeat Recall@10, users with repeat target) | 1,867 | repeat_lr -> repeat_lri | 0.2740 | 0.2739 | -0.0001 | [-0.0012, +0.0009] | [-0.0013, +0.0010] | (CI only) |
| repeat_lri_vs_repeat_star_repeat_recall (repeat Recall@10, users with repeat target) | 1,867 | repeat_star -> repeat_lri | 0.2497 | 0.2739 | +0.0242 | [+0.0156, +0.0333] | [+0.0146, +0.0347] | (CI only) |
| final_lri_ucf_vs_final_lr_ucf_recall (Recall@10, all users) | 2,069 | final_lr_ucf -> final_lri_ucf | 0.1315 | 0.1316 | +0.0001 | [-0.0005, +0.0007] | [-0.0005, +0.0008] | (CI only) |

Repeat Recall@10 R2 -> R2': observed 0.2740 -> 0.2739 (expected approx. 0.2740 -> 0.2739 from the ICF_feature_check study).

R2' (C = 1.0, intercept -5.0381; standardized coefficients):

| feature | coefficient |
| --- | --- |
| dec_g0.5 | -0.1200 |
| dec_g0.8 | +0.3105 |
| dec_g1.0 | +0.0525 |
| recency | -0.3018 |
| mean_gap | -0.3150 |
| once | -0.4549 |
| overdue | -0.3935 |
| item_repeat_rate | +0.1669 |
| log_pop | +0.1715 |
| user_repeat_rate | +0.2431 |
| log_hist_len | -0.1956 |
| share | +0.0585 |
| icf | -0.0124 |

By history-length bin (H14: C4 vs C3, NDCG@10; diff [95 % CI]):

| bin | users | H14 NDCG@10 |
| --- | --- | --- |
| (0,20] | 138 | +0.0006 [-0.0011, +0.0025] (n = 138) |
| (20,50] | 547 | -0.0008 [-0.0018, +0.0002] (n = 547) |
| (50,100] | 653 | +0.0001 [-0.0009, +0.0011] (n = 653) |
| (100,inf) | 731 | +0.0009 [+0.0000, +0.0018] (n = 731) |

### Exploration policy on C4 (PREREG v1.5.1, `policy_c4` block; T = 4, r_short = 9, r_default = r (C4) = 10; 1 users with history_len <= 4) next to the policy on C3 (`policy` block; r_default = 10)

Decision wording of PREREG v1.4 at 95 % / 98.33 % (secondary replication; the confirmatory policy result is H11-H13 on holdout3):

| base | hypothesis | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C4 H11_ndcg (NDCG@10, all users) | 2,069 | final_lri_ucf -> P_T | 0.2048 | 0.2048 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | non-inferior / non-inferior |
| C4 H12_new_hit_share (new-item hit share@10, all users) | 2,069 | final_lri_ucf -> P_T | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | no difference established / no difference established |
| C4 H13_new_hit_share (new-item hit share@10, users with new target) | 1,971 | flat_explore -> usercf | 0.0791 | 0.0979 | +0.0188 | [+0.0061, +0.0315] | [+0.0037, +0.0345] | improvement observed / improvement observed |
| C3 H11_ndcg (NDCG@10, all users) | 2,069 | final_lr_ucf -> P_T | 0.2046 | 0.2046 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | non-inferior / non-inferior |
| C3 H12_new_hit_share (new-item hit share@10, all users) | 2,069 | final_lr_ucf -> P_T | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | no difference established / no difference established |
| C3 H13_new_hit_share (new-item hit share@10, users with new target) | 1,971 | flat_explore -> usercf | 0.0791 | 0.0979 | +0.0188 | [+0.0061, +0.0315] | [+0.0037, +0.0345] | improvement observed / improvement observed |

Variants (CI only):

| base | variant | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C4 P_T4_r8 (T = 4, r_short = 8): NDCG@10, all users | 2,069 | final_lri_ucf -> P_T4_r8 | 0.2048 | 0.2048 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| C4 P_T4_r8 (T = 4, r_short = 8): new-item hit share@10, all users | 2,069 | final_lri_ucf -> P_T4_r8 | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| C4 P_T6_r9 (T = 6, r_short = 9): NDCG@10, all users | 2,069 | final_lri_ucf -> P_T6_r9 | 0.2048 | 0.2048 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| C4 P_T6_r9 (T = 6, r_short = 9): new-item hit share@10, all users | 2,069 | final_lri_ucf -> P_T6_r9 | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| C4 P_T10_r9 (T = 10, r_short = 9): NDCG@10, all users | 2,069 | final_lri_ucf -> P_T10_r9 | 0.2048 | 0.2047 | -0.0001 | [-0.0002, +0.0001] | [-0.0003, +0.0001] | (CI only) |
| C4 P_T10_r9 (T = 10, r_short = 9): new-item hit share@10, all users | 2,069 | final_lri_ucf -> P_T10_r9 | 0.0000 | 0.0005 | +0.0005 | [+0.0000, +0.0019] | [+0.0000, +0.0019] | (CI only) |
| C3 P_T4_r8 (T = 4, r_short = 8): NDCG@10, all users | 2,069 | final_lr_ucf -> P_T4_r8 | 0.2046 | 0.2046 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| C3 P_T4_r8 (T = 4, r_short = 8): new-item hit share@10, all users | 2,069 | final_lr_ucf -> P_T4_r8 | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| C3 P_T6_r9 (T = 6, r_short = 9): NDCG@10, all users | 2,069 | final_lr_ucf -> P_T6_r9 | 0.2046 | 0.2046 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| C3 P_T6_r9 (T = 6, r_short = 9): new-item hit share@10, all users | 2,069 | final_lr_ucf -> P_T6_r9 | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| C3 P_T10_r9 (T = 10, r_short = 9): NDCG@10, all users | 2,069 | final_lr_ucf -> P_T10_r9 | 0.2046 | 0.2046 | -0.0000 | [-0.0001, +0.0001] | [-0.0002, +0.0002] | (CI only) |
| C3 P_T10_r9 (T = 10, r_short = 9): new-item hit share@10, all users | 2,069 | final_lr_ucf -> P_T10_r9 | 0.0000 | 0.0005 | +0.0005 | [+0.0000, +0.0019] | [+0.0000, +0.0019] | (CI only) |

## Inputs

| file | sha256 | git_commit |
| --- | --- | --- |
| C:\dev_ML-wt-w19\reports\metrics\medium_instacart_tuning.json | e5bf8d7e56481bca1e9f1315df310b50138f9bdc49f4018c74559d0329cf72b6 | 655bb325aecd1a201c26cee65d92f425438495c5 |
| C:\dev_ML-wt-w19\reports\metrics\medium_instacart_eval.json | ec354b6378c7d00751b0f7fb9b723706f7d11168d26d7ac62808e83cf073a355 | 655bb325aecd1a201c26cee65d92f425438495c5 |
| C:\dev_ML-wt-w8\reports\metrics\medium_instacart_tuning.json | 87c84ea24f1d2953c272148c87ce15fd6f05ca231f79f2f0a4ad2aa71c04b25f | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C:\dev_ML-wt-w8\reports\metrics\medium_instacart_eval.json | 4e29f0d65d86947b5bd558fd8c3402708192f1571d73c2623b675bab4100582b | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C:\dev_ML-wt-w19\artifacts\medium\instacart_lr_repeat.joblib | 0bf51f8852cb80e410f80d2109ecb0fd0e889d79d1f7cd34b74413f4ee01cf9e |  |
| C:\dev_ML-wt-w19\artifacts\medium\instacart_lr_repeat_icf.joblib | d9da752a481de4068a9fbcf94552f9a67bf371cbde920713bfacb3a44360abdc |  |
| C:\dev_ML-wt-w19\reports\metrics\medium_complete_journey_tuning.json | 4cc370187cb7845523da26d5ddf0761604641684aa3b8672eac9a6495635e160 | 655bb325aecd1a201c26cee65d92f425438495c5 |
| C:\dev_ML-wt-w19\reports\metrics\medium_complete_journey_eval.json | fcf01cf479b487a6cda0d611d416197cc0b88c3bbedab0bdd12a9e3092da0dbf | 655bb325aecd1a201c26cee65d92f425438495c5 |
| C:\dev_ML-wt-w8\reports\metrics\medium_complete_journey_tuning.json | 0a215302c9e4778e0459a5ff52b19243fca04443f0f7108f708812e0467cdfec | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C:\dev_ML-wt-w8\reports\metrics\medium_complete_journey_eval.json | 01ec2b6f03bc97364af4f9e41d045eb878a54ea5059e98ebaf1a7bb0cc199b1c | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C:\dev_ML-wt-w19\artifacts\medium\complete_journey_lr_repeat.joblib | c71680afc3bbb98e5ed024e7a40b23d4bc1cd7c5c2c6df44016e4d47eb808dd4 |  |
| C:\dev_ML-wt-w19\artifacts\medium\complete_journey_lr_repeat_icf.joblib | d2966ce0700ef0d6ff8257d7cc3f577283e8377aeea8dc9579df1091e970e97e |  |
