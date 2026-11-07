# Exploration policy (PREREG v1.4): development check on medium

**development check of the PREREG v1.4 code; not confirmatory.** The policy code of W17 (nbr.policy, `python -m nbr evaluate --skip-l3 --policy`) is run on the development profile with the pre-registered settings T = 4, delta = 0.003, r_short = 9 (variants r_short = 8 and T in {6, 10}) and compared with the W8b selections and the W15 post-hoc numbers. This is a code check, not a selection: nothing is changed on its basis. The holdout3 run (reports/final3) is the confirmatory test.

- Code commit `0c7aa8960e34d2d1a6ada57b7d7c072430080603`, 2026-11-07T07:14:11.852067+00:00; profile `medium`; CI levels 0.95, 0.9833 (primary level 0.9833 = 1 - 0.05/3); paired bootstrap n = 1000, seed 42.
- Tuning documents reused from the W15 worktree (python -m nbr tune, commit `78ba1fd0dd6b366afb9c84806dba157759ef156b`), verified against the W8b reference `C:\dev_ML-wt-w8\reports\metrics`; W15 JSON commit `8964e01a252242f4029596fabb052a41b06e2854`.
- Expected from W15 (Instacart, approximate): overall delta NDCG@10 about -0.0013, delta new-hit share about +0.010.
- All checks passed: **yes**.

## Instacart

### Setup check (tuning selections vs the W8b reference; LR artifact)

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
| LR artifact sha256 (tuning / file / eval) | 0bf51f8852cb | 0bf51f8852cb / 0bf51f8852cb | yes |

### Reproduction check (C3 = final_lr_ucf NDCG@10, four decimals)

| expected | observed | policy comparator mean | matched |
| --- | --- | --- | --- |
| 0.4274 | 0.427383 | 0.427383 | yes |

### Policy block (20,000 users; 16,319 with a new target item, 18,643 with a repeat target item; 4,143 with history_len <= 4; by T: T = 4: 4,143, T = 6: 7,076, T = 10: 10,917; r* = 10)

Primary (decision at 95 % / 98.33 %; H11 rule: non-inferior if the 98.33 % lower bound > -0.003):

| hypothesis | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H11_ndcg (NDCG@10, all users) | 20,000 | final_lr_ucf -> P_T | 0.4274 | 0.4261 | -0.0013 | [-0.0015, -0.0011] | [-0.0015, -0.0011] | non-inferior / non-inferior |
| H12_new_hit_share (new-item hit share@10, all users) | 20,000 | final_lr_ucf -> P_T | 0.0040 | 0.0143 | +0.0103 | [+0.0089, +0.0117] | [+0.0087, +0.0119] | improvement observed / improvement observed |
| H13_new_hit_share (new-item hit share@10, users with new target) | 16,319 | flat_explore -> usercf | 0.1704 | 0.2227 | +0.0523 | [+0.0471, +0.0571] | [+0.0461, +0.0580] | improvement observed / improvement observed |

Secondary (CI only):

| comparison | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P_T_vs_final_lr_ucf_recall (Recall@10, all users) | 20,000 | final_lr_ucf -> P_T | 0.3515 | 0.3497 | -0.0018 | [-0.0021, -0.0015] | [-0.0022, -0.0014] | (CI only) |
| P_T_vs_final_lr_ucf_new_recall (new-item Recall@10, users with new target) | 16,319 | final_lr_ucf -> P_T | 0.0022 | 0.0042 | +0.0020 | [+0.0016, +0.0023] | [+0.0015, +0.0024] | (CI only) |
| P_T_vs_final_lr_ucf_repeat_recall (repeat Recall@10, users with repeat target) | 18,643 | final_lr_ucf -> P_T | 0.5757 | 0.5695 | -0.0062 | [-0.0068, -0.0055] | [-0.0069, -0.0053] | (CI only) |
| P_T_vs_final_lr_ucf_new_hit_share_new_users (new-item hit share@10, users with new target) | 16,319 | final_lr_ucf -> P_T | 0.0050 | 0.0175 | +0.0126 | [+0.0108, +0.0142] | [+0.0104, +0.0147] | (CI only) |
| usercf_vs_flat_explore_new_recall (new-item Recall@10, users with new target) | 16,319 | flat_explore -> usercf | 0.0398 | 0.0593 | +0.0195 | [+0.0175, +0.0214] | [+0.0171, +0.0218] | (CI only) |
| P_T4_r8 (T = 4, r_short = 8; 4,143 short users): NDCG@10, all users | 20,000 | final_lr_ucf -> P_T4_r8 | 0.4274 | 0.4243 | -0.0031 | [-0.0033, -0.0028] | [-0.0034, -0.0027] | (CI only) |
| P_T4_r8 (T = 4, r_short = 8; 4,143 short users): new-item hit share@10, all users | 20,000 | final_lr_ucf -> P_T4_r8 | 0.0040 | 0.0203 | +0.0163 | [+0.0144, +0.0180] | [+0.0141, +0.0184] | (CI only) |
| P_T4_r8 (T = 4, r_short = 8; 4,143 short users): Recall@10, all users | 20,000 | final_lr_ucf -> P_T4_r8 | 0.3515 | 0.3475 | -0.0040 | [-0.0045, -0.0035] | [-0.0046, -0.0034] | (CI only) |
| P_T4_r8 (T = 4, r_short = 8; 4,143 short users): new-item Recall@10, users with new target | 16,319 | final_lr_ucf -> P_T4_r8 | 0.0022 | 0.0056 | +0.0034 | [+0.0029, +0.0039] | [+0.0028, +0.0040] | (CI only) |
| P_T4_r8 (T = 4, r_short = 8; 4,143 short users): repeat Recall@10, users with repeat target | 18,643 | final_lr_ucf -> P_T4_r8 | 0.5757 | 0.5631 | -0.0126 | [-0.0136, -0.0116] | [-0.0138, -0.0114] | (CI only) |
| P_T4_r8 (T = 4, r_short = 8; 4,143 short users): new-item hit share@10, users with new target | 16,319 | final_lr_ucf -> P_T4_r8 | 0.0050 | 0.0249 | +0.0199 | [+0.0178, +0.0220] | [+0.0173, +0.0226] | (CI only) |
| P_T6_r9 (T = 6, r_short = 9; 7,076 short users): NDCG@10, all users | 20,000 | final_lr_ucf -> P_T6_r9 | 0.4274 | 0.4248 | -0.0026 | [-0.0028, -0.0023] | [-0.0029, -0.0023] | (CI only) |
| P_T6_r9 (T = 6, r_short = 9; 7,076 short users): new-item hit share@10, all users | 20,000 | final_lr_ucf -> P_T6_r9 | 0.0040 | 0.0210 | +0.0170 | [+0.0152, +0.0188] | [+0.0149, +0.0192] | (CI only) |
| P_T6_r9 (T = 6, r_short = 9; 7,076 short users): Recall@10, all users | 20,000 | final_lr_ucf -> P_T6_r9 | 0.3515 | 0.3480 | -0.0035 | [-0.0040, -0.0031] | [-0.0041, -0.0030] | (CI only) |
| P_T6_r9 (T = 6, r_short = 9; 7,076 short users): new-item Recall@10, users with new target | 16,319 | final_lr_ucf -> P_T6_r9 | 0.0022 | 0.0055 | +0.0033 | [+0.0029, +0.0038] | [+0.0028, +0.0040] | (CI only) |
| P_T6_r9 (T = 6, r_short = 9; 7,076 short users): repeat Recall@10, users with repeat target | 18,643 | final_lr_ucf -> P_T6_r9 | 0.5757 | 0.5649 | -0.0108 | [-0.0116, -0.0100] | [-0.0118, -0.0099] | (CI only) |
| P_T6_r9 (T = 6, r_short = 9; 7,076 short users): new-item hit share@10, users with new target | 16,319 | final_lr_ucf -> P_T6_r9 | 0.0050 | 0.0258 | +0.0208 | [+0.0186, +0.0232] | [+0.0181, +0.0235] | (CI only) |
| P_T10_r9 (T = 10, r_short = 9; 10,917 short users): NDCG@10, all users | 20,000 | final_lr_ucf -> P_T10_r9 | 0.4274 | 0.4231 | -0.0043 | [-0.0046, -0.0039] | [-0.0047, -0.0039] | (CI only) |
| P_T10_r9 (T = 10, r_short = 9; 10,917 short users): new-item hit share@10, all users | 20,000 | final_lr_ucf -> P_T10_r9 | 0.0040 | 0.0296 | +0.0255 | [+0.0234, +0.0278] | [+0.0231, +0.0284] | (CI only) |
| P_T10_r9 (T = 10, r_short = 9; 10,917 short users): Recall@10, all users | 20,000 | final_lr_ucf -> P_T10_r9 | 0.3515 | 0.3458 | -0.0057 | [-0.0063, -0.0051] | [-0.0065, -0.0051] | (CI only) |
| P_T10_r9 (T = 10, r_short = 9; 10,917 short users): new-item Recall@10, users with new target | 16,319 | final_lr_ucf -> P_T10_r9 | 0.0022 | 0.0081 | +0.0058 | [+0.0051, +0.0066] | [+0.0050, +0.0067] | (CI only) |
| P_T10_r9 (T = 10, r_short = 9; 10,917 short users): repeat Recall@10, users with repeat target | 18,643 | final_lr_ucf -> P_T10_r9 | 0.5757 | 0.5598 | -0.0159 | [-0.0169, -0.0150] | [-0.0171, -0.0148] | (CI only) |
| P_T10_r9 (T = 10, r_short = 9; 10,917 short users): new-item hit share@10, users with new target | 16,319 | final_lr_ucf -> P_T10_r9 | 0.0050 | 0.0363 | +0.0313 | [+0.0288, +0.0339] | [+0.0284, +0.0346] | (CI only) |

By history-length bin (P_T vs C3 and S vs flat; diff [95 % CI]; bins above T are identical lists):

| bin | users | H11 NDCG@10 | H12 new-hit share | H13 S vs flat new-hit share | Recall@10 | new-item Recall@10 | repeat Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 4,143 | -0.0063 [-0.0072, -0.0055] (n = 4,143) | +0.0495 [+0.0430, +0.0562] (n = 4,143) | +0.0588 [+0.0482, +0.0706] (n = 3,654) | -0.0086 [-0.0101, -0.0070] (n = 4,143) | +0.0088 [+0.0072, +0.0105] (n = 3,654) | -0.0320 [-0.0354, -0.0286] (n = 3,598) |
| (4,6] | 2,933 | +0.0000 [+0.0000, +0.0000] (n = 2,933) | +0.0000 [+0.0000, +0.0000] (n = 2,933) | +0.0580 [+0.0452, +0.0709] (n = 2,498) | +0.0000 [+0.0000, +0.0000] (n = 2,933) | +0.0000 [+0.0000, +0.0000] (n = 2,498) | +0.0000 [+0.0000, +0.0000] (n = 2,667) |
| (6,10] | 3,841 | +0.0000 [+0.0000, +0.0000] (n = 3,841) | +0.0000 [+0.0000, +0.0000] (n = 3,841) | +0.0554 [+0.0447, +0.0663] (n = 3,196) | +0.0000 [+0.0000, +0.0000] (n = 3,841) | +0.0000 [+0.0000, +0.0000] (n = 3,196) | +0.0000 [+0.0000, +0.0000] (n = 3,582) |
| (10,20] | 4,470 | +0.0000 [+0.0000, +0.0000] (n = 4,470) | +0.0000 [+0.0000, +0.0000] (n = 4,470) | +0.0520 [+0.0415, +0.0626] (n = 3,612) | +0.0000 [+0.0000, +0.0000] (n = 4,470) | +0.0000 [+0.0000, +0.0000] (n = 3,612) | +0.0000 [+0.0000, +0.0000] (n = 4,271) |
| (20,inf) | 4,613 | +0.0000 [+0.0000, +0.0000] (n = 4,613) | +0.0000 [+0.0000, +0.0000] (n = 4,613) | +0.0384 [+0.0283, +0.0482] (n = 3,359) | +0.0000 [+0.0000, +0.0000] (n = 4,613) | +0.0000 [+0.0000, +0.0000] (n = 3,359) | +0.0000 [+0.0000, +0.0000] (n = 4,525) |

### Agreement with the W15 part C numbers (R2 + E2; |gap| <= 1e-09): **yes** (25 checkable comparisons, 0 not applicable because T is not a bin edge of this dataset)

| comparison | metric | policy diff | W15-implied diff | gap | agree |
| --- | --- | --- | --- | --- | --- |
| P_T vs C3 (H11) | ndcg | -0.001313 | -0.001313 | 0.0e+00 | yes |
| P_T vs C3 (H12) | new_hit_share | +0.010250 | +0.010250 | 0.0e+00 | yes |
| P_T vs C3 | recall | -0.001771 | -0.001771 | 0.0e+00 | yes |
| P_T vs C3 | new_recall | +0.001967 | +0.001967 | 0.0e+00 | yes |
| P_T vs C3 | repeat_recall | -0.006175 | -0.006175 | 0.0e+00 | yes |
| P_T4_r8 vs C3 | ndcg | -0.003054 | -0.003054 | 0.0e+00 | yes |
| P_T4_r8 vs C3 | new_hit_share | +0.016250 | +0.016250 | 0.0e+00 | yes |
| P_T4_r8 vs C3 | recall | -0.003989 | -0.003989 | 8.7e-19 | yes |
| P_T4_r8 vs C3 | new_recall | +0.003393 | +0.003393 | 4.3e-19 | yes |
| P_T4_r8 vs C3 | repeat_recall | -0.012564 | -0.012564 | 0.0e+00 | yes |
| P_T6_r9 vs C3 | ndcg | -0.002565 | -0.002565 | 4.3e-19 | yes |
| P_T6_r9 vs C3 | new_hit_share | +0.017000 | +0.017000 | 0.0e+00 | yes |
| P_T6_r9 vs C3 | recall | -0.003534 | -0.003534 | 0.0e+00 | yes |
| P_T6_r9 vs C3 | new_recall | +0.003319 | +0.003319 | 0.0e+00 | yes |
| P_T6_r9 vs C3 | repeat_recall | -0.010804 | -0.010804 | 0.0e+00 | yes |
| P_T10_r9 vs C3 | ndcg | -0.004269 | -0.004269 | 0.0e+00 | yes |
| P_T10_r9 vs C3 | new_hit_share | +0.025550 | +0.025550 | 0.0e+00 | yes |
| P_T10_r9 vs C3 | recall | -0.005729 | -0.005729 | 8.7e-19 | yes |
| P_T10_r9 vs C3 | new_recall | +0.005822 | +0.005822 | 8.7e-19 | yes |
| P_T10_r9 vs C3 | repeat_recall | -0.015905 | -0.015905 | 3.5e-18 | yes |

First bin (same users and bootstrap draw as W15):

| key | policy diff [95 % CI] (n) | W15 diff [95 % CI] (n) | agree |
| --- | --- | --- | --- |
| H11_ndcg | -0.0063 [-0.0072, -0.0055] (4,143) | -0.0063 [-0.0072, -0.0055] (4,143) | yes |
| H12_new_hit_share | +0.0495 [+0.0430, +0.0562] (4,143) | +0.0495 [+0.0430, +0.0562] (4,143) | yes |
| P_T_vs_final_lr_ucf_recall | -0.0086 [-0.0101, -0.0070] (4,143) | -0.0086 [-0.0101, -0.0070] (4,143) | yes |
| P_T_vs_final_lr_ucf_new_recall | +0.0088 [+0.0072, +0.0105] (3,654) | +0.0088 [+0.0072, +0.0105] (3,654) | yes |
| P_T_vs_final_lr_ucf_repeat_recall | -0.0320 [-0.0354, -0.0286] (3,598) | -0.0320 [-0.0354, -0.0286] (3,598) | yes |

C3 list metrics against the W15 R2 r = 10 row: ndcg 0.4274 vs 0.4274 (equal), recall 0.3515 vs 0.3515 (equal), new_recall 0.0022 vs 0.0022 (equal), new_hit_share 0.0040 vs 0.0040 (equal), repeat_recall 0.5757 vs 0.5757 (equal)

## Complete Journey

### Setup check (tuning selections vs the W8b reference; LR artifact)

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
| LR artifact sha256 (tuning / file / eval) | c71680afc3bb | c71680afc3bb / c71680afc3bb | yes |

### Reproduction check (C3 = final_lr_ucf NDCG@10, four decimals)

| expected | observed | policy comparator mean | matched |
| --- | --- | --- | --- |
| 0.2046 | 0.204637 | 0.204637 | yes |

### Policy block (2,069 users; 1,971 with a new target item, 1,867 with a repeat target item; 1 with history_len <= 4; by T: T = 4: 1, T = 6: 3, T = 10: 31; r* = 10)

Primary (decision at 95 % / 98.33 %; H11 rule: non-inferior if the 98.33 % lower bound > -0.003):

| hypothesis | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H11_ndcg (NDCG@10, all users) | 2,069 | final_lr_ucf -> P_T | 0.2046 | 0.2046 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | non-inferior / non-inferior |
| H12_new_hit_share (new-item hit share@10, all users) | 2,069 | final_lr_ucf -> P_T | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | no difference established / no difference established |
| H13_new_hit_share (new-item hit share@10, users with new target) | 1,971 | flat_explore -> usercf | 0.0791 | 0.0979 | +0.0188 | [+0.0061, +0.0315] | [+0.0037, +0.0345] | improvement observed / improvement observed |

Secondary (CI only):

| comparison | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P_T_vs_final_lr_ucf_recall (Recall@10, all users) | 2,069 | final_lr_ucf -> P_T | 0.1315 | 0.1315 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T_vs_final_lr_ucf_new_recall (new-item Recall@10, users with new target) | 1,971 | final_lr_ucf -> P_T | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T_vs_final_lr_ucf_repeat_recall (repeat Recall@10, users with repeat target) | 1,867 | final_lr_ucf -> P_T | 0.2740 | 0.2740 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T_vs_final_lr_ucf_new_hit_share_new_users (new-item hit share@10, users with new target) | 1,971 | final_lr_ucf -> P_T | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| usercf_vs_flat_explore_new_recall (new-item Recall@10, users with new target) | 1,971 | flat_explore -> usercf | 0.0130 | 0.0145 | +0.0015 | [-0.0014, +0.0044] | [-0.0021, +0.0051] | (CI only) |
| P_T4_r8 (T = 4, r_short = 8; 1 short users): NDCG@10, all users | 2,069 | final_lr_ucf -> P_T4_r8 | 0.2046 | 0.2046 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T4_r8 (T = 4, r_short = 8; 1 short users): new-item hit share@10, all users | 2,069 | final_lr_ucf -> P_T4_r8 | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T4_r8 (T = 4, r_short = 8; 1 short users): Recall@10, all users | 2,069 | final_lr_ucf -> P_T4_r8 | 0.1315 | 0.1315 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T4_r8 (T = 4, r_short = 8; 1 short users): new-item Recall@10, users with new target | 1,971 | final_lr_ucf -> P_T4_r8 | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T4_r8 (T = 4, r_short = 8; 1 short users): repeat Recall@10, users with repeat target | 1,867 | final_lr_ucf -> P_T4_r8 | 0.2740 | 0.2740 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T4_r8 (T = 4, r_short = 8; 1 short users): new-item hit share@10, users with new target | 1,971 | final_lr_ucf -> P_T4_r8 | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T6_r9 (T = 6, r_short = 9; 3 short users): NDCG@10, all users | 2,069 | final_lr_ucf -> P_T6_r9 | 0.2046 | 0.2046 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T6_r9 (T = 6, r_short = 9; 3 short users): new-item hit share@10, all users | 2,069 | final_lr_ucf -> P_T6_r9 | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T6_r9 (T = 6, r_short = 9; 3 short users): Recall@10, all users | 2,069 | final_lr_ucf -> P_T6_r9 | 0.1315 | 0.1315 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T6_r9 (T = 6, r_short = 9; 3 short users): new-item Recall@10, users with new target | 1,971 | final_lr_ucf -> P_T6_r9 | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T6_r9 (T = 6, r_short = 9; 3 short users): repeat Recall@10, users with repeat target | 1,867 | final_lr_ucf -> P_T6_r9 | 0.2740 | 0.2740 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T6_r9 (T = 6, r_short = 9; 3 short users): new-item hit share@10, users with new target | 1,971 | final_lr_ucf -> P_T6_r9 | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | (CI only) |
| P_T10_r9 (T = 10, r_short = 9; 31 short users): NDCG@10, all users | 2,069 | final_lr_ucf -> P_T10_r9 | 0.2046 | 0.2046 | -0.0000 | [-0.0001, +0.0001] | [-0.0002, +0.0002] | (CI only) |
| P_T10_r9 (T = 10, r_short = 9; 31 short users): new-item hit share@10, all users | 2,069 | final_lr_ucf -> P_T10_r9 | 0.0000 | 0.0005 | +0.0005 | [+0.0000, +0.0019] | [+0.0000, +0.0019] | (CI only) |
| P_T10_r9 (T = 10, r_short = 9; 31 short users): Recall@10, all users | 2,069 | final_lr_ucf -> P_T10_r9 | 0.1315 | 0.1316 | +0.0001 | [-0.0001, +0.0003] | [-0.0001, +0.0004] | (CI only) |
| P_T10_r9 (T = 10, r_short = 9; 31 short users): new-item Recall@10, users with new target | 1,971 | final_lr_ucf -> P_T10_r9 | 0.0000 | 0.0001 | +0.0001 | [+0.0000, +0.0004] | [+0.0000, +0.0005] | (CI only) |
| P_T10_r9 (T = 10, r_short = 9; 31 short users): repeat Recall@10, users with repeat target | 1,867 | final_lr_ucf -> P_T10_r9 | 0.2740 | 0.2739 | -0.0001 | [-0.0004, +0.0000] | [-0.0004, +0.0000] | (CI only) |
| P_T10_r9 (T = 10, r_short = 9; 31 short users): new-item hit share@10, users with new target | 1,971 | final_lr_ucf -> P_T10_r9 | 0.0000 | 0.0005 | +0.0005 | [+0.0000, +0.0015] | [+0.0000, +0.0020] | (CI only) |

By history-length bin (P_T vs C3 and S vs flat; diff [95 % CI]; bins above T are identical lists):

| bin | users | H11 NDCG@10 | H12 new-hit share | H13 S vs flat new-hit share | Recall@10 | new-item Recall@10 | repeat Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (0,20] | 138 | +0.0000 [+0.0000, +0.0000] (n = 138) | +0.0000 [+0.0000, +0.0000] (n = 138) | +0.0073 [-0.0513, +0.0657] (n = 137) | +0.0000 [+0.0000, +0.0000] (n = 138) | +0.0000 [+0.0000, +0.0000] (n = 137) | +0.0000 [+0.0000, +0.0000] (n = 100) |
| (20,50] | 547 | +0.0000 [+0.0000, +0.0000] (n = 547) | +0.0000 [+0.0000, +0.0000] (n = 547) | +0.0296 [+0.0000, +0.0610] (n = 541) | +0.0000 [+0.0000, +0.0000] (n = 547) | +0.0000 [+0.0000, +0.0000] (n = 541) | +0.0000 [+0.0000, +0.0000] (n = 465) |
| (50,100] | 653 | +0.0000 [+0.0000, +0.0000] (n = 653) | +0.0000 [+0.0000, +0.0000] (n = 653) | +0.0208 [-0.0000, +0.0431] (n = 626) | +0.0000 [+0.0000, +0.0000] (n = 653) | +0.0000 [+0.0000, +0.0000] (n = 626) | +0.0000 [+0.0000, +0.0000] (n = 600) |
| (100,inf) | 731 | +0.0000 [+0.0000, +0.0000] (n = 731) | +0.0000 [+0.0000, +0.0000] (n = 731) | +0.0105 [-0.0075, +0.0285] (n = 667) | +0.0000 [+0.0000, +0.0000] (n = 731) | +0.0000 [+0.0000, +0.0000] (n = 667) | +0.0000 [+0.0000, +0.0000] (n = 702) |

### Agreement with the W15 part C numbers (R2 + E2; |gap| <= 1e-09): **yes** (0 checkable comparisons, 25 not applicable because T is not a bin edge of this dataset)

| comparison | metric | policy diff | W15-implied diff | gap | agree |
| --- | --- | --- | --- | --- | --- |
| P_T vs C3 (H11) | ndcg | +0.000000 | - | - | n/a |
| P_T vs C3 (H12) | new_hit_share | +0.000000 | - | - | n/a |
| P_T vs C3 | recall | +0.000000 | - | - | n/a |
| P_T vs C3 | new_recall | +0.000000 | - | - | n/a |
| P_T vs C3 | repeat_recall | +0.000000 | - | - | n/a |
| P_T4_r8 vs C3 | ndcg | +0.000000 | - | - | n/a |
| P_T4_r8 vs C3 | new_hit_share | +0.000000 | - | - | n/a |
| P_T4_r8 vs C3 | recall | +0.000000 | - | - | n/a |
| P_T4_r8 vs C3 | new_recall | +0.000000 | - | - | n/a |
| P_T4_r8 vs C3 | repeat_recall | +0.000000 | - | - | n/a |
| P_T6_r9 vs C3 | ndcg | +0.000000 | - | - | n/a |
| P_T6_r9 vs C3 | new_hit_share | +0.000000 | - | - | n/a |
| P_T6_r9 vs C3 | recall | +0.000000 | - | - | n/a |
| P_T6_r9 vs C3 | new_recall | +0.000000 | - | - | n/a |
| P_T6_r9 vs C3 | repeat_recall | +0.000000 | - | - | n/a |
| P_T10_r9 vs C3 | ndcg | -0.000014 | - | - | n/a |
| P_T10_r9 vs C3 | new_hit_share | +0.000483 | - | - | n/a |
| P_T10_r9 vs C3 | recall | +0.000059 | - | - | n/a |
| P_T10_r9 vs C3 | new_recall | +0.000127 | - | - | n/a |
| P_T10_r9 vs C3 | repeat_recall | -0.000138 | - | - | n/a |

First bin (same users and bootstrap draw as W15):

| key | policy diff [95 % CI] (n) | W15 diff [95 % CI] (n) | agree |
| --- | --- | --- | --- |
| H11_ndcg | - | - | n/a (not applicable: the first bin edge is not T = 4) |
| H12_new_hit_share | - | - | n/a (not applicable: the first bin edge is not T = 4) |
| P_T_vs_final_lr_ucf_recall | - | - | n/a (not applicable: the first bin edge is not T = 4) |
| P_T_vs_final_lr_ucf_new_recall | - | - | n/a (not applicable: the first bin edge is not T = 4) |
| P_T_vs_final_lr_ucf_repeat_recall | - | - | n/a (not applicable: the first bin edge is not T = 4) |

C3 list metrics against the W15 R2 r = 10 row: ndcg 0.2046 vs 0.2046 (equal), recall 0.1315 vs 0.1315 (equal), new_recall 0.0000 vs 0.0000 (equal), new_hit_share 0.0000 vs 0.0000 (equal), repeat_recall 0.2740 vs 0.2740 (equal)

## Inputs

| file | sha256 | git_commit |
| --- | --- | --- |
| C:\dev_ML-wt-w17\reports\posthoc\explore_slot_medium.json | 28d4b3188775fd327006e84d1a9f51087fe269bcdbb174692579892fb25a3b99 | 8964e01a252242f4029596fabb052a41b06e2854 |
| C:\dev_ML-wt-w17\reports\metrics\medium_instacart_tuning.json | d7b15f217ff0c1ff133d57c9bad0b96b3394fb0d5e33b3f7da21c396243e2e8e | 78ba1fd0dd6b366afb9c84806dba157759ef156b |
| C:\dev_ML-wt-w17\reports\metrics\medium_instacart_eval.json | e954178002d0973e0e70c77bde6b61f211a79139811cdc399dfcd913d87c95d5 | 0c7aa8960e34d2d1a6ada57b7d7c072430080603 |
| C:\dev_ML-wt-w8\reports\metrics\medium_instacart_tuning.json | 87c84ea24f1d2953c272148c87ce15fd6f05ca231f79f2f0a4ad2aa71c04b25f | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C:\dev_ML-wt-w17\artifacts\medium\instacart_lr_repeat.joblib | 0bf51f8852cb80e410f80d2109ecb0fd0e889d79d1f7cd34b74413f4ee01cf9e |  |
| C:\dev_ML-wt-w17\reports\metrics\medium_complete_journey_tuning.json | 8b83576501e82f53f96bd2281c114f63145494f20566087a725dbff970032c58 | 78ba1fd0dd6b366afb9c84806dba157759ef156b |
| C:\dev_ML-wt-w17\reports\metrics\medium_complete_journey_eval.json | 62f2cf1751b82f85caba7b09ea8b44984878cdc24f62df287a90a3b015547b4f | 0c7aa8960e34d2d1a6ada57b7d7c072430080603 |
| C:\dev_ML-wt-w8\reports\metrics\medium_complete_journey_tuning.json | 0a215302c9e4778e0459a5ff52b19243fca04443f0f7108f708812e0467cdfec | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C:\dev_ML-wt-w17\artifacts\medium\complete_journey_lr_repeat.joblib | c71680afc3bbb98e5ed024e7a40b23d4bc1cd7c5c2c6df44016e4d47eb808dd4 |  |
