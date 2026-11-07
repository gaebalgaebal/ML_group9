# T6 module catalogue

module and combination status catalogue (adopted / evaluated-not-adopted / not-examined / not run) with basis hypotheses

| id | group | name | list | status | basis | evidence_stage | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | explore | hierarchical explore (theta_u(c), then within-category popularity; L2) | explore_star | previous final (PREREG v1.2) | H2, H4 (holdout) | confirmatory holdout; confirmatory holdout2; confirmatory holdout3; confirmatory holdout4; development medium | explore module of C0, the H7' comparator |
| E2 | explore | user-based CF (cosine neighbours, k, binary/log1p weighting) | usercf | adopted (final system, PREREG v1.3) | H8 (holdout2); H8, H9 (medium) | confirmatory holdout2; confirmatory holdout3; confirmatory holdout4; development medium | explore module of C3 |
| E3 | explore | flat item-CF (item cosine, 20 neighbours) | itemcf_flat | evaluated-not-adopted | H5 (holdout, secondary confirmatory) | confirmatory holdout; confirmatory holdout2; confirmatory holdout3; confirmatory holdout4; development medium | its neighbour lists feed the icf feature of R2' (PREREG v1.5) |
| E4 | explore | theta x item-CF within category | explore_cf | evaluated-not-adopted | secondary explore_cf_vs_explore_star (holdout, medium) |  |  |
| E5 | explore | matrix factorisation (MF) |  | not-examined |  |  |  |
| E6 | explore | product-name tf-idf similarity |  | not-examined |  |  |  |
| E7 | explore | health-attribute match |  | not-examined |  |  |  |
| E8 | explore | recent-basket item-CF |  | not-examined |  |  |  |
| flat_pop | explore | flat popularity excluding history items (comparator) | flat_explore | evaluated-not-adopted | comparator of H2, H5, H8 | confirmatory holdout; confirmatory holdout2; confirmatory holdout3; confirmatory holdout4; development medium | baseline |
| R1 | repeat | decayed purchase frequency s_rep(gamma*) | repeat_star | previous final (PREREG v1.2) | H1 (holdout) | confirmatory holdout; confirmatory holdout2; confirmatory holdout3; confirmatory holdout4; development medium | repeat module of C0; comparator of H6 on holdout2 |
| R2 | repeat | L2 logistic-regression repeat ranker (12 features) | repeat_lr | adopted (final system, PREREG v1.3) | H6 (holdout2); H6, H7 (medium) | confirmatory holdout2; confirmatory holdout3; confirmatory holdout4; development medium | repeat module of C3; comparator of the R2' secondary on holdout4 |
| R2' | repeat | L2 logistic-regression repeat ranker + item-CF similarity feature icf (13 features; 20 item neighbours) | repeat_lri | confirmatory test on holdout4 (PREREG v1.5) | H14, H15 (holdout4) | confirmatory holdout4 | repeat module of C4; item-CF enters as a feature (PREREG v1.5 item 2b) |
| R3 | repeat | logistic ranker with binned cycle features (30 features) | repeat_r3 | evaluated-not-adopted | H10 (medium) | development medium |  |
| R4 | repeat | decision tree repeat ranker |  | not-examined |  |  |  |
| R5 | repeat | personal purchase frequency (gamma = 1; comparator) | repeat_pf | evaluated-not-adopted | comparator of H1 | confirmatory holdout; confirmatory holdout2; confirmatory holdout3; confirmatory holdout4; development medium | baseline |
| R6 | repeat | recent-N-basket candidate set |  | not-examined |  |  |  |
| C0 | combination | E1 + R1, r = 10 (stage-3 selection) | final | previous final (PREREG v1.2) | H3 (holdout) | confirmatory holdout; confirmatory holdout2; confirmatory holdout3; confirmatory holdout4; development medium | comparator of H7' on holdout2 |
| C1 | combination | E1 + R2, r = 10 fixed | final_lr | evaluated-not-adopted | H7 (medium) | confirmatory holdout2; confirmatory holdout3; confirmatory holdout4; development medium |  |
| C2 | combination | E1 + R2, r re-selected on the fit split (LR validation users) | final_lr_r | evaluated-not-adopted | secondary final_lr_r_vs_final_lr (medium) |  | r = 10 selected |
| C3 | combination | E2 + R2, r re-selected on the fit split (LR validation users) | final_lr_ucf | adopted (final system, PREREG v1.3) | H7' (holdout2); H9 = C3 vs C1 (medium, holdout2 secondary) | confirmatory holdout2; confirmatory holdout3; confirmatory holdout4; development medium | r = 10 selected |
| C3b | combination | E2 + R1, r re-selected on the fit split (all fit users) | final_star_ucf | evaluated-not-adopted | secondary final_star_ucf_vs_final (medium) |  | r = 10 selected |
| P_T | combination | E2 + R2, r_u = 9 for users with at most T = 4 history baskets, r* otherwise (PREREG v1.4 policy on C3) | P_T | confirmatory test on holdout3 (PREREG v1.4) | H11, H12 (holdout3) | confirmatory holdout3; confirmatory holdout4 | T = 4 and delta = 0.003 fixed from the W15 post-hoc study; S = E2 on its own is H13 |
| P_T(C4) | combination | E2 + R2', r_u = 9 for users with at most T = 4 history baskets, r (C4) otherwise (PREREG v1.5.1 policy on C4) | P_T | secondary replication on holdout4 (PREREG v1.5.1) |  |  | the v1.4 policy repeated on the v1.5 final system; no confirmatory claim (H11-H13 on holdout3 remain the confirmatory policy result) |
| C4 | combination | E2 + R2', r re-selected on the fit split (LR validation users, train-only R2') | final_lri_ucf | confirmatory test on holdout4 (PREREG v1.5; two CF methods) | H14, H15 (holdout4) | confirmatory holdout4 | R2' = R2 + item-CF similarity feature icf (20 item neighbours); E2 = user-based CF; C3 is the H14 comparator. Replaces the earlier plan E3 + R2 (never run) |
| E4+R1 | combination | E4 + R1, r = 10 (stage-3 selection) | final_cf | evaluated-not-adopted | secondary final_cf_vs_final (holdout v1.2 auxiliary) |  | reference combination |

Hidden columns (see the CSV): .
