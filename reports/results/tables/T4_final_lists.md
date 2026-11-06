# T4 final lists

Recall@10 and NDCG@10 over all users for the combinations C0-C3, C3b, the reference E4 + R1 and the baselines (medium); C0, E4 + R1 and baselines on the holdout; C4 not run

| combination | list | repeat_module | explore_module | r_rule | r | stage | run | dataset | users_n | recall10 | ndcg10 | note | source_file | git_commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | final | R1 | E1 | r = 10 (stage-3 selection) | 10 | confirmatory holdout | holdout | instacart | 20000 | 0.348505 | 0.422635 |  | reports/final/holdout_instacart_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba |
| E4+R1 | final_cf | R1 | E4 | r = 10 (stage-3 selection) | 10 | confirmatory holdout | holdout | instacart | 20000 | 0.348256 | 0.422487 | reference combination | reports/final/holdout_instacart_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba |
| personal_frequency | personal_frequency |  |  |  |  | confirmatory holdout | holdout | instacart | 20000 | 0.32611 | 0.394681 |  | reports/final/holdout_instacart_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba |
| popularity | popularity |  |  |  |  | confirmatory holdout | holdout | instacart | 20000 | 0.069444 | 0.098257 |  | reports/final/holdout_instacart_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba |
| C0 | final | R1 | E1 | r = 10 (stage-3 selection) | 10 | confirmatory holdout | holdout | complete_journey | 2069 | 0.118726 | 0.185898 |  | reports/final/holdout_complete_journey_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba |
| E4+R1 | final_cf | R1 | E4 | r = 10 (stage-3 selection) | 10 | confirmatory holdout | holdout | complete_journey | 2069 | 0.118726 | 0.185898 | reference combination | reports/final/holdout_complete_journey_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba |
| personal_frequency | personal_frequency |  |  |  |  | confirmatory holdout | holdout | complete_journey | 2069 | 0.118726 | 0.185898 |  | reports/final/holdout_complete_journey_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba |
| popularity | popularity |  |  |  |  | confirmatory holdout | holdout | complete_journey | 2069 | 0.044681 | 0.077358 |  | reports/final/holdout_complete_journey_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba |
| C0 | final | R1 | E1 | r = 10 (stage-3 selection) | 10 | development medium | W8b | instacart | 20000 | 0.345919 | 0.420777 |  | ../dev_ML-wt-w8/reports/metrics/medium_instacart_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C1 | final_lr | R2 | E1 | r = 10 fixed | 10 | development medium | W8b | instacart | 20000 | 0.350928 | 0.427116 |  | ../dev_ML-wt-w8/reports/metrics/medium_instacart_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C2 | final_lr_r | R2 | E1 | r re-selected on the fit split (LR validation users) | 10 | development medium | W8b | instacart | 20000 | 0.350928 | 0.427116 | r = 10 selected | ../dev_ML-wt-w8/reports/metrics/medium_instacart_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C3 | final_lr_ucf | R2 | E2 | r re-selected on the fit split (LR validation users) | 10 | development medium | W8b | instacart | 20000 | 0.35152 | 0.427383 | r = 10 selected | ../dev_ML-wt-w8/reports/metrics/medium_instacart_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C3b | final_star_ucf | R1 | E2 | r re-selected on the fit split (all fit users) | 10 | development medium | W8b | instacart | 20000 | 0.346511 | 0.421044 | r = 10 selected | ../dev_ML-wt-w8/reports/metrics/medium_instacart_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C4 |  | R2 | E3 |  |  | development medium |  | instacart |  |  |  | not run: E3 + R2 not implemented |  |  |
| E4+R1 | final_cf | R1 | E4 | r = 10 (stage-3 selection) | 10 | development medium | W8b | instacart | 20000 | 0.34577 | 0.420693 | reference combination | ../dev_ML-wt-w8/reports/metrics/medium_instacart_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| personal_frequency | personal_frequency |  |  |  |  | development medium | W8b | instacart | 20000 | 0.323844 | 0.3929 |  | ../dev_ML-wt-w8/reports/metrics/medium_instacart_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| popularity | popularity |  |  |  |  | development medium | W8b | instacart | 20000 | 0.071254 | 0.099767 |  | ../dev_ML-wt-w8/reports/metrics/medium_instacart_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C0 | final | R1 | E1 | r = 10 (stage-3 selection) | 10 | development medium | W8b | complete_journey | 2069 | 0.118726 | 0.185898 |  | ../dev_ML-wt-w8/reports/metrics/medium_complete_journey_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C1 | final_lr | R2 | E1 | r = 10 fixed | 10 | development medium | W8b | complete_journey | 2069 | 0.131525 | 0.204637 |  | ../dev_ML-wt-w8/reports/metrics/medium_complete_journey_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C2 | final_lr_r | R2 | E1 | r re-selected on the fit split (LR validation users) | 10 | development medium | W8b | complete_journey | 2069 | 0.131525 | 0.204637 | r = 10 selected | ../dev_ML-wt-w8/reports/metrics/medium_complete_journey_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C3 | final_lr_ucf | R2 | E2 | r re-selected on the fit split (LR validation users) | 10 | development medium | W8b | complete_journey | 2069 | 0.131525 | 0.204637 | r = 10 selected | ../dev_ML-wt-w8/reports/metrics/medium_complete_journey_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C3b | final_star_ucf | R1 | E2 | r re-selected on the fit split (all fit users) | 10 | development medium | W8b | complete_journey | 2069 | 0.118726 | 0.185898 | r = 10 selected | ../dev_ML-wt-w8/reports/metrics/medium_complete_journey_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| C4 |  | R2 | E3 |  |  | development medium |  | complete_journey |  |  |  | not run: E3 + R2 not implemented |  |  |
| E4+R1 | final_cf | R1 | E4 | r = 10 (stage-3 selection) | 10 | development medium | W8b | complete_journey | 2069 | 0.118726 | 0.185898 | reference combination | ../dev_ML-wt-w8/reports/metrics/medium_complete_journey_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| personal_frequency | personal_frequency |  |  |  |  | development medium | W8b | complete_journey | 2069 | 0.118726 | 0.185898 |  | ../dev_ML-wt-w8/reports/metrics/medium_complete_journey_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |
| popularity | popularity |  |  |  |  | development medium | W8b | complete_journey | 2069 | 0.044681 | 0.077358 |  | ../dev_ML-wt-w8/reports/metrics/medium_complete_journey_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 |

Hidden columns (see the CSV): source_sha256.
