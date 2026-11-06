# T4 final lists

Recall@10 and NDCG@10 over all users for the combinations C0-C4, C3b and the baselines (medium); C0 and baselines on the holdout

| combination | list | repeat_module | explore_module | r_rule | r | stage | run | dataset | users_n | recall10 | ndcg10 | source_file | git_commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | final | R1 | E1 | r* (stage 3 of the pre-registered selection) | 10 | confirmatory holdout | holdout | instacart | 20000 | 0.348505 | 0.422635 | C:\dev_ML-wt-w4\reports\final\holdout_instacart_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba |
| personal_frequency | personal_frequency |  |  |  |  | confirmatory holdout | holdout | instacart | 20000 | 0.32611 | 0.394681 | C:\dev_ML-wt-w4\reports\final\holdout_instacart_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba |
| popularity | popularity |  |  |  |  | confirmatory holdout | holdout | instacart | 20000 | 0.069444 | 0.098257 | C:\dev_ML-wt-w4\reports\final\holdout_instacart_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba |
| C0 | final | R1 | E1 | r* (stage 3 of the pre-registered selection) | 10 | confirmatory holdout | holdout | complete_journey | 2069 | 0.118726 | 0.185898 | C:\dev_ML-wt-w4\reports\final\holdout_complete_journey_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba |
| personal_frequency | personal_frequency |  |  |  |  | confirmatory holdout | holdout | complete_journey | 2069 | 0.118726 | 0.185898 | C:\dev_ML-wt-w4\reports\final\holdout_complete_journey_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba |
| popularity | popularity |  |  |  |  | confirmatory holdout | holdout | complete_journey | 2069 | 0.044681 | 0.077358 | C:\dev_ML-wt-w4\reports\final\holdout_complete_journey_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba |
| C0 | final | R1 | E1 | r* (stage 3 of the pre-registered selection) | 10 | development medium | W8 | instacart | 20000 | 0.345919 | 0.420777 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_instacart_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| C1 | final_cf | R1 | E4 | r* (stage 3) | 10 | development medium | W8 | instacart | 20000 | 0.34577 | 0.420693 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_instacart_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| C2 | final_star_ucf | R1 | E2 | r re-selected on all fit users (W8) | 10 | development medium | W8 | instacart | 20000 | 0.346511 | 0.421044 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_instacart_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| C3 | final_lr | R2 | E1 | r* (stage 3) | 10 | development medium | W8 | instacart | 20000 | 0.350928 | 0.427116 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_instacart_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| C3b | final_lr_r | R2 | E1 | r re-selected on LR validation users (W8) | 10 | development medium | W8 | instacart | 20000 | 0.350928 | 0.427116 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_instacart_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| C4 | final_lr_ucf | R2 | E2 | r re-selected on LR validation users (W8) | 10 | development medium | W8 | instacart | 20000 | 0.35152 | 0.427383 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_instacart_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| personal_frequency | personal_frequency |  |  |  |  | development medium | W8 | instacart | 20000 | 0.323844 | 0.3929 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_instacart_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| popularity | popularity |  |  |  |  | development medium | W8 | instacart | 20000 | 0.071254 | 0.099767 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_instacart_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| C0 | final | R1 | E1 | r* (stage 3 of the pre-registered selection) | 10 | development medium | W8 | complete_journey | 2069 | 0.118726 | 0.185898 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_complete_journey_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| C1 | final_cf | R1 | E4 | r* (stage 3) | 10 | development medium | W8 | complete_journey | 2069 | 0.118726 | 0.185898 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_complete_journey_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| C2 | final_star_ucf | R1 | E2 | r re-selected on all fit users (W8) | 10 | development medium | W8 | complete_journey | 2069 | 0.118726 | 0.185898 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_complete_journey_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| C3 | final_lr | R2 | E1 | r* (stage 3) | 10 | development medium | W8 | complete_journey | 2069 | 0.131525 | 0.204637 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_complete_journey_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| C3b | final_lr_r | R2 | E1 | r re-selected on LR validation users (W8) | 10 | development medium | W8 | complete_journey | 2069 | 0.131525 | 0.204637 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_complete_journey_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| C4 | final_lr_ucf | R2 | E2 | r re-selected on LR validation users (W8) | 10 | development medium | W8 | complete_journey | 2069 | 0.131525 | 0.204637 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_complete_journey_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| personal_frequency | personal_frequency |  |  |  |  | development medium | W8 | complete_journey | 2069 | 0.118726 | 0.185898 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_complete_journey_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |
| popularity | popularity |  |  |  |  | development medium | W8 | complete_journey | 2069 | 0.044681 | 0.077358 | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_complete_journey_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 |

Hidden columns (see the CSV): source_sha256.
