# Consolidated results (W11)

- W8b outputs reflected: **no** (W8b not complete at generation time; development figures use run W8; re-run `python scripts/make_results.py` after W8b finishes)
- holdout2 results: **not included** (holdout2 has not been run; the holdout2 data folder was not opened)
- Stage wording: `confirmatory holdout` = PREREG v1.2 final run (Instacart holdout; Complete Journey refit on medium data); `development medium` = development runs, not confirmatory.
- Development integration run used for T3-T5, F2-F5: `W8`; W9 provides R3/H10 in every case.
- Combination labels C0-C4/C3b and the module catalogue E1-E8/R1-R6 are defined in `scripts/results_sources.yaml` (mapping assumed by W11; confirm with the team). Modules without an implementation are `not-examined`.

## Tables (reports/results/tables)

| table | rows | description |
| --- | --- | --- |
| T1_master_hypotheses | 199 | one row per hypothesis or secondary/auxiliary comparison; diff = mean_model - mean_comparator; decision as recorded in the source |
| T2_selections | 64 | selected values per dataset x run (long format) with grid and grid-edge flag |
| T3_module_screening | 30 | explore modules by new-item Recall@10, repeat modules by repeat Recall@10; diff and 95% CI vs the comparator named in the row |
| T4_final_lists | 22 | Recall@10 and NDCG@10 over all users for the combinations C0-C4, C3b and the baselines (medium); C0 and baselines on the holdout |
| T5_history_length | 117 | per history-length bin: diff and 95% CI of H1, H3, H6, H7, H8, H9, H10 |
| T6_module_catalog | 17 | module status catalogue from results_sources.yaml (adopted / evaluated-not-adopted / excluded / not-examined) with basis hypotheses |
| T7_datasets_eda | 2 | dataset summary (eval history) copied from the EDA branch feat/w5a-eda |

## Figures (reports/results/figures, PNG 300 dpi + CSV of the plotted numbers)

| figure | points | description |
| --- | --- | --- |
| F1_holdout_forest | 12 | holdout confirmatory H1-H5 forest plot, Instacart and Complete Journey refit side by side (95% thick, 99.375% thin) |
| F2_dev_forest | 14 | development (medium) H6-H10 forest plot, same format |
| F3_module_screening | 30 | module means: explore (new-item Recall@10) and repeat (repeat Recall@10), holdout vs medium |
| F4_r_grid | 66 | NDCG@10 by r (0-10) for C2, C3/C3b, C4 on the fit split, one panel per dataset |
| F5_history_length | 27 | per-bin differences with 95% CIs for H3, H7, H8, one panel per dataset |
| F6_overdue_effect | 14 | W9 R3 log-odds effect per overdue bin (from the W9 CSV), one panel per dataset |

## Inputs (read-only)

| key | run | stage | path | git_commit | sha256 |
| --- | --- | --- | --- | --- | --- |
| holdout/instacart_tuning | holdout | confirmatory holdout | C:\dev_ML-wt-w4\reports\final\holdout_instacart_tuning.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba | bc103b88546554debe55f7480d51d4c98ac8b41e324f44b3be5d5b38e629f931 |
| holdout/instacart_eval | holdout | confirmatory holdout | C:\dev_ML-wt-w4\reports\final\holdout_instacart_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba | c4c27a9d10fceb7df7950366806d34824bec8d35abcbf5d039c0dd744f153d7f |
| holdout/complete_journey_tuning | holdout | confirmatory holdout | C:\dev_ML-wt-w4\reports\final\holdout_complete_journey_tuning.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba | 2659a34da8f49ded8e0e75ab0403e4e2d144bbe18688f1f4236b87f25f8faef0 |
| holdout/complete_journey_eval | holdout | confirmatory holdout | C:\dev_ML-wt-w4\reports\final\holdout_complete_journey_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba | 9907f6813ac3fd071e3934d5a6063b4868e1b1bdfc0699791e4555153d9c2a26 |
| holdout/complete_journey_eval_transfer | holdout | confirmatory holdout | C:\dev_ML-wt-w4\reports\final\holdout_complete_journey_eval_transfer.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba | 0f7f31d926835c4be6aec1d20e0cb4efb2f1cd7b4fb2cbf573ed04bb3bd17a14 |
| w4a/instacart_tuning | W4a | development medium | C:\dev_ML-wt-w6\reports\metrics\_w4_backup\medium_instacart_tuning.json | 7baa6a1ec320c8bd0e304e4260797f59daf28807 | fbc20b2c7be30519be2452491fd4a5b9c5fe8ec566261ed04413a2a1d19c4163 |
| w4a/instacart_eval | W4a | development medium | C:\dev_ML-wt-w6\reports\metrics\_w4_backup\medium_instacart_eval.json | 7baa6a1ec320c8bd0e304e4260797f59daf28807 | 8c5acc3adc1a3f923fc070f2f69625864527df8f1338e7951cbb137d98747f22 |
| w4a/complete_journey_tuning | W4a | development medium | C:\dev_ML-wt-w6\reports\metrics\_w4_backup\medium_complete_journey_tuning.json | 7baa6a1ec320c8bd0e304e4260797f59daf28807 | cffdbc39d6de128bcd49fc17683fc31abf16a14a35fd148cf2e4ee0dbecc894b |
| w4a/complete_journey_eval | W4a | development medium | C:\dev_ML-wt-w6\reports\metrics\_w4_backup\medium_complete_journey_eval.json | 7baa6a1ec320c8bd0e304e4260797f59daf28807 | b6300726df02605d85df7e12df24d31b0e669fa56032b42575263e4a16d9dc76 |
| w4a/complete_journey_eval_transfer | W4a | development medium | C:\dev_ML-wt-w6\reports\metrics\_w4_backup\medium_complete_journey_eval_transfer.json | 7baa6a1ec320c8bd0e304e4260797f59daf28807 | 5cc2c8aad19d4a422159ddeb685c9f4bcfe4a84639e0972c11b71126cd297df1 |
| w6/instacart_tuning | W6 | development medium | C:\dev_ML-wt-w6\reports\metrics\medium_instacart_tuning.json | 87f6c6aa57fd09816205f1dddc027629ecacf17a | 9e41a28eb6a4fcba4540881aad8bd1ef27b87085ee80841ed4521ca337848160 |
| w6/instacart_eval | W6 | development medium | C:\dev_ML-wt-w6\reports\metrics\medium_instacart_eval.json | 87f6c6aa57fd09816205f1dddc027629ecacf17a | 97e88b0c8d2d5fac03db26895caab5732666cf9e72025d9e6a03f003c88f9908 |
| w6/complete_journey_tuning | W6 | development medium | C:\dev_ML-wt-w6\reports\metrics\medium_complete_journey_tuning.json | 87f6c6aa57fd09816205f1dddc027629ecacf17a | fec5dd4e2b979a59c633c12a02eb458845ee85875be47db56cbc0936ca9e5be0 |
| w6/complete_journey_eval | W6 | development medium | C:\dev_ML-wt-w6\reports\metrics\medium_complete_journey_eval.json | 87f6c6aa57fd09816205f1dddc027629ecacf17a | df35f1833b54d566fbf7c81abdf563b0fe0c7b08c3a606059e9e3875e84ae139 |
| w7/instacart_tuning | W7 | development medium | C:\dev_ML-wt-w7\reports\metrics\medium_instacart_usercf_tuning.json | c6733677c5d80630dd5c9cd8b61bf66437fb089f | 720091a3fdc7ac299ae74ea199a419380fb79a0ad4ac5a25c1b1e38e8408d151 |
| w7/instacart_eval | W7 | development medium | C:\dev_ML-wt-w7\reports\metrics\medium_instacart_usercf_eval.json | c6733677c5d80630dd5c9cd8b61bf66437fb089f | 87e81816d55f38b7e83e54203447c4fcc73fe3e452652c3547f3f1ab6023d8ba |
| w7/complete_journey_tuning | W7 | development medium | C:\dev_ML-wt-w7\reports\metrics\medium_complete_journey_usercf_tuning.json | c6733677c5d80630dd5c9cd8b61bf66437fb089f | 0276d30bc6b8f0402b4f8ca9cb24482f81b7c42eadecfaa1942d0a715787a9a8 |
| w7/complete_journey_eval | W7 | development medium | C:\dev_ML-wt-w7\reports\metrics\medium_complete_journey_usercf_eval.json | c6733677c5d80630dd5c9cd8b61bf66437fb089f | f6e30ede8f40bec158a55eadad4f92e007c6b4543a554a725b0c39931e36f3cd |
| w8/instacart_tuning | W8 | development medium | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_instacart_tuning.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 | 893c4d50c15f1ba9036f2a747f044e906aa638377faef60b396d193d26d487d0 |
| w8/instacart_eval | W8 | development medium | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_instacart_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 | dd0b0d0d9832ea0c8f630049caeeb016f3652feb6a64b3b72f5f1f4369733e22 |
| w8/complete_journey_tuning | W8 | development medium | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_complete_journey_tuning.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 | 1acc4755ce047113e4a076c6409cdc943639bd08f017c2d01910295ca69aba70 |
| w8/complete_journey_eval | W8 | development medium | C:\dev_ML-wt-w8\reports\metrics\_w8_backup\medium_complete_journey_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 | 73715dca71304404c406b12771ae77e8e8ed5565228f6ae3902c3053ccc07571 |
| w9/instacart_r3 | W9 | development medium | C:\dev_ML-wt-w9\reports\metrics\medium_instacart_r3.json | 037f37ef3b6ce082653e285d7eec3149f5b21d32 | a3f9376b8837f9fcec998c6fa5279501f5ead84c9342fabc8facce1b6184c5be |
| w9/complete_journey_r3 | W9 | development medium | C:\dev_ML-wt-w9\reports\metrics\medium_complete_journey_r3.json | 037f37ef3b6ce082653e285d7eec3149f5b21d32 | d29c57c95fe1a97218bee35c25c72af26709bb4aba188fd0a849c07b539d8102 |
| w9/instacart_overdue_csv | W9 | development medium | C:\dev_ML-wt-w9\reports\tables\medium_instacart_r3_overdue_coef.csv |  | 1ca70c87085ac136756d035b08f3d7f8b997e97c6d3ea4815e091a9143953295 |
| w9/complete_journey_overdue_csv | W9 | development medium | C:\dev_ML-wt-w9\reports\tables\medium_complete_journey_r3_overdue_coef.csv |  | 864d27e71068d78a06508227a26c85724f97827cf6b4df153f7841c2e585f96c |
| w9/instacart_coefficients_csv | W9 | development medium | C:\dev_ML-wt-w9\reports\tables\medium_instacart_r3_coefficients.csv |  | 9ab3828c46f484929ed215a9f5c96c7c7b37070b631fabbab5b2bcf6b0cdbb41 |
| w9/complete_journey_coefficients_csv | W9 | development medium | C:\dev_ML-wt-w9\reports\tables\medium_complete_journey_r3_coefficients.csv |  | 1a898989550222cade7b0dc4ae9b5ae6e35ebb995c9289d6f2dc214d54b86cf9 |
| eda/table_datasets | W5a | development medium | feat\w5a-eda:reports\eda\table_datasets.csv | 5ce3e7ed007b3584ba5bbb905175058b63604c80 | ec54f8ddcad4ed25485d776a681657bfca3122f3e5c9813889cbe3cfd76e6b7d |
| eda/fig_history_length | W5a | development medium | feat\w5a-eda:reports\eda\fig_history_length.csv | 5ce3e7ed007b3584ba5bbb905175058b63604c80 | f83dfaa2b9c187f0d92222a0b740426c0a44de422d8679c909a47fefd756269f |
| eda/fig_repeat_share_by_history | W5a | development medium | feat\w5a-eda:reports\eda\fig_repeat_share_by_history.csv | 5ce3e7ed007b3584ba5bbb905175058b63604c80 | c3c5063b7bbcc194fb8aa535b8689bea1047fe46a3f7aff066f821c4e873820d |
| eda/fig_new_item_level | W5a | development medium | feat\w5a-eda:reports\eda\fig_new_item_level.csv | 5ce3e7ed007b3584ba5bbb905175058b63604c80 | 84600d285dc888db5514e6eade4f78ea90dcb9713396546864be8f2464ec2105 |
| eda/eda_summary | W5a | development medium | feat\w5a-eda:reports\eda\eda_summary.md | 5ce3e7ed007b3584ba5bbb905175058b63604c80 | f25323b9e611c09af74de601172ea6670af185c93a6879cd09fe5b287155f1ae |

### Skipped inputs

- W8b: not complete: meta.git_commit still 24ce85f in instacart_tuning, instacart_eval

## Verification

- T1 diff identity: 199 rows, max |diff - (mean_model - mean_comparator)| = 1.041e-16 (tolerance 1e-12).
- Every figure is drawn from its own CSV (written first, then read back).
- Determinism (`--verify`): 26 files regenerated, 0 digest mismatches.

## Generator

`python scripts/make_results.py` (scripts/results_sources.yaml lists every input). Values are never typed in by hand.
