# Consolidated results (W11; updated in W11b with the PREREG v1.3 run on holdout2, in W17 with the PREREG v1.4 exploration-policy test on holdout3 and in W19 with the PREREG v1.5 two-CF final run on holdout4)

- W8b outputs reflected: **yes**
- holdout4 results: **included** (stage `confirmatory holdout4`, PREREG v1.5 two-CF final run, reports/final4; primary H14 (non-inferiority, delta = 0.003) and H15 on Instacart holdout4 with 97.5 % intervals in the T1 columns `ci975_lo` / `ci975_hi` / `decision_975` (the profile's bootstrap levels, so every row of that stage carries them); everything else of that stage is secondary)
- holdout3 results: **included** (stage `confirmatory holdout3`, PREREG v1.4 exploration-policy test, reports/final3; primary H11 (non-inferiority, delta = 0.003), H12, H13 on Instacart holdout3 with 98.33 % intervals in the T1 columns `ci9833_lo` / `ci9833_hi` / `decision_9833`; everything else of that stage is secondary)
- holdout2 results: **included** (stage `confirmatory holdout2`, PREREG v1.3 final run, reports/final2; T1 columns `prereg` and `tier` carry the pre-registered classification, see `classification` in results_sources.yaml)
- Stage wording: `confirmatory holdout4` = PREREG v1.5 two-CF final run (Instacart holdout4; Complete Journey refit on medium data, not independent of development); `confirmatory holdout3` = PREREG v1.4 exploration-policy test (Instacart holdout3; Complete Journey refit on medium data with the same policy, not independent of development); `confirmatory holdout2` = PREREG v1.3 final run (Instacart holdout2; Complete Journey refit on medium data, not independent of development); `confirmatory holdout` = PREREG v1.2 final run (Instacart holdout; Complete Journey refit on medium data); `development medium` = development runs, not confirmatory.
- Tier (T1 `tier`): primary = pre-registered primary test of that stage (v1.5: H14, H15 on Instacart holdout4; v1.4: H11, H12, H13 on Instacart holdout3; v1.3: H6, H7' NDCG, H7' Recall, H8 on Instacart holdout2; v1.2: H1-H4 on Instacart holdout and the Complete Journey refit); secondary-confirmatory = H5 (v1.2); secondary = every other confirmatory-stage row (H9, H7, the H1-H5 replication, the H6/H7'/H8 replication, the policy variants on holdout3, the R2'/C4 secondaries and the policy replication (base C3 and base C4, ids policy_c4:*) on holdout4, transfer, CI-only comparisons); development = development rows.
- Development integration run used for T3-T5, F2-F5: `W8b`; W9 provides R3/H10 in every case.
- Change note (W19, PREREG v1.5 / v1.5.1): structural changes to the existing tables are limited to T1 (new columns `ci975_lo`, `ci975_hi`, `decision_975`, empty outside the holdout4 stage), T4 (the two empty `C4` placeholder rows "not run: E3 + R2 not implemented" of the development stage are no longer produced because C4 now denotes the evaluated list `final_lri_ucf`; the new rows C4, P_T and P_T(C4) of the holdout4 stage are added) and T6 (new rows R2' and P_T(C4); `evidence_stage` of the existing modules gains `confirmatory holdout4` where their hypotheses were replicated; the notes of E3, R2 and C4 updated). C4 (PREREG v1.5) = R2' + E2; the earlier C4 placeholder (E3 + R2) was never evaluated.
- Combination labels (fixed by the team): C0 = E1 + R1, r = 10 (`final`); C1 = E1 + R2, r = 10 (`final_lr`); C2 = E1 + R2, r re-selected (`final_lr_r`); C3 = E2 + R2, r re-selected (`final_lr_ucf`; H9 = C3 vs C1); C3b = E2 + R1, r re-selected (`final_star_ucf`); C4 = E2 + R2' (R2 with the item-CF similarity feature), r re-selected (`final_lri_ucf`, holdout4, PREREG v1.5); reference E4 + R1, r = 10 (`final_cf`); P_T = C3 with the PREREG v1.4 slot policy, r_u = 9 for users with at most 4 history baskets (`P_T`, holdout3; repeated on holdout4 as a secondary reference); P_T(C4) = C4 with the same policy (`P_T` of the policy_c4 block, holdout4, PREREG v1.5.1). Module codes and statuses are listed in T6 (`scripts/results_sources.yaml`).
- **For the slides and the report use only T1 rows with `current = true`.** A row is current when it comes from the newest run for the same stage, dataset, comparator, model and metric (holdout4, holdout3, holdout2 and holdout are separate confirmatory stages; development: W8b > W9 > W8 > W7, W6 > W4a); `superseded_reason` explains every other row.

## Tables (reports/results/tables)

| table | rows | description |
| --- | --- | --- |
| T1_master_hypotheses | 754 | one row per hypothesis or secondary/auxiliary comparison; diff = mean_model - mean_comparator; decision as recorded in the source; prereg / tier = pre-registered classification (results_sources.yaml); current = newest run for that comparison (use only these rows) |
| T2_selections | 160 | selected values per dataset x run (long format) with grid and grid-edge flag |
| T3_module_screening | 80 | explore modules by new-item Recall@10, repeat modules by repeat Recall@10; diff and 95% CI vs the comparator named in the row |
| T4_final_lists | 80 | Recall@10 and NDCG@10 over all users for the combinations C0-C3, C3b, C4 (holdout4, PREREG v1.5), the policy lists P_T (holdout3 and holdout4, base C3) and P_T(C4) (holdout4, PREREG v1.5.1), the reference E4 + R1 and the baselines (holdout4, holdout3, holdout2 and medium); C0, E4 + R1 and baselines on the holdout (v1.2) |
| T5_history_length | 423 | per history-length bin: diff and 95% CI of H1, H3, H6, H7, H8, H9, H10, (holdout3, PREREG v1.4) H11, H12, H13 and (holdout4, PREREG v1.5) H14 |
| T6_module_catalog | 25 | module and combination status catalogue (adopted / evaluated-not-adopted / not-examined / not run) with basis hypotheses |
| T7_datasets_eda | 2 | dataset summary (eval history) copied from the EDA branch feat/w5a-eda |

## Figures (reports/results/figures, PNG 300 dpi + CSV of the plotted numbers)

| figure | points | description |
| --- | --- | --- |
| F1_holdout_forest | 12 | holdout confirmatory H1-H5 forest plot, Instacart and Complete Journey refit side by side (95% thick, 99.375% thin) |
| F7_holdout2_forest | 30 | holdout2 confirmatory forest plot (PREREG v1.3): primary H6, H7' (NDCG, Recall), H8 and the H1-H5 replication; Instacart holdout2, Complete Journey refit and transfer side by side (95% thick, 99.375% thin) |
| F8_holdout3_forest | 30 | holdout3 confirmatory forest plot (PREREG v1.4): primary H11 (non-inferiority; dashed tick = margin -0.003), H12, H13, the secondary P_T-vs-C3 metrics and the H6, H7', H8 replication; Instacart holdout3, Complete Journey refit and transfer side by side (95% thick; 98.33% thin for policy rows, 99.375% thin for replication rows) |
| F9_holdout4_forest | 27 | holdout4 confirmatory forest plot (PREREG v1.5): primary H14 (non-inferiority; dashed tick = margin -0.003), H15, the secondary R2'-vs-R2 / R2'-vs-R1 repeat Recall@10 and C4-vs-C3 Recall@10 and the H6, H7', H8 replication; Instacart holdout4, Complete Journey refit and transfer side by side (95% thick, 97.5% thin) |
| F2_dev_forest | 14 | development (medium) H6-H10 forest plot, same format |
| F3_module_screening | 80 | module means: explore (new-item Recall@10) and repeat (repeat Recall@10), holdout2 and holdout vs medium |
| F4_r_grid | 66 | NDCG@10 by r (0-10) for C2, C3, C3b on the fit split, one panel per dataset |
| F5_history_length | 27 | per-bin differences with 95% CIs for H3, H7, H8, one panel per dataset |
| F6_overdue_effect | 14 | W9 R3 log-odds effect per overdue bin (from the W9 CSV), one panel per dataset |

## Inputs (read-only)

| key | run | stage | path | git_commit | sha256 |
| --- | --- | --- | --- | --- | --- |
| holdout4/instacart_tuning | holdout4 | confirmatory holdout4 | reports/final4/holdout4_instacart_tuning.json | ba0f38745560575d5e15c3a8f0abd3948ff28354 | f1a9a45d38e35a25988b8fa20ad914942d5af0571048d544cc09d5ae88c5ce49 |
| holdout4/instacart_eval | holdout4 | confirmatory holdout4 | reports/final4/holdout4_instacart_eval.json | 5c40aab2a4c8aca08a93eaa90b0ae100dececd5c | 5c210d5bd41d290196971cd2b393bf8cad7d0fcc7f4ee435106399a9e9b0dbe2 |
| holdout4/complete_journey_tuning | holdout4 | confirmatory holdout4 | reports/final4/holdout4_complete_journey_tuning.json | 5c40aab2a4c8aca08a93eaa90b0ae100dececd5c | 648ba30f88be21a9b768648b5a9de49b8b9ca00bc52a8e9261be91e0490dde70 |
| holdout4/complete_journey_eval | holdout4 | confirmatory holdout4 | reports/final4/holdout4_complete_journey_eval.json | 5c40aab2a4c8aca08a93eaa90b0ae100dececd5c | 9908edc2d1d1b5f32f365057c4b47e42bf4ebe90fd9c078d59f35927f24c28ad |
| holdout4/complete_journey_eval_transfer | holdout4 | confirmatory holdout4 | reports/final4/holdout4_complete_journey_eval_transfer.json | 5c40aab2a4c8aca08a93eaa90b0ae100dececd5c | 243a684cee8d3e39dd6122fc2a0e0fdefc01c0bed8155c2ac29a25bdf3ce0231 |
| holdout3/instacart_tuning | holdout3 | confirmatory holdout3 | reports/final3/holdout3_instacart_tuning.json | fececaab45da83498cc0dd5f9b5e98e491e0edaf | e23b24f0f90d9b34ad45c862e827f3be926c3ca62a1e64f172f45920dea572ab |
| holdout3/instacart_eval | holdout3 | confirmatory holdout3 | reports/final3/holdout3_instacart_eval.json | 4a88bc01f17d8ce85280bb66d69366914c8abe04 | d2eb70bf4a48a0e0bed0e414d4b984675324defbd563922de9382f857149d68e |
| holdout3/complete_journey_tuning | holdout3 | confirmatory holdout3 | reports/final3/holdout3_complete_journey_tuning.json | 4a88bc01f17d8ce85280bb66d69366914c8abe04 | d94dd8fa88b477c1af932710b72103a64f44c91cb61f41d34be972e3f64127eb |
| holdout3/complete_journey_eval | holdout3 | confirmatory holdout3 | reports/final3/holdout3_complete_journey_eval.json | 4a88bc01f17d8ce85280bb66d69366914c8abe04 | cde59bf031022af05e3b4c77a9fa631eadca033498fad03a9b76dd2edc132105 |
| holdout3/complete_journey_eval_transfer | holdout3 | confirmatory holdout3 | reports/final3/holdout3_complete_journey_eval_transfer.json | 4a88bc01f17d8ce85280bb66d69366914c8abe04 | bde82a8aa3a72c81b2fa1c3de43d4dee2e172671dd1c9798f16d806ab98f3bd6 |
| holdout2/instacart_tuning | holdout2 | confirmatory holdout2 | reports/final2/holdout2_instacart_tuning.json | a0f130c8770d36a5ae4e0851780e2f536505a46c | ffdb99c6e72ee2c5af403d4c58352d80e47fb6a2396a13541903ba9636ae1257 |
| holdout2/instacart_eval | holdout2 | confirmatory holdout2 | reports/final2/holdout2_instacart_eval.json | e7ad59d3c220609cbfc1d101177c02ee9abecd79 | a6745a1090750f3e5941ee23e3161fd1398fd271b8f4e418019108e9e37eaa09 |
| holdout2/complete_journey_tuning | holdout2 | confirmatory holdout2 | reports/final2/holdout2_complete_journey_tuning.json | e7ad59d3c220609cbfc1d101177c02ee9abecd79 | b8e71b97bf771b6c275f964246bcede675911ed5cf33900624b6bcb4a4b6d016 |
| holdout2/complete_journey_eval | holdout2 | confirmatory holdout2 | reports/final2/holdout2_complete_journey_eval.json | e7ad59d3c220609cbfc1d101177c02ee9abecd79 | b558d1c8c4b5e58b30dc5264a415bd03d20f3d2d125f2ee1881f99448cf2f13e |
| holdout2/complete_journey_eval_transfer | holdout2 | confirmatory holdout2 | reports/final2/holdout2_complete_journey_eval_transfer.json | e7ad59d3c220609cbfc1d101177c02ee9abecd79 | 8e95abebe6a1389a0f461130cb5907d5b67920e0039f4b4b59e9186102606907 |
| holdout/instacart_tuning | holdout | confirmatory holdout | reports/final/holdout_instacart_tuning.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba | bc103b88546554debe55f7480d51d4c98ac8b41e324f44b3be5d5b38e629f931 |
| holdout/instacart_eval | holdout | confirmatory holdout | reports/final/holdout_instacart_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba | c4c27a9d10fceb7df7950366806d34824bec8d35abcbf5d039c0dd744f153d7f |
| holdout/complete_journey_tuning | holdout | confirmatory holdout | reports/final/holdout_complete_journey_tuning.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba | 2659a34da8f49ded8e0e75ab0403e4e2d144bbe18688f1f4236b87f25f8faef0 |
| holdout/complete_journey_eval | holdout | confirmatory holdout | reports/final/holdout_complete_journey_eval.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba | 9907f6813ac3fd071e3934d5a6063b4868e1b1bdfc0699791e4555153d9c2a26 |
| holdout/complete_journey_eval_transfer | holdout | confirmatory holdout | reports/final/holdout_complete_journey_eval_transfer.json | 9d51446207bd8d803cd4e3c2a3708f8276752dba | 0f7f31d926835c4be6aec1d20e0cb4efb2f1cd7b4fb2cbf573ed04bb3bd17a14 |
| w4a/instacart_tuning | W4a | development medium | ../dev_ML-wt-w6/reports/metrics/_w4_backup/medium_instacart_tuning.json | 7baa6a1ec320c8bd0e304e4260797f59daf28807 | fbc20b2c7be30519be2452491fd4a5b9c5fe8ec566261ed04413a2a1d19c4163 |
| w4a/instacart_eval | W4a | development medium | ../dev_ML-wt-w6/reports/metrics/_w4_backup/medium_instacart_eval.json | 7baa6a1ec320c8bd0e304e4260797f59daf28807 | 8c5acc3adc1a3f923fc070f2f69625864527df8f1338e7951cbb137d98747f22 |
| w4a/complete_journey_tuning | W4a | development medium | ../dev_ML-wt-w6/reports/metrics/_w4_backup/medium_complete_journey_tuning.json | 7baa6a1ec320c8bd0e304e4260797f59daf28807 | cffdbc39d6de128bcd49fc17683fc31abf16a14a35fd148cf2e4ee0dbecc894b |
| w4a/complete_journey_eval | W4a | development medium | ../dev_ML-wt-w6/reports/metrics/_w4_backup/medium_complete_journey_eval.json | 7baa6a1ec320c8bd0e304e4260797f59daf28807 | b6300726df02605d85df7e12df24d31b0e669fa56032b42575263e4a16d9dc76 |
| w4a/complete_journey_eval_transfer | W4a | development medium | ../dev_ML-wt-w6/reports/metrics/_w4_backup/medium_complete_journey_eval_transfer.json | 7baa6a1ec320c8bd0e304e4260797f59daf28807 | 5cc2c8aad19d4a422159ddeb685c9f4bcfe4a84639e0972c11b71126cd297df1 |
| w6/instacart_tuning | W6 | development medium | ../dev_ML-wt-w6/reports/metrics/medium_instacart_tuning.json | 87f6c6aa57fd09816205f1dddc027629ecacf17a | 9e41a28eb6a4fcba4540881aad8bd1ef27b87085ee80841ed4521ca337848160 |
| w6/instacart_eval | W6 | development medium | ../dev_ML-wt-w6/reports/metrics/medium_instacart_eval.json | 87f6c6aa57fd09816205f1dddc027629ecacf17a | 97e88b0c8d2d5fac03db26895caab5732666cf9e72025d9e6a03f003c88f9908 |
| w6/complete_journey_tuning | W6 | development medium | ../dev_ML-wt-w6/reports/metrics/medium_complete_journey_tuning.json | 87f6c6aa57fd09816205f1dddc027629ecacf17a | fec5dd4e2b979a59c633c12a02eb458845ee85875be47db56cbc0936ca9e5be0 |
| w6/complete_journey_eval | W6 | development medium | ../dev_ML-wt-w6/reports/metrics/medium_complete_journey_eval.json | 87f6c6aa57fd09816205f1dddc027629ecacf17a | df35f1833b54d566fbf7c81abdf563b0fe0c7b08c3a606059e9e3875e84ae139 |
| w7/instacart_tuning | W7 | development medium | ../dev_ML-wt-w7/reports/metrics/medium_instacart_usercf_tuning.json | c6733677c5d80630dd5c9cd8b61bf66437fb089f | 720091a3fdc7ac299ae74ea199a419380fb79a0ad4ac5a25c1b1e38e8408d151 |
| w7/instacart_eval | W7 | development medium | ../dev_ML-wt-w7/reports/metrics/medium_instacart_usercf_eval.json | c6733677c5d80630dd5c9cd8b61bf66437fb089f | 87e81816d55f38b7e83e54203447c4fcc73fe3e452652c3547f3f1ab6023d8ba |
| w7/complete_journey_tuning | W7 | development medium | ../dev_ML-wt-w7/reports/metrics/medium_complete_journey_usercf_tuning.json | c6733677c5d80630dd5c9cd8b61bf66437fb089f | 0276d30bc6b8f0402b4f8ca9cb24482f81b7c42eadecfaa1942d0a715787a9a8 |
| w7/complete_journey_eval | W7 | development medium | ../dev_ML-wt-w7/reports/metrics/medium_complete_journey_usercf_eval.json | c6733677c5d80630dd5c9cd8b61bf66437fb089f | f6e30ede8f40bec158a55eadad4f92e007c6b4543a554a725b0c39931e36f3cd |
| w8/instacart_tuning | W8 | development medium | ../dev_ML-wt-w8/reports/metrics/_w8_backup/medium_instacart_tuning.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 | 893c4d50c15f1ba9036f2a747f044e906aa638377faef60b396d193d26d487d0 |
| w8/instacart_eval | W8 | development medium | ../dev_ML-wt-w8/reports/metrics/_w8_backup/medium_instacart_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 | dd0b0d0d9832ea0c8f630049caeeb016f3652feb6a64b3b72f5f1f4369733e22 |
| w8/complete_journey_tuning | W8 | development medium | ../dev_ML-wt-w8/reports/metrics/_w8_backup/medium_complete_journey_tuning.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 | 1acc4755ce047113e4a076c6409cdc943639bd08f017c2d01910295ca69aba70 |
| w8/complete_journey_eval | W8 | development medium | ../dev_ML-wt-w8/reports/metrics/_w8_backup/medium_complete_journey_eval.json | 24ce85f65f8904eb76c42f1a77f8dbd1fa7ae694 | 73715dca71304404c406b12771ae77e8e8ed5565228f6ae3902c3053ccc07571 |
| w8b/instacart_tuning | W8b | development medium | ../dev_ML-wt-w8/reports/metrics/medium_instacart_tuning.json | 6936873dd9b541a3a177643e97700182b8d729d8 | 87c84ea24f1d2953c272148c87ce15fd6f05ca231f79f2f0a4ad2aa71c04b25f |
| w8b/instacart_eval | W8b | development medium | ../dev_ML-wt-w8/reports/metrics/medium_instacart_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 | 4e29f0d65d86947b5bd558fd8c3402708192f1571d73c2623b675bab4100582b |
| w8b/complete_journey_tuning | W8b | development medium | ../dev_ML-wt-w8/reports/metrics/medium_complete_journey_tuning.json | 6936873dd9b541a3a177643e97700182b8d729d8 | 0a215302c9e4778e0459a5ff52b19243fca04443f0f7108f708812e0467cdfec |
| w8b/complete_journey_eval | W8b | development medium | ../dev_ML-wt-w8/reports/metrics/medium_complete_journey_eval.json | 6936873dd9b541a3a177643e97700182b8d729d8 | 01ec2b6f03bc97364af4f9e41d045eb878a54ea5059e98ebaf1a7bb0cc199b1c |
| w9/instacart_r3 | W9 | development medium | ../dev_ML-wt-w9/reports/metrics/medium_instacart_r3.json | 037f37ef3b6ce082653e285d7eec3149f5b21d32 | a3f9376b8837f9fcec998c6fa5279501f5ead84c9342fabc8facce1b6184c5be |
| w9/complete_journey_r3 | W9 | development medium | ../dev_ML-wt-w9/reports/metrics/medium_complete_journey_r3.json | 037f37ef3b6ce082653e285d7eec3149f5b21d32 | d29c57c95fe1a97218bee35c25c72af26709bb4aba188fd0a849c07b539d8102 |
| w9/instacart_overdue_csv | W9 | development medium | ../dev_ML-wt-w9/reports/tables/medium_instacart_r3_overdue_coef.csv |  | 1ca70c87085ac136756d035b08f3d7f8b997e97c6d3ea4815e091a9143953295 |
| w9/complete_journey_overdue_csv | W9 | development medium | ../dev_ML-wt-w9/reports/tables/medium_complete_journey_r3_overdue_coef.csv |  | 864d27e71068d78a06508227a26c85724f97827cf6b4df153f7841c2e585f96c |
| w9/instacart_coefficients_csv | W9 | development medium | ../dev_ML-wt-w9/reports/tables/medium_instacart_r3_coefficients.csv |  | 9ab3828c46f484929ed215a9f5c96c7c7b37070b631fabbab5b2bcf6b0cdbb41 |
| w9/complete_journey_coefficients_csv | W9 | development medium | ../dev_ML-wt-w9/reports/tables/medium_complete_journey_r3_coefficients.csv |  | 1a898989550222cade7b0dc4ae9b5ae6e35ebb995c9289d6f2dc214d54b86cf9 |
| eda/table_datasets | W5a | development medium | feat/w5a-eda:reports/eda/table_datasets.csv | 5ce3e7ed007b3584ba5bbb905175058b63604c80 | ec54f8ddcad4ed25485d776a681657bfca3122f3e5c9813889cbe3cfd76e6b7d |
| eda/fig_history_length | W5a | development medium | feat/w5a-eda:reports/eda/fig_history_length.csv | 5ce3e7ed007b3584ba5bbb905175058b63604c80 | f83dfaa2b9c187f0d92222a0b740426c0a44de422d8679c909a47fefd756269f |
| eda/fig_repeat_share_by_history | W5a | development medium | feat/w5a-eda:reports/eda/fig_repeat_share_by_history.csv | 5ce3e7ed007b3584ba5bbb905175058b63604c80 | c3c5063b7bbcc194fb8aa535b8689bea1047fe46a3f7aff066f821c4e873820d |
| eda/fig_new_item_level | W5a | development medium | feat/w5a-eda:reports/eda/fig_new_item_level.csv | 5ce3e7ed007b3584ba5bbb905175058b63604c80 | 84600d285dc888db5514e6eade4f78ea90dcb9713396546864be8f2464ec2105 |
| eda/eda_summary | W5a | development medium | feat/w5a-eda:reports/eda/eda_summary.md | 5ce3e7ed007b3584ba5bbb905175058b63604c80 | f25323b9e611c09af74de601172ea6670af185c93a6879cd09fe5b287155f1ae |

## Verification

- T1 diff identity: 754 rows, max |diff - (mean_model - mean_comparator)| = 1.318e-16 (tolerance 1e-12); current rows: 624.
- Every figure is drawn from its own CSV (written first, then read back).
- Determinism (`--verify`): 32 files regenerated, 0 digest mismatches.

## Generator

`python scripts/make_results.py` (scripts/results_sources.yaml lists every input). Values are never typed in by hand.
