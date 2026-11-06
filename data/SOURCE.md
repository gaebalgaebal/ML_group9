# Data provenance

- Copied on 2026-11-05 from `C:\dev\ML_group9\data\processed` (copy, source unchanged). Checksums: `MANIFEST.sha256` (24 Parquet files; source == copy verified).
- Source repository state at copy time: working branch `feat/hierarchical` (5d304cb); release branch `release/term-project` (e11b365). `data/` is not tracked by Git, so the generating code version is inferred from file times: histories/targets 2026-11-03, `items.parquet` (with `ingredient_type`) 2026-11-04.
- Profiles: `sample` (Instacart 1,500 users), `medium` (Instacart 20,000 users; Complete Journey 2,069 households). 227 users overlap between sample and medium.
- Raw archives (not copied): Instacart `instacart.zip` (orders, order_products__prior/train, products, aisles, departments); Complete Journey `archive.zip` = dunnhumby "The Complete Journey" source files (8 CSVs incl. transaction_data.csv, product.csv), not the 84.51 Complete Journey 2.0 R package cited in the proposal.
- The `medium` eval split was already used in earlier experiments (RE, HIER, H*' analyses); see the development plan for the fresh Instacart holdout profile.

## holdout profile (2026-11-05)
- Instacart only. 20,000 train-set users drawn with seed 42 from the 109,936 users not in medium or sample (131,209 eligible − 21,273 used). Overlap with used profiles: 0. Selected-user digest (sha256 of sorted ids): 43a1578613b2e1bd7f04db87ed3e6b078fe9537e944d650bf7872bbce13f7a06.
- Built by `scripts/make_holdout.py` with the unmodified preprocessing snapshot `vendor/ml_group9_e11b365` (release/term-project e11b365); the wrapper only removes used users before the seeded draw and adds ingredient-type columns with the same mapping functions (mapping v2.1, version check passed).
- Checks: the unmodified draw reproduces the medium user set; `items.parquet` and `attribute_categories.parquet` are identical to medium; all six Parquet schemas equal medium; split checks in `reports/data_check/holdout.json`.
- Environment: Python 3.10.12, pandas 2.2.3, numpy 1.26.4, pyarrow 17.0.0 (pinned versions of the ML_group9 pyproject). Runtime 26.7 s, peak RSS 1.8 GB.
- Rule: holdout is used once, for the final evaluation after hyperparameters are fixed on fit splits.

## holdout2 profile (2026-11-06)
- Confirmatory sample of PREREG v1.3 (item 1). Instacart only. 20,000 train-set users drawn with seed 42 from the 89,936 users not in sample, medium or holdout (131,209 eligible − 41,273 used: sample 1,500, medium 20,000, holdout 20,000, with the 227 sample users that are also in medium counted once). Overlap with sample, medium and holdout: 0 each. Selected-user digest (sha256 of sorted ids): 359eba2005a3b0e69aee7fbe3179c6ee5a1d466c54c14094e8911bfc8bedb2cc.
- Command (worktree of branch feat/w10a-holdout2, code commit a0882fb): `NBR_DATA_DIR=C:\dev_ML\data\processed python scripts/make_holdout.py --raw-dir C:\dev\ML_group9\data\raw --name holdout2 --exclude sample,medium,holdout --size 20000 --seed 42 --expect-remaining 89936`. Report: `reports/holdout_prep/holdout2_selection.json`.
- Exclusions: the sample and medium user ids were read from their eval targets (user_id column only). The holdout user ids were not read from its files: they were reconstructed by re-running the v1.2 draw (eligible users minus medium and sample, seed 42, size 20,000), and the digest of the reconstruction equals the one recorded in `reports/holdout_prep/holdout_selection.json` (43a15786...).
- Same preprocessing snapshot `vendor/ml_group9_e11b365` (release/term-project e11b365) and mapping v2.1 (version check passed) as holdout. Checks: the unmodified draw reproduces the medium user set; `items.parquet` and `attribute_categories.parquet` are identical to medium (and byte-identical to the holdout copies, see `MANIFEST.sha256`). No `reports/data_check/holdout2.json`: the split checks read target content, which stays unread until the final run.
- Environment: Python 3.12.16 (Windows), pandas 2.2.3, numpy 1.26.4, pyarrow 17.0.0. Runtime 37.0 s, peak working set 1.5 GB.
- Rule: holdout2 is used once, for the final evaluation of PREREG v1.3; its Instacart eval split is read only with `--final`.
