# nbr — hierarchical next-basket recommendation (ML Team 9)

Repurchase and exploration are ranked separately; exploration goes through the category hierarchy (L2 -> item).
Build data: Instacart; test data: dunnhumby The Complete Journey (refit). Plan and hypotheses: `docs/PREREG.md`.

- `data/` processed Parquet (not tracked; see `data/SOURCE.md`, `data/MANIFEST.sha256`)
- `vendor/ml_group9_e11b365/` unmodified preprocessing snapshot (ML_group9 release/term-project e11b365)
- `scripts/make_holdout.py` holdout profile builder; `scripts/check_data.py` split checks
- `src/nbr/` recommender package (in progress)
