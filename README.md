# nbr — hierarchical next-basket recommendation (ML Team 9)

Next-basket recommendation that ranks **repurchases** and **exploration** separately and merges the
two lists into one Top-10 list per user. The repeat branch scores items the user has bought before;
the explore branch scores items the user has never bought, with a hierarchical model that goes through
the category hierarchy (user's category profile -> within-category popularity). Everything is
pre-registered (`docs/PREREG.md`): hypotheses, selection rules and the single read of the holdout
split were fixed before the final run.

## Task

Given a user's purchase history (a sequence of baskets), recommend 10 items for the next basket.
Evaluation: Recall@10 and NDCG@10 against the next basket, plus component metrics (repeat-component
Recall@10 on users with a repeated target item, new-item Recall@10 on users with a new target item)
and paired bootstrap confidence intervals over users. Pre-registered question: does separating
repurchase from exploration, with exploration ranked through the category hierarchy, improve
next-basket Top-10 recommendation over personal purchase frequency?

## Data

| Dataset | Source | Role |
|---|---|---|
| Instacart (orders, products, aisles, departments) | public Instacart release (Kaggle mirror); see `docs/CITATIONS.md` | development (`medium`: 20,000 users), confirmatory (`holdout`: 20,000 unused users), smoke (`sample`: 1,500 users) |
| dunnhumby The Complete Journey (transaction_data, product) | dunnhumby source files (Kaggle mirror); see `docs/CITATIONS.md` | development and refit evaluation (`medium`: 2,069 households; no unused households, limitation recorded in PREREG v1.2) |

Processed Parquet files (`items`, `fit_history`, `fit_target`, `eval_history`, `eval_target` per
profile and dataset) are produced by the unmodified ML_group9 preprocessing snapshot in
`vendor/ml_group9_e11b365` and are **not tracked** by git. Provenance and checksums:
`data/SOURCE.md`, `data/MANIFEST.sha256`. Hierarchy: Instacart department -> aisle (L2) -> ingredient
type -> item; Complete Journey DEPARTMENT -> commodity (L2) -> sub-commodity -> item.

## Structure (summary)

Repeat branch: R1 decayed purchase frequency (adopted), R2 L2 logistic regression on 12 history
features (development; selected on development data, confirmation planned). Explore branch: E1
hierarchical category-profile x within-category popularity (adopted), E2 user-based CF (development;
selected on development data, confirmation planned), E3 flat item-CF (secondary). Slot combination:
the first `r` slots from the repeat list, the rest from the explore list. Full description with a
diagram, profiles, commands, selection procedure and leakage rules: **`docs/ARCHITECTURE.md`**.
Module codes (E1-E8, R1-R6, C0-C4, C3b) and their status: **`docs/MODULES.md`**.

## How to run

All commands run from the repository root. The commands marked "verified" were executed on the
`sample` profile while preparing this release (see "Verified commands" at the end of this section).

### 1. Environment

Python 3.10-3.12 (the runs used 3.12.16). Dependencies are pinned in `pyproject.toml`.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1   /   POSIX: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Without installing the package, set `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH = "src"`).

### 2. Data placement

Copy the processed Parquet files to `data/processed/<profile>/<dataset>/` (datasets `instacart`,
`complete_journey`; profiles `sample`, `medium`, `holdout`), or point the environment variable
`NBR_DATA_DIR` at a folder with that layout (`NBR_DATA_DIR` wins over `configs/common.yaml:data_dir`;
the git worktrees of this project share one copy). Verify with:

```bash
python scripts/check_data.py sample        # -> reports/data_check/sample.json (medium, holdout likewise)
python -m nbr check --profile sample       # split statistics; the holdout eval split is skipped by the guard
```

### 3. Holdout profile (done once; recorded in `data/SOURCE.md`)

```bash
python scripts/make_holdout.py --raw-dir <ML_group9/data/raw> --size 20000 --seed 42
```

Draws 20,000 Instacart users disjoint from `medium` and `sample` with the vendored preprocessing
code and writes `reports/holdout_prep/holdout_selection.json`. The peak-memory field uses the
POSIX-only `resource` module; on Windows it is recorded as `null` and the draw is unchanged. The
recorded holdout build ran on a non-Windows interpreter (`data/SOURCE.md`). This command was not
re-run for the release (it needs the raw archives and would rewrite the holdout data).

### 4. Tune and evaluate

```bash
# development (medium): fit-split selection, then eval-split evaluation
python -m nbr tune --profile medium --dataset instacart
python -m nbr evaluate --profile medium --dataset instacart
python -m nbr tune --profile medium --dataset complete_journey
python -m nbr evaluate --profile medium --dataset complete_journey
python -m nbr evaluate --profile medium --dataset complete_journey --transfer-from instacart

# confirmatory (holdout, PREREG v1.2): the Instacart eval split is read once; --final is required
python -m nbr tune --profile holdout --dataset instacart
python -m nbr evaluate --profile holdout --dataset instacart --final
```

`tune` writes `reports/metrics/<profile>_<dataset>_tuning.json` and
`artifacts/<profile>/<dataset>_lr_repeat.joblib`; `evaluate` writes
`reports/metrics/<profile>_<dataset>_eval.json` (`_eval_transfer.json` with `--transfer-from`).
Optional flags: `--no-itemcf`, `--no-lr`, `--no-integration`, `--skip-l3`. The exact command
sequence of the final run is in `reports/final/RUN.md`; its JSONs and digests are in `reports/final/`.

Development-only screening runners (module screening entry points; both refuse holdout profiles):

```bash
python -m nbr.usercf_eval --profile sample --dataset instacart --stage all   # user-CF, H8 (W7)
python -m nbr.r3_eval --profile sample --dataset instacart                   # R3 cycle features, H10 (W9)
```

Both read the base selections (gamma, gamma_c, alpha, r) from `reports/metrics/<profile>_<dataset>_tuning.json`
written by `python -m nbr tune`; another folder can be given with the environment variable `NBR_TUNING_DIR`
or the flags `--tuning-dir` / `--gamma-dir`.

### 5. Tests, code audit, consolidated results

```bash
python -m pytest                        # synthetic-data tests (tests/), 164 tests
python scripts/audit_code.py            # static audit -> docs/CODE_AUDIT.md
python scripts/make_results.py --verify # tables T1-T7 and figures F1-F6 -> reports/results/ (+ determinism check)
python scripts/make_eda.py              # EDA tables and figures -> reports/eda/ (reads the medium profile)
```

`scripts/make_results.py` reads the final JSONs from `reports/final/` and the development JSONs from
the sibling git worktrees listed in `scripts/results_sources.yaml` (folder given by `NBR_WORKTREES`,
default: the parent folder of the repository); those development JSONs are untracked run outputs.

### Verified commands (release preparation, `sample` profile)

Run in this order from the repository root with `NBR_DATA_DIR` set: `python -m nbr check --profile sample`,
`python scripts/check_data.py sample`, `python -m nbr tune` and `python -m nbr evaluate --skip-l3` for
both datasets, `python -m nbr.r3_eval` and `python -m nbr.usercf_eval --stage all` for both datasets,
`python -m pytest`, `python scripts/audit_code.py`, `python scripts/make_results.py --verify`.
Not run: `pip install -e`, `scripts/make_holdout.py` (needs the raw archives), `scripts/make_eda.py`
and every `medium` / `holdout` command (development and confirmatory runs are not repeated).

## Results

Confirmatory results (PREREG v1.2, `holdout` profile; Instacart holdout read once, Complete Journey
refit on its `medium` data). Numbers are the `current = true` rows of
`reports/results/tables/T1_master_hypotheses.csv`; diff = model - comparator, paired bootstrap over
users (n = 1000, seed 42); decision wording as pre-registered. H5 is a secondary confirmatory
comparison with a 95 % interval only.

| Hypothesis (metric) | Dataset | Users (n) | Comparator -> model | diff | 95 % CI | Decision |
|---|---|---|---|---|---|---|
| H1 (repeat Recall@10) | Instacart holdout | 18,677 | 0.5355 -> 0.5643 | +0.0288 | [0.0256, 0.0319] | improvement observed |
| H2 (new-item Recall@10) | Instacart holdout | 16,262 | 0.0404 -> 0.0422 | +0.0018 | [0.0009, 0.0028] | improvement observed |
| H3 (NDCG@10) | Instacart holdout | 20,000 | 0.3947 -> 0.4226 | +0.0280 | [0.0263, 0.0297] | improvement observed |
| H3 (Recall@10) | Instacart holdout | 20,000 | 0.3261 -> 0.3485 | +0.0224 | [0.0205, 0.0242] | improvement observed |
| H4 (category hit@10) | Instacart holdout | 16,262 | 0.3697 -> 0.4072 | +0.0375 | [0.0339, 0.0413] | improvement observed |
| H5 (new-item Recall@10, secondary) | Instacart holdout | 16,262 | 0.0404 -> 0.0476 | +0.0073 | [0.0057, 0.0091] | improvement observed |
| H1 (repeat Recall@10) | Complete Journey refit | 1,867 | 0.2497 -> 0.2497 | 0.0000 | [0.0000, 0.0000] | no difference established |
| H2 (new-item Recall@10) | Complete Journey refit | 1,971 | 0.0130 -> 0.0127 | -0.0003 | [-0.0031, 0.0026] | no difference established |
| H3 (NDCG@10) | Complete Journey refit | 2,069 | 0.1859 -> 0.1859 | 0.0000 | [0.0000, 0.0000] | no difference established |
| H3 (Recall@10) | Complete Journey refit | 2,069 | 0.1187 -> 0.1187 | 0.0000 | [0.0000, 0.0000] | no difference established |
| H4 (category hit@10) | Complete Journey refit | 1,971 | 0.2112 -> 0.2405 | +0.0293 | [0.0166, 0.0417] | improvement observed |
| H5 (new-item Recall@10, secondary) | Complete Journey refit | 1,971 | 0.0130 -> 0.0143 | +0.0013 | [-0.0011, 0.0037] | no difference established |
| H1 (repeat Recall@10) | Complete Journey, Instacart selections transferred | 1,867 | 0.2497 -> 0.2193 | -0.0303 | [-0.0431, -0.0185] | lower |
| H2 (new-item Recall@10) | Complete Journey, Instacart selections transferred | 1,971 | 0.0130 -> 0.0133 | +0.0003 | [-0.0016, 0.0026] | no difference established |
| H3 (NDCG@10) | Complete Journey, Instacart selections transferred | 2,069 | 0.1859 -> 0.1704 | -0.0155 | [-0.0225, -0.0087] | lower |
| H3 (Recall@10) | Complete Journey, Instacart selections transferred | 2,069 | 0.1187 -> 0.1065 | -0.0122 | [-0.0175, -0.0066] | lower |
| H4 (category hit@10) | Complete Journey, Instacart selections transferred | 1,971 | 0.2112 -> 0.2461 | +0.0349 | [0.0254, 0.0448] | improvement observed |

Comparators: H1 personal purchase frequency (repeat list), H2 and H5 flat popularity excluding
history items, H3 personal purchase frequency (full list), H4 population category ranking. Model
lists: H1 `repeat_star` (R1), H2 `explore_star` (E1), H3 `final` (C0), H4 `category_ranking`, H5
`itemcf_flat` (E3).

Development results (`medium` profile; H6-H10 for R2, E2, R3 and the combinations C1-C3b) are
labelled **development** in `reports/results/RESULTS.md` (stage `development medium`) and are not
confirmatory. A second confirmatory run on new Instacart users (holdout2) for the combination
selected on development data is **planned (PREREG v1.3)**; the amendment is not committed and nothing
has been run (`docs/MODULES.md`, section 5).

## Folder structure

```
configs/     common.yaml (grids, bootstrap, category levels), sample/medium/holdout.yaml, usercf.yaml, r3.yaml
data/        SOURCE.md, MANIFEST.sha256 (processed Parquet untracked)
docs/        PREREG.md, ARCHITECTURE.md, MODULES.md, LIBRARIES.md, CITATIONS.md, CODE_AUDIT.md
reports/     final/ (holdout JSONs, RUN.md, SHA256SUMS), results/ (tables T1-T7, figures F1-F6, RESULTS.md),
             eda/, data_check/, holdout_prep/; metrics/, figures/, tables/ (untracked run outputs)
scripts/     make_holdout.py, check_data.py, make_eda.py, make_results.py (+ results_sources.yaml), audit_code.py
src/nbr/     package: config, data, io, baselines, repeat, explore, itemcf, combine, metrics,
             features, lr_repeat, usercf, usercf_eval, features_cycle, r3_eval, pipeline, __main__
tests/       pytest suite (synthetic data)
vendor/      ml_group9_e11b365 (unmodified preprocessing snapshot, release/term-project e11b365)
```

## Documents

| Document | Content |
|---|---|
| `docs/PREREG.md` | pre-registration v1.2 (question, model, selection, hypotheses, deviations) |
| `docs/ARCHITECTURE.md` | system structure (diagram), profiles, commands and files, selection procedure, leakage rules |
| `docs/MODULES.md` | module and combination catalogue with status, hypothesis basis and filtering-method classification |
| `docs/LIBRARIES.md` | third-party modules and functions used, with parameters and the related course module |
| `docs/CITATIONS.md` | data sources, vendored code, prior work, AI-tool disclosure (TBD) |
| `docs/CODE_AUDIT.md` | generated static audit (docstrings, library calls, attribution comments, observations) |
| `reports/final/RUN.md` | final run log (PREREG v1.2) |
| `reports/results/RESULTS.md` | consolidated result tables and figures with source files and digests |
| `reports/eda/eda_summary.md` | exploratory data analysis of the development data (Korean) |

## Team and contributions

TBD (team members, roles and contribution shares to be filled in by the team).

## AI-tool usage

TBD (see `docs/CITATIONS.md`, section 5). Git history: commits carry a `Co-Authored-By` trailer where
an AI assistant co-wrote the change.
