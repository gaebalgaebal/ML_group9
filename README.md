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
| Instacart (orders, products, aisles, departments) | public Kaggle release; see `docs/CITATIONS.md` | development (`medium`: 20,000 users), confirmatory (`holdout`: 20,000 unused users), smoke (`sample`: 1,500 users) |
| dunnhumby The Complete Journey (transaction_data, product) | dunnhumby source files; see `docs/CITATIONS.md` | development and refit evaluation (`medium`: 2,069 households; no unused households, limitation recorded in PREREG v1.2) |

Processed Parquet files (`items`, `fit_history`, `fit_target`, `eval_history`, `eval_target` per
profile and dataset) are produced by the unmodified ML_group9 preprocessing snapshot in
`vendor/ml_group9_e11b365` and are **not tracked** by git. Provenance and checksums:
`data/SOURCE.md`, `data/MANIFEST.sha256`. Hierarchy: Instacart department -> aisle (L2) -> ingredient
type -> item; Complete Journey DEPARTMENT -> commodity (L2) -> sub-commodity -> item.

## Structure (summary)

Repeat branch: R1 decayed purchase frequency (adopted), R2 L2 logistic regression on 12 history
features (development). Explore branch: E1 hierarchical category-profile x within-category popularity
(adopted), E2 user-based CF (development), E3 flat item-CF (secondary). Slot combination: the first
`r` slots from the repeat list, the rest from the explore list. Full description with a diagram,
profiles, commands, selection procedure and leakage rules: **`docs/ARCHITECTURE.md`**. Module codes
(E1-E8, R1-R6, C0-C4, C3b) and their status: **`docs/MODULES.md`**.

## How to run

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
`NBR_DATA_DIR` at a folder with that layout (the git worktrees of this project share one copy).
Verify with:

```bash
python scripts/check_data.py medium        # -> reports/data_check/medium.json
python -m nbr check --profile medium       # split statistics; holdout eval split is skipped by the guard
```

### 3. Holdout profile (done once; recorded in `data/SOURCE.md`)

```bash
python scripts/make_holdout.py --raw-dir <ML_group9/data/raw> --size 20000 --seed 42
```

Draws 20,000 Instacart users disjoint from `medium` and `sample` with the vendored preprocessing
code and writes `reports/holdout_prep/holdout_selection.json`. The script imports the POSIX-only
`resource` module (it was run on a non-Windows interpreter).

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

Development-only tools: `python -m nbr.usercf_eval --profile medium --dataset D --stage all`
(stand-alone user-CF run, W7) and, on branch `feat/w9-r3-cycle`,
`python -m nbr.r3_eval --profile medium --dataset D` (R3 screening, W9). Both refuse holdout profiles.

### 5. Tests and code audit

```bash
python -m pytest                       # synthetic-data tests (tests/)
python scripts/audit_code.py           # static audit -> docs/CODE_AUDIT.md
```

## Results

See `reports/results/RESULTS.md` (branch `feat/w11-results`; tables T1-T7 and figures F1-F6 with
source files and SHA-256 digests). Use only T1 rows with `current = true`. Confirmatory results
(Instacart holdout, Complete Journey refit) and development results (`medium`) are labelled
separately; no numbers are repeated in this README.

## Folder structure

```
configs/     common.yaml (grids, bootstrap, category levels), sample/medium/holdout.yaml, usercf.yaml
data/        SOURCE.md, MANIFEST.sha256 (processed Parquet untracked)
docs/        PREREG.md, ARCHITECTURE.md, MODULES.md, LIBRARIES.md, CITATIONS.md, CODE_AUDIT.md
reports/     final/ (holdout JSONs, RUN.md, SHA256SUMS), data_check/, holdout_prep/
             metrics/, figures/, tables/ (untracked run outputs); results/ on feat/w11-results
scripts/     make_holdout.py, check_data.py, audit_code.py
src/nbr/     package: config, data, io, baselines, repeat, explore, itemcf, combine, metrics,
             features, lr_repeat, usercf, usercf_eval, pipeline, __main__
tests/       pytest suite
vendor/      ml_group9_e11b365 (unmodified preprocessing snapshot, release/term-project e11b365)
```

## Documents

| Document | Content |
|---|---|
| `docs/PREREG.md` | pre-registration v1.2 (question, model, selection, hypotheses, deviations) |
| `docs/ARCHITECTURE.md` | system structure (diagram), profiles, commands and files, selection procedure, leakage rules |
| `docs/MODULES.md` | module and combination catalogue with status, hypothesis basis and filtering-method classification |
| `docs/LIBRARIES.md` | third-party modules and functions used, with parameters (course coverage: TBD) |
| `docs/CITATIONS.md` | data sources, vendored code, prior work, AI-tool disclosure (TBD) |
| `docs/CODE_AUDIT.md` | generated static audit (docstrings, library calls, attribution comments, observations) |
| `reports/final/RUN.md` | final run log (PREREG v1.2) |
| `reports/results/RESULTS.md` | consolidated result tables and figures (branch `feat/w11-results`) |

## Team and contributions

TBD.

## AI-tool usage

TBD (see `docs/CITATIONS.md`).
