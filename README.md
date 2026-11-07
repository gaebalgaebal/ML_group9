# nbr — hierarchical next-basket recommendation (ML Team 9)

Next-basket recommendation that ranks **repurchases** and **exploration** separately and merges the
two lists into one Top-10 list per user. The repeat branch scores items the user has bought before;
the explore branch scores items the user has never bought, with a hierarchical model that goes through
the category hierarchy (user's category profile -> within-category popularity). Everything is
pre-registered (`docs/PREREG.md`): hypotheses, selection rules and the single read of each holdout
split were fixed before the four final runs (v1.2 on `holdout`, v1.3 on `holdout2`, v1.4 exploration
policy on `holdout3`, v1.5 two-CF final system on `holdout4` with the v1.5.1 policy replication).

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
| Instacart (orders, products, aisles, departments) | public Instacart release (Kaggle mirror); see `docs/CITATIONS.md` | development (`medium`: 20,000 users), confirmatory (`holdout`: 20,000 unused users, PREREG v1.2; `holdout2`: 20,000 further unused users, PREREG v1.3; `holdout3`: 20,000 further unused users, PREREG v1.4; `holdout4`: 20,000 further unused users, PREREG v1.5), smoke (`sample`: 1,500 users) |
| dunnhumby The Complete Journey (transaction_data, product) | dunnhumby source files (Kaggle mirror); see `docs/CITATIONS.md` | development and refit evaluation (`medium`: 2,069 households; no unused households, limitation recorded in PREREG v1.2) |

Processed Parquet files (`items`, `fit_history`, `fit_target`, `eval_history`, `eval_target` per
profile and dataset; profiles `sample`, `medium`, `holdout`, `holdout2`, `holdout3`, `holdout4`) are produced by the
unmodified ML_group9 preprocessing snapshot in `vendor/ml_group9_e11b365` and are **not tracked** by git.
Provenance and checksums: `data/SOURCE.md`, `data/MANIFEST.sha256`. Hierarchy: Instacart department ->
aisle (L2) -> ingredient type -> item; Complete Journey DEPARTMENT -> commodity (L2) -> sub-commodity -> item.

## Structure (summary)

Final system (PREREG v1.5, tested on `holdout4`): C4 = R2' + E2, two collaborative-filtering methods.
Repeat branch: R2' = the R2 L2 logistic regression on 12 history features plus the item-CF similarity
feature `icf` (`src/nbr/features_icf.py`; 20 item neighbours, fixed); R2 without the feature is the
PREREG v1.3 final repeat module (C3 = R2 + E2, confirmed on `holdout2`, is the H14 comparator), and R1
decayed purchase frequency is the previous final repeat module of PREREG v1.2 and the H6 comparator.
Explore branch: E2 user-based CF (adopted; E1 hierarchical category-profile x within-category
popularity is the previous final explore module and, as part of C0, the H7' comparator), E3 flat
item-CF (secondary; its neighbour lists feed the `icf` feature). Slot combination: the first `r`
slots from the repeat list, the rest from the explore list (`r* = 10` on every fit split, T2, so
the explore list fills only the users with fewer than 10 repeat candidates). Exploration policy
(PREREG v1.4, tested on `holdout3`; `src/nbr/policy.py`): P_T = C3 with one exploration slot
(r_u = 9) for users with at most T = 4 history baskets and r* otherwise; S = the E2 list on its own as a
separate 10-item exploration list; repeated on `holdout4` with C4 as the base (PREREG v1.5.1, secondary).
Diagram of the PREREG v1.3 final system: `docs/figures/nbr_architecture_final.png` (Mermaid source `.mmd`
next to it; C4 adds the `icf` feature to its repeat ranker). Full description with the branch diagram,
profiles, commands, selection procedure and leakage rules: **`docs/ARCHITECTURE.md`**. Module codes
(E1-E8, R1-R6, R2', C0-C4, C3b) and their status: **`docs/MODULES.md`**.

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
`complete_journey`; profiles `sample`, `medium`, `holdout`, `holdout2`), or point the environment variable
`NBR_DATA_DIR` at a folder with that layout (profiles `sample`, `medium`, `holdout`, `holdout2`,
`holdout3`; `NBR_DATA_DIR` wins over `configs/common.yaml:data_dir`;
the git worktrees of this project share one copy). Verify with:

```bash
python scripts/check_data.py sample        # -> reports/data_check/sample.json (medium, holdout likewise)
python -m nbr check --profile sample       # split statistics; the holdout eval split is skipped by the guard
```

### 3. Holdout profiles (each done once; recorded in `data/SOURCE.md`)

```bash
# holdout (PREREG v1.2, 2026-11-05): 20,000 Instacart users disjoint from medium and sample
python scripts/make_holdout.py --raw-dir <ML_group9/data/raw> --size 20000 --seed 42
# holdout2 (PREREG v1.3, 2026-11-06): 20,000 Instacart users disjoint from sample, medium and holdout
python scripts/make_holdout.py --raw-dir <ML_group9/data/raw> --name holdout2 --exclude sample,medium,holdout --size 20000 --seed 42 --expect-remaining 89936
# holdout3 (PREREG v1.4, 2026-11-07): 20,000 Instacart users disjoint from sample, medium, holdout and holdout2
python scripts/make_holdout.py --raw-dir <ML_group9/data/raw> --name holdout3 --exclude sample,medium,holdout,holdout2 --size 20000 --seed 42 --expect-remaining 69936
# holdout4 (PREREG v1.5, 2026-11-07): 20,000 Instacart users disjoint from sample, medium, holdout, holdout2 and holdout3
python scripts/make_holdout.py --raw-dir <ML_group9/data/raw> --name holdout4 --exclude sample,medium,holdout,holdout2,holdout3 --size 20000 --seed 42 --expect-remaining 49936
```

All four draw the users with the vendored preprocessing code and write
`reports/holdout_prep/<name>_selection.json`. With `--exclude holdout` (or `holdout2`, `holdout3`) those user ids are
not read from their files: the earlier draws are re-run and checked against the digests recorded in
their selection reports; `--expect-remaining` stops before the draw unless exactly that many users
remain. Peak memory is `ru_maxrss` on POSIX and the peak working set on Windows. None of the commands is
re-run for a release (they need the raw archives and would rewrite the data).

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

# confirmatory (holdout2, PREREG v1.3): the five commands of reports/final2/RUN.md, each run once
python -m nbr tune --profile holdout2 --dataset instacart
python -m nbr evaluate --profile holdout2 --dataset instacart --final --skip-l3
python -m nbr tune --profile holdout2 --dataset complete_journey
python -m nbr evaluate --profile holdout2 --dataset complete_journey --skip-l3
python -m nbr evaluate --profile holdout2 --dataset complete_journey --transfer-from instacart --skip-l3

# confirmatory (holdout3, PREREG v1.4 exploration policy): the five commands of reports/final3/RUN.md, each run once
python -m nbr tune --profile holdout3 --dataset instacart
python -m nbr evaluate --profile holdout3 --dataset instacart --final --skip-l3 --policy
python -m nbr tune --profile holdout3 --dataset complete_journey
python -m nbr evaluate --profile holdout3 --dataset complete_journey --skip-l3 --policy
python -m nbr evaluate --profile holdout3 --dataset complete_journey --transfer-from instacart --skip-l3 --policy
# confirmatory (holdout4, PREREG v1.5 two-CF final system + v1.5.1 policy replication): the five commands of
# reports/final4/RUN.md, each run once; configs/holdout4.yaml sets lr_repeat.itemcf_feature: true (R2', C4, "v15" block)
python -m nbr tune --profile holdout4 --dataset instacart
python -m nbr evaluate --profile holdout4 --dataset instacart --final --skip-l3 --policy
python -m nbr tune --profile holdout4 --dataset complete_journey
python -m nbr evaluate --profile holdout4 --dataset complete_journey --skip-l3 --policy
python -m nbr evaluate --profile holdout4 --dataset complete_journey --transfer-from instacart --skip-l3 --policy
```

`tune` writes `reports/metrics/<profile>_<dataset>_tuning.json` and
`artifacts/<profile>/<dataset>_lr_repeat.joblib`; `evaluate` writes
`reports/metrics/<profile>_<dataset>_eval.json` (`_eval_transfer.json` with `--transfer-from`).
Optional flags: `--no-itemcf`, `--no-lr`, `--no-integration`, `--skip-l3`, `--policy` (adds the PREREG v1.4
exploration-policy block `policy` with H11-H13, secondary metrics, history-length bins and the variants;
needs the LR ranker and the integration selections). The exact command sequences of the final runs are in
`reports/final/RUN.md` (v1.2), `reports/final2/RUN.md` (v1.3) and `reports/final3/RUN.md` (v1.4); their
JSONs, digests and run logs are in `reports/final/`, `reports/final2/` and `reports/final3/`. The
development check of the policy code on `medium` (`python scripts/check_policy_dev.py --reference-tuning-dir <W8b metrics>`)
is `reports/posthoc/policy_dev_medium.md` (not confirmatory).

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
python -m pytest                        # synthetic-data tests (tests/), 265 tests
python scripts/audit_code.py            # static audit -> docs/CODE_AUDIT.md
python scripts/make_results.py --verify # tables T1-T7 and figures F1-F8 -> reports/results/ (+ determinism check)
python scripts/check_readme_results.py  # README result tables vs T1, cell by cell (0 mismatches; --render prints them)
python scripts/make_eda.py              # EDA tables and figures -> reports/eda/ (reads the medium profile)
```

`scripts/make_results.py` reads the final JSONs from `reports/final/` (PREREG v1.2), `reports/final2/`
(PREREG v1.3) and `reports/final3/` (PREREG v1.4) and the development JSONs from
the sibling git worktrees listed in `scripts/results_sources.yaml` (folder given by `NBR_WORKTREES`,
default: the parent folder of the repository); those development JSONs are untracked run outputs.

### Verified commands (release preparation, `sample` profile)

Run in this order from the repository root with `NBR_DATA_DIR` set: `python -m nbr check --profile sample`,
`python scripts/check_data.py sample`, `python -m nbr tune` and `python -m nbr evaluate --skip-l3` for
both datasets, `python -m nbr.r3_eval` and `python -m nbr.usercf_eval --stage all` for both datasets,
`python -m pytest`, `python scripts/audit_code.py`, `python scripts/make_results.py --verify`; after the
holdout2 run and again after the holdout3 run: `python -m pytest`, `python scripts/make_results.py --verify`,
`python scripts/check_readme_results.py`.
Not run: `pip install -e`, `scripts/make_holdout.py` (needs the raw archives), `scripts/make_eda.py`
and every `medium` / `holdout` / `holdout2` / `holdout3` command (development and confirmatory runs are not repeated).

## Results

Numbers are the `current = true` rows of `reports/results/tables/T1_master_hypotheses.csv`, written by
`scripts/make_results.py`; the tables below are rendered from T1 and checked against it cell by cell
by `python scripts/check_readme_results.py`. diff = model - comparator, paired bootstrap over users
(n = 1000, seed 42); decision wording as pre-registered: "improvement observed" when the lower bound
is above 0, "lower" when the upper bound is below 0, otherwise "no difference established". The
classification of every row (`prereg`, `tier`) is explicit in `scripts/results_sources.yaml`.

### Final system with two CF methods (PREREG v1.5, holdout4)

Instacart `holdout4`: 20,000 users not in `sample`, `medium`, `holdout`, `holdout2` or `holdout3`; eval split read
once (command 2 of `reports/final4/RUN.md`). PREREG v1.5 fixes the final system with two collaborative-filtering
methods: C4 (PREREG v1.5) = R2' + E2, where R2' (`repeat_lri`) is the logistic-regression repeat ranker R2 with
one extra feature, the item-CF similarity `icf(u, i) = log(1 + sum over the user's other items j of sim(j, i))`
taken from the top-20 cosine item-neighbour lists of the scored history (self excluded; neighbours computed on the
fit history for training rows and on the eval history for lists), and E2 is the user-based CF explore list. The
C of R2' is selected separately from R2 on the same validation users (holdout4: C = 0.01 for R2', 0.01 for
R2); r of C4 is re-selected like r of C3 (train-only R2' bundle, LR validation users, NDCG@10; holdout4: r = 10
for C4, 10 for C3; user-CF k = 100, log1p). The label C4 had denoted a planned, never evaluated
combination E3 + R2 in the consolidated tables before v1.5; that placeholder was never run. Primary family of 2; the
97.5 % interval (1 - 0.05/2) is the pre-registered one, and the profile's bootstrap levels are 0.95 and 0.975, so
every row of this stage carries it. H14: NDCG@10 of C4 minus C3 over all users, "non-inferior" when the 97.5 % lower
bound is above -0.003 (the v1.4 margin), otherwise "non-inferiority not established". H15: NDCG@10 of C4 minus
personal frequency (padded with popularity) over all users, with the improvement wording above. Decisions at the
97.5 % level: H14 non-inferior, H15 improvement observed.

<!-- results-table: holdout4_primary -->
| Hypothesis (metric) | Dataset | Users (n) | Comparator -> model | diff | 95 % CI | 97.5 % CI | Decision (95 % / 97.5 %) |
|---|---|---|---|---|---|---|---|
| H14 (NDCG@10, non-inferiority): C4 vs C3 | Instacart holdout4 | 20,000 | 0.4258 -> 0.4258 | 0.0000 | [-0.0001, 0.0001] | [-0.0001, 0.0001] | non-inferior / non-inferior |
| H15 (NDCG@10): C4 vs personal frequency | Instacart holdout4 | 20,000 | 0.3934 -> 0.4258 | +0.0324 | [0.0307, 0.0340] | [0.0305, 0.0343] | improvement observed / improvement observed |

Deviation PREREG v1.5.2: the first execution of command 2 was terminated by the host tool after it had loaded the eval
split and before any output; the authorised single re-run (started outside the host tool) produced the result, so the
holdout4 Instacart eval split was loaded twice and the first load produced no output (`reports/final4/RUN.md`).

Secondary (no confirmatory claim; CI only): repeat Recall@10 of R2' against R2 and against R1 (18,701 users
with a repeat target item), Recall@10 of C4 against C3, the standardized coefficients of R2' (icf coefficient
-0.002 on Instacart, -0.012 on Complete Journey; `v15.lr_repeat_icf` of the eval JSON) and the
history-length bins of H14 (T5, figure F9). The replication of H6, H7' and H8 on holdout4 follows (CI levels 95 % and
97.5 % as written by the code; secondary under v1.5). The development check of the v1.5 code on `medium` is
`reports/posthoc/icf_dev_medium.md`.

<!-- results-table: holdout4_secondary -->
| Hypothesis (metric) | Dataset | Users (n) | Comparator -> model | diff | 95 % CI | 97.5 % CI | Decision (95 % / 97.5 %) |
|---|---|---|---|---|---|---|---|
| R2' vs R2 (repeat Recall@10) | Instacart holdout4 | 18,701 | 0.5750 -> 0.5749 | -0.0002 | [-0.0004, 0.0000] | [-0.0004, 0.0001] | (CI only) |
| R2' vs R1 (repeat Recall@10) | Instacart holdout4 | 18,701 | 0.5628 -> 0.5749 | +0.0121 | [0.0102, 0.0141] | [0.0099, 0.0144] | (CI only) |
| C4 vs C3 (Recall@10) | Instacart holdout4 | 20,000 | 0.3522 -> 0.3521 | -0.0001 | [-0.0002, 0.0000] | [-0.0002, 0.0000] | (CI only) |

<!-- results-table: holdout4_replication -->
| Hypothesis (metric) | Dataset | Users (n) | Comparator -> model | diff | 95 % CI | 97.5 % CI | Decision (95 % / 97.5 %) |
|---|---|---|---|---|---|---|---|
| H6 (repeat Recall@10) | Instacart holdout4 | 18,701 | 0.5628 -> 0.5750 | +0.0122 | [0.0103, 0.0143] | [0.0100, 0.0146] | improvement observed / improvement observed |
| H7' (NDCG@10) | Instacart holdout4 | 20,000 | 0.4197 -> 0.4258 | +0.0061 | [0.0051, 0.0070] | [0.0050, 0.0072] | improvement observed / improvement observed |
| H7' (Recall@10, reported) | Instacart holdout4 | 20,000 | 0.3466 -> 0.3522 | +0.0056 | [0.0044, 0.0068] | [0.0042, 0.0071] | improvement observed / improvement observed |
| H8 (new-item Recall@10) | Instacart holdout4 | 16,335 | 0.0408 -> 0.0590 | +0.0181 | [0.0162, 0.0200] | [0.0158, 0.0202] | improvement observed / improvement observed |

Complete Journey refit (same `medium` data as development, not independent; C = 1 for R2', r = 10
for C4) and transfer of the Instacart holdout4 selections and artifacts, secondary under PREREG v1.5.

<!-- results-table: holdout4_complete_journey -->
| Hypothesis (metric) | Dataset | Users (n) | Comparator -> model | diff | 95 % CI | 97.5 % CI | Decision (95 % / 97.5 %) |
|---|---|---|---|---|---|---|---|
| H14 (NDCG@10, non-inferiority): C4 vs C3 | Complete Journey refit | 2,069 | 0.2046 -> 0.2048 | +0.0002 | [-0.0004, 0.0007] | [-0.0005, 0.0008] | non-inferior / non-inferior |
| H15 (NDCG@10): C4 vs personal frequency | Complete Journey refit | 2,069 | 0.1859 -> 0.2048 | +0.0189 | [0.0147, 0.0231] | [0.0138, 0.0235] | improvement observed / improvement observed |
| R2' vs R2 (repeat Recall@10) | Complete Journey refit | 1,867 | 0.2740 -> 0.2739 | -0.0001 | [-0.0012, 0.0009] | [-0.0013, 0.0010] | (CI only) |
| H14 (NDCG@10, non-inferiority): C4 vs C3 | Complete Journey, Instacart selections transferred | 2,069 | 0.1992 -> 0.1993 | +0.0001 | [-0.0003, 0.0006] | [-0.0004, 0.0006] | non-inferior / non-inferior |
| H15 (NDCG@10): C4 vs personal frequency | Complete Journey, Instacart selections transferred | 2,069 | 0.1859 -> 0.1993 | +0.0134 | [0.0084, 0.0182] | [0.0078, 0.0189] | improvement observed / improvement observed |
| R2' vs R2 (repeat Recall@10) | Complete Journey, Instacart selections transferred | 1,867 | 0.2686 -> 0.2683 | -0.0003 | [-0.0010, 0.0005] | [-0.0011, 0.0006] | (CI only) |

Exploration policy on the final system (secondary replication, holdout4; PREREG v1.5.1): the v1.4 policy is
repeated with C4 as the base, P_T(C4) = C4 with r_u = 9 for users with at most T = 4 history baskets (4,316 of
the 20,000 users; 1 of the 2,069 Complete Journey households) and r of C4 otherwise; S = E2 on its own.
T = 4 and delta = 0.003 are unchanged, the intervals are 95 % and 98.33 % with the v1.4 decision wording, and no
confirmatory claim is made: the confirmatory policy result remains H11-H13 on `holdout3` (first three rows of the
table). The v1.4 policy on C3 is repeated on holdout4 as a reference (last two rows); the Complete Journey values are
in T1 (`policy_c4:*` rows of stage `confirmatory holdout4`).

<!-- results-table: holdout4_policy -->
| Hypothesis (metric) | Dataset | Users (n) | Comparator -> model | diff | 95 % CI | 98.33 % CI | Decision (95 % / 98.33 %) |
|---|---|---|---|---|---|---|---|
| H11 (NDCG@10, non-inferiority): P_T vs C3 | Instacart holdout3 | 20,000 | 0.4274 -> 0.4261 | -0.0013 | [-0.0015, -0.0011] | [-0.0016, -0.0011] | non-inferior / non-inferior |
| H12 (new-item hit share@10): P_T vs C3 | Instacart holdout3 | 20,000 | 0.0042 -> 0.0143 | +0.0101 | [0.0087, 0.0115] | [0.0084, 0.0118] | improvement observed / improvement observed |
| H13 (new-item hit share@10): S vs flat popularity | Instacart holdout3 | 16,340 | 0.1681 -> 0.2151 | +0.0470 | [0.0420, 0.0520] | [0.0413, 0.0528] | improvement observed / improvement observed |
| as H11 (NDCG@10): P_T(C4) vs C4 | Instacart holdout4 | 20,000 | 0.4258 -> 0.4243 | -0.0015 | [-0.0017, -0.0012] | [-0.0017, -0.0012] | non-inferior / non-inferior |
| as H12 (new-item hit share@10): P_T(C4) vs C4 | Instacart holdout4 | 20,000 | 0.0049 -> 0.0163 | +0.0115 | [0.0101, 0.0129] | [0.0097, 0.0132] | improvement observed / improvement observed |
| as H13 (new-item hit share@10): S vs flat popularity | Instacart holdout4 | 16,335 | 0.1726 -> 0.2243 | +0.0517 | [0.0467, 0.0565] | [0.0457, 0.0575] | improvement observed / improvement observed |
| reference: P_T vs C3 (NDCG@10) | Instacart holdout4 | 20,000 | 0.4258 -> 0.4243 | -0.0015 | [-0.0017, -0.0013] | [-0.0017, -0.0012] | non-inferior / non-inferior |
| reference: P_T vs C3 (new-item hit share@10) | Instacart holdout4 | 20,000 | 0.0049 -> 0.0163 | +0.0115 | [0.0101, 0.0129] | [0.0097, 0.0132] | improvement observed / improvement observed |

### Exploration policy (PREREG v1.4, holdout3)

Instacart `holdout3`: 20,000 users not in `sample`, `medium`, `holdout` or `holdout2`; eval split read
once (command 2 of `reports/final3/RUN.md`). P_T = C3 (`final_lr_ucf`) with r_u = 9 for users with at
most T = 4 distinct history baskets (4,252 of the 20,000 users) and r_u = r*
otherwise; S = E2 (`usercf`) top 10 on its own. T = 4 and delta = 0.003 were fixed in PREREG v1.4 from
the W15 post-hoc study on `medium` (development data) and were not re-selected. Primary family of 3; the
98.33 % interval (1 - 0.05/3) is the pre-registered one. H11: NDCG@10 of P_T minus C3 over all users,
"non-inferior" when the 98.33 % lower bound is above -0.003, otherwise "non-inferiority not established".
H12: share of users with at least one new-item hit in the top 10, P_T minus C3, all users (a user without
a new target item counts 0). H13: the same share in the 10-item exploration list, S minus flat popularity
excluding history items, users with a new target item. H12 and H13 use the improvement wording above.

<!-- results-table: holdout3_primary -->
| Hypothesis (metric) | Dataset | Users (n) | Comparator -> model | diff | 95 % CI | 98.33 % CI | Decision (95 % / 98.33 %) |
|---|---|---|---|---|---|---|---|
| H11 (NDCG@10, non-inferiority) | Instacart holdout3 | 20,000 | 0.4274 -> 0.4261 | -0.0013 | [-0.0015, -0.0011] | [-0.0016, -0.0011] | non-inferior / non-inferior |
| H12 (new-item hit share@10) | Instacart holdout3 | 20,000 | 0.0042 -> 0.0143 | +0.0101 | [0.0087, 0.0115] | [0.0084, 0.0118] | improvement observed / improvement observed |
| H13 (new-item hit share@10, S vs flat popularity) | Instacart holdout3 | 16,340 | 0.1681 -> 0.2151 | +0.0470 | [0.0420, 0.0520] | [0.0413, 0.0528] | improvement observed / improvement observed |

Secondary (no confirmatory claim; CI only): Recall@10, new-item Recall@10 and repeat Recall@10 of P_T
against C3, the r_short = 8 variant (two exploration slots for the same users) and the sensitivity values
T = 6 and T = 10 (7,155 and 10,837 users
with one exploration slot). The replication of H6, H7' and H8 on holdout3 follows (CI levels 95 % and
99.375 % as written by the code; secondary under v1.4). History-length bins are in T5 and figure F8
(`reports/results/`); the development check of the policy code on `medium` is
`reports/posthoc/policy_dev_medium.md`.

<!-- results-table: holdout3_secondary -->
| Hypothesis (metric) | Dataset | Users (n) | Comparator -> model | diff | 95 % CI | 98.33 % CI | Decision (95 % / 98.33 %) |
|---|---|---|---|---|---|---|---|
| P_T vs C3 (Recall@10) | Instacart holdout3 | 20,000 | 0.3522 -> 0.3505 | -0.0017 | [-0.0021, -0.0014] | [-0.0022, -0.0013] | (CI only) |
| P_T vs C3 (new-item Recall@10) | Instacart holdout3 | 16,340 | 0.0026 -> 0.0049 | +0.0023 | [0.0019, 0.0027] | [0.0018, 0.0028] | (CI only) |
| P_T vs C3 (repeat Recall@10) | Instacart holdout3 | 18,738 | 0.5741 -> 0.5680 | -0.0061 | [-0.0068, -0.0054] | [-0.0069, -0.0052] | (CI only) |
| r_short = 8, T = 4 vs C3 (NDCG@10) | Instacart holdout3 | 20,000 | 0.4274 -> 0.4244 | -0.0031 | [-0.0034, -0.0028] | [-0.0035, -0.0028] | (CI only) |
| r_short = 8, T = 4 vs C3 (new-item hit share@10) | Instacart holdout3 | 20,000 | 0.0042 -> 0.0226 | +0.0184 | [0.0166, 0.0203] | [0.0163, 0.0205] | (CI only) |
| T = 6 vs C3 (NDCG@10) | Instacart holdout3 | 20,000 | 0.4274 -> 0.4250 | -0.0025 | [-0.0027, -0.0022] | [-0.0028, -0.0022] | (CI only) |
| T = 6 vs C3 (new-item hit share@10) | Instacart holdout3 | 20,000 | 0.0042 -> 0.0211 | +0.0170 | [0.0152, 0.0188] | [0.0148, 0.0191] | (CI only) |
| T = 10 vs C3 (NDCG@10) | Instacart holdout3 | 20,000 | 0.4274 -> 0.4232 | -0.0043 | [-0.0046, -0.0040] | [-0.0047, -0.0039] | (CI only) |
| T = 10 vs C3 (new-item hit share@10) | Instacart holdout3 | 20,000 | 0.0042 -> 0.0282 | +0.0240 | [0.0218, 0.0261] | [0.0213, 0.0268] | (CI only) |

<!-- results-table: holdout3_replication -->
| Hypothesis (metric) | Dataset | Users (n) | Comparator -> model | diff | 95 % CI | 99.375 % CI | Decision (95 % / 99.375 %) |
|---|---|---|---|---|---|---|---|
| H6 (repeat Recall@10) | Instacart holdout3 | 18,738 | 0.5611 -> 0.5741 | +0.0129 | [0.0108, 0.0149] | [0.0100, 0.0156] | improvement observed / improvement observed |
| H7' (NDCG@10) | Instacart holdout3 | 20,000 | 0.4206 -> 0.4274 | +0.0068 | [0.0058, 0.0078] | [0.0055, 0.0082] | improvement observed / improvement observed |
| H7' (Recall@10, reported) | Instacart holdout3 | 20,000 | 0.3460 -> 0.3522 | +0.0062 | [0.0049, 0.0074] | [0.0046, 0.0078] | improvement observed / improvement observed |
| H8 (new-item Recall@10) | Instacart holdout3 | 16,340 | 0.0404 -> 0.0577 | +0.0174 | [0.0152, 0.0193] | [0.0147, 0.0200] | improvement observed / improvement observed |

Complete Journey refit (same `medium` data as development, not independent) and transfer with the same
policy, secondary under PREREG v1.4. Only 1 of the 2,069 households have at most
4 history baskets, so P_T coincides with C3 for the other households.

<!-- results-table: holdout3_complete_journey -->
| Hypothesis (metric) | Dataset | Users (n) | Comparator -> model | diff | 95 % CI | 98.33 % CI | Decision (95 % / 98.33 %) |
|---|---|---|---|---|---|---|---|
| H11 (NDCG@10, non-inferiority) | Complete Journey refit | 2,069 | 0.2046 -> 0.2046 | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] | non-inferior / non-inferior |
| H12 (new-item hit share@10) | Complete Journey refit | 2,069 | 0.0000 -> 0.0000 | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] | no difference established / no difference established |
| H13 (new-item hit share@10, S vs flat popularity) | Complete Journey refit | 1,971 | 0.0791 -> 0.0979 | +0.0188 | [0.0061, 0.0315] | [0.0037, 0.0345] | improvement observed / improvement observed |
| H11 (NDCG@10, non-inferiority) | Complete Journey, Instacart selections transferred | 2,069 | 0.1996 -> 0.1996 | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] | non-inferior / non-inferior |
| H12 (new-item hit share@10) | Complete Journey, Instacart selections transferred | 2,069 | 0.0000 -> 0.0000 | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] | no difference established / no difference established |
| H13 (new-item hit share@10, S vs flat popularity) | Complete Journey, Instacart selections transferred | 1,971 | 0.0791 -> 0.1050 | +0.0259 | [0.0137, 0.0381] | [0.0108, 0.0396] | improvement observed / improvement observed |

### Final system (PREREG v1.3, confirmed on holdout2)

Instacart `holdout2`: 20,000 users not in `sample`, `medium` or `holdout`; eval split read once
(command 2 of `reports/final2/RUN.md`). Primary family of 3; the 99.375 % interval is the
multiplicity-adjusted one. H6: R2 vs R1 (repeat Recall@10, users with a repeat target item).
H7': C3 (`final_lr_ucf`) vs C0 (`final`, the v1.2 final list), NDCG@10 with Recall@10 reported
(all users). H8: E2 (`usercf`) vs flat popularity excluding history items (new-item Recall@10,
users with a new target item).

<!-- results-table: holdout2_primary -->
| Hypothesis (metric) | Dataset | Users (n) | Comparator -> model | diff | 95 % CI | 99.375 % CI | Decision (95 % / 99.375 %) |
|---|---|---|---|---|---|---|---|
| H6 (repeat Recall@10) | Instacart holdout2 | 18,686 | 0.5606 -> 0.5724 | +0.0119 | [0.0097, 0.0139] | [0.0088, 0.0149] | improvement observed / improvement observed |
| H7' (NDCG@10) | Instacart holdout2 | 20,000 | 0.4192 -> 0.4255 | +0.0063 | [0.0053, 0.0072] | [0.0051, 0.0076] | improvement observed / improvement observed |
| H7' (Recall@10, reported) | Instacart holdout2 | 20,000 | 0.3461 -> 0.3517 | +0.0056 | [0.0043, 0.0069] | [0.0039, 0.0074] | improvement observed / improvement observed |
| H8 (new-item Recall@10) | Instacart holdout2 | 16,270 | 0.0413 -> 0.0602 | +0.0189 | [0.0171, 0.0209] | [0.0165, 0.0217] | improvement observed / improvement observed |

The C3-vs-C0 difference comes almost entirely from the repeat ranker: replacing R1 by R2 at a fixed
explore list (H7, C1 vs C0) accounts for nearly all of it, and adding E2 on top of R2 (H9, C3 vs C1)
changes the full-list metrics only for the users with fewer than 10 repeat candidates, because
`r* = 10` (secondary rows below; no confirmatory claim).

<!-- results-table: holdout2_decomposition -->
| Hypothesis (metric) | Dataset | Users (n) | Comparator -> model | diff | 95 % CI | 99.375 % CI | Decision (95 % / 99.375 %) |
|---|---|---|---|---|---|---|---|
| H7 (NDCG@10): C1 vs C0 | Instacart holdout2 | 20,000 | 0.4192 -> 0.4253 | +0.0061 | [0.0051, 0.0070] | [0.0048, 0.0074] | improvement observed / improvement observed |
| H7 (Recall@10): C1 vs C0 | Instacart holdout2 | 20,000 | 0.3461 -> 0.3512 | +0.0050 | [0.0037, 0.0063] | [0.0034, 0.0067] | improvement observed / improvement observed |
| H9 (NDCG@10): C3 vs C1 | Instacart holdout2 | 20,000 | 0.4253 -> 0.4255 | +0.0003 | [0.0001, 0.0004] | [0.0001, 0.0004] | improvement observed / improvement observed |
| H9 (Recall@10): C3 vs C1 | Instacart holdout2 | 20,000 | 0.3512 -> 0.3517 | +0.0006 | [0.0003, 0.0009] | [0.0002, 0.0010] | improvement observed / improvement observed |
| C3 vs personal frequency (NDCG@10) | Instacart holdout2 | 20,000 | 0.3917 -> 0.4255 | +0.0338 | [0.0322, 0.0354] | [0.0317, 0.0362] | (CI only) |
| E2 vs flat item-CF (new-item Recall@10) | Instacart holdout2 | 16,270 | 0.0475 -> 0.0602 | +0.0127 | [0.0110, 0.0146] | [0.0104, 0.0153] | (CI only) |

Complete Journey has no unused households: its refit uses the same `medium` data as development
(not independent) and the transfer run applies the Instacart holdout2 selections; both are secondary
under PREREG v1.3. The H1-H5 replication on holdout2 and every history-length bin are in T1, T5 and
figure F7 (`reports/results/`).

<!-- results-table: holdout2_complete_journey -->
| Hypothesis (metric) | Dataset | Users (n) | Comparator -> model | diff | 95 % CI | 99.375 % CI | Decision (95 % / 99.375 %) |
|---|---|---|---|---|---|---|---|
| H6 (repeat Recall@10) | Complete Journey refit | 1,867 | 0.2497 -> 0.2740 | +0.0243 | [0.0158, 0.0333] | [0.0137, 0.0362] | improvement observed / improvement observed |
| H7' (NDCG@10) | Complete Journey refit | 2,069 | 0.1859 -> 0.2046 | +0.0187 | [0.0146, 0.0228] | [0.0126, 0.0247] | improvement observed / improvement observed |
| H7' (Recall@10, reported) | Complete Journey refit | 2,069 | 0.1187 -> 0.1315 | +0.0128 | [0.0090, 0.0167] | [0.0077, 0.0183] | improvement observed / improvement observed |
| H8 (new-item Recall@10) | Complete Journey refit | 1,971 | 0.0130 -> 0.0145 | +0.0015 | [-0.0014, 0.0044] | [-0.0026, 0.0055] | no difference established / no difference established |
| H6 (repeat Recall@10) | Complete Journey, Instacart selections transferred | 1,867 | 0.2193 -> 0.2684 | +0.0490 | [0.0394, 0.0587] | [0.0367, 0.0618] | improvement observed / improvement observed |
| H7' (NDCG@10) | Complete Journey, Instacart selections transferred | 2,069 | 0.1704 -> 0.1993 | +0.0289 | [0.0246, 0.0332] | [0.0231, 0.0350] | improvement observed / improvement observed |
| H7' (Recall@10, reported) | Complete Journey, Instacart selections transferred | 2,069 | 0.1065 -> 0.1279 | +0.0214 | [0.0178, 0.0257] | [0.0159, 0.0270] | improvement observed / improvement observed |
| H8 (new-item Recall@10) | Complete Journey, Instacart selections transferred | 1,971 | 0.0130 -> 0.0162 | +0.0032 | [0.0003, 0.0061] | [-0.0009, 0.0072] | improvement observed / no difference established |

### Earlier confirmatory run (PREREG v1.2, holdout)

`holdout` profile (Instacart holdout read once, Complete Journey refit on its `medium` data); H1-H4
primary, H5 secondary confirmatory with a 95 % interval only. Comparators: H1 personal purchase
frequency (repeat list), H2 and H5 flat popularity excluding history items, H3 personal purchase
frequency (full list), H4 population category ranking. Model lists: H1 `repeat_star` (R1), H2
`explore_star` (E1), H3 `final` (C0), H4 `category_ranking`, H5 `itemcf_flat` (E3).

<!-- results-table: holdout_v12 -->
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

Development results (`medium` profile; H6-H10 for R2, E2, R3 and the combinations C1-C3b) are
labelled **development** in `reports/results/RESULTS.md` (stage `development medium`) and are not
confirmatory; they are the basis on which C3 was selected before the holdout2 run.

## Folder structure

```
configs/     common.yaml (grids, bootstrap, category levels, lr_repeat.itemcf_feature), sample/medium/holdout/holdout2/holdout3/holdout4.yaml, usercf.yaml, r3.yaml
data/        SOURCE.md, MANIFEST.sha256 (processed Parquet untracked)
docs/        PREREG.md, ARCHITECTURE.md, MODULES.md, LIBRARIES.md, CITATIONS.md, CODE_AUDIT.md, figures/ (final-system diagram)
reports/     final/ (holdout JSONs, RUN.md, SHA256SUMS; PREREG v1.2), final2/ (holdout2 JSONs, RUN.md, SHA256SUMS, logs; PREREG v1.3),
             final3/ (holdout3 JSONs, RUN.md, SHA256SUMS, logs; PREREG v1.4 exploration policy),
             final4/ (holdout4 JSONs, RUN.md, SHA256SUMS, logs; PREREG v1.5 two-CF final system, v1.5.1 policy replication),
             results/ (tables T1-T7, figures F1-F8, RESULTS.md), posthoc/ (development case studies), eda/, data_check/, holdout_prep/;
             metrics/, figures/, tables/ (untracked run outputs)
scripts/     make_holdout.py, check_data.py, make_eda.py, make_results.py (+ results_sources.yaml), check_readme_results.py, audit_code.py,
             check_policy_dev.py, posthoc_*.py, make_basket_days.py
src/nbr/     package: config, data, io, baselines, repeat, explore, itemcf, combine, metrics,
             features, lr_repeat, usercf, usercf_eval, features_cycle, r3_eval, day_decay, policy, pipeline, __main__
tests/       pytest suite (synthetic data)
vendor/      ml_group9_e11b365 (unmodified preprocessing snapshot, release/term-project e11b365)
```

## Documents

| Document | Content |
|---|---|
| `docs/PREREG.md` | pre-registration v1.5.1 (question, model, selection, hypotheses, deviations v1.1-v1.5.1; v1.5 = the two-CF final system C4 tested on holdout4, v1.5.1 = the policy replication on C4) |
| `docs/ARCHITECTURE.md` | system structure (diagram), profiles, commands and files, selection procedure, leakage rules |
| `docs/MODULES.md` | module and combination catalogue with status, hypothesis basis and filtering-method classification |
| `docs/LIBRARIES.md` | third-party modules and functions used, with parameters and the related course module |
| `docs/CITATIONS.md` | data sources, vendored code, prior work, AI-tool disclosure (TBD) |
| `docs/CODE_AUDIT.md` | generated static audit (docstrings, library calls, attribution comments, observations) |
| `reports/final/RUN.md` | final run log (PREREG v1.2, holdout) |
| `reports/final2/RUN.md` | final run log (PREREG v1.3, holdout2: pre-checks, selections fixed before the eval read, commands, CJ reproducibility check) |
| `reports/final3/RUN.md` | run log of the exploration-policy test (PREREG v1.4, holdout3: pre-specified keys, pre-checks, selections fixed before the eval read, commands, CJ reproducibility check) |
| `reports/final4/RUN.md` | run log of the two-CF final run (PREREG v1.5 / v1.5.1, holdout4: pre-specified keys, pre-checks, selections fixed before the eval read, commands, CJ reproducibility check) |
| `reports/posthoc/policy_dev_medium.md` | development check of the policy code on medium against the W8b selections and the W15 numbers (not confirmatory) |
| `reports/results/RESULTS.md` | consolidated result tables and figures with source files and digests |
| `reports/eda/eda_summary.md` | exploratory data analysis of the development data (Korean) |

## Team and contributions

TBD (team members, roles and contribution shares to be filled in by the team).
