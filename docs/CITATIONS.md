# Citations, data sources and disclosures

## 1. Data

| Dataset | What we use | Public source | Terms of use |
|---|---|---|---|
| Instacart Online Grocery Shopping Dataset 2017 | `orders.csv`, `order_products__prior.csv`, `order_products__train.csv`, `products.csv`, `aisles.csv`, `departments.csv` (archive `instacart.zip`, read by the vendored preprocessing code) | Kaggle competition "Instacart Market Basket Analysis", https://www.kaggle.com/competitions/instacart-market-basket-analysis (the vendored README records the URL as https://www.kaggle.com/competitions/basket-analysis; the original Instacart release page is no longer online) | to be confirmed (Kaggle competition rules / Instacart dataset terms) |
| dunnhumby "The Complete Journey" (source files) | `transaction_data.csv`, `product.csv` and the other CSVs of the dunnhumby source-files archive (`archive.zip`); not the 84.51° "Complete Journey 2.0" R package | https://www.dunnhumby.com/source-files/ | to be confirmed (dunnhumby source-files licence) |

Processed copies, checksums and the holdout draw are documented in `data/SOURCE.md` and
`data/MANIFEST.sha256`; the ingredient-type mapping used for L3 comes from the separate project
`food_recipe_preprocessing_gu` (mapping version 2026-11-02.v2.1, see the vendored README).

## 2. Vendored code

| Component | Origin | Use in this repository |
|---|---|---|
| `vendor/ml_group9_e11b365/` | ML_group9 team repository, branch `release/term-project`, commit `e11b3652c5fc92aa5d22d4309a2c6368ed8a464d` (`vendor/ml_group9_e11b365/COMMIT`) | unmodified preprocessing snapshot: builds the processed Parquet splits and is imported by `scripts/make_holdout.py` for the holdout profile |
| `src/nbr/itemcf.py` | block structure adapted from `src/ml_group9/recommender.py` (`item_cosine_neighbors`) of the same snapshot | flat item-CF; the matrix here is user x item as specified in PREREG v1.2 (the original uses basket x item) |
| `src/nbr/usercf.py` | block/top-k structure adapted from `src/nbr/itemcf.py` (`_topn_with_ties`) | user-based CF |

Attribution comments are listed automatically in `docs/CODE_AUDIT.md`, section 8.

## 3. Prior work (reference list fixed by the team; nothing added)

Bibliographic details were checked against arXiv / publisher pages on 2026-11-06; items marked
"to be confirmed" could not be fully verified.

1. Li, M., Jullien, S., Ariannezhad, M., de Rijke, M. **A Next Basket Recommendation Reality
   Check.** ACM Transactions on Information Systems (TOIS), 2023. arXiv:2109.14233.
   https://arxiv.org/abs/2109.14233 — repeat/explore split of NBR metrics; motivates the separate
   repeat-component and new-item metrics used here.
2. Hu, H., He, X., Gao, J., Zhang, Z.-L. **Modeling Personalized Item Frequency Information for
   Next-basket Recommendation.** SIGIR 2020. arXiv:2006.00556. https://arxiv.org/abs/2006.00556 —
   TIFU-KNN: personalised item frequency with time decay plus a nearest-neighbour component; the
   decayed repeat score R1 is of this family.
3. Ariannezhad, M., Jullien, S., Li, M., Fang, M., Schelter, S., de Rijke, M. **ReCANet: A Repeat
   Consumption-Aware Neural Network for Next Basket Recommendation in Grocery Shopping.** SIGIR 2022
   (DOI to be confirmed; full text on the UvA DARE repository) — repeat-consumption modelling in
   grocery NBR.
4. Katz, O., Barkan, O., Koenigstein, N., Zabari, N. **Learning to Ride a Buy-Cycle: A
   Hyper-Convolutional Model for Next Basket Repurchase Recommendation.** RecSys 2022,
   https://doi.org/10.1145/3523227.3546763. Extended version: **Personalized Cadence Awareness for
   Next Basket Recommendation**, ACM Transactions on Recommender Systems 3(1), 2024 (author list of
   the extended version to be confirmed) — purchase cadence for repurchase recommendation; background
   for the R3 cycle features.
5. Romanov, A., Lashinin, O., Ananyeva, M., Kolesnikov, S. **Time-Aware Item Weighting for the Next
   Basket Recommendations.** RecSys 2023 (venue details to be confirmed).
   arXiv:2307.16297. https://arxiv.org/abs/2307.16297 — time-aware item weighting for repurchases.
6. Cao, Y., Ranjan, A., Subramaniam, S., Korpeoglu, E., Nag, K., Achan, K. **CASE: Cadence-Aware
   Set Encoding for Large-Scale Next Basket Repurchase Recommendation.** arXiv:2604.06718 (2026;
   the PDF header refers to SIGIR '26, venue to be confirmed). https://arxiv.org/abs/2604.06718 —
   per-item cadence learning for repurchase recommendation.

## 4. Software

numpy 1.26.4, pandas 2.2.3, scipy 1.13.1, scikit-learn 1.5.2, pyarrow 17.0.0, PyYAML 6.0.2,
joblib 1.4.2, matplotlib 3.9.2, pytest 8.3.3 (`pyproject.toml`). Their use is described in
`docs/LIBRARIES.md`.

## 5. AI-tool usage disclosure

TBD.
