# Third-party modules and functions used in the code

Companion to the assignment requirement "explain the modules and functions (with their
parameters) that were not covered in class". One row per third-party call found by the static
audit (`scripts/audit_code.py`, output `docs/CODE_AUDIT.md`, sections 5 and 6; the audit lists
every call site with file and line). The column **Related course module** names the course module
a function corresponds to (`—` = no direct course module). Pandas' basic table operations are
summarised in one section at the end.

Pinned versions (`pyproject.toml`, same as the preprocessing snapshot): numpy 1.26.4, pandas 2.2.3,
scipy 1.13.1, scikit-learn 1.5.2, pyarrow 17.0.0, PyYAML 6.0.2, joblib 1.4.2, matplotlib 3.9.2;
pytest 8.3.3 for the tests. Files are named relative to the repository root; every audited file
(`src/nbr/*.py`, `scripts/*.py`, including `features_cycle.py`, `r3_eval.py`, `make_results.py`
and `make_eda.py`) is in this repository.

## 1. scikit-learn

| Function / class | What we use it for | Parameters used and their meaning | Files | Related course module |
|---|---|---|---|---|
| `sklearn.linear_model.LogisticRegression(penalty, C, solver, max_iter, random_state)` | the R2/R3 repeat ranker: binary classifier "pair is bought again" over standardised features; its linear decision function is the ranking score | `penalty="l2"` (ridge penalty on the coefficients), `C` (inverse regularisation strength; grid {0.001, ..., 10} selected on validation users), `solver="lbfgs"` (deterministic quasi-Newton optimiser), `max_iter` (iteration cap; 1000, r3: 2000), `random_state=seed` (unused by lbfgs; recorded for completeness) | `src/nbr/lr_repeat.py` (`_fit`), used by `pipeline.py`, `r3_eval.py` | module 2-2 logistic regression; `penalty="l2"` / `C`: module 5-1 regularization; `solver="lbfgs"`: module 5-2 gradient descent (related) |
| `LogisticRegression.fit(X, y)` | fit on the standardised training rows; `y` is 0/1 | positional `X` (n x p float array), `y` (labels) | `src/nbr/lr_repeat.py` | module 2-2 logistic regression |
| `LogisticRegression.decision_function(X)` | raw linear score `w.x + b` per pair (higher = more likely repurchase); used instead of probabilities because only the ranking matters | positional `X` | `src/nbr/lr_repeat.py` (`score_frame`) | module 2-2 logistic regression |
| `LogisticRegression.coef_`, `.intercept_`, `.n_iter_` (attributes) | coefficient table in the tuning JSON and the R3 coefficient tables; convergence check | - | `src/nbr/lr_repeat.py`, `r3_eval.py` | module 2-2 logistic regression |
| `sklearn.preprocessing.StandardScaler()` | per-column standardisation (zero mean, unit variance) so that the L2 penalty treats every feature on the same scale and coefficients are comparable | no arguments (defaults `with_mean=True`, `with_std=True`) | `src/nbr/lr_repeat.py` | — |
| `StandardScaler.fit(X)`, `.transform(X)` | learn mean/std on the training rows, apply to training, validation and eval rows | positional `X` | `src/nbr/lr_repeat.py` | — |
| `StandardScaler.mean_`, `.scale_` (attributes) | convert standardised coefficients to per-unit log-odds effects (`coef / scale_`) | - | `r3_eval.py` (`coefficient_table`, `overdue_table`) | — |

## 2. SciPy (sparse matrices)

| Function / class | What we use it for | Parameters used and their meaning | Files | Related course module |
|---|---|---|---|---|
| `scipy.sparse.csr_matrix((data, (rows, cols)), shape=)` | build the user x item purchase matrix (binary or log1p weights), the item x item co-occurrence products, the user x user neighbour-similarity matrix and the "already bought" mask without a dense matrix | COO-style triplets `data`, `rows`, `cols`; `shape=(n_users, n_items)` fixes the matrix size even when the last rows/columns are empty | `src/nbr/itemcf.py` (`item_neighbors`), `src/nbr/usercf.py` (`user_vectors`, `_score_blocks`) | module 6 recommendation (cosine similarity / collaborative filtering) |
| `csr_matrix.T`, `@` (matrix product), slicing `[:, a:b]`, `[a:b]` | cosine numerators block by block: `(M[:, block].T @ M)` for items, `(normalized[block] @ normalized.T)` for users, `adjacency[block] @ X` for user-CF scores | - | `src/nbr/itemcf.py`, `src/nbr/usercf.py` | module 6 recommendation (cosine similarity / collaborative filtering) |
| `csr_matrix.tocsc()`, `.tocsr()`, `.tocoo()` | format conversions: CSC for fast column slicing, CSR for row slicing, COO to read `(row, col, data)` triplets of the non-zeros | - | `src/nbr/itemcf.py`, `src/nbr/usercf.py` | — |
| `csr_matrix.toarray()` | densify one block (block_size x n columns) of similarities for the top-k selection | - | `src/nbr/itemcf.py`, `src/nbr/usercf.py` | — |
| `csr_matrix.sum(axis=)` | column sums (number of users per item) and row sums of squares (vector norms) | `axis=0` columns, `axis=1` rows | `src/nbr/itemcf.py`, `src/nbr/usercf.py` | — |
| `csr_matrix.multiply(other)` | element-wise product: `X.multiply(X)` for squared norms; `scores.multiply(seen)` to remove items the user already bought | positional `other` (same shape) | `src/nbr/usercf.py` | — |
| `csr_matrix.sort_indices()`, `.eliminate_zeros()` | canonical CSR layout; drop explicit zeros after subtracting the mask | - | `src/nbr/usercf.py` | — |
| `scipy.sparse.diags(values)` | diagonal matrix of inverse norms, so `diags(1/norm) @ X` row-normalises the user vectors for cosine similarity | positional `values` (1-D array) | `src/nbr/usercf.py` (`user_neighbors`) | module 6 recommendation (cosine similarity / collaborative filtering) |
| `scipy.sparse.isspmatrix_csr(X)` | input check that the labelled matrix is CSR | positional `X` | `src/nbr/usercf.py` (`_labels`) | — |

## 3. NumPy

| Function | What we use it for | Parameters used and their meaning | Files | Related course module |
|---|---|---|---|---|
| `numpy.power(base, exponent)` | decay weights `gamma ** age` for the repeat score, the LR decay features and the category profile | positional | `src/nbr/repeat.py`, `features.py`, `explore.py` | — |
| `numpy.log1p(x)`, `numpy.expm1(x)` | `log(1 + pop)` and `log(1 + n_baskets)` features; log1p weighting of user vectors; `expm1` recovers pop from `log_pop` for tie-breaking | positional | `src/nbr/features.py`, `usercf.py`, `lr_repeat.py` | — |
| `numpy.log2(x)` | NDCG position discount `1 / log2(rank + 1)` | positional | `src/nbr/metrics.py` | — |
| `numpy.minimum(a, b)`, `numpy.maximum(a, b)` | caps (`overdue`, `cadence_ratio` at 10; take counts); raise the top-k threshold above zero | positional | `src/nbr/features.py`, `features_cycle.py`, `explore.py`, `itemcf.py` | — |
| `numpy.partition(a, kth, axis=1)` | per row, the value at position `kth` after a partial sort: the n-th largest similarity without sorting the whole row (exact top-k with ties) | `kth = cols - n`, `axis=1` (row-wise) | `src/nbr/itemcf.py` (`_topn_with_ties`, reused by `usercf.py`) | — |
| `numpy.nextafter(x, y)` | the smallest float32 above 0, used as a strictly positive threshold floor | positional (`np.float32(0)`, `np.float32(1)`) | `src/nbr/itemcf.py` | — |
| `numpy.nonzero(mask)` | row/column indices of `True` entries, in row-major order (so within a row the columns are ascending) | positional | `src/nbr/itemcf.py` | — |
| `numpy.searchsorted(a, v, side=)` | bin index for right-open bins (`side="right"`, minus one); first position of each row id in a sorted array (`side="left"`) | `side="right"` / `"left"` chooses which side of equal values the insertion point goes to | `features_cycle.py` (`bin_index`), `src/nbr/itemcf.py` | — |
| `numpy.repeat(a, counts)`, `numpy.cumsum(a)`, `numpy.arange(n)` | vectorised expansion of "gaps" of unbought within-category ranks into candidate rows; index vectors, rank numbers, bar positions | positional | `src/nbr/explore.py` (`_unbought_top`), `metrics.py`, `itemcf.py`, `usercf_eval.py`, `r3_eval.py` | — |
| `numpy.random.default_rng(seed)` | the modern random Generator; seeded so that the bootstrap and the LR user split are reproducible | positional `seed` (42) | `src/nbr/metrics.py`, `lr_repeat.py` | module 7 inferential statistics (bootstrap confidence intervals) |
| `Generator.integers(low, high, size=)` | bootstrap resampling: `size=(chunk, n_users)` user indices drawn with replacement | `low=0`, `high=n_users`, `size` (shape of the draw) | `src/nbr/metrics.py` (`paired_bootstrap`) | module 7 inferential statistics (bootstrap confidence intervals) |
| `Generator.permutation(n)` | random order of users for the 80/20 train/validation split | positional `n` | `src/nbr/lr_repeat.py` (`split_users`) | — |
| `numpy.percentile(a, q)` | percentile bootstrap CI bounds (`q` = 2.5/97.5 for 95 %, 0.3125/99.6875 for 99.375 %) | positional `a`, `q` (list of two percentiles) | `src/nbr/metrics.py` | module 7 inferential statistics (bootstrap confidence intervals) |
| `numpy.divide(a, b, out=, where=)` | safe `1 / norm` with zeros where the norm is 0 (users without purchases) | `out=np.zeros_like(norms)` (result buffer, default value 0), `where=norms > 0` (compute only there) | `src/nbr/usercf.py` | module 6 recommendation (cosine similarity / collaborative filtering) |
| `numpy.sqrt(x)` | vector norms for cosine similarity | positional | `src/nbr/itemcf.py`, `usercf.py` | module 6 recommendation (cosine similarity / collaborative filtering) |
| `numpy.asarray(x, dtype=)`, `numpy.array(x)`, `numpy.ones(n, dtype=)`, `numpy.empty(n, dtype=)`, `numpy.zeros_like(x)`, `numpy.float32` | array construction and dtype control (`float32` for the dense similarity blocks to halve memory; `float64` everywhere else) | `dtype="float64"` / `np.float32` | `src/nbr/itemcf.py`, `usercf.py`, `metrics.py`, `lr_repeat.py`, `features_cycle.py` | — |
| `numpy.concatenate(parts)`, `numpy.sort(a)`, `numpy.array_equal(a, b)`, `numpy.isfinite(x)` | join block results; sorted user arrays; check that R2 and R3 training rows are identical; input validation | positional | `src/nbr/itemcf.py`, `lr_repeat.py`, `r3_eval.py`, `metrics.py`, `features_cycle.py` | — |
| `ndarray.astype`, `.sum(axis=)`, `.mean(axis=)`, `.ravel()`, `.tolist()`, `numpy.generic.item()` | dtype casts, block sums, bootstrap means per replicate, flattening sparse sums, JSON serialisation of numpy scalars and arrays | `axis=1` row-wise | `src/nbr/itemcf.py`, `usercf.py`, `metrics.py`, `io.py` | — |

## 4. joblib, PyYAML, pyarrow

| Function | What we use it for | Parameters used and their meaning | Files | Related course module |
|---|---|---|---|---|
| `joblib.dump(obj, path)`, `joblib.load(path)` | persist the fitted scaler + logistic model (and R3 bundle) between `tune` and `evaluate`; pickle-based, numpy-aware; the file's SHA-256 is recorded in the JSON | positional | `src/nbr/pipeline.py` (`save_lr_repeat`, `load_lr_repeat`), `r3_eval.py` | — |
| `yaml.safe_load(stream)` | read `configs/*.yaml` into dicts; `safe_load` constructs only plain YAML types (no arbitrary Python objects) | positional file handle | `src/nbr/config.py`, `usercf_eval.py`, `r3_eval.py`, `scripts/make_holdout.py` | — |
| pyarrow (not called directly) | engine behind `pandas.read_parquet` / `DataFrame.to_parquet` for the processed Parquet files | - | via `src/nbr/data.py`, `scripts/*.py` | — |

## 5. matplotlib (R3 screening figure, `src/nbr/r3_eval.py`)

| Function | What we use it for | Parameters used and their meaning | Files | Related course module |
|---|---|---|---|---|
| `matplotlib.use("Agg")` | select the file-only backend before importing pyplot (no display needed) | backend name | `r3_eval.py` (`plot_overdue`) | — |
| `matplotlib.pyplot.subplots(figsize=, dpi=)` | one figure with one axes | `figsize=(7.5, 4.2)` inches, `dpi=150` | `r3_eval.py` | — |
| `Axes.bar(x, height, width=, color=, edgecolor=, zorder=)` | log-odds effect per overdue bin | `width=0.62`, per-bar `color` list (grey for the once-bought bin), `edgecolor="none"`, `zorder=3` (draw above the grid) | `r3_eval.py` | — |
| `Axes.axhline(y, color=, linewidth=, zorder=)`, `Axes.text(x, y, s, ha=, va=, fontsize=, color=)` | zero line; "ref." label on the reference bin | `ha`/`va` horizontal/vertical alignment | `r3_eval.py` | — |
| `Axes.set_xticks`, `set_xticklabels(labels, fontsize=)`, `set_xlabel(s, fontsize=, color=)`, `set_ylabel`, `set_title(s, fontsize=, color=, loc=)` | axis labels in English; `loc="left"` left-aligned title | - | `r3_eval.py` | — |
| `Axes.yaxis.grid(True, color=, linewidth=, zorder=)`, `Axes.set_axisbelow(True)`, `Axes.spines[side].set_visible(False)` / `.set_color(c)`, `Axes.tick_params(axis=, colors=, labelsize=)` | recessive grid and axes styling | `axis="both"` | `r3_eval.py` | — |
| `Figure.tight_layout()`, `Figure.savefig(path)`, `matplotlib.pyplot.close(fig)` | fit labels inside the canvas; write the PNG; release the figure | - | `r3_eval.py` | — |

### 5.1 Report scripts (`scripts/make_results.py`, `scripts/make_eda.py`)

Exact call sites: `docs/CODE_AUDIT.md`, section 5 (matplotlib, numpy, pandas, yaml rows of the two scripts).

| Function | What we use it for | Parameters used and their meaning | Files | Related course module |
|---|---|---|---|---|
| `matplotlib.pyplot.subplots(figsize=, dpi=, sharey=, squeeze=, layout=)` | one figure with one panel per dataset | `sharey` (one y scale for all panels), `squeeze=False` (always a 2-D axes array), `layout` (automatic spacing) | `scripts/make_results.py`, `scripts/make_eda.py` | — |
| `Axes.bar(x, height, width=, color=, edgecolor=, hatch=, label=, linewidth=, zorder=)`, `Axes.step(x, y, where=, color=, linestyle=, linewidth=, label=)`, `Axes.errorbar`, `Axes.plot`, `Axes.axvline`, `Axes.axhline`, `Axes.text(x, y, s, ha=, va=, fontsize=, color=)` | forest plots (point estimate + CI segments), bar charts of module means, step curves of the r grid and of the history-length distribution | `hatch` (pattern that marks the stage without relying on colour), `where` (step position), `zorder` (draw order) | same | — |
| `Axes.set(title=, xlabel=, ylabel=, xscale=, ylim=)`, `Axes.set_xticks` / `set_xticklabels` / `set_yticks` / `set_yticklabels`, `Axes.legend(loc=, frameon=)`, `Axes.grid(axis=, color=, linewidth=)`, `Axes.set_axisbelow`, `Axes.spines[...].set_visible`, `Axes.xaxis/yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter / ScalarFormatter)` | axis labels, English tick labels, percentages on the EDA axes, recessive grid and spines | `xscale="log"` (log axis), `frameon=False` (legend without a box) | same | — |
| `Figure.suptitle(t, x=, ha=, fontsize=, color=)`, `Figure.text(...)`, `Figure.savefig(path, dpi=, bbox_inches=)`, `Figure.get_figwidth` / `get_figheight`, `matplotlib.pyplot.close(fig)`, `matplotlib.pyplot.rcParams.update(dict)`, `matplotlib.use("Agg")` | left-aligned titles, footers naming the source files, PNG export, global font/colour defaults, file-only backend | `dpi=300`, `bbox_inches="tight"` (crop to the drawn content) | same | — |
| `pandas.read_csv(path)`, `DataFrame.to_csv(path, index=, lineterminator=, encoding=)`, `DataFrame.iterrows()`, `DataFrame.itertuples(index=, name=)`, `pandas.io.common.BytesIO`, `pandas.cut(x, bins, labels=, right=)`, `pandas.isna`, `pandas.read_parquet(path, columns=)` | every figure is drawn from the CSV written just before; row iteration for the Markdown tables; history-length bins of the EDA; reading a `git show` byte stream | `lineterminator="
"` (LF on every platform), `right=False` (right-open bins), `columns` (read only the needed Parquet columns) | same | — |
| `numpy.atleast_1d`, `numpy.where`, `numpy.array`; `yaml.safe_load(stream)` | axes arrays with a single panel, conditional note column; reading `scripts/results_sources.yaml` | - | `scripts/make_results.py` | — |

## 6. pandas (table operations)

The code is written with vectorised pandas operations instead of per-user Python loops; the
metric functions (`per_user_recall`, `per_user_ndcg`, `per_user_category_hit`) are the exception.
Top-level constructors and functions used: `pandas.DataFrame`, `pandas.Series(values, index=,
dtype=, name=)`, `pandas.Index(values, name=)`, `pandas.MultiIndex.from_frame(frame)`,
`pandas.concat(parts, ignore_index=True)`, `pandas.factorize(values, sort=True)`,
`pandas.read_parquet(path, columns=)`, `pandas.unique(values)`.

| Operation | What we use it for | Parameters used | Related course module |
|---|---|---|---|
| `groupby(keys, sort=, as_index=)` with `.size()`, `.sum()`, `.min()`, `.max()`, `.mean()`, `.median()`, `.agg(name=(column, func))`, `.transform(func)`, `.cumcount()`, `.diff()`, `.shift()`, `.rank(ascending=, method="first")`, `.first()` | basket counts per item (`pop`), per-user aggregates, basket ages (rank of `basket_seq`), rank numbering within a user (`cumcount() + 1`), purchase gaps (`diff`), previous bought order (`shift`), named aggregations of the LR features | `sort=True/False` (sorted group keys), `as_index=False` (keys as columns), `method="first"` (ties by order of appearance) | — |
| `merge(right, on=, how=, indicator=)` | joins; `how="cross"` for user x category / user x top-M grids; `how="left"` + `indicator=True` then keeping `_merge == "left_only"` as an anti-join (remove the user's history items) | `on` (key columns), `how` in {`inner`, `left`, `cross`}, `indicator` adds the `_merge` column | — |
| `drop_duplicates(subset=)`, `sort_values(by, ascending=, kind="mergesort")`, `reset_index(drop=)`, `set_index`, `sort_index` | de-duplicate (basket, item) events; deterministic ordering (mergesort is stable, so the tie rule is reproducible); clean integer indexes | `kind="mergesort"` (stable sort), `drop=True` (discard the old index) | — |
| `map(series_or_dict)`, `fillna(value)`, `reindex(index, fill_value=)`, `isin(values)`, `Index.get_indexer(labels)`, `Index.isin`, `Index.intersection`, `MultiIndex.isin` | look-ups of pop / medians / neighbour ids; membership of (user, item) pairs (repeat vs new target rows, LR labels); label to integer position for sparse matrices | `fill_value=0` | — |
| `to_numpy(dtype=)`, `astype(dtype)`, `assign(column=value)`, `rename(columns=)`, `rename_axis(name)`, `drop(columns=)`, `head(n)`, `copy()`, `nunique()`, `unique()`, `value_counts()`, `quantile(q)`, `equals(other)`, `eq`, `notna`, `isna`, `any`, `join(other, on=)` | array extraction, dtype control (`float64`, `str`), column bookkeeping, counts in the JSON summaries, data checks | `dtype="float64"` | — |
| `to_csv(path, index=False)`, `to_dict(orient="records")`, `to_parquet(path, index=False)` | R3 coefficient tables, JSON-friendly records, holdout item table | `index=False`, `orient="records"` | — |

## 7. Standard-library modules (for completeness; not third-party)

`argparse` (CLI), `pathlib` (paths), `json` (result files), `hashlib.sha256` (artifact and user-set
digests), `subprocess` (`git rev-parse HEAD` for provenance), `math.isclose` (tie tolerance 1e-12),
`time.perf_counter` (timings), `datetime` (UTC timestamps), `ctypes` (`psapi.GetProcessMemoryInfo`
for the peak working set on Windows), `resource` (POSIX peak RSS in `make_holdout.py` and
`r3_eval.py`), `zipfile`, `platform`, `ast` and `re` (`scripts/audit_code.py`).
