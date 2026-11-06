"""History-only user CF with labelled sparse vectors and bounded score blocks.

The CSR returned by user_vectors carries user_index/item_index attributes
(sorted pandas Index labels); pass that same matrix to the other functions.
Neighbour cosine uses normalized vectors, while scores use original weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from nbr.baselines import item_popularity, pad_list, popularity_list
from nbr.itemcf import _topn_with_ties
from nbr.repeat import LIST_COLUMNS, PAIR_KEY, rank_scores

NEIGHBOR_COLUMNS = ["user_id", "neighbor_id", "sim"]
SCORE_COLUMNS = PAIR_KEY + ["score"]


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def user_vectors(history: pd.DataFrame, weighting: str) -> tuple:
    """Return (labelled CSR U x I, sorted user Index, sorted item Index) of ``history`` with ``weighting`` "binary" or "log1p".

    scipy.sparse.csr_matrix((data, (rows, cols)), shape) stores only observed
    user/item weights. numpy.log1p(x) computes log(1+x) for basket counts.
    Duplicate rows in one basket count once, regardless of quantity.
    """
    if weighting not in ("binary", "log1p"):
        raise ValueError("weighting must be binary or log1p")
    events = history[["user_id", "basket_seq", "item_id"]].drop_duplicates()
    counts = events.groupby(PAIR_KEY, sort=True).size().rename("count").reset_index()
    users = pd.Index(sorted(history["user_id"].unique()), name="user_id")
    items = pd.Index(sorted(history["item_id"].unique()), name="item_id")
    weights = (np.ones(len(counts), dtype="float64") if weighting == "binary"
               else np.log1p(counts["count"].to_numpy(dtype="float64")))
    matrix = sparse.csr_matrix(
        (weights, (users.get_indexer(counts["user_id"]), items.get_indexer(counts["item_id"]))),
        shape=(len(users), len(items)),
    )
    matrix.sort_indices()
    matrix.user_index, matrix.item_index = users, items
    return matrix, users, items


def _labels(X: sparse.csr_matrix) -> tuple[pd.Index, pd.Index]:
    if not sparse.isspmatrix_csr(X) or not hasattr(X, "user_index") or not hasattr(X, "item_index"):
        raise ValueError("X must be the labelled CSR returned by user_vectors")
    if X.shape != (len(X.user_index), len(X.item_index)):
        raise ValueError("X labels do not match its shape")
    return X.user_index, X.item_index


def user_neighbors(X: sparse.csr_matrix, k: int, block_size: int = 256) -> pd.DataFrame:
    """Top-``k`` positive cosine neighbours of every row of ``X`` (labelled CSR of user_vectors), excluding self; ties use neighbour ID.

    scipy.sparse.diags(diagonals) builds a diagonal row-scaling matrix.
    CSR.multiply(other) multiplies elementwise; sum(axis=1) gives row norms.
    CSR.toarray() densifies only block_size x n_users similarities.
    The block/top-k structure and _topn_with_ties are adapted from nbr.itemcf;
    that helper uses numpy.partition(a, kth, axis=1) to select exact top-k ties.
    """
    _positive_integer(k, "k")
    _positive_integer(block_size, "block_size")
    users, _ = _labels(X)
    if not len(users):
        return pd.DataFrame(columns=NEIGHBOR_COLUMNS).astype({"sim": "float64"})
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    normalized = (sparse.diags(inverse) @ X).tocsr()
    transpose = normalized.T.tocsc()
    parts = []
    for start in range(0, len(users), block_size):
        stop = min(start + block_size, len(users))
        block = (normalized[start:stop] @ transpose).toarray()
        block[np.arange(stop - start), np.arange(start, stop)] = 0.0
        rows, cols = _topn_with_ties(block, k)
        parts.append(pd.DataFrame({
            "user_id": users.to_numpy()[rows + start],
            "neighbor_id": users.to_numpy()[cols], "sim": block[rows, cols],
        }))
    return pd.concat(parts, ignore_index=True).sort_values(
        ["user_id", "sim", "neighbor_id"], ascending=[True, False, True], kind="mergesort"
    )[NEIGHBOR_COLUMNS].reset_index(drop=True)


def _score_blocks(history, neighbors, X, block_size):
    """Yield unseen-item score frames, bounding intermediates to user blocks.

    scipy.sparse.csr_matrix((data, (rows, cols)), shape) encodes neighbour
    similarities; CSR.tocoo() exposes row/column coordinates of nonzeros.
    CSR.multiply(mask) removes purchased entries without a dense U x I matrix.
    """
    _positive_integer(block_size, "block_size")
    users, items = _labels(X)
    rows = users.get_indexer(neighbors["user_id"])
    cols = users.get_indexer(neighbors["neighbor_id"])
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError("neighbour IDs are absent from X")
    adjacency = sparse.csr_matrix(
        (neighbors["sim"].to_numpy(dtype="float64"), (rows, cols)),
        shape=(len(users), len(users)),
    )
    bought = history[PAIR_KEY].drop_duplicates()
    bought_rows, bought_cols = users.get_indexer(bought["user_id"]), items.get_indexer(bought["item_id"])
    if (bought_rows < 0).any() or (bought_cols < 0).any():
        raise ValueError("history IDs are absent from X")
    seen = sparse.csr_matrix((np.ones(len(bought)), (bought_rows, bought_cols)), shape=X.shape)
    for start in range(0, len(users), block_size):
        stop = min(start + block_size, len(users))
        scores = adjacency[start:stop] @ X
        scores = scores - scores.multiply(seen[start:stop])
        scores.eliminate_zeros()
        coordinates = scores.tocoo()
        positive = coordinates.data > 0
        yield pd.DataFrame({
            "user_id": users.to_numpy()[coordinates.row[positive] + start],
            "item_id": items.to_numpy()[coordinates.col[positive]],
            "score": coordinates.data[positive],
        }).sort_values(PAIR_KEY, kind="mergesort").reset_index(drop=True)


def ucf_scores(history: pd.DataFrame, neighbors: pd.DataFrame, X: sparse.csr_matrix,
               block_size: int = 256) -> pd.DataFrame:
    """Sum sim(u,v)*X[v,i] over the ``neighbors`` v of u for items i unseen in ``history``, ``block_size`` users at a time.

    Intermediates are blocked; this API materializes all returned candidates.
    ucf_list instead ranks each block immediately to keep only k per user.
    """
    parts = list(_score_blocks(history, neighbors, X, block_size))
    if not parts:
        return pd.DataFrame(columns=SCORE_COLUMNS).astype({"score": "float64"})
    return pd.concat(parts, ignore_index=True)[SCORE_COLUMNS]


def _list_from_vectors(history, neighbors, X, k, pop, block_size=256):
    """Shared list construction; tuning reuses fit vectors and neighbour prefixes."""
    _positive_integer(k, "k")
    parts = [rank_scores(scores, pop, k, "usercf")
             for scores in _score_blocks(history, neighbors, X, block_size)]
    lists = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=LIST_COLUMNS)
    users, _ = _labels(X)
    sizes = lists.groupby("user_id").size().reindex(users, fill_value=0)
    short = sizes[sizes < k].index
    if len(short):
        # k filler candidates suffice: at most m overlap the existing m entries.
        filler = popularity_list(history, short, k, pop, exclude_history=True)
        lists = pad_list(lists, filler, k)
    return lists[LIST_COLUMNS].sort_values(["user_id", "rank"], kind="mergesort").reset_index(drop=True)


def ucf_list(history: pd.DataFrame, k_neighbors: int, weighting: str, k: int = 10,
             pop: pd.Series | None = None, *, block_size: int = 256) -> pd.DataFrame:
    """Top-``k`` unseen user-CF items per user of ``history`` (``k_neighbors`` cosine neighbours on ``weighting``
    vectors, scored ``block_size`` users at a time), padded with history-excluding popularity.

    If fewer than k distinct unseen catalogue items exist, returns all of them.
    pop defaults to history basket popularity; target is never accepted or read.
    """
    _positive_integer(k, "k")
    X, _, _ = user_vectors(history, weighting)
    neighbors = user_neighbors(X, k_neighbors, block_size)
    if pop is None:
        pop = item_popularity(history)
    return _list_from_vectors(history, neighbors, X, k, pop, block_size)
