# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : History-only user-based collaborative filtering (E2) with labelled sparse vectors
          and bounded score blocks.
          Neighbour cosine uses normalized vectors, while scores use original weights.
Inputs  : A history frame (columns user_id, basket_seq, item_id), a weighting ("binary" or
          "log1p"), the neighbour count, the list length k and an optional popularity Series.
          The CSR returned by user_vectors carries user_index/item_index attributes
          (sorted pandas Index labels); pass that same matrix to the other functions.
Outputs : user_vectors returns (labelled CSR, user Index, item Index); user_neighbors returns
          [user_id, neighbor_id, sim]; ucf_scores returns [user_id, item_id, score];
          ucf_list returns a list frame with the columns of nbr.repeat.LIST_COLUMNS.
Usage   :
    from nbr.usercf import ucf_list
    usercf_fit = ucf_list(history, k_star, weighting_star, K, pop, block_size=block_size)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy import sparse

from nbr.baselines import item_popularity, pad_list, popularity_list
from nbr.itemcf import _topn_with_ties
from nbr.repeat import LIST_COLUMNS, PAIR_KEY, rank_scores

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
NEIGHBOR_COLUMNS = ["user_id", "neighbor_id", "sim"]
SCORE_COLUMNS = PAIR_KEY + ["score"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _positive_integer(value: int, name: str) -> None:
    '''Raise ValueError unless ``value`` is a positive integer; ``name`` labels the error.'''
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


# ---------------------------------------------------------------------------
# user vectors
# ---------------------------------------------------------------------------
def user_vectors(history: pd.DataFrame, weighting: str) -> tuple:
    """Return (labelled CSR U x I, sorted user Index, sorted item Index) of ``history`` with ``weighting`` "binary" or "log1p".

    scipy.sparse.csr_matrix((data, (rows, cols)), shape) stores only observed
    user/item weights. numpy.log1p(x) computes log(1+x) for basket counts.
    Duplicate rows in one basket count once, regardless of quantity.
    """
    # Reject an unknown weighting.
    if weighting not in ("binary", "log1p"):
        raise ValueError("weighting must be binary or log1p")
    # Keep one row per (user, basket, item).
    events = history[["user_id", "basket_seq", "item_id"]].drop_duplicates()
    # Count the baskets of every (user, item) pair.
    counts = events.groupby(PAIR_KEY, sort=True).size().rename("count").reset_index()
    # Build the sorted user and item labels.
    users = pd.Index(sorted(history["user_id"].unique()), name="user_id")
    items = pd.Index(sorted(history["item_id"].unique()), name="item_id")
    # Use weight 1 for binary weighting and log(1 + count) for log1p.
    weights = (np.ones(len(counts), dtype="float64") if weighting == "binary"
               else np.log1p(counts["count"].to_numpy(dtype="float64")))
    # Build the sparse user x item matrix of the weights.
    matrix = sparse.csr_matrix(
        (weights, (users.get_indexer(counts["user_id"]), items.get_indexer(counts["item_id"]))),
        shape=(len(users), len(items)),
    )
    # Sort the column indices of each row and attach the labels to the matrix.
    matrix.sort_indices()
    matrix.user_index, matrix.item_index = users, items
    return matrix, users, items


def _labels(X: sparse.csr_matrix) -> tuple[pd.Index, pd.Index]:
    '''Return the (user Index, item Index) labels of ``X``; reject a matrix without them.'''
    # Require a CSR matrix that carries both label attributes.
    if not sparse.isspmatrix_csr(X) or not hasattr(X, "user_index") or not hasattr(X, "item_index"):
        raise ValueError("X must be the labelled CSR returned by user_vectors")
    # Require the label lengths to match the matrix shape.
    if X.shape != (len(X.user_index), len(X.item_index)):
        raise ValueError("X labels do not match its shape")
    return X.user_index, X.item_index


# ---------------------------------------------------------------------------
# user neighbours
# ---------------------------------------------------------------------------
def user_neighbors(X: sparse.csr_matrix, k: int, block_size: int = 256) -> pd.DataFrame:
    """Top-``k`` positive cosine neighbours of every row of ``X`` (labelled CSR of user_vectors), excluding self; ties use neighbour ID.

    scipy.sparse.diags(diagonals) builds a diagonal row-scaling matrix.
    CSR.multiply(other) multiplies elementwise; sum(axis=1) gives row norms.
    CSR.toarray() densifies only block_size x n_users similarities.
    The block/top-k structure and _topn_with_ties are adapted from nbr.itemcf;
    that helper uses numpy.partition(a, kth, axis=1) to select exact top-k ties.
    """
    # Validate the neighbour count and the block size.
    _positive_integer(k, "k")
    _positive_integer(block_size, "block_size")
    # Read the user labels of the matrix.
    users, _ = _labels(X)
    # Return an empty neighbour frame when there is no user.
    if not len(users):
        return pd.DataFrame(columns=NEIGHBOR_COLUMNS).astype({"sim": "float64"})
    # Compute the Euclidean norm of every row.
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    # Invert the norms, leaving zero for empty rows.
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    # Scale every row to unit length.
    normalized = (sparse.diags(inverse) @ X).tocsr()
    # Store the transpose in column-compressed form for the block products.
    transpose = normalized.T.tocsc()
    parts = []
    for start in range(0, len(users), block_size):
        # Take the next block of users.
        stop = min(start + block_size, len(users))
        # Compute the dense cosine block of these users against all users.
        block = (normalized[start:stop] @ transpose).toarray()
        # Set the self-similarity to zero.
        block[np.arange(stop - start), np.arange(start, stop)] = 0.0
        # Select the top-k positive neighbours of every row with the tie rule.
        rows, cols = _topn_with_ties(block, k)
        # Store the neighbour rows of the block with their user labels.
        parts.append(pd.DataFrame({
            "user_id": users.to_numpy()[rows + start],
            "neighbor_id": users.to_numpy()[cols], "sim": block[rows, cols],
        }))
    # Stack the blocks and sort by user, similarity descending and neighbour ID ascending.
    return pd.concat(parts, ignore_index=True).sort_values(
        ["user_id", "sim", "neighbor_id"], ascending=[True, False, True], kind="mergesort"
    )[NEIGHBOR_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------
def _score_blocks(history, neighbors, X, block_size):
    """Yield unseen-item score frames, bounding intermediates to user blocks.

    scipy.sparse.csr_matrix((data, (rows, cols)), shape) encodes neighbour
    similarities; CSR.tocoo() exposes row/column coordinates of nonzeros.
    CSR.multiply(mask) removes purchased entries without a dense U x I matrix.
    """
    # Validate the block size and read the labels.
    _positive_integer(block_size, "block_size")
    users, items = _labels(X)
    # Map the neighbour frame to row and column positions.
    rows = users.get_indexer(neighbors["user_id"])
    cols = users.get_indexer(neighbors["neighbor_id"])
    # Reject neighbour IDs that are not labels of the matrix.
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError("neighbour IDs are absent from X")
    # Build the sparse user x user matrix of neighbour similarities.
    adjacency = sparse.csr_matrix(
        (neighbors["sim"].to_numpy(dtype="float64"), (rows, cols)),
        shape=(len(users), len(users)),
    )
    # Map the distinct (user, item) pairs of the history to matrix positions.
    bought = history[PAIR_KEY].drop_duplicates()
    bought_rows, bought_cols = users.get_indexer(bought["user_id"]), items.get_indexer(bought["item_id"])
    # Reject history IDs that are not labels of the matrix.
    if (bought_rows < 0).any() or (bought_cols < 0).any():
        raise ValueError("history IDs are absent from X")
    # Build the sparse mask of bought (user, item) pairs.
    seen = sparse.csr_matrix((np.ones(len(bought)), (bought_rows, bought_cols)), shape=X.shape)
    for start in range(0, len(users), block_size):
        # Take the next block of users.
        stop = min(start + block_size, len(users))
        # Sum the weights of the neighbours times their similarities for the block.
        scores = adjacency[start:stop] @ X
        # Remove the scores of items the user already bought.
        scores = scores - scores.multiply(seen[start:stop])
        scores.eliminate_zeros()
        # Read the remaining nonzero scores as coordinates and mark the positive ones.
        coordinates = scores.tocoo()
        positive = coordinates.data > 0
        # Yield the positive scores of the block as a frame sorted by user and item.
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
    # Collect the score frames of all blocks.
    parts = list(_score_blocks(history, neighbors, X, block_size))
    # Return an empty score frame when there is no block.
    if not parts:
        return pd.DataFrame(columns=SCORE_COLUMNS).astype({"score": "float64"})
    # Stack the blocks into one frame.
    return pd.concat(parts, ignore_index=True)[SCORE_COLUMNS]


# ---------------------------------------------------------------------------
# recommendation lists
# ---------------------------------------------------------------------------
def _list_from_vectors(history, neighbors, X, k, pop, block_size=256):
    """Shared list construction; tuning reuses fit vectors and neighbour prefixes."""
    # Validate the list length.
    _positive_integer(k, "k")
    # Rank the scores of each block into top-k lists with the source "usercf".
    parts = [rank_scores(scores, pop, k, "usercf")
             for scores in _score_blocks(history, neighbors, X, block_size)]
    # Stack the block lists, or start from an empty list frame.
    lists = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=LIST_COLUMNS)
    # Count the list rows of every user of the matrix.
    users, _ = _labels(X)
    sizes = lists.groupby("user_id").size().reindex(users, fill_value=0)
    # Find the users with fewer than k rows.
    short = sizes[sizes < k].index
    # Pad the lists of these users with popularity that excludes their history.
    if len(short):
        # k filler candidates suffice: at most m overlap the existing m entries.
        filler = popularity_list(history, short, k, pop, exclude_history=True)
        lists = pad_list(lists, filler, k)
    # Return the list columns sorted by user and rank.
    return lists[LIST_COLUMNS].sort_values(["user_id", "rank"], kind="mergesort").reset_index(drop=True)


def ucf_list(history: pd.DataFrame, k_neighbors: int, weighting: str, k: int = 10,
             pop: pd.Series | None = None, *, block_size: int = 256) -> pd.DataFrame:
    """Top-``k`` unseen user-CF items per user of ``history`` (``k_neighbors`` cosine neighbours on ``weighting``
    vectors, scored ``block_size`` users at a time), padded with history-excluding popularity.

    If fewer than k distinct unseen catalogue items exist, returns all of them.
    pop defaults to history basket popularity; target is never accepted or read.
    """
    # Validate the list length.
    _positive_integer(k, "k")
    # Build the user vectors and the neighbour frame.
    X, _, _ = user_vectors(history, weighting)
    neighbors = user_neighbors(X, k_neighbors, block_size)
    # Default the popularity to the item popularity of the history.
    if pop is None:
        pop = item_popularity(history)
    # Build the padded lists from the vectors and the neighbours.
    return _list_from_vectors(history, neighbors, X, k, pop, block_size)
