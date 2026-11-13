# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Flat item-based collaborative filtering (E3, secondary variant).
          Block structure adapted from ML_group9 e11b365 src/ml_group9/recommender.py
          (item_cosine_neighbors); here the binary matrix is user x item as specified.
Inputs  : A history frame (columns user_id, item_id), the neighbour count, the list length k
          and a popularity Series for the tie rule.
Outputs : item_neighbors returns [item_id, neighbor_id, sim]; cf_scores returns
          [user_id, item_id, score]; cf_list returns [user_id, rank, item_id, source].
Usage   :
    from nbr.itemcf import cf_scores, item_neighbors
    neighbors = item_neighbors(history, n_neighbors)
    cf = cf_scores(history, neighbors)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy import sparse

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
PAIR_KEY = ["user_id", "item_id"]
LIST_COLUMNS = ["user_id", "rank", "item_id", "source"]
NEIGHBOR_COLUMNS = ["item_id", "neighbor_id", "sim"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _topn_with_ties(block: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise top-n column indices of ``block`` (values > 0), ties resolved by smaller column index.

    Uses numpy.partition(a, kth, axis=1) to find each row's n-th largest value
    without a full sort; columns strictly above that value are kept, and
    columns equal to it are added in ascending column order until n are chosen.
    Returns (row_indices, column_indices).
    """
    # Read the number of columns.
    _, cols = block.shape
    # Return every positive entry when the block has at most n columns.
    if cols <= n:
        row_index, col_index = np.nonzero(block > 0)
        return row_index, col_index
    # Find the n-th largest value of every row.
    threshold = np.partition(block, cols - n, axis=1)[:, cols - n]
    # Raise the threshold to the smallest positive float32 so that zeros never qualify.
    threshold = np.maximum(threshold, np.nextafter(np.float32(0), np.float32(1)))
    # Select the entries strictly above the threshold.
    above = block > threshold[:, None]
    row_above, col_above = np.nonzero(above)
    # Count how many more entries each row needs.
    need = n - above.sum(axis=1)
    # Find the positive entries equal to the threshold.
    tie_rows, tie_cols = np.nonzero((block == threshold[:, None]) & (block > 0))
    # Number the tied entries inside each row.
    # nonzero returns row-major order, so within a row the columns are ascending.
    position = np.arange(len(tie_rows)) - np.searchsorted(tie_rows, tie_rows, side="left")
    # Keep the first needed ties of every row.
    keep = position < need[tie_rows]
    # Join the entries above the threshold with the kept ties.
    row_index = np.concatenate([row_above, tie_rows[keep]])
    col_index = np.concatenate([col_above, tie_cols[keep]])
    return row_index, col_index


# ---------------------------------------------------------------------------
# item neighbours
# ---------------------------------------------------------------------------
def item_neighbors(history: pd.DataFrame, n_neighbors: int = 20, block_size: int = 256) -> pd.DataFrame:
    """Top-``n_neighbors`` cosine neighbours of every item of ``history`` from the binary user x item matrix.

    scipy.sparse.csr_matrix((data, (rows, cols)), shape) builds the sparse
    user x item matrix; cosine(i, j) = |U_i & U_j| / sqrt(|U_i| |U_j|) is
    computed block by block (``block_size`` items at a time) as a sparse
    product densified only for the block, bounding memory to
    block_size x n_items floats. Self-similarity is excluded and only
    neighbours with sim > 0 are kept. Output: [item_id, neighbor_id, sim],
    sorted by item_id, sim desc, neighbor_id asc.
    """
    # Reject a neighbour count below 1.
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
    # Keep the distinct (user, item) pairs and encode both IDs as sorted integer codes.
    pairs = history[PAIR_KEY].drop_duplicates()
    user_codes, _ = pd.factorize(pairs["user_id"], sort=True)
    item_codes, item_values = pd.factorize(pairs["item_id"], sort=True)
    n_items = len(item_values)
    # Build the binary sparse user x item matrix.
    matrix = sparse.csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (user_codes, item_codes)),
        shape=(int(user_codes.max()) + 1, n_items),
    )
    # Compute the norm of every item column.
    norms = np.sqrt(np.asarray(matrix.sum(axis=0)).ravel()).astype(np.float32)
    # Store the matrix in column-compressed form for the block products.
    matrix_csc = matrix.tocsc()

    # Create the output buffers.
    items_out, neighbors_out, sims_out = [], [], []
    for start in range(0, n_items, block_size):
        # Take the next block of items.
        stop = min(start + block_size, n_items)
        # Count the common users of the block items with all items.
        block = (matrix_csc[:, start:stop].T @ matrix).toarray().astype(np.float32)
        # Divide by the norms to get the cosine.
        block /= norms[start:stop, None] * norms[None, :]
        # Set the self-similarity to zero.
        block[np.arange(stop - start), np.arange(start, stop)] = 0.0
        # Select the top neighbours of every block item with the tie rule.
        row_index, col_index = _topn_with_ties(block, n_neighbors)
        # Store the item codes, the neighbour codes and the similarities of the block.
        items_out.append(row_index + start)
        neighbors_out.append(col_index)
        sims_out.append(block[row_index, col_index].astype("float64"))

    # Build the frame with the original item IDs.
    result = pd.DataFrame({
        "item_id": item_values[np.concatenate(items_out)],
        "neighbor_id": item_values[np.concatenate(neighbors_out)],
        "sim": np.concatenate(sims_out),
    })
    # Sort by item, similarity descending and neighbour ID ascending.
    result = result.sort_values(["item_id", "sim", "neighbor_id"], ascending=[True, False, True], kind="mergesort")
    return result[NEIGHBOR_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# scores and lists
# ---------------------------------------------------------------------------
def cf_scores(history: pd.DataFrame, neighbors: pd.DataFrame, chunk_users: int = 2000) -> pd.DataFrame:
    """score(u, i) = sum over the items j bought by u in ``history`` of sim(j, i), restricted to unbought i.

    Each bought item j contributes to its stored neighbours i (rows of
    ``neighbors`` with item_id = j, neighbor_id = i); candidates are items
    not in I(H_u). Users are processed in blocks of ``chunk_users`` (sorted
    user_id) to bound the intermediate join; the result does not depend on
    the block size. Output: [user_id, item_id, score(float64)], sorted.
    """
    # Keep the distinct (user, item) pairs sorted, and the users in that order.
    bought = history[PAIR_KEY].drop_duplicates().sort_values(PAIR_KEY, kind="mergesort")
    users = bought["user_id"].drop_duplicates().to_numpy()
    parts = []
    for start in range(0, len(users), max(1, chunk_users)):
        # Take the bought pairs of the next block of users.
        block = bought[bought["user_id"].isin(users[start : start + chunk_users])]
        # Spread every bought item to its stored neighbours.
        spread = block.merge(neighbors, on="item_id", how="inner")
        spread = spread.drop(columns="item_id").rename(columns={"neighbor_id": "item_id"})
        # Sum the similarities per (user, candidate item).
        scores = spread.groupby(PAIR_KEY, sort=True, as_index=False)["sim"].sum().rename(columns={"sim": "score"})
        # Drop the candidates the user already bought.
        scores = scores.merge(block, on=PAIR_KEY, how="left", indicator=True)
        parts.append(scores[scores["_merge"] == "left_only"].drop(columns="_merge"))
    # Stack the blocks, or build an empty score frame.
    if parts:
        scores = pd.concat(parts, ignore_index=True)
    else:
        scores = pd.DataFrame({"user_id": [], "item_id": [], "score": []})
    # Sort by user and item and store the scores as float64.
    scores = scores.sort_values(PAIR_KEY, kind="mergesort")
    scores["score"] = scores["score"].astype("float64")
    return scores[PAIR_KEY + ["score"]].reset_index(drop=True)


def cf_list(history: pd.DataFrame, neighbors: pd.DataFrame, k: int, pop: pd.Series) -> pd.DataFrame:
    """Top-k flat item-CF list per user of ``history`` from cf_scores(history, neighbors) (source = "itemcf");
    ties: score desc, pop desc, item_id asc."""
    # Reject a list length below 1.
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    # Compute the scores and attach the item popularity, with 0 for unknown items.
    scores = cf_scores(history, neighbors)
    scores["pop"] = scores["item_id"].map(pop).fillna(0).to_numpy(dtype="float64")
    # Order by user, score descending, popularity descending and item ID ascending.
    ordered = scores.sort_values(
        ["user_id", "score", "pop", "item_id"], ascending=[True, False, False, True], kind="mergesort"
    )
    # Number the rows of each user from 1 and keep ranks up to k.
    ordered["rank"] = ordered.groupby("user_id", sort=False).cumcount() + 1
    ordered = ordered[ordered["rank"] <= k]
    # Label the rows with the source "itemcf".
    ordered["source"] = "itemcf"
    return ordered[LIST_COLUMNS].reset_index(drop=True)
