# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-07
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Item-CF similarity feature of the logistic-regression repeat ranker with the item-CF
          similarity feature (R2') (PREREG v1.5, item 2b and item 4).
          icf(u, i) = log(1 + sum over j in I(H_u) of sim(j, i)) for every (user, item) pair
          (u, i) of a history, where sim(j, i) is the cosine similarity stored in j's
          top-``n_neighbors`` neighbour list of nbr.itemcf.item_neighbors(history, n_neighbors)
          (binary user x item matrix of the same history; self excluded; only neighbours with
          sim > 0 are stored). icf = 0 when i is in no neighbour list of the user's other
          items; a single-item user has icf = 0. The neighbour count is fixed at 20
          (configs/common.yaml ``explore.itemcf.neighbors``); no variant of the feature is
          selected.
          The sum is a sparse product (user x item) @ (item x item) restricted to the purchased
          positions, computed in blocks of ``block_size`` users; the result does not depend on
          the block size.
Inputs  : A history DataFrame (columns user_id, item_id), the neighbour count, the block size
          and optionally a precomputed neighbour frame ([item_id, neighbor_id, sim]).
          No file and no target is read; the function only uses the frame it receives.
Outputs : A DataFrame [user_id, item_id, icf] with one row per (user, item) pair of the
          history, sorted by user_id, item_id; icf is float64.
Usage   :
    from nbr.features_icf import icf_feature
    icf = icf_feature(history, n_neighbors, neighbors=neighbors)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy import sparse

from nbr.itemcf import item_neighbors
from nbr.repeat import PAIR_KEY

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
ICF_COLUMN = "icf"
COLUMNS = PAIR_KEY + [ICF_COLUMN]
DEFAULT_NEIGHBORS = 20


# ---------------------------------------------------------------------------
# item-CF similarity feature
# ---------------------------------------------------------------------------
def icf_feature(
    history: pd.DataFrame,
    n_neighbors: int = DEFAULT_NEIGHBORS,
    block_size: int = 2000,
    neighbors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """[user_id, item_id, icf] for every (user, item) pair of ``history`` (sorted by user_id, item_id; icf float64).

    ``neighbors`` may pass a precomputed item_neighbors(history, n_neighbors) frame ([item_id, neighbor_id, sim])
    so that the eval-history neighbours are shared with the flat item-CF list; otherwise the neighbours are
    computed here with ``n_neighbors``. Rows of ``neighbors`` whose item_id or neighbor_id is not an item of
    ``history`` are ignored. Users are processed in blocks of ``block_size`` (sorted user_id).
    """
    # Reject a neighbour count or a block size below 1.
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    # Keep one row per (user, item) pair, sorted by user and item with a stable sort.
    pairs = history[PAIR_KEY].drop_duplicates().sort_values(PAIR_KEY, kind="mergesort").reset_index(drop=True)
    # Compute the item neighbours when no precomputed frame was passed.
    if neighbors is None:
        neighbors = item_neighbors(history, n_neighbors)
    # Number the users and the items in sorted order.
    user_codes, user_values = pd.factorize(pairs["user_id"], sort=True)
    item_codes, item_values = pd.factorize(pairs["item_id"], sort=True)
    n_users, n_items = len(user_values), len(item_values)
    user_codes = np.asarray(user_codes, dtype="int64")
    item_codes = np.asarray(item_codes, dtype="int64")

    # Look up the item number of both ends of every neighbour row (-1 for unknown items).
    items = pd.Index(item_values)
    source = items.get_indexer(neighbors["item_id"])
    target = items.get_indexer(neighbors["neighbor_id"])
    # Keep the neighbour rows whose two items both occur in the history.
    keep = (source >= 0) & (target >= 0)
    # Build the sparse item x item similarity matrix.
    similarity = sparse.csr_matrix(
        (neighbors["sim"].to_numpy(dtype="float64")[keep], (source[keep], target[keep])), shape=(n_items, n_items)
    )
    # Build the sparse binary user x item purchase matrix.
    bought = sparse.csr_matrix((np.ones(len(pairs), dtype="float64"), (user_codes, item_codes)), shape=(n_users, n_items))

    # Start with a similarity sum of 0 for every pair.
    raw = np.zeros(len(pairs), dtype="float64")
    # pairs are sorted by user_id and factorize(sort=True) numbers users in sorted order: user_codes is non-decreasing.
    for start in range(0, n_users, block_size):
        # Find the rows of ``pairs`` that belong to the users of this block.
        stop = min(start + block_size, n_users)
        lo, hi = np.searchsorted(user_codes, [start, stop])
        # Skip a block without rows.
        if lo == hi:
            continue
        # Multiply the block's purchase rows by the similarity matrix.
        product = (bought[start:stop] @ similarity).tocsr()
        # Read the product at the purchased (user, item) positions of the block.
        raw[lo:hi] = np.asarray(product[user_codes[lo:hi] - start, item_codes[lo:hi]]).ravel()

    # Store log(1 + sum) as the feature and return the output columns.
    out = pairs.copy()
    out[ICF_COLUMN] = np.log1p(raw)
    return out[COLUMNS]
