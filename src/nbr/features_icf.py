"""Item-CF similarity feature of the LR repeat ranker R2' (PREREG v1.5, item 2b and item 4).

icf(u, i) = log(1 + sum over j in I(H_u) of sim(j, i)) for every (user, item) pair (u, i) of a history,
where sim(j, i) is the cosine similarity stored in j's top-``n_neighbors`` neighbour list of
nbr.itemcf.item_neighbors(history, n_neighbors) (binary user x item matrix of the same history; self
excluded; only neighbours with sim > 0 are stored). icf = 0 when i is in no neighbour list of the user's
other items; a single-item user has icf = 0. The neighbour count is fixed at 20 (configs/common.yaml
``explore.itemcf.neighbors``); no variant of the feature is selected.

The sum is a sparse product (user x item) @ (item x item) restricted to the purchased positions, computed
in blocks of ``block_size`` users; the result does not depend on the block size. No file and no target
is read; the function only uses the frame it receives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from nbr.itemcf import item_neighbors
from nbr.repeat import PAIR_KEY

ICF_COLUMN = "icf"
COLUMNS = PAIR_KEY + [ICF_COLUMN]
DEFAULT_NEIGHBORS = 20


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
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    pairs = history[PAIR_KEY].drop_duplicates().sort_values(PAIR_KEY, kind="mergesort").reset_index(drop=True)
    if neighbors is None:
        neighbors = item_neighbors(history, n_neighbors)
    user_codes, user_values = pd.factorize(pairs["user_id"], sort=True)
    item_codes, item_values = pd.factorize(pairs["item_id"], sort=True)
    n_users, n_items = len(user_values), len(item_values)
    user_codes = np.asarray(user_codes, dtype="int64")
    item_codes = np.asarray(item_codes, dtype="int64")

    items = pd.Index(item_values)
    source = items.get_indexer(neighbors["item_id"])
    target = items.get_indexer(neighbors["neighbor_id"])
    keep = (source >= 0) & (target >= 0)
    similarity = sparse.csr_matrix(
        (neighbors["sim"].to_numpy(dtype="float64")[keep], (source[keep], target[keep])), shape=(n_items, n_items)
    )
    bought = sparse.csr_matrix((np.ones(len(pairs), dtype="float64"), (user_codes, item_codes)), shape=(n_users, n_items))

    raw = np.zeros(len(pairs), dtype="float64")
    # pairs are sorted by user_id and factorize(sort=True) numbers users in sorted order: user_codes is non-decreasing.
    for start in range(0, n_users, block_size):
        stop = min(start + block_size, n_users)
        lo, hi = np.searchsorted(user_codes, [start, stop])
        if lo == hi:
            continue
        product = (bought[start:stop] @ similarity).tocsr()
        raw[lo:hi] = np.asarray(product[user_codes[lo:hi] - start, item_codes[lo:hi]]).ravel()

    out = pairs.copy()
    out[ICF_COLUMN] = np.log1p(raw)
    return out[COLUMNS]
