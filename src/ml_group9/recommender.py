"""Item-CF, content profiles, candidate generation, and transferable features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


FEATURE_COLUMNS = [
    "user_item_rate",
    "user_item_count_log",
    "recency_normalized",
    "mean_gap_normalized",
    "item_popularity",
    "item_repeat_rate",
    "item_cf_max",
    "item_cf_sum",
    "content_affinity",
    "source_history",
    "source_cf",
    "source_content",
    "source_popular",
]


@dataclass
class RecommendationContext:
    """Dataset-local statistics transformed into domain-independent scores."""

    user_baskets: dict[str, int]
    user_next_seq: dict[str, int]
    user_item: dict[tuple[str, str], tuple[int, int, float]]
    item_popularity: dict[str, float]
    item_repeat_rate: dict[str, float]
    item_categories: dict[str, tuple[str, str]]
    coarse_affinity: dict[tuple[str, str], float]
    fine_affinity: dict[tuple[str, str], float]
    neighbors: dict[str, list[tuple[str, float]]]
    category_popular: dict[str, list[str]]
    global_popular: list[str]
    recent_items: dict[str, list[str]]


def item_cosine_neighbors(
    history: pd.DataFrame, top_k: int = 20, block_size: int = 256
) -> dict[str, list[tuple[str, float]]]:
    """Compute sparse item cosine neighbors without materializing a dense matrix."""
    interactions = history[["basket_id", "item_id"]].drop_duplicates()
    basket_codes, _ = pd.factorize(interactions["basket_id"], sort=True)
    item_codes, item_values = pd.factorize(interactions["item_id"], sort=True)
    matrix = sparse.csr_matrix(
        (np.ones(len(interactions), dtype=np.float32), (basket_codes, item_codes)),
        shape=(basket_codes.max() + 1, len(item_values)),
    )
    matrix.data[:] = 1.0
    norms = np.sqrt(np.asarray(matrix.sum(axis=0)).ravel()).astype(np.float32)
    norms[norms == 0] = 1.0
    result: dict[str, list[tuple[str, float]]] = {}
    for start in range(0, matrix.shape[1], block_size):
        stop = min(start + block_size, matrix.shape[1])
        cooccurrence = (matrix[:, start:stop].T @ matrix).tocsr()
        for local_row in range(stop - start):
            item_index = start + local_row
            row_start, row_stop = cooccurrence.indptr[local_row : local_row + 2]
            indices = cooccurrence.indices[row_start:row_stop]
            values = cooccurrence.data[row_start:row_stop].astype(np.float32, copy=True)
            keep = indices != item_index
            indices, values = indices[keep], values[keep]
            if len(indices):
                values /= norms[item_index] * norms[indices]
                if len(indices) > top_k:
                    chosen = np.argpartition(values, -top_k)[-top_k:]
                    indices, values = indices[chosen], values[chosen]
                order = np.argsort(-values, kind="stable")
                result[str(item_values[item_index])] = [
                    (str(item_values[index]), float(score))
                    for index, score in zip(indices[order], values[order])
                ]
            else:
                result[str(item_values[item_index])] = []
    return result


def build_context(
    history: pd.DataFrame, items: pd.DataFrame, config: dict[str, object]
) -> RecommendationContext:
    """Build leakage-safe statistics using history only."""
    frame = history.copy()
    for column in ["user_id", "basket_id", "item_id", "coarse_category", "fine_category"]:
        frame[column] = frame[column].astype("string").fillna("UNKNOWN")
    implicit = frame.drop_duplicates(["user_id", "basket_id", "item_id"])
    user_baskets_series = implicit.groupby("user_id", observed=True)["basket_id"].nunique()
    user_next_series = implicit.groupby("user_id", observed=True)["basket_seq"].max().add(1)

    ui = (
        implicit.groupby(["user_id", "item_id"], observed=True)
        .agg(item_count=("basket_id", "nunique"), last_seq=("basket_seq", "max"))
        .reset_index()
    )
    ordered = implicit[["user_id", "item_id", "basket_seq"]].drop_duplicates().sort_values(
        ["user_id", "item_id", "basket_seq"]
    )
    ordered["gap"] = ordered.groupby(["user_id", "item_id"], observed=True)["basket_seq"].diff()
    mean_gap = ordered.groupby(["user_id", "item_id"], observed=True)["gap"].mean()
    ui = ui.merge(mean_gap.rename("mean_gap").reset_index(), on=["user_id", "item_id"], how="left")

    item_baskets = implicit.groupby("item_id", observed=True)["basket_id"].nunique()
    item_popularity = item_baskets.rank(method="average", pct=True).to_dict()
    repeat_buyers = ui.assign(repeated=ui["item_count"].gt(1)).groupby("item_id", observed=True)["repeated"].mean()

    user_total = implicit.groupby("user_id", observed=True).size()
    coarse_counts = implicit.groupby(["user_id", "coarse_category"], observed=True).size()
    fine_counts = implicit.groupby(["user_id", "fine_category"], observed=True).size()
    coarse_affinity = (coarse_counts / coarse_counts.index.get_level_values(0).map(user_total)).to_dict()
    fine_affinity = (fine_counts / fine_counts.index.get_level_values(0).map(user_total)).to_dict()

    item_table = items.copy()
    item_table["item_id"] = item_table["item_id"].astype("string")
    item_table["coarse_category"] = item_table["coarse_category"].astype("string").fillna("UNKNOWN")
    item_table["fine_category"] = item_table["fine_category"].astype("string").fillna("UNKNOWN")
    item_categories = item_table.set_index("item_id")[["coarse_category", "fine_category"]].apply(tuple, axis=1).to_dict()

    popularity_frame = item_baskets.rename("count").reset_index().merge(
        item_table[["item_id", "fine_category"]], on="item_id", how="left"
    )
    category_popular = {
        str(category): group.nlargest(int(config["candidates"]["category_popular"]), "count")["item_id"].astype(str).tolist()
        for category, group in popularity_frame.groupby("fine_category", observed=True)
    }
    global_popular = item_baskets.nlargest(int(config["candidates"]["global_popular"])).index.astype(str).tolist()
    recent = implicit.sort_values(["user_id", "basket_seq"]).groupby("user_id", observed=True)["item_id"].apply(list)
    recent_items = {str(user): list(dict.fromkeys(reversed([str(item) for item in values]))) for user, values in recent.items()}

    neighbors = item_cosine_neighbors(
        implicit,
        top_k=int(config["item_cf"]["top_k"]),
        block_size=int(config["item_cf"]["block_size"]),
    )
    user_item = {
        (str(row.user_id), str(row.item_id)): (
            int(row.item_count),
            int(row.last_seq),
            float(row.mean_gap) if pd.notna(row.mean_gap) else 0.0,
        )
        for row in ui.itertuples(index=False)
    }
    return RecommendationContext(
        user_baskets={str(k): int(v) for k, v in user_baskets_series.items()},
        user_next_seq={str(k): int(v) for k, v in user_next_series.items()},
        user_item=user_item,
        item_popularity={str(k): float(v) for k, v in item_popularity.items()},
        item_repeat_rate={str(k): float(v) for k, v in repeat_buyers.items()},
        item_categories={str(k): (str(v[0]), str(v[1])) for k, v in item_categories.items()},
        coarse_affinity={(str(k[0]), str(k[1])): float(v) for k, v in coarse_affinity.items()},
        fine_affinity={(str(k[0]), str(k[1])): float(v) for k, v in fine_affinity.items()},
        neighbors=neighbors,
        category_popular=category_popular,
        global_popular=global_popular,
        recent_items=recent_items,
    )


def _content_score(context: RecommendationContext, user: str, item: str) -> float:
    coarse, fine = context.item_categories.get(item, ("UNKNOWN", "UNKNOWN"))
    return 0.3 * context.coarse_affinity.get((user, coarse), 0.0) + 0.7 * context.fine_affinity.get((user, fine), 0.0)


def build_feature_frame(
    history: pd.DataFrame,
    items: pd.DataFrame,
    target: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, set[str]], RecommendationContext]:
    """Generate bounded candidates and transferable features for target users."""
    context = build_context(history, items, config)
    target_sets = {
        str(user): set(group["item_id"].astype(str))
        for user, group in target.groupby("user_id", observed=True)
    }
    history_by_user: dict[str, set[str]] = {}
    for user, group in history.groupby("user_id", observed=True):
        history_by_user[str(user)] = set(group["item_id"].astype(str))

    rows: list[dict[str, object]] = []
    limits = config["candidates"]
    for user, truth in target_sets.items():
        if user not in context.user_baskets:
            continue
        sources: dict[str, set[str]] = {
            "history": set(history_by_user.get(user, set())),
            "cf": set(),
            "content": set(),
            "popular": set(context.global_popular),
        }
        cf_max: dict[str, float] = {}
        cf_sum: dict[str, float] = {}
        for recent_item in context.recent_items.get(user, [])[: int(limits["recent_items"])]:
            for neighbor, score in context.neighbors.get(recent_item, [])[: int(limits["neighbors_per_item"])]:
                sources["cf"].add(neighbor)
                cf_max[neighbor] = max(cf_max.get(neighbor, 0.0), score)
                cf_sum[neighbor] = cf_sum.get(neighbor, 0.0) + score

        preferred = sorted(
            ((key[1], score) for key, score in context.fine_affinity.items() if key[0] == user),
            key=lambda value: (-value[1], value[0]),
        )[: int(limits["preferred_categories"])]
        for category, _ in preferred:
            sources["content"].update(context.category_popular.get(category, []))

        all_candidates = set().union(*sources.values())
        prescore: list[tuple[str, float]] = []
        basket_count = max(context.user_baskets[user], 1)
        for item in all_candidates:
            ui_count = context.user_item.get((user, item), (0, 0, 0.0))[0]
            score = (
                ui_count / basket_count
                + cf_max.get(item, 0.0)
                + _content_score(context, user, item)
                + 0.1 * context.item_popularity.get(item, 0.0)
            )
            prescore.append((item, score))
        prescore.sort(key=lambda value: (-value[1], value[0]))

        for item, _ in prescore[: int(limits["max_per_user"])]:
            count, last_seq, mean_gap = context.user_item.get((user, item), (0, 0, 0.0))
            baskets = max(context.user_baskets[user], 1)
            recency = (context.user_next_seq[user] - last_seq) / baskets if count else 1.0
            row = {
                "user_id": user,
                "item_id": item,
                "label": int(item in truth),
                "user_item_rate": count / baskets,
                "user_item_count_log": np.log1p(count),
                "recency_normalized": min(max(recency, 0.0), 1.0),
                "mean_gap_normalized": min(mean_gap / baskets, 1.0) if count else 1.0,
                "item_popularity": context.item_popularity.get(item, 0.0),
                "item_repeat_rate": context.item_repeat_rate.get(item, 0.0),
                "item_cf_max": cf_max.get(item, 0.0),
                "item_cf_sum": cf_sum.get(item, 0.0),
                "content_affinity": _content_score(context, user, item),
                "source_history": int(item in sources["history"]),
                "source_cf": int(item in sources["cf"]),
                "source_content": int(item in sources["content"]),
                "source_popular": int(item in sources["popular"]),
            }
            rows.append(row)
    return pd.DataFrame(rows), target_sets, context


def score_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach comparable baseline scores without using labels."""
    scored = frame.copy()
    scored["score_popularity"] = scored["item_popularity"]
    scored["score_repeat"] = scored["user_item_rate"]
    scored["score_item_cf"] = scored["item_cf_max"]
    scored["score_content"] = scored["content_affinity"]
    scored["score_weighted"] = (
        0.4 * scored["user_item_rate"]
        + 0.3 * scored["item_cf_max"]
        + 0.2 * scored["content_affinity"]
        + 0.1 * scored["item_popularity"]
    )
    return scored
