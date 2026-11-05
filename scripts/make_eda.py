"""Create descriptive medium-profile tables and slide-ready PNG figures.

Run with NBR_DATA_DIR pointing to the shared processed-data root. Only medium
items/eval_history/eval_target Parquet files are read. Holdout selection counts
are quoted from committed data/SOURCE.md; no holdout directory is accessed.
Third-party imports are limited to pandas and matplotlib (pandas uses the
already installed Parquet engine). No nbr package or raw-data rebuild is used.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "eda"
VENDOR = "vendor/ml_group9_e11b365"
DATA_CODE = f"{VENDOR}/src/ml_group9/data.py"
SETTINGS = f"{VENDOR}/configs/medium.yaml"
CHECK = "reports/data_check/medium.json"
DATASETS = {
    "instacart": {
        "label": "Instacart",
        "levels": ["coarse_category", "fine_category", "ingredient_type"],
        "check_levels": ["coarse_category", "fine_category", "ingredient_type"],
        "bins": [0, 4, 6, 10, 20, math.inf],
        "color": "#2166AC",
    },
    "complete_journey": {
        "label": "Complete Journey",
        "levels": ["DEPARTMENT", "commodity", "fine_category"],
        "check_levels": ["DEPARTMENT", "commodity", "fine_category"],
        "bins": [0, 20, 50, 100, math.inf],
        "color": "#B35806",
    },
}


def committed_text(relative: str) -> str:
    """Read a tracked source from HEAD, independent of unrelated working edits."""
    return subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=True,
        capture_output=True, encoding="utf-8",
    ).stdout


def source_reference(relative: str, function: str | None = None) -> str:
    """Return a file reference with an AST-derived function line range if supplied."""
    if function is None:
        return relative
    tree = ast.parse(committed_text(relative))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
    return f"{relative}:{node.lineno}-{node.end_lineno} ({function})"


def read_medium(data_root: Path, dataset: str, name: str, columns: list[str]) -> pd.DataFrame:
    """Read only allowlisted medium files; reject a resolved holdout path."""
    if dataset not in DATASETS or name not in {"items", "eval_history", "eval_target"}:
        raise ValueError("Only allowlisted medium data is permitted")
    path = (data_root / "medium" / dataset / f"{name}.parquet").resolve()
    if any(part.lower() == "holdout" for part in path.parts):
        raise PermissionError("Holdout paths are not permitted for EDA")
    frame = pd.read_parquet(path, columns=columns)
    if frame[columns].isna().any().any():
        raise ValueError(f"Missing required values in medium/{dataset}/{name}")
    return frame


def describe_dataset(history: pd.DataFrame, items: pd.DataFrame, dataset: str) -> tuple[dict, pd.Series]:
    """Count unique (user_id, basket_seq) baskets and distinct items per basket."""
    unique = history.drop_duplicates(["user_id", "basket_seq", "item_id"])
    baskets = unique.groupby(["user_id", "basket_seq"], sort=True)["item_id"].nunique()
    lengths = baskets.groupby(level="user_id").size().rename("history_baskets")
    if lengths.empty or lengths.le(0).any():
        raise ValueError("History must contain positive-length user histories")
    if items["item_id"].duplicated().any() or not history["item_id"].isin(items["item_id"]).all():
        raise ValueError("Items must map every history item exactly once")
    observed = items.loc[items["item_id"].isin(unique["item_id"].unique())]
    summary = {
        "dataset": dataset,
        "users": int(len(lengths)),
        "baskets": int(len(baskets)),
        "items": int(unique["item_id"].nunique()),
        **{f"L{index}": int(observed[column].nunique())
           for index, column in enumerate(DATASETS[dataset]["levels"], start=1)},
        "history_baskets_min": int(lengths.min()),
        "history_baskets_p10": float(lengths.quantile(0.1)),
        "history_baskets_median": float(lengths.median()),
        "history_baskets_p90": float(lengths.quantile(0.9)),
        "history_baskets_max": int(lengths.max()),
        "basket_size_median": float(baskets.median()),
    }
    return summary, lengths


def history_distribution(lengths: pd.Series, dataset: str) -> pd.DataFrame:
    """Keep an exact frequency table and its cumulative distribution for plotting."""
    result = lengths.value_counts().sort_index().rename_axis("history_baskets").rename("users").reset_index()
    result.insert(0, "dataset", dataset)
    result["user_share"] = result["users"] / len(lengths)
    result["cumulative_users"] = result["users"].cumsum()
    result["cumulative_user_share"] = result["cumulative_users"] / len(lengths)
    return result


def repeat_by_history(history: pd.DataFrame, target: pd.DataFrame, lengths: pd.Series, dataset: str):
    """Average per-user distinct-target-item repeat shares in right-closed bins.

    pandas.MultiIndex.from_frame(df) encodes user/item pairs for exact membership;
    pandas.cut(x, bins, right=True) assigns the requested (lower, upper] intervals.
    """
    truth = target[["user_id", "item_id"]].drop_duplicates().copy()
    seen = pd.MultiIndex.from_frame(history[["user_id", "item_id"]].drop_duplicates())
    truth["repeat"] = pd.MultiIndex.from_frame(truth[["user_id", "item_id"]]).isin(seen)
    per_user = truth.groupby("user_id").agg(target_items=("item_id", "size"), repeat_items=("repeat", "sum"))
    if not per_user.index.isin(lengths.index).all() or len(per_user) != len(lengths):
        raise ValueError("Target and history user sets differ")
    per_user["history_baskets"] = lengths.reindex(per_user.index)
    per_user["repeat_share"] = per_user["repeat_items"] / per_user["target_items"]
    boundaries = DATASETS[dataset]["bins"]
    labels = [f"({low}, {high if math.isfinite(high) else 'inf'})" if math.isinf(high)
              else f"({low}, {high}]" for low, high in zip(boundaries, boundaries[1:])]
    per_user["history_bin"] = pd.cut(per_user["history_baskets"], boundaries, labels=labels, right=True)
    if per_user["history_bin"].isna().any():
        raise ValueError("History bins do not cover all target users")
    grouped = per_user.groupby("history_bin", observed=False).agg(
        users=("repeat_share", "size"), mean_repeat_share=("repeat_share", "mean"),
        target_items=("target_items", "sum"), repeat_items=("repeat_items", "sum"),
    ).reset_index()
    grouped.insert(0, "dataset", dataset)
    grouped["pooled_repeat_share"] = grouped["repeat_items"].div(grouped["target_items"].replace(0, math.nan))
    return grouped, truth


def new_item_levels(history: pd.DataFrame, truth: pd.DataFrame, items: pd.DataFrame, dataset: str, check: dict):
    """Reuse committed JSON ratios and reconcile every ratio with medium data."""
    spec = DATASETS[dataset]
    columns = spec["levels"]
    lookup = items.set_index("item_id")[columns]
    new = truth.loc[~truth["repeat"], ["user_id", "item_id"]].join(lookup, on="item_id")
    past = history[["user_id", "item_id"]].drop_duplicates().join(lookup, on="item_id")
    if new[columns].isna().any().any() or past[columns].isna().any().any():
        raise ValueError("Incomplete hierarchy mappings")
    rows = []
    for index, (column, check_column) in enumerate(zip(columns, spec["check_levels"]), start=1):
        key = f"new_item_rows_in_seen_{check_column}"
        value = float(check[dataset]["eval"][key])
        seen = pd.MultiIndex.from_frame(past[["user_id", column]].drop_duplicates())
        count = int(pd.MultiIndex.from_frame(new[["user_id", column]]).isin(seen).sum())
        if not math.isclose(count / len(new), value, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Committed JSON mismatch: {dataset}/{key}")
        rows.append({"dataset": dataset, "level": f"L{index}", "item_column": column,
                     "new_target_items": len(new), "in_seen_category_items": count,
                     "share": value, "source": f"{CHECK} -> {dataset}.eval.{key}"})
    return rows


def preprocessing_table(check: dict) -> pd.DataFrame:
    """Document the snapshot rules without inferring unavailable raw cleaning counts.

    Split descriptions are adapted from ML_group9 e11b365 src/ml_group9/data.py;
    configured constants come from its configs/medium.yaml. Selection counts are
    quoted from data/SOURCE.md, not recomputed from any holdout data.
    """
    settings = committed_text(SETTINGS)
    source = committed_text("data/SOURCE.md")
    exclusions = re.search(r"  exclude_departments:\n(.*?)  min_target_items:", settings, re.S).group(1)
    departments = re.findall(r'^\s+- "([^"]+)"', exclusions, re.M)
    def scalar(key):
        return int(re.search(rf"^\s*{key}: (\d+)", settings, re.M).group(1))
    cutoff, fit_week, minimum = (scalar(key) for key in ("complete_journey_history_week", "fit_history_week", "min_target_items"))
    pool = re.search(r"from the ([\d,]+) users not in medium or sample \(([\d,]+) eligible.*?([\d,]+) used\)", source)
    draw = re.search(r"(\d[\d,]*) train-set users drawn with seed (\d+)", source)
    if pool is None or draw is None or not departments:
        raise ValueError("Source documentation format changed; review preprocessing descriptions")
    remaining, eligible, used = (int(value.replace(",", "")) for value in pool.groups())
    selected, seed = (int(value.replace(",", "")) for value in draw.groups())
    if eligible - used != remaining:
        raise ValueError("Documented holdout user counts do not reconcile")
    rows = []

    def add(dataset, step, input_name, rule, function=None, **values):
        rows.append({"dataset": dataset, "step": step, "input": input_name, "rule": rule,
                     "rows_before": None, "rows_after": None, "users_before": None,
                     "users_after": None, "history_rows_after": None, "target_rows_after": None,
                     "status": "규칙 확인·수치 확인 필요", "note": "원시 단계별 행·사용자 수는 허용 자료에서 확인할 수 없음",
                     "source": source_reference(DATA_CODE, function) if function else "data/SOURCE.md",
                     **values})

    ic = check["instacart"]
    add("instacart", "1. 사용자 선택", "orders.csv; medium 설정",
        "공식 train 사용자를 문자열 정렬 후 seed 42로 비복원 추출; medium 20,000명",
        "prepare_instacart", users_before=eligible, users_after=ic["eval"]["users"],
        status="부분 확인", note="추출 전 수는 SOURCE.md 기록; 추출 후 수는 medium JSON; 원시 행 수 확인 필요",
        source=f"{source_reference(DATA_CODE, 'prepare_instacart')}; {source_reference(DATA_CODE, '_sample_ids')}; {SETTINGS}; data/SOURCE.md; {CHECK}")
    add("instacart", "2. 공통 스키마·계층", "prior/train 상품 행; products/aisles/departments",
        "order 메타와 상품 계층을 조인; user_id·item_id·basket_id는 문자열, basket_seq는 int32, quantity=1",
        "prepare_instacart")
    add("instacart", "3. eval 분할", "선택 사용자의 prior 및 공식 train 주문",
        "eval_history=모든 prior 주문; eval_target=공식 train 주문", "prepare_instacart",
        rows_after=ic["eval"]["history_rows"] + ic["eval"]["target_rows"],
        users_after=ic["eval"]["users"], history_rows_after=ic["eval"]["history_rows"],
        target_rows_after=ic["eval"]["target_rows"], status="부분 확인",
        note="분할 후 수는 medium JSON; 전체 원시 입력 행 수 확인 필요", source=f"{source_reference(DATA_CODE, 'prepare_instacart')}; {CHECK}")
    add("instacart", "4. fit 분할", "eval_history의 prior 주문",
        "마지막 prior 주문=fit_target; 이전 주문=fit_history; fit_history 바스켓 2개 이상인 사용자 유지",
        "prepare_instacart", rows_before=ic["eval"]["history_rows"],
        rows_after=ic["fit"]["history_rows"] + ic["fit"]["target_rows"],
        users_before=ic["eval"]["users"], users_after=ic["fit"]["users"],
        history_rows_after=ic["fit"]["history_rows"], target_rows_after=ic["fit"]["target_rows"],
        status="확인", note="분할 전후 합계는 medium JSON; 원시 단계별 로그를 뜻하지 않음",
        source=f"{source_reference(DATA_CODE, 'prepare_instacart')}; {CHECK}")
    rules = [
        ("5. 미등록 상품 제거", "PRODUCT_ID가 product.csv에 없는 행 제거"),
        ("6. 부서 누락 제거", "공백 DEPARTMENT를 결측으로 바꾼 후 부서 결측 행 제거"),
        ("7. 제외 부서 제거", "DEPARTMENT가 다음 값이면 제거: " + "; ".join(departments)),
        ("8. 수량 필터", "QUANTITY <= 0 제거; 양수 수량이면 SALES_VALUE <= 0인 행도 유지"),
    ]
    for step, rule in rules:
        source = f"{source_reference(DATA_CODE, 'clean_complete_journey')}; {SETTINGS}"
        if step.startswith("6."):
            source += f"; {source_reference(DATA_CODE, '_read_complete_journey')}"
        add("complete_journey", step, "transaction_data.csv + product.csv; 앞 단계 통과 행", rule,
            "clean_complete_journey", source=source)
    add("complete_journey", "9. 바스켓 정렬", "정제 거래 행",
        "household_key, DAY, TRANS_TIME, BASKET_ID 순 정렬; 가구별 basket_seq 부여; 공통 스키마로 변환",
        "_complete_journey_common")
    cj = check["complete_journey"]
    add("complete_journey", "10. 사용자·정답 적격성", "90주 전후 거래가 있는 가구",
        f"각 분할 history 바스켓 >=3; 미래의 서로 다른 상품 >={minimum}인 최초 바스켓을 정답으로 선택; fit/eval 모두 가능한 가구의 교집합; 필터 후 표본 추출(null=전체)",
        "prepare_complete_journey", users_after=cj["eval"]["users"],
        status="부분 확인", note="최종 가구 수만 medium JSON으로 확인; 단계별 탈락 수 확인 필요",
        source=f"{source_reference(DATA_CODE, 'prepare_complete_journey')}; {source_reference(DATA_CODE, '_history_and_target')}; {source_reference(DATA_CODE, 'first_future_basket')}; {SETTINGS}; {CHECK}")
    for split, rule in [
        ("fit", f"history WEEK_NO<={fit_week}; target=({fit_week}, {cutoff}]의 최초 적격 바스켓"),
        ("eval", f"history WEEK_NO<={cutoff}; target={cutoff}주 이후의 최초 적격 바스켓"),
    ]:
        stats = cj[split]
        add("complete_journey", f"{11 if split == 'fit' else 12}. {split} 주차 분할", "적격 가구의 정제 거래 행", rule,
            "_complete_journey_splits", rows_after=stats["history_rows"] + stats["target_rows"],
            users_after=stats["users"], history_rows_after=stats["history_rows"], target_rows_after=stats["target_rows"],
            status="부분 확인", note="분할 후 수는 medium JSON; 분할 전 정제 거래 수 확인 필요",
            source=f"{source_reference(DATA_CODE, '_complete_journey_splits')}; {SETTINGS}; {CHECK}")
    add("instacart_holdout_rule_only", "13. 기존 사용자 제외", "공식 train 적격 사용자",
        "medium 또는 sample에 속한 사용자 합집합을 추출 전에 제외",
        users_before=eligible, users_after=remaining, status="기록 확인",
        note=f"SOURCE.md 기록: 기존 사용자 {used:,}명 제외; holdout 파일은 열지 않음; 행 수 확인 필요")
    add("instacart_holdout_rule_only", "14. holdout 사용자 추출", "기존 사용자 제외 후 풀",
        f"문자열 정렬된 잔여 사용자에서 seed {seed}로 {selected:,}명 비복원 추출; medium/sample 중복 0명(문서 기록)",
        users_before=remaining, users_after=selected, status="기록 확인",
        note="SOURCE.md 기록만 인용; holdout 원자료 검증 및 분할 행 수 확인 필요",
        source=f"data/SOURCE.md; {source_reference(DATA_CODE, '_sample_ids')}")
    result = pd.DataFrame(rows)
    for column in ["rows_before", "rows_after", "users_before", "users_after", "history_rows_after", "target_rows_after"]:
        result[column] = result[column].astype("Int64")
    return result


def write_csv(frame: pd.DataFrame, name: str) -> None:
    """Write Excel-readable UTF-8 CSV; unavailable numeric values remain blank."""
    frame.to_csv(OUTPUT / name, index=False, encoding="utf-8-sig")


def style_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DCE1E7", linewidth=0.7)
    ax.set_axisbelow(True)


def figures(check: dict) -> None:
    """Render all chart values directly from their companion CSV files."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titlesize": 16, "axes.labelsize": 13,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    distribution = pd.read_csv(OUTPUT / "fig_history_length.csv")
    fig, ax = plt.subplots(figsize=(10.8, 5.8), layout="constrained")
    for index, (dataset, spec) in enumerate(DATASETS.items()):
        part = distribution.loc[distribution["dataset"].eq(dataset)]
        ax.step(part["history_baskets"], part["cumulative_user_share"], where="post",
                color=spec["color"], linestyle="-" if index == 0 else "--", linewidth=2.5, label=spec["label"])
    ax.set(xscale="log", ylim=(0, 1.025), xlabel="History baskets per user (log scale)",
           ylabel="Cumulative share of users", title="Distribution of purchase-history length")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    style_axis(ax)
    ax.legend(loc="lower right", frameon=False)
    fig.text(0.12, -0.025, "Medium profile | eval history | one observation per user", fontsize=10, color="#4D5968")
    fig.savefig(OUTPUT / "fig_history_length.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    repeat = pd.read_csv(OUTPUT / "fig_repeat_share_by_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.7), sharey=True, layout="constrained")
    for ax, (dataset, spec) in zip(axes, DATASETS.items()):
        part = repeat.loc[repeat["dataset"].eq(dataset)]
        positions = list(range(len(part)))
        bars = ax.bar(positions, part["mean_repeat_share"], color=spec["color"], width=0.67)
        ax.set_xticks(positions, [f"{row.history_bin}\nn = {row.users:,}" for row in part.itertuples()])
        for bar, value, users in zip(bars, part["mean_repeat_share"], part["users"]):
            label = f"{value:.1%}" if users else "No users"
            ax.text(bar.get_x() + bar.get_width() / 2, (value if users else 0) + 0.025,
                    label, ha="center", fontsize=11)
        ax.set(title=spec["label"], xlabel="History baskets per user", ylim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        style_axis(ax)
    axes[0].set_ylabel("Mean repeat-item share in target basket")
    fig.suptitle("Repeat-item share by history length", fontsize=17)
    fig.text(0.10, -0.03, "Medium eval target | distinct items | equal weight per user | n = users", fontsize=10, color="#4D5968")
    fig.text(0.10, -0.075, "Descriptive; history bins differ between datasets.", fontsize=10, color="#4D5968")
    fig.savefig(OUTPUT / "fig_repeat_share_by_history.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    levels = pd.read_csv(OUTPUT / "fig_new_item_level.csv")
    fig, ax = plt.subplots(figsize=(10.8, 5.8), layout="constrained")
    width = 0.32
    for offset, (dataset, spec) in zip([-width / 2, width / 2], DATASETS.items()):
        part = levels.loc[levels["dataset"].eq(dataset)]
        x = [index + offset for index in range(3)]
        bars = ax.bar(x, part["share"], width=width, label=spec["label"], color=spec["color"],
                      hatch="" if dataset == "instacart" else "//", edgecolor="white")
        for bar, value in zip(bars, part["share"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.1%}", ha="center", fontsize=11)
    ax.set_xticks(range(3), ["L1", "L2", "L3"])
    ax.set(ylim=(0, 1.12), xlabel="Hierarchy level", ylabel="Share of new target items",
           title="New target items in previously purchased categories")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    style_axis(ax)
    ax.legend(loc="upper right", frameon=False)
    fig.text(0.12, -0.025, "Medium eval target | item-weighted | source: committed data-check report", fontsize=10, color="#4D5968")
    fallback_pct = round(100 * (1 - check["instacart"]["eval"]["history_rows_l3_mapped"]))
    notes = (
        "Category levels are dataset-specific (Instacart: department / aisle / ingredient type;\n"
        "CJ: department / commodity / sub-commodity); bar heights are not comparable across datasets.\n"
        f"Instacart L3 falls back to the aisle for unmapped items (≈{fallback_pct}% of history rows)."
    )
    fig.text(0.12, -0.065, notes, fontsize=10, color="#4D5968", va="top", linespacing=1.5)
    fig.savefig(OUTPUT / "fig_new_item_level.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without an optional tabulate dependency."""
    def value(cell):
        if pd.isna(cell):
            return ""
        if isinstance(cell, float):
            return f"{cell:.6g}"
        return str(cell).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(value(cell) for cell in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def summary_markdown(datasets, repeats, levels, preprocessing, quality, check):
    """Retain definitions, verified descriptive findings, and explicit evidence gaps."""
    lines = ["# 데이터 탐색·전처리 요약", "", "## 범위와 계산 기준", "",
             "medium의 Instacart·Complete Journey(CJ)를 기술한다. 데이터셋 규모·history 길이·바스켓 크기는 eval_history 기준이다. 정답 통계는 eval_target의 기술 통계이며 모델 평가·선택에는 사용하지 않는다.", "",
             "- 바스켓은 서로 다른 (user_id, basket_seq), 바스켓 크기는 그 안의 서로 다른 item_id 수다. 상품·L1/L2/L3 수는 eval_history에서 실제 관측한 상품과 그 상품의 items 매핑으로 센다. 전체 catalog 크기와 구분한다.",
             "- history 길이의 p10·p90은 pandas 선형 보간 분위수다. 그림 1은 사용자 한 명을 관측값 하나로 한 누적분포(ECDF)이며 x축은 로그 축이다.",
             "- 재구매 상품은 같은 사용자의 history에 있던 상품이다. target의 (user_id, item_id) 중복을 제거하고 사용자별 재구매 상품 수 / 정답 상품 수를 계산한 뒤 사용자에 같은 가중치를 주어 평균낸다. 구간은 왼쪽 열림·오른쪽 닫힘이며 inf는 상한이 없음을 뜻한다.",
             "- 신규 상품의 기존 범주 비율은 신규 target 상품 행에 같은 가중치를 준 비율이다. 사용자 평균과 다르다. medium.json의 eval 값을 그대로 사용하고 원자료 계산과 대조했다.",
             "- Instacart: L1=department(coarse_category), L2=aisle(fine_category), L3=ingredient_type(aisle fallback 포함). CJ: L1=DEPARTMENT, L2=commodity(coarse_category 별칭), L3=fine_category.",
             "- holdout 폴더와 원시 ZIP은 열지 않았다. holdout 추출 수치는 커밋된 data/SOURCE.md 기록만 인용했다.", "",
             "## 표 1: 관측 history 규모", "", markdown_table(datasets), ""]
    cj_p90 = datasets.loc[datasets["dataset"].eq("complete_journey"), "history_baskets_p90"].iloc[0]
    lines += [f"PPT 표시값: CJ p90 = {cj_p90:.0f}개 바스켓(정수 반올림). 표와 CSV의 원래 계산값 {cj_p90:g}는 유지한다.", ""]
    for row in datasets.itertuples():
        label = DATASETS[row.dataset]["label"]
        lines.append(f"- {label}: 사용자 {row.users:,}명, 바스켓 {row.baskets:,}개, 상품 {row.items:,}개. 사용자당 history 바스켓 중앙값 {row.history_baskets_median:g}개(p10={row.history_baskets_p10:g}, p90={row.history_baskets_p90:g}), 바스켓별 서로 다른 상품 수 중앙값 {row.basket_size_median:g}개.")
        lines.append(f"  전체 catalog는 {check[row.dataset]['items']:,}개 상품이며, 위 관측 상품 수와 범위가 다르다.")
    lines += ["", "## 그림 2: history 길이별 정답 재구매 비중", "", markdown_table(repeats), "",
              "구간별 표본 수와 평균을 함께 제시한다. history 길이와 재구매 비중의 관계는 관측 자료의 기술 결과이며 인과관계로 해석하지 않는다. pooled_repeat_share는 전체 상품 행을 합친 검산용 값이며 막대는 mean_repeat_share를 사용한다.", ""]
    for dataset, spec in DATASETS.items():
        part = repeats.loc[repeats["dataset"].eq(dataset)]
        first, last = part.iloc[0], part.iloc[-1]
        lines.append(f"- {spec['label']}: {first['history_bin']} 구간의 사용자 평균 재구매 비중은 {first['mean_repeat_share']:.2%}(n={first['users']:,}), {last['history_bin']} 구간은 {last['mean_repeat_share']:.2%}(n={last['users']:,})다.")
    lines += ["", "## 그림 3: 신규 상품의 기존 범주 포함 비율", "", markdown_table(levels[["dataset", "level", "new_target_items", "in_seen_category_items", "share"]]), ""]
    fallback_pct = round(100 * (1 - check["instacart"]["eval"]["history_rows_l3_mapped"]))
    lines += [
        "- 범주 수준은 데이터셋마다 다르다. Instacart는 department / aisle / ingredient type, CJ는 department / commodity / sub-commodity이므로 데이터셋 간 막대 높이를 직접 비교할 수 없다.",
        f"- Instacart L3는 매핑되지 않은 상품에 aisle을 대신 사용한다. medium eval_history 행의 약 {fallback_pct}%가 이 fallback에 해당한다(커밋 medium.json의 1 − history_rows_l3_mapped를 정수 %로 반올림).", "",
    ]
    for dataset, spec in DATASETS.items():
        part = levels.loc[levels["dataset"].eq(dataset)]
        values = ", ".join(f"{row.level} {row.share:.2%}" for row in part.itertuples())
        lines.append(f"- {spec['label']}: {values}. 분모는 해당 사용자의 history에 없던 신규 정답 상품 {int(part.iloc[0]['new_target_items']):,}개다.")
    lines += ["", "## 표 5: 전처리 단계", "",
              "rows_after는 해당 단계 출력 전체 행 수이며 분할 단계에서는 history_rows_after + target_rows_after다. 빈 수치는 0이 아니라 확인할 수 없는 값이다. 기록 확인은 문서 인용이며 원자료 재검증을 뜻하지 않는다.", "",
              markdown_table(preprocessing), "", "## 검산", "", markdown_table(pd.DataFrame(quality)), "",
              "medium eval history/target 행 수·사용자 수 및 전체 target 재구매 행 비중을 커밋 JSON과 대조했다. 신규 상품의 L1/L2/L3 비율 6개는 전체 medium 자료로 재계산해 절대오차 1e-12 이내 일치를 확인했다. 각 그림은 저장한 동명 CSV를 다시 읽어 그린다.", "",
              "## 확인 필요", "",
              "- 원시 Instacart 조인·샘플링 전후 상품 행 수와 CJ의 미등록 상품·부서 누락·제외 부서·수량 필터별 제거 행 수 및 사용자 수. 허용된 자료에 해당 단계별 로그가 없어 CSV에서 빈칸으로 두었다.",
              "- CJ의 주차 경계 통과, history 최소 바스켓 수, 정답 최소 상품 수, fit/eval 교집합 조건 각각으로 탈락한 가구 수와 정제 거래 전체 행 수.",
              "- data/SOURCE.md는 기존 medium/sample 생성 코드 버전을 파일 시각으로 추정한다고 명시한다. vendor e11b365는 확인한 규칙의 근거이며 복사본의 모든 생성 단계를 확정하는 실행 로그는 아니다.",
              "- holdout의 131,209 → 109,936 → 20,000명 및 기존 사용자 21,273명 제외·중복 0명은 SOURCE.md 기록이다. 이 작업에서는 holdout 파일을 열거나 해당 수치를 재검증하지 않았다.", "",
              "## 재현과 출처", "", "```powershell", '$env:NBR_DATA_DIR = "C:\\dev_ML\\data\\processed"',
              r"C:\dev_ML\.venv\Scripts\python.exe scripts/make_eda.py", "```", "",
              f"pandas {pd.__version__}; matplotlib {matplotlib.__version__}. 제3자 라이브러리를 직접 불러오는 것은 이 두 패키지뿐이며 nbr에 의존하지 않는다.", ""]
    for relative in ["data/SOURCE.md", DATA_CODE, SETTINGS, CHECK]:
        digest = hashlib.sha256(committed_text(relative).encode("utf-8")).hexdigest()
        lines.append(f"- `{relative}` (HEAD에 커밋된 내용), SHA-256 `{digest}`")
    lines += ["", "입력 Parquet은 NBR_DATA_DIR/medium/{instacart,complete_journey}/{items,eval_history,eval_target}.parquet 여섯 파일로 제한한다. fit 수치는 medium.json만 이용한다.", "",
              "## 산출물", "", "- table_datasets.csv", "- table_preprocessing.csv", "- fig_history_length.png / fig_history_length.csv",
              "- fig_repeat_share_by_history.png / fig_repeat_share_by_history.csv", "- fig_new_item_level.png / fig_new_item_level.csv", ""]
    (OUTPUT / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Build all requested artifacts from the explicit, read-only source scope."""
    data_root = Path(os.environ.get("NBR_DATA_DIR", str(ROOT / "data" / "processed")))
    check = json.loads(committed_text(CHECK))
    descriptions, distributions, repeats, levels, quality = [], [], [], [], []
    for dataset, spec in DATASETS.items():
        history = read_medium(data_root, dataset, "eval_history", ["user_id", "basket_id", "basket_seq", "item_id"])
        target = read_medium(data_root, dataset, "eval_target", ["user_id", "basket_id", "basket_seq", "item_id"])
        items = read_medium(data_root, dataset, "items", ["item_id", *spec["levels"]])
        description, lengths = describe_dataset(history, items, dataset)
        stats = check[dataset]["eval"]
        if (len(history), len(target), len(lengths)) != (stats["history_rows"], stats["target_rows"], stats["users"]):
            raise ValueError(f"Medium input counts disagree with committed JSON: {dataset}")
        grouped, truth = repeat_by_history(history, target, lengths, dataset)
        pooled = float(truth["repeat"].mean())
        if not math.isclose(pooled, stats["target_repeat_row_share"], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Target repeat share disagrees with committed JSON: {dataset}")
        if grouped["users"].sum() != len(lengths) or grouped["target_items"].sum() != len(truth):
            raise ValueError("Grouped counts do not reconcile")
        descriptions.append(description)
        distributions.append(history_distribution(lengths, dataset))
        repeats.append(grouped)
        levels.extend(new_item_levels(history, truth, items, dataset, check))
        quality.append({"dataset": dataset, "history_rows": len(history), "target_rows": len(target),
                        "duplicate_history_user_basket_item": int(history.duplicated(["user_id", "basket_seq", "item_id"]).sum()),
                        "duplicate_target_user_item": int(target.duplicated(["user_id", "item_id"]).sum()),
                        "target_users": int(target["user_id"].nunique()), "json_repeat_share": pooled,
                        "json_counts_match": True, "level_ratios_match": True})
        print(f"{dataset}: users={len(lengths):,}, baskets={description['baskets']:,}, items={description['items']:,}")
    description_frame = pd.DataFrame(descriptions)
    repeat_frame = pd.concat(repeats, ignore_index=True)
    level_frame = pd.DataFrame(levels)
    preprocessing = preprocessing_table(check)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(description_frame, "table_datasets.csv")
    write_csv(pd.concat(distributions, ignore_index=True), "fig_history_length.csv")
    write_csv(repeat_frame, "fig_repeat_share_by_history.csv")
    write_csv(level_frame, "fig_new_item_level.csv")
    write_csv(preprocessing, "table_preprocessing.csv")
    figures(check)
    summary_markdown(description_frame, repeat_frame, level_frame, preprocessing, quality, check)
    print("Wrote 5 CSV files, 3 PNG figures, and eda_summary.md to reports/eda")


if __name__ == "__main__":
    main()
