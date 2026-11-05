"""Create comparison tables and a figure from sample/medium repeat-explore reports.

Usage:
  python scripts/make_health_candidate_report.py \
    reports/metrics/sample_re_instacart.json \
    reports/metrics/medium_re_instacart.json \
    --out-dir reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL = "re_health_candidate_soft"
BASELINE = "re_beta0"
ATTRIBUTES = ["fat_reduced", "gluten_free", "low_sodium"]
LABELS = {"fat_reduced": "Low fat", "gluten_free": "Gluten free", "low_sodium": "Low sodium"}


def read_results(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall: list[dict[str, object]] = []
    attributes: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        profile = str(payload["profile"])
        for scope, scope_report in payload["scopes"].items():
            row = scope_report["repeat_explore"]["instacart"]
            base = row["metrics"][BASELINE]
            model = row["metrics"][MODEL]
            ci = row["bootstrap_ci"]["new_item_recall_health_candidate_soft_minus_beta0"]
            overall.append(
                {
                    "profile": profile,
                    "scope": scope,
                    "lambda": row["selected"]["health_candidate_lambda"],
                    "base_new_recall": base["new_item_recall_at_k"],
                    "model_new_recall": model["new_item_recall_at_k"],
                    "delta_new_recall": model["new_item_recall_at_k"] - base["new_item_recall_at_k"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "base_ndcg": base["ndcg_at_k"],
                    "model_ndcg": model["ndcg_at_k"],
                    "delta_ndcg": model["ndcg_at_k"] - base["ndcg_at_k"],
                }
            )
            for attribute in ATTRIBUTES:
                attr = row["attribute_hit_rates"][attribute]
                attributes.append(
                    {
                        "profile": profile,
                        "scope": scope,
                        "attribute": attribute,
                        "users": attr["users"],
                        "candidate_recall": attr["candidate_recall"],
                        "base_hit_rate": attr[BASELINE],
                        "model_hit_rate": attr[MODEL],
                        "delta_hit_rate": attr[MODEL] - attr[BASELINE],
                    }
                )
    return pd.DataFrame(overall), pd.DataFrame(attributes)


def make_figure(overall: pd.DataFrame, attributes: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    labels = [f"{row.profile}\n{row.scope}" for row in overall.itertuples()]
    x = np.arange(len(overall))
    delta = overall["delta_new_recall"].to_numpy()
    low = delta - overall["ci_low"].to_numpy()
    high = overall["ci_high"].to_numpy() - delta
    colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in delta]
    axes[0].bar(x, delta, color=colors, width=0.65)
    axes[0].errorbar(x, delta, yerr=np.vstack([low, high]), fmt="none", ecolor="#263238", capsize=4)
    axes[0].axhline(0, color="#263238", linewidth=1)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Delta new-item Recall@10")
    axes[0].set_title("Overall effect (95% bootstrap CI)")

    medium_food = attributes[(attributes["profile"] == "medium") & (attributes["scope"] == "food")]
    x2 = np.arange(len(medium_food))
    width = 0.36
    axes[1].bar(x2 - width / 2, medium_food["base_hit_rate"], width, label="Item-CF", color="#9aa0a6")
    axes[1].bar(x2 + width / 2, medium_food["model_hit_rate"], width, label="Health candidates", color="#457b9d")
    axes[1].set_xticks(x2, [LABELS[name] for name in medium_food["attribute"]])
    axes[1].set_ylabel("Attribute new-item hit rate@10")
    axes[1].set_title("Medium food-scope attribute effect")
    axes[1].legend(frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown(overall: pd.DataFrame, attributes: pd.DataFrame) -> str:
    lines = [
        "# 건강 선호 후보 추천 실험 결과 요약",
        "",
        "이 문서는 고정된 sample/medium Instacart 평가 JSON에서 자동 생성했다. 평가 정답은 사용자 선호, "
        "후보 생성, 가중치 선택에 사용하지 않았다.",
        "",
        "## 실험 목적과 변경 사항",
        "",
        "- 외부 재료 매핑 없이 상품명과 aisle만 사용해 저지방, 글루텐프리, 저염 속성을 탐색 추천에 반영했다.",
        "- 사용자 이력에서 속성 적용 상품을 10회 이상 구매했고 모집단보다 선호 비율이 0.05 이상 높을 때만 "
        "건강 선호를 활성화했다.",
        "- 사용자가 과거에 구매한 aisle의 미구매 건강 상품을 후보에 추가한 뒤, fit split에서 선택한 lambda로 "
        "Item-CF 점수와 결합했다.",
        "- 재구매 상품의 순위와 슬롯은 변경하지 않았고, 유형 내 AUC 근거가 없었던 저당은 추천 규칙에서 제외했다.",
        "",
        "## 전체 다음 장바구니 지표",
        "",
        "| 프로필 | 범위 | lambda | 기준 신규 Recall@10 | 건강 모델 신규 Recall@10 | 차이 | 차이의 95% CI | NDCG@10 차이 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"| {row['profile']} | {row['scope']} | {row['lambda']:g} | {row['base_new_recall']:.6f} | "
            f"{row['model_new_recall']:.6f} | {row['delta_new_recall']:+.6f} | "
            f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | {row['delta_ndcg']:+.6f} |"
        )
    lines += [
        "",
        "## Medium food 범위의 건강 속성별 성과",
        "",
        "| 속성 | 평가 사용자 | 후보 재현율 | 기준 hit@10 | 건강 모델 hit@10 | 절대 차이 | 상대 변화 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    subset = attributes[(attributes["profile"] == "medium") & (attributes["scope"] == "food")]
    for row in subset.itertuples():
        relative = row.delta_hit_rate / row.base_hit_rate if row.base_hit_rate else float("nan")
        lines.append(
            f"| {LABELS[row.attribute]} | {row.users} | {row.candidate_recall:.6f} | "
            f"{row.base_hit_rate:.6f} | {row.model_hit_rate:.6f} | {row.delta_hit_rate:+.6f} | "
            f"{relative:+.1%} |"
        )
    lines += [
        "",
        "## 결론",
        "",
        "- **확인된 성과:** medium food 범위에서 건강 속성 신상품 hit@10이 저지방 "
        "0.027215→0.035625(+30.9%), 글루텐프리 0.005880→0.007010(+19.2%), 저염 "
        "0.008140→0.012791(+57.1%)로 모두 상승했다.",
        "- **전체 정확도:** sample에서는 신규 Recall@10이 아주 작게 상승했지만 신뢰구간이 0을 포함했다. "
        "medium food에서는 -0.000097이었고 역시 신뢰구간이 0을 포함했다. medium all은 튜닝이 lambda=0을 "
        "선택하여 기준 모델로 안전하게 되돌아갔다.",
        "- **해석:** 건강 후보 규칙은 일반적인 다음 장바구니 정확도 향상으로 볼 수 없다. 대신 전체 정확도를 거의 "
        "유지하면서 검증된 건강 선호 사용자의 관련 신상품 노출을 높인 개인화 보조 규칙으로 해석한다.",
        "- **한계:** 속성별 hit 평가는 대상 사용자 수가 서로 다르고 별도 신뢰구간을 아직 계산하지 않았다. 따라서 "
        "상대 변화율만으로 효과 크기를 단정하지 않고, full 또는 반복 표본 평가에서 재확인해야 한다.",
        "",
        "## 재현 산출물",
        "",
        "이 스크립트는 이 요약과 함께 `reports/tables/health_candidate_overall.csv`, "
        "`reports/tables/health_candidate_attributes.csv`, `reports/figures/health_candidate_comparison.png`를 생성한다. "
        "표·그림과 원본 평가 JSON은 생성물로 Git에서 제외한다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    overall, attributes = read_results(args.reports)
    tables = args.out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    overall.to_csv(tables / "health_candidate_overall.csv", index=False)
    attributes.to_csv(tables / "health_candidate_attributes.csv", index=False)
    make_figure(overall, attributes, args.out_dir / "figures" / "health_candidate_comparison.png")
    (args.out_dir / "HEALTH_CANDIDATE_RESULTS.md").write_text(
        markdown(overall, attributes), encoding="utf-8"
    )
    print(f"Wrote health-candidate report under {args.out_dir}")


if __name__ == "__main__":
    main()
