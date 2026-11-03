# Sample experiment results

이 문서는 seed 42로 선택한 Instacart 1,500명과 Complete Journey 500명 표본의 검증 결과입니다. 전체 데이터 결과가 아니므로 최종 발표에서는 `sample`로 명시해야 합니다.

## Data and candidate coverage

| Dataset | Evaluated users | Candidate rows | Candidate recall |
|---|---:|---:|---:|
| Instacart | 1,500 | 541,128 | 0.6735 |
| Complete Journey | 498 | 240,096 | 0.3961 |

## Top-10 comparison

| Dataset / model | Precision@10 | Recall@10 | F1@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Instacart personal repeat | 0.2687 | 0.3139 | 0.2511 | 0.3852 |
| Instacart LR Hybrid | **0.2739** | **0.3204** | **0.2556** | **0.3907** |
| Complete Journey personal repeat | **0.0994** | **0.1371** | **0.0850** | **0.1688** |
| Complete Journey LR Hybrid | 0.0819 | 0.1125 | 0.0695 | 0.1387 |

Instacart 내부 평가에서는 LR Hybrid가 가장 강한 개인 재구매 baseline을 근소하게 개선했다. 반면 Complete Journey에서는 재구매 baseline보다 낮았다. 외부 평가 전후 모델 SHA-256은 동일하여 Complete Journey 라벨로 재학습하지 않았음을 확인했다.

외부 성능 저하의 첫 번째 원인은 candidate recall 차이다. Complete Journey 후보 집합은 실제 다음 장바구니 상품의 39.6%만 포함해 Instacart의 67.3%보다 낮다. 두 번째 원인은 Instacart의 aisle/department와 Complete Journey의 commodity/sub-commodity가 서로 다른 상품 체계라는 점이다. 세 번째 원인은 구매 빈도와 장바구니 구성의 분포 차이로, Instacart에서 학습한 feature 가중치가 Complete Journey의 반복 구매 패턴을 충분히 반영하지 못한 것으로 해석할 수 있다.

다음 실험에서는 Complete Journey의 카테고리 후보 수 확대, recency의 시간 단위 feature 추가, dataset별 score percentile 정규화를 우선 검토한다. 단, 엄격한 전이 검증을 유지하기 위해 선택 기준은 Instacart validation으로 제한한다.

상세 수치는 `reports/metrics/sample_instacart.json`과 `reports/metrics/sample_complete_journey.json`, 시각화는 `reports/figures/`에 저장된다.
