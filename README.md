# ML Group 9 - Hybrid Next-Basket Recommender

Instacart에서 학습한 `Item-based Collaborative Filtering + Content-based Filtering + Logistic Regression` 모델을 Complete Journey에 재학습 없이 적용하는 개인화 다음 장바구니 추천 프로젝트입니다.

## 프로젝트 목표

- 사용자별 다음 장바구니 상품 Top-10 추천
- 재구매 상품과 신규 상품을 함께 후보로 생성
- Instacart 공식 `train` 주문으로 내부 평가
- Complete Journey 1~90주 이력과 91~102주 정답으로 외부 전이 평가
- Precision@10, Recall@10, F1@10, NDCG@10, candidate recall, catalog coverage 비교

## 데이터 배치

원본 ZIP은 Git에 포함하지 않습니다. ZIP 이름은 자유롭게 정할 수 있으며 프로그램이 내부 CSV 목록으로 데이터셋을 식별합니다.

```text
data/raw/
├── instacart/<archive>.zip
└── complete_journey/<archive>.zip
```

Instacart ZIP에는 `orders.csv`, `order_products__prior.csv`, `order_products__train.csv`, `products.csv`, `aisles.csv`, `departments.csv`가 필요합니다. Complete Journey ZIP에는 최소 `transaction_data.csv`, `product.csv`가 필요합니다.

데이터 출처는 [Kaggle Instacart Market Basket Analysis](https://www.kaggle.com/competitions/basket-analysis)와 [dunnhumby Source Files - The Complete Journey](https://www.dunnhumby.com/source-files/)입니다. 원본 데이터의 이용 조건은 각 배포 페이지를 따릅니다.

## 설치

Python 3.10을 기준으로 하며 3.12까지 호환됩니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 실행

먼저 작은 사용자 표본으로 전체 흐름을 검증합니다.

```bash
python -m ml_group9 prepare --dataset all --profile sample
python -m ml_group9 train --profile sample
python -m ml_group9 evaluate --dataset instacart --profile sample
python -m ml_group9 evaluate --dataset complete_journey --profile sample
```

한 번에 실행하려면 다음 명령을 사용합니다.

```bash
python -m ml_group9 run --profile sample
```

sample 실행이 완료된 후 전체 데이터로 최종 실험합니다.

```bash
python -m ml_group9 run --profile full
```

`full`은 3천만 건 이상의 interaction으로 Item-CF를 계산하므로 충분한 메모리와 실행 시간이 필요합니다. 설정은 `configs/sample.yaml`, `configs/full.yaml`에서 관리합니다.

## 모델 구조

후보 집합은 사용자의 과거 구매 상품, 최근 상품의 Item-CF cosine 이웃, 사용자 선호 카테고리의 인기 상품, 전체 인기 상품 fallback의 합집합입니다.

각 후보에 구매 비율, recency, 평균 구매 간격, 상품 인기도, 재구매율, CF 점수, content affinity, 후보 출처 feature를 생성합니다. 상품 ID와 카테고리 문자열은 모델에 직접 입력하지 않으므로 Instacart에서 학습한 결합 방식을 Complete Journey에 적용할 수 있습니다.

최종 분류기는 scikit-learn `SGDClassifier(loss="log_loss")`입니다. 이는 대규모 후보를 처리할 수 있는 L2 정규화 Logistic Regression 구현입니다. 사용자 단위 holdout에서 NDCG@10이 가장 높은 alpha를 선택합니다.

## 시간 분할과 누수 방지

- Instacart 학습: 마지막 `prior` 주문을 정답으로 분리하고 이전 주문만 feature 계산에 사용
- Instacart 평가: 모든 `prior` 주문을 이력으로 사용하고 공식 `train` 주문을 정답으로 사용
- Complete Journey 평가: 1~90주만 이력으로 사용하고 91주 이후 사용자의 첫 장바구니를 정답으로 사용
- Complete Journey 평가 중 저장된 LR 파일의 SHA-256을 전후 비교하여 재학습이 없음을 검증

## 결과 파일

- `data/processed/<profile>/`: 공통 스키마 Parquet와 시간 분할 결과
- `artifacts/<profile>/`: 모델, 후보 feature, Top-10 예측
- `reports/metrics/`: 모델별 JSON 지표
- `reports/figures/`: 모델 비교 그래프

원본·중간 데이터와 모델은 Git에서 제외하고, 재현된 최종 JSON 지표와 비교 그림은 Git에서 추적합니다.

## 코드 구성

- `ml_group9.data`: ZIP 탐색, 공통 스키마 변환, 시간 분할
- `ml_group9.recommender`: Item-CF, content profile, 후보 및 feature 생성
- `ml_group9.modeling`: Logistic Regression 선택·학습·저장
- `ml_group9.metrics`: Top-K 지표
- `ml_group9.pipeline`: 학습·평가·보고서 오케스트레이션

## 테스트

```bash
pytest
```

테스트는 시간 누수 방지, sparse cosine 이웃, content 점수, 후보 중복·개수, Top-K 수식, 모델 고정 checksum과 재현성을 확인합니다.

## 사용 라이브러리 설명

- pandas/PyArrow: CSV 스트리밍, 표 연산, Parquet 저장
- NumPy/SciPy: 수치 계산 및 sparse basket-item matrix
- scikit-learn: 표준화, Logistic Regression, 사용자 단위 분할
- Matplotlib/Seaborn: 평가 결과 시각화
- joblib: 학습 모델 직렬화
- PyYAML: sample/full 실험 설정

외부 코드를 복사하지 않고 프로젝트 전용 구현으로 작성했습니다. 강의에서 다루지 않은 API는 docstring과 본 문서에 역할을 설명합니다.
