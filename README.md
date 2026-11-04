# ML Group 9 - Next-Basket Recommendation and Health-Attribute Analysis

Instacart와 dunnhumby Complete Journey(CJ)의 구매 이력으로 사용자별 다음 장바구니 Top-10을 추천하고, 건강 속성(지방 감소, 당 감소, 글루텐 프리, 저나트륨) 구매가 사용자 수준에서 얼마나 지속되는지 분석하는 프로젝트입니다.

## 구성 요소

- **기준 모델(LR hybrid)**: Item-CF, content 점수, 구매 이력 feature를 Logistic Regression으로 결합합니다. Instacart fit split에서만 학습하고, CJ에는 재학습 없이 적용합니다(모델 파일 SHA-256을 평가 전후로 비교).
- **재구매/탐색 분리(repeat-explore)**: 최근 가중 재구매 점수와 item-CF 탐색 점수를 따로 만들고, 파라미터(`gamma`, `kappa`, `beta`, `global_r`)를 fit split에서 순서대로 고릅니다(`repeat_explore`, `re_pipeline`).
- **계층형 추천(HIER)**: aisle(CJ는 commodity) → 재료 유형 → 속성 조합 → 상품 순서로 확률을 곱합니다. 각 단계는 사용자 이력, 사용자 KNN 이웃, 모집단 비율로 shrinkage합니다(`hierarchical`, `hier_pipeline`). 재료 유형은 외부 매핑 파일에서 가져옵니다(`ingredient_types`).
- **건강 속성 분석(H\*, H\*′)**: 이력을 앞뒤 절반으로 나누어 사용자별 속성 구매 비율의 지속성(Spearman ρ, 첫 구매 상품만 쓴 버전 포함)과 구매자 지속성(odds ratio)을 봅니다(`hstar`, `hstar_followup`, `hstar_robust`). 같은 재료 유형 안에서 사용자의 속성 선호가 실제 속성 선택과 연관되는지를 유형 내 AUC로 확인합니다(`hier_followup`). 건강 속성 구매자 하위집단 분석은 `health_subgroup`, `hs_followup`에 있습니다.
- **CJ 외부 데이터**: CJ에는 상품명 열이 없어 Instacart식 건강 속성을 만들 수 없으므로, CJ 속성은 PB(`BRAND == "Private"`, `private_label`) 하나만 씁니다. CJ 카테고리 텍스트의 건강·맛 키워드는 `cj_keyword_audit`로 기술 통계만 냅니다(모델에는 쓰지 않음).

설계 대부분(재구매/탐색, HIER, 하위집단, H\* 후속 분석)은 앞선 결과를 본 뒤 정한 사후(post-hoc) 분석이며, 파라미터는 fit split에서만 고르고 eval split으로 보고합니다.

## 데이터 배치

원본 데이터는 Git에 포함하지 않습니다(`data/raw/`, `data/processed/`, `artifacts/`는 `.gitignore`). ZIP 이름은 자유이며 프로그램이 ZIP 내부 CSV 목록으로 데이터셋을 식별합니다.

```text
data/raw/
├── instacart/<archive>.zip          # orders.csv, order_products__prior.csv, order_products__train.csv,
│                                    # products.csv, aisles.csv, departments.csv
├── complete_journey/<archive>.zip   # transaction_data.csv, product.csv (최소)
└── mapping/product_ingredient_mapping_v2.parquet
```

- 데이터 출처: [Kaggle Instacart Market Basket Analysis](https://www.kaggle.com/competitions/basket-analysis), [dunnhumby Source Files - The Complete Journey](https://www.dunnhumby.com/source-files/). 이용 조건은 각 배포 페이지를 따릅니다.
- 재료 유형 매핑(`mapping/product_ingredient_mapping_v2.parquet`, 기대 버전 `2026-11-02.v2.1`)은 별도 프로젝트 `food_recipe_preprocessing_gu`가 만든 Instacart `product_id` → 재료 매핑입니다. `prepare-types`는 `configs/<profile>.yaml`의 `hierarchical.mapping.source` 경로에서 이 파일을 읽어 `data/raw/mapping/`에 복사하고 SHA-256을 확인합니다. source 경로가 없으면 이미 있는 복사본을 경고와 함께 사용합니다.

## 설치

Python 3.10–3.12를 지원합니다.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

패키지를 설치하지 않고 실행할 때는 `PYTHONPATH=src`(PowerShell: `$env:PYTHONPATH = "src"`)를 설정합니다.

## 프로필

`configs/sample.yaml`(Instacart 1,500명, CJ 500명), `configs/medium.yaml`(Instacart 20,000명, CJ 전체), `configs/full.yaml`(전체). 모든 명령은 `--profile`로 프로필을 고르며, 결과 파일 이름 앞에 프로필 이름이 붙습니다(`<p>` = sample, medium, full).

## 실행 순서

먼저 `sample`로 전체 흐름을 확인한 뒤 같은 순서로 `medium`을 실행합니다. 아래 표의 순서가 의존 순서입니다.

| # | 명령 | 선행 명령 | 주 입력 | 출력 |
|---|---|---|---|---|
| 1 | `python -m ml_group9 prepare --dataset all --profile <p>` | - | `data/raw/instacart`, `data/raw/complete_journey` ZIP | `data/processed/<p>/<dataset>/{items,fit_history,fit_target,eval_history,eval_target,attribute_categories}.parquet`, `reports/metrics/<p>_attributes.json` |
| 2 | `python -m ml_group9 prepare-types --profile <p>` | 1 | 매핑 parquet, `data/processed/<p>/*/items.parquet` | `items.parquet`에 유형 열 추가(덮어씀), `reports/metrics/<p>_types.json` |
| 3 | `python -m ml_group9 train --profile <p>` | 1 | Instacart fit split | `artifacts/<p>/models/hybrid_lr.joblib`, `hybrid_lr_pref.joblib`, `artifacts/<p>/training_candidates.parquet`, `reports/metrics/<p>_training.json` |
| 4 | `python -m ml_group9 evaluate --dataset {instacart,complete_journey} --profile <p>` | 3 | eval split, 모델 | `reports/metrics/<p>_<dataset>.json`, `reports/figures/<p>_<dataset>.png`, `artifacts/<p>/predictions/<dataset>_top10.parquet` |
| 5 | `python -m ml_group9 tune-re --profile <p>` | 1 | fit split | `reports/metrics/<p>_re_tuning.json` |
| 6 | `python -m ml_group9 eval-re --dataset {instacart,complete_journey} --profile <p>` | 3, 5 | eval split, 모델, re 튜닝 | `reports/metrics/<p>_re_<dataset>.json` |
| 7 | `python -m ml_group9 tune-hs --profile <p>` | 5 | Instacart fit split, re 튜닝 | `reports/metrics/<p>_hs_tuning.json` |
| 8 | `python -m ml_group9 eval-hs --profile <p>` | 3, 7 | Instacart eval split, 모델, hs 튜닝 | `reports/metrics/<p>_hs_instacart.json` |
| 9 | `python -m ml_group9 tune-hier --profile <p>` | 2, 5 | fit split, re 튜닝(gamma) | `reports/metrics/<p>_hier_tuning.json` |
| 10 | `python -m ml_group9 eval-hier --dataset {instacart,complete_journey} --profile <p>` | 3, 9 | eval split, 모델, hier·re 튜닝 | `reports/metrics/<p>_hier_<dataset>.json` |
| 11 | `python -m ml_group9 analyze-hstar --profile <p>` | 1 | Instacart eval split | `reports/metrics/<p>_hstar.json`, `reports/figures/<p>_hstar_m1.png`, `<p>_hstar_m3_lorenz.png` |
| 12 | `python -m ml_group9.hstar_followup --profile <p> [--reference reports/metrics/<p>_hstar.json]` | 11 | Instacart eval history | `reports/metrics/<p>_hstar_followup.json`, `reports/figures/<p>_hstar_delta.png` |
| 13 | `python -m ml_group9.hstar_robust --profile <p> [--reference reports/metrics/<p>_hstar_followup.json]` | 12 | Instacart eval history | `reports/metrics/<p>_hstar_robust.json` |
| 14 | `python -m ml_group9.hier_followup --profile <p> --dataset {instacart,complete_journey} [--reference-dir reports/metrics]` | 9, 10 | eval split, hier·re 튜닝, `<p>_hier_<dataset>.json` | `reports/metrics/<p>_hier_followup.json`(Instacart), `<p>_hier_followup_complete_journey.json` |
| 15 | `python -m ml_group9.cj_keyword_audit --profile <p>` | - | `data/raw/complete_journey` ZIP | `reports/metrics/cj_keyword_audit.json`, `reports/tables/cj_keyword_matches.csv` (프로필 접두어 없음) |
| 16 | `python -m ml_group9.hs_followup --profile <p>` | 8 | `<p>_hs_instacart.json`, Instacart eval split | `reports/metrics/<p>_hs_followup.json`, `<p>_hs_followup_peruser_<label>.parquet` |

- `--reference`, `--reference-dir`를 주면 이전 결과를 다시 계산해 일치하는지 확인합니다. 생략할 수 있습니다.
- `python -m ml_group9 run --profile <p>`는 1, 3, 4를 한 번에 실행합니다.
- `python -m ml_group9 inspect-cj --profile <p>`는 CJ DEPARTMENT별 거래 비중을 표준 출력으로만 보여 줍니다(파일 없음).
- `hs_followup`은 현재 특정 worktree 경로와 브랜치에서만 실행되도록 고정되어 있습니다(코드 상단 `WORKTREE`, `BRANCH`).
- `medium`, `full`은 실행 시간과 메모리가 많이 듭니다. 특히 `full`은 3천만 건 이상의 interaction으로 Item-CF를 계산합니다.

## 그림 스크립트

두 스크립트는 결과 JSON만 읽고 `ml_group9`를 import하지 않습니다. 저장소 루트에서 실행합니다.

```bash
# H*' 요약 그림과 표: <out-dir>/figures/<prefix>_*.png, HSTARP_FIGURES.md, <out-dir>/tables/<prefix>_*.csv
python scripts/make_hstarp_figures.py \
  --hstar reports/metrics/medium_hstar.json \
  --robust reports/metrics/medium_hstar_robust.json \
  --hier-followup reports/metrics/medium_hier_followup.json \
  --out-dir reports

# 보고서용 그림 3개(fig1_persistence.png, fig2_pooled_vs_within.png, fig3_within_type_auc.png)
python scripts/make_report_figures.py reports/metrics/medium_hstar_robust.json \
  reports/metrics/medium_hier_followup.json reports/figures
```

`make_hstarp_figures.py`의 `--hier-followup`은 생략할 수 있으며, 생략하면 유형 내 AUC 그림을 만들지 않습니다. `--prefix` 기본값은 `medium_hstarp`입니다.

## 시간 분할과 누수 방지

- Instacart fit split: 마지막 `prior` 주문을 정답으로 분리하고 그 이전 주문만 이력으로 사용합니다(학습·튜닝용).
- Instacart eval split: 모든 `prior` 주문을 이력, 공식 `train` 주문을 정답으로 사용합니다.
- CJ: 1~90주를 이력, 91주 이후 사용자의 첫 장바구니를 정답으로 사용합니다.
- 모든 확률·선호·이웃은 해당 split의 이력에서만 계산하고, 정답은 지표 계산에만 씁니다.

## 결과 파일

- `data/processed/<p>/`: 공통 스키마 Parquet와 시간 분할 결과
- `artifacts/<p>/`: 모델, 후보 feature, Top-10 예측
- `reports/metrics/`, `reports/figures/`, `reports/tables/`: 위 표의 출력 파일

`sample` 기준 모델 결과(`reports/metrics/sample_{training,instacart,complete_journey}.json`과 그림 2개)만 Git에서 추적합니다. `medium`·`full` 결과, `reports/tables/`, 보고서 문서(`*.docx`)는 별도 제출하며 `.gitignore`로 제외합니다.

## 코드 구성

- `ml_group9.data`: ZIP 탐색, 공통 스키마 변환, CJ 정제, 시간 분할
- `ml_group9.attributes`: 상품명 기반 속성, 적용 카테고리 C_a, 사용자 선호
- `ml_group9.recommender`, `ml_group9.modeling`, `ml_group9.metrics`, `ml_group9.pipeline`: 기준 모델의 후보·feature, LR 학습, Top-K 지표, 학습·평가
- `ml_group9.repeat_explore`, `ml_group9.re_pipeline`: 재구매/탐색 추천과 튜닝·평가
- `ml_group9.ingredient_types`, `ml_group9.hierarchical`, `ml_group9.hier_pipeline`: 재료 유형, 계층형 추천, 튜닝·평가·단계별 지표
- `ml_group9.health_subgroup`, `ml_group9.hs_followup`: 건강 속성 구매자 하위집단
- `ml_group9.hstar`, `ml_group9.hstar_followup`, `ml_group9.hstar_robust`, `ml_group9.hier_followup`: 건강 속성 지속성과 유형 내 AUC
- `ml_group9.cj_keyword_audit`: CJ 카테고리 텍스트 키워드 집계
- `scripts/`: 결과 JSON에서 보고서 그림을 만드는 스크립트

## 테스트

```bash
python -m pytest
```

`pyproject.toml`이 `pythonpath = ["src"]`를 설정하므로 저장소 루트에서 바로 실행됩니다. 테스트는 대부분 합성 데이터로 시간 누수 방지, 확률 정규화, 손계산 예제, 후보 생성, 지표 수식, 모델 고정 checksum 등을 확인합니다. `integration` 마커 테스트는 `data/processed/sample/`이 있을 때만 실행되고, 없으면 건너뜁니다.

## 사용 라이브러리

- pandas/PyArrow: CSV 읽기, 표 연산, Parquet 저장
- NumPy/SciPy: 수치 계산, sparse 행렬, 최적화
- scikit-learn: 표준화, Logistic Regression, 사용자 단위 분할
- Matplotlib/Seaborn: 그림
- joblib: 모델 직렬화
- PyYAML: 프로필 설정
