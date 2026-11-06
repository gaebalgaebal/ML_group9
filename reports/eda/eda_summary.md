# 데이터 탐색·전처리 요약

## 범위와 계산 기준

medium의 Instacart·Complete Journey(CJ)를 기술한다. 데이터셋 규모·history 길이·바스켓 크기는 eval_history 기준이다. 정답 통계는 eval_target의 기술 통계이며 모델 평가·선택에는 사용하지 않는다.

- 바스켓은 서로 다른 (user_id, basket_seq), 바스켓 크기는 그 안의 서로 다른 item_id 수다. 상품·L1/L2/L3 수는 eval_history에서 실제 관측한 상품과 그 상품의 items 매핑으로 센다. 전체 catalog 크기와 구분한다.
- history 길이의 p10·p90은 pandas 선형 보간 분위수다. 그림 1은 사용자 한 명을 관측값 하나로 한 누적분포(ECDF)이며 x축은 로그 축이다.
- 재구매 상품은 같은 사용자의 history에 있던 상품이다. target의 (user_id, item_id) 중복을 제거하고 사용자별 재구매 상품 수 / 정답 상품 수를 계산한 뒤 사용자에 같은 가중치를 주어 평균낸다. 구간은 왼쪽 열림·오른쪽 닫힘이며 inf는 상한이 없음을 뜻한다.
- 신규 상품의 기존 범주 비율은 신규 target 상품 행에 같은 가중치를 준 비율이다. 사용자 평균과 다르다. medium.json의 eval 값을 그대로 사용하고 원자료 계산과 대조했다.
- Instacart: L1=department(coarse_category), L2=aisle(fine_category), L3=ingredient_type(aisle fallback 포함). CJ: L1=DEPARTMENT, L2=commodity(coarse_category 별칭), L3=fine_category.
- holdout 폴더와 원시 ZIP은 열지 않았다. holdout 추출 수치는 커밋된 data/SOURCE.md 기록만 인용했다.

## 표 1: 관측 history 규모

| dataset | users | baskets | items | L1 | L2 | L3 | history_baskets_min | history_baskets_p10 | history_baskets_median | history_baskets_p90 | history_baskets_max | basket_size_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instacart | 20000 | 313281 | 41250 | 21 | 134 | 3209 | 3 | 3 | 9 | 37 | 99 | 8 |
| complete_journey | 2069 | 206682 | 82354 | 32 | 302 | 2323 | 4 | 25 | 76 | 197.2 | 1167 | 5 |

PPT 표시값: CJ p90 = 197개 바스켓(정수 반올림). 표와 CSV의 원래 계산값 197.2는 유지한다.

- Instacart: 사용자 20,000명, 바스켓 313,281개, 상품 41,250개. 사용자당 history 바스켓 중앙값 9개(p10=3, p90=37), 바스켓별 서로 다른 상품 수 중앙값 8개.
  전체 catalog는 49,688개 상품이며, 위 관측 상품 수와 범위가 다르다.
- Complete Journey: 사용자 2,069명, 바스켓 206,682개, 상품 82,354개. 사용자당 history 바스켓 중앙값 76개(p10=25, p90=197.2), 바스켓별 서로 다른 상품 수 중앙값 5개.
  전체 catalog는 91,691개 상품이며, 위 관측 상품 수와 범위가 다르다.

## 그림 2: history 길이별 정답 재구매 비중

| dataset | history_bin | users | mean_repeat_share | target_items | repeat_items | pooled_repeat_share |
| --- | --- | --- | --- | --- | --- | --- |
| instacart | (0, 4] | 4143 | 0.443009 | 40374 | 16885 | 0.418215 |
| instacart | (4, 6] | 2933 | 0.517845 | 29805 | 14637 | 0.491092 |
| instacart | (6, 10] | 3841 | 0.581116 | 40284 | 22846 | 0.567123 |
| instacart | (10, 20] | 4470 | 0.647133 | 48540 | 31392 | 0.646724 |
| instacart | (20, inf) | 4613 | 0.765314 | 52357 | 40428 | 0.77216 |
| complete_journey | (0, 20] | 138 | 0.23301 | 1977 | 474 | 0.239757 |
| complete_journey | (20, 50] | 547 | 0.33416 | 8217 | 3074 | 0.374102 |
| complete_journey | (50, 100] | 653 | 0.472826 | 8981 | 4464 | 0.497049 |
| complete_journey | (100, inf) | 731 | 0.572478 | 9536 | 5623 | 0.58966 |

구간별 표본 수와 평균을 함께 제시한다. history 길이와 재구매 비중의 관계는 관측 자료의 기술 결과이며 인과관계로 해석하지 않는다. pooled_repeat_share는 전체 상품 행을 합친 검산용 값이며 막대는 mean_repeat_share를 사용한다.

- Instacart: (0, 4] 구간의 사용자 평균 재구매 비중은 44.30%(n=4,143), (20, inf) 구간은 76.53%(n=4,613)다.
- Complete Journey: (0, 20] 구간의 사용자 평균 재구매 비중은 23.30%(n=138), (100, inf) 구간은 57.25%(n=731)다.

## 그림 3: 신규 상품의 기존 범주 포함 비율

| dataset | level | new_target_items | in_seen_category_items | share |
| --- | --- | --- | --- | --- |
| instacart | L1 | 85172 | 75467 | 0.886054 |
| instacart | L2 | 85172 | 51060 | 0.599493 |
| instacart | L3 | 85172 | 17121 | 0.201017 |
| complete_journey | L1 | 15076 | 14928 | 0.990183 |
| complete_journey | L2 | 15076 | 12708 | 0.842929 |
| complete_journey | L3 | 15076 | 8878 | 0.588883 |

- 범주 수준은 데이터셋마다 다르다. Instacart는 department / aisle / ingredient type, CJ는 department / commodity / sub-commodity이므로 데이터셋 간 막대 높이를 직접 비교할 수 없다.
- Instacart L3는 매핑되지 않은 상품에 aisle을 대신 사용한다. medium eval_history 행의 약 19%가 이 fallback에 해당한다(커밋 medium.json의 1 − history_rows_l3_mapped를 정수 %로 반올림).

- Instacart: L1 88.61%, L2 59.95%, L3 20.10%. 분모는 해당 사용자의 history에 없던 신규 정답 상품 85,172개다.
- Complete Journey: L1 99.02%, L2 84.29%, L3 58.89%. 분모는 해당 사용자의 history에 없던 신규 정답 상품 15,076개다.

## 표 5: 전처리 단계

rows_after는 해당 단계 출력 전체 행 수이며 분할 단계에서는 history_rows_after + target_rows_after다. 빈 수치는 0이 아니라 확인할 수 없는 값이다. 기록 확인은 문서 인용이며 원자료 재검증을 뜻하지 않는다.

| dataset | step | input | rule | rows_before | rows_after | users_before | users_after | history_rows_after | target_rows_after | status | note | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instacart | 1. 사용자 선택 | orders.csv; medium 설정 | 공식 train 사용자를 문자열 정렬 후 seed 42로 비복원 추출; medium 20,000명 |  |  | 131209 | 20000 |  |  | 부분 확인 | 추출 전 수는 SOURCE.md 기록; 추출 후 수는 medium JSON; 원시 행 수 확인 필요 | vendor/ml_group9_e11b365/src/ml_group9/data.py:208-314 (prepare_instacart); vendor/ml_group9_e11b365/src/ml_group9/data.py:173-178 (_sample_ids); vendor/ml_group9_e11b365/configs/medium.yaml; data/SOURCE.md; reports/data_check/medium.json |
| instacart | 2. 공통 스키마·계층 | prior/train 상품 행; products/aisles/departments | order 메타와 상품 계층을 조인; user_id·item_id·basket_id는 문자열, basket_seq는 int32, quantity=1 |  |  |  |  |  |  | 규칙 확인·수치 확인 필요 | 원시 단계별 행·사용자 수는 허용 자료에서 확인할 수 없음 | vendor/ml_group9_e11b365/src/ml_group9/data.py:208-314 (prepare_instacart) |
| instacart | 3. eval 분할 | 선택 사용자의 prior 및 공식 train 주문 | eval_history=모든 prior 주문; eval_target=공식 train 주문 |  | 3355343 |  | 20000 | 3143983 | 211360 | 부분 확인 | 분할 후 수는 medium JSON; 전체 원시 입력 행 수 확인 필요 | vendor/ml_group9_e11b365/src/ml_group9/data.py:208-314 (prepare_instacart); reports/data_check/medium.json |
| instacart | 4. fit 분할 | eval_history의 prior 주문 | 마지막 prior 주문=fit_target; 이전 주문=fit_history; fit_history 바스켓 2개 이상인 사용자 유지 | 3143983 | 3143983 | 20000 | 20000 | 2937801 | 206182 | 확인 | 분할 전후 합계는 medium JSON; 원시 단계별 로그를 뜻하지 않음 | vendor/ml_group9_e11b365/src/ml_group9/data.py:208-314 (prepare_instacart); reports/data_check/medium.json |
| complete_journey | 5. 미등록 상품 제거 | transaction_data.csv + product.csv; 앞 단계 통과 행 | PRODUCT_ID가 product.csv에 없는 행 제거 |  |  |  |  |  |  | 규칙 확인·수치 확인 필요 | 원시 단계별 행·사용자 수는 허용 자료에서 확인할 수 없음 | vendor/ml_group9_e11b365/src/ml_group9/data.py:94-122 (clean_complete_journey); vendor/ml_group9_e11b365/configs/medium.yaml |
| complete_journey | 6. 부서 누락 제거 | transaction_data.csv + product.csv; 앞 단계 통과 행 | 공백 DEPARTMENT를 결측으로 바꾼 후 부서 결측 행 제거 |  |  |  |  |  |  | 규칙 확인·수치 확인 필요 | 원시 단계별 행·사용자 수는 허용 자료에서 확인할 수 없음 | vendor/ml_group9_e11b365/src/ml_group9/data.py:94-122 (clean_complete_journey); vendor/ml_group9_e11b365/configs/medium.yaml; vendor/ml_group9_e11b365/src/ml_group9/data.py:69-91 (_read_complete_journey) |
| complete_journey | 7. 제외 부서 제거 | transaction_data.csv + product.csv; 앞 단계 통과 행 | DEPARTMENT가 다음 값이면 제거: KIOSK-GAS; MISC SALES TRAN; MISC. TRANS.; COUP/STR & MFG; AUTOMOTIVE; GM MERCH EXP; CNTRL/STORE SUP; MEAT-WHSE; PHARMACY SUPPLY; ELECT &PLUMBING; CHARITABLE CONT |  |  |  |  |  |  | 규칙 확인·수치 확인 필요 | 원시 단계별 행·사용자 수는 허용 자료에서 확인할 수 없음 | vendor/ml_group9_e11b365/src/ml_group9/data.py:94-122 (clean_complete_journey); vendor/ml_group9_e11b365/configs/medium.yaml |
| complete_journey | 8. 수량 필터 | transaction_data.csv + product.csv; 앞 단계 통과 행 | QUANTITY <= 0 제거; 양수 수량이면 SALES_VALUE <= 0인 행도 유지 |  |  |  |  |  |  | 규칙 확인·수치 확인 필요 | 원시 단계별 행·사용자 수는 허용 자료에서 확인할 수 없음 | vendor/ml_group9_e11b365/src/ml_group9/data.py:94-122 (clean_complete_journey); vendor/ml_group9_e11b365/configs/medium.yaml |
| complete_journey | 9. 바스켓 정렬 | 정제 거래 행 | household_key, DAY, TRANS_TIME, BASKET_ID 순 정렬; 가구별 basket_seq 부여; 공통 스키마로 변환 |  |  |  |  |  |  | 규칙 확인·수치 확인 필요 | 원시 단계별 행·사용자 수는 허용 자료에서 확인할 수 없음 | vendor/ml_group9_e11b365/src/ml_group9/data.py:317-335 (_complete_journey_common) |
| complete_journey | 10. 사용자·정답 적격성 | 90주 전후 거래가 있는 가구 | 각 분할 history 바스켓 >=3; 미래의 서로 다른 상품 >=3인 최초 바스켓을 정답으로 선택; fit/eval 모두 가능한 가구의 교집합; 필터 후 표본 추출(null=전체) |  |  |  | 2069 |  |  | 부분 확인 | 최종 가구 수만 medium JSON으로 확인; 단계별 탈락 수 확인 필요 | vendor/ml_group9_e11b365/src/ml_group9/data.py:382-472 (prepare_complete_journey); vendor/ml_group9_e11b365/src/ml_group9/data.py:338-353 (_history_and_target); vendor/ml_group9_e11b365/src/ml_group9/data.py:125-131 (first_future_basket); vendor/ml_group9_e11b365/configs/medium.yaml; reports/data_check/medium.json |
| complete_journey | 11. fit 주차 분할 | 적격 가구의 정제 거래 행 | history WEEK_NO<=80; target=(80, 90]의 최초 적격 바스켓 |  | 1853327 |  | 2069 | 1822633 | 30694 | 부분 확인 | 분할 후 수는 medium JSON; 분할 전 정제 거래 수 확인 필요 | vendor/ml_group9_e11b365/src/ml_group9/data.py:356-379 (_complete_journey_splits); vendor/ml_group9_e11b365/configs/medium.yaml; reports/data_check/medium.json |
| complete_journey | 12. eval 주차 분할 | 적격 가구의 정제 거래 행 | history WEEK_NO<=90; target=90주 이후의 최초 적격 바스켓 |  | 2125351 |  | 2069 | 2096640 | 28711 | 부분 확인 | 분할 후 수는 medium JSON; 분할 전 정제 거래 수 확인 필요 | vendor/ml_group9_e11b365/src/ml_group9/data.py:356-379 (_complete_journey_splits); vendor/ml_group9_e11b365/configs/medium.yaml; reports/data_check/medium.json |
| instacart_holdout_rule_only | 13. 기존 사용자 제외 | 공식 train 적격 사용자 | medium 또는 sample에 속한 사용자 합집합을 추출 전에 제외 |  |  | 131209 | 109936 |  |  | 기록 확인 | SOURCE.md 기록: 기존 사용자 21,273명 제외; holdout 파일은 열지 않음; 행 수 확인 필요 | data/SOURCE.md |
| instacart_holdout_rule_only | 14. holdout 사용자 추출 | 기존 사용자 제외 후 풀 | 문자열 정렬된 잔여 사용자에서 seed 42로 20,000명 비복원 추출; medium/sample 중복 0명(문서 기록) |  |  | 109936 | 20000 |  |  | 기록 확인 | SOURCE.md 기록만 인용; holdout 원자료 검증 및 분할 행 수 확인 필요 | data/SOURCE.md; vendor/ml_group9_e11b365/src/ml_group9/data.py:173-178 (_sample_ids) |

## 검산

| dataset | history_rows | target_rows | duplicate_history_user_basket_item | duplicate_target_user_item | target_users | json_repeat_share | json_counts_match | level_ratios_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instacart | 3143983 | 211360 | 0 | 0 | 20000 | 0.597029 | True | True |
| complete_journey | 2096640 | 28711 | 0 | 0 | 2069 | 0.474905 | True | True |

medium eval history/target 행 수·사용자 수 및 전체 target 재구매 행 비중을 커밋 JSON과 대조했다. 신규 상품의 L1/L2/L3 비율 6개는 전체 medium 자료로 재계산해 절대오차 1e-12 이내 일치를 확인했다. 각 그림은 저장한 동명 CSV를 다시 읽어 그린다.

## 확인 필요

- 원시 Instacart 조인·샘플링 전후 상품 행 수와 CJ의 미등록 상품·부서 누락·제외 부서·수량 필터별 제거 행 수 및 사용자 수. 허용된 자료에 해당 단계별 로그가 없어 CSV에서 빈칸으로 두었다.
- CJ의 주차 경계 통과, history 최소 바스켓 수, 정답 최소 상품 수, fit/eval 교집합 조건 각각으로 탈락한 가구 수와 정제 거래 전체 행 수.
- data/SOURCE.md는 기존 medium/sample 생성 코드 버전을 파일 시각으로 추정한다고 명시한다. vendor e11b365는 확인한 규칙의 근거이며 복사본의 모든 생성 단계를 확정하는 실행 로그는 아니다.
- holdout의 131,209 → 109,936 → 20,000명 및 기존 사용자 21,273명 제외·중복 0명은 SOURCE.md 기록이다. 이 작업에서는 holdout 파일을 열거나 해당 수치를 재검증하지 않았다.

## 재현과 출처

```powershell
$env:NBR_DATA_DIR = "<processed-data folder>"   # default: data/processed
python scripts/make_eda.py
```

pandas 2.2.3; matplotlib 3.9.2. 제3자 라이브러리를 직접 불러오는 것은 이 두 패키지뿐이며 nbr에 의존하지 않는다.

- `data/SOURCE.md` (HEAD에 커밋된 내용), SHA-256 `68fa2d527662b22bb54cf85a3ca05c97156bb404b621ea61482552bf4e1fbef2`
- `vendor/ml_group9_e11b365/src/ml_group9/data.py` (HEAD에 커밋된 내용), SHA-256 `38acb04cf1d6376eabdab431365298819c0cdfd5e929446df480a8945abb4fc7`
- `vendor/ml_group9_e11b365/configs/medium.yaml` (HEAD에 커밋된 내용), SHA-256 `d6edfa6e8d63aa348a34df8daa264a7198cf83e7b2e6660a8adba082a58b1f17`
- `reports/data_check/medium.json` (HEAD에 커밋된 내용), SHA-256 `c919b380c87b7ffa0409bdc5c8f3457a85e3f225823ae660a610ca235ef0b9dd`

입력 Parquet은 NBR_DATA_DIR/medium/{instacart,complete_journey}/{items,eval_history,eval_target}.parquet 여섯 파일로 제한한다. fit 수치는 medium.json만 이용한다.

## 산출물

- table_datasets.csv
- table_preprocessing.csv
- fig_history_length.png / fig_history_length.csv
- fig_repeat_share_by_history.png / fig_repeat_share_by_history.csv
- fig_new_item_level.png / fig_new_item_level.csv
