# TabDDPM 기반 Tabular Auto-labeling Engine 설계서
# (Cursor가 이 문서를 읽고 자동으로 코드를 생성하는 것을 목표로 함)

---

## 1. 프로젝트 개요

- 프로젝트명: **TabDDPM 기반 합성데이터 Few-shot Auto-labeling Engine**
- 목적:
  - 라벨이 적은(tabular) 데이터셋에서 TabDDPM 기반 합성 데이터를 활용하여 **자동 라벨링 + 분류 모델 학습**까지 한 번에 수행하는 엔진을 만드는 것임.
  - 최종적으로는, 사용자가 **CSV 파일만 넣으면**:
    1) 전처리
    2) TabDDPM 학습
    3) 합성 데이터 생성
    4) Few-shot 기반 자동 라벨링
    5) 최종 분류 모델 학습·평가
    를 자동으로 실행하는 파이프라인이 돌아가도록 하는 것이 목표임.

- 연구/구현 키워드:
  - TabDDPM (Tabular Diffusion Model)
  - Synthetic Data Generation
  - Few-shot Labeling / Semi-supervised Learning
  - Auto-labeling Pipeline
  - Tabular Classification (XGBoost, LightGBM 등)

---

## 2. 데이터 가정 및 입출력 스펙

### 2.1 입력 데이터 가정

- 입력 파일: `data/input.csv`
- 구조:
  - **행(row)**: 샘플
  - **열(column)**: 피처 + 라벨
- 컬럼 규칙:
  - `target`:
    - 분류 문제의 정답 라벨 컬럼명으로 고정
    - 일부 행은 `target` 값이 비어 있거나 `NaN`일 수 있음 → **unlabeled 데이터**
  - 나머지 컬럼:
    - 수치형(numeric) + 범주형(categorical) 혼합 가능
- 예시 컬럼:
  - `age`, `income`, `channel`, `campaign_type`, `clicks`, `impressions`, `time_on_site`, ..., `target`

### 2.2 출력 결과

1. **합성 데이터 파일**
   - 경로: `outputs/synthetic_data.csv`
   - 내용: TabDDPM이 생성한 합성 탭울러 데이터 (라벨 포함)

2. **자동 라벨링 결과**
   - 경로: `outputs/auto_labeled_data.csv`
   - 내용: 원본 데이터 + (모델이 부여한 pseudo-label, confidence score 컬럼 포함)

3. **최종 학습 데이터**
   - 경로: `outputs/train_final.csv`
   - 내용: 실제 라벨 + 고신뢰 pseudo-label + 합성데이터를 포함하는 최종 학습용 데이터셋

4. **모델 및 지표**
   - `models/tabddpm/` : 학습된 TabDDPM 체크포인트
   - `models/classifier/` : 최종 분류 모델 (예: XGBoost or LightGBM)
   - `reports/metrics.json` : Accuracy, F1, Precision, Recall 등
   - `reports/feature_importance.csv` (선택)

---

## 3. 전체 파이프라인 개요

전체 플로우는 다음과 같이 DAG 형태로 구성함 (Airflow까지는 필수 아님, 우선 Python 모듈/함수 기준으로 설계).

1. `load_data`  
2. `split_labeled_unlabeled`  
3. `preprocess_tabular` (수치형/범주형 처리, 스케일링, 인코딩)
4. `train_tabddpm` (labeled 데이터 + 조건부 라벨 y 사용)
5. `generate_synthetic_samples` (클래스별 합성 데이터 생성)
6. `build_fewshot_autolabel_engine`
   - 6-1. `train_base_classifier` (실제 소량 라벨 + 합성데이터)
   - 6-2. `pseudo_label_unlabeled` (unlabeled에 자동 라벨/신뢰도 부여)
   - 6-3. `filter_high_confidence_samples` (threshold 기반)
7. `train_final_classifier` (실제 라벨 + 고신뢰 pseudo-label + 합성데이터)
8. `evaluate_and_save_reports`

각 스텝은 독립적인 함수로 구성하여, 나중에 Airflow로 옮기기 쉽게 설계함.

---

## 4. 모듈 및 디렉토리 구조 제안

Cursor가 이 구조를 기준으로 자동 생성/보완할 수 있도록, 기본 구조를 명시함.

```text
project_root/
  data/
    input.csv
  src/
    __init__.py
    config.py
    data_loader.py
    preprocessing.py
    tabddpm_trainer.py
    synthetic_generator.py
    autolabel_engine.py
    classifier_trainer.py
    evaluation.py
    pipeline.py
  models/
    tabddpm/
    classifier/
  outputs/
    synthetic_data.csv
    auto_labeled_data.csv
    train_final.csv
  reports/
    metrics.json
  run_pipeline.py
```

각 파일의 책임은 아래에 정의함.

---

## 5. 모듈별 요구사항 (코딩 가이드)

### 5.1 `config.py`

- 공통 설정값 관리
  - 랜덤 시드
  - 입력/출력 경로
  - TabDDPM 하이퍼파라미터
  - 분류 모델 하이퍼파라미터
  - pseudo-label confidence threshold 등

- 예시 항목:
  - `DATA_PATH = "data/input.csv"`
  - `SYNTHETIC_OUTPUT_PATH = "outputs/synthetic_data.csv"`
  - `AUTO_LABELED_OUTPUT_PATH = "outputs/auto_labeled_data.csv"`
  - `FINAL_TRAIN_PATH = "outputs/train_final.csv"`
  - `CONFIDENCE_THRESHOLD = 0.9`

### 5.2 `data_loader.py`

- 기능:
  - CSV 로드 (pandas)
  - 기본적인 null 체크, info 로그 출력
- 주요 함수:
  - `load_data(path: str) -> pd.DataFrame`

### 5.3 `preprocessing.py`

- 기능:
  - `target` 컬럼 분리
  - 수치형 / 범주형 자동 구분
  - 범주형 One-Hot 인코딩 혹은 Target Encoding (단, TabDDPM 구현체에 맞게 처리)
  - 수치형 표준화/정규화
- 주요 함수:
  - `split_labeled_unlabeled(df, target_col="target") -> (df_labeled, df_unlabeled)`
  - `fit_preprocessor(df_labeled) -> preprocessor` (ex: sklearn ColumnTransformer)
  - `transform(preprocessor, df) -> np.ndarray`

- 요구사항:
  - TabDDPM의 입력은 **모든 피처 + 라벨(y)을 포함하는 텐서**가 되도록 설계 (단, 라벨은 조건부로 넣을 수 있도록 별도 처리 가능)
  - 역변환(디코딩)을 위해 preprocessor 객체를 저장할 것 (pickle 등)

### 5.4 `tabddpm_trainer.py`

- 기능:
  - TabDDPM 모델 초기화, 학습, 저장/로드
- 구현 방식:
  - 실제 TabDDPM 구현체는 다음 중 하나를 사용하거나, 단순 placeholder 클래스를 만들어둘 수 있음
    - 예: `ydata-synthetic`, `sdv`, 혹은 직접 구현한 Diffusion 모델 래퍼
- 필수 함수 (인터페이스 기준만 맞추면 됨):
  - `train_tabddpm(X: np.ndarray, y: np.ndarray, config) -> TabDDPMModel`
  - `save_tabddpm(model, path: str)`
  - `load_tabddpm(path: str) -> TabDDPMModel`

- `TabDDPMModel` 클래스 요구사항:
  - `sample(num_samples: int, class_condition: Optional[int or array] = None) -> np.ndarray`
    - 클래스별 합성 생성이 가능하도록 `class_condition` 인자를 받을 수 있어야 함.

### 5.5 `synthetic_generator.py`

- 기능:
  - 학습된 TabDDPM으로부터 합성 데이터를 생성하고, 디코딩하여 DataFrame으로 반환
- 주요 함수:
  - `generate_synthetic_data(model, preprocessor, n_per_class: int, classes: List[int]) -> pd.DataFrame`

- 로직 개요:
  1. `for c in classes:`
     - `model.sample(n_per_class, class_condition=c)` 호출
  2. 합성된 피처를 preprocessor의 inverse_transform으로 되돌림
  3. `target` 컬럼을 해당 클래스 값으로 채움
  4. 모든 클래스를 concat 후 `outputs/synthetic_data.csv`로 저장

### 5.6 `autolabel_engine.py`

- 기능:
  - Few-shot + 합성데이터를 활용한 자동 라벨링 엔진
- 주요 함수:
  - `train_base_classifier(X_labeled, y_labeled, config) -> ClassifierModel`
  - `pseudo_label_unlabeled(classifier, X_unlabeled, threshold) -> (pseudo_labels, confidences)`
  - `build_autolabeled_dataset(df_labeled, df_unlabeled, pseudo_labels, confidences, threshold) -> pd.DataFrame`

- 로직 개요:
  1. (옵션) **Few-shot 설정**:
     - labeled 데이터 중 소량만 사용하거나, 클래스별 최소 샘플 수만 사용하도록 샘플링 옵션 추가
  2. labeled + synthetic 데이터를 묶어서 base classifier 학습
  3. unlabeled 데이터에 대해 예측 확률 추론
  4. `max_prob >= CONFIDENCE_THRESHOLD` 인 샘플만 pseudo-label 채택
  5. pseudo-label이 붙은 행만 골라 labeled 데이터셋에 추가
  6. 결과를 `auto_labeled_data.csv`로 저장

### 5.7 `classifier_trainer.py`

- 기능:
  - 최종 분류 모델 학습
- 주요 함수:
  - `train_final_classifier(X_train, y_train, config) -> ClassifierModel`
  - `evaluate_classifier(model, X_test, y_test) -> Dict[str, float]`
  - `save_classifier(model, path)`

- 모델 후보:
  - XGBoost, LightGBM, RandomForest 등
  - 첫 버전은 `XGBoostClassifier`로 고정해도 무방함.

### 5.8 `evaluation.py`

- 기능:
  - 성능지표 계산 및 저장
- 주요 함수:
  - `compute_metrics(y_true, y_pred) -> Dict[str, float]` (Accuracy, Precision, Recall, F1)
  - (선택) ROC-AUC, confusion matrix 등 추가
  - 결과를 `reports/metrics.json`으로 저장

---

## 6. 파이프라인 오케스트레이션 (`pipeline.py`, `run_pipeline.py`)

### 6.1 `pipeline.py`

- `run_full_pipeline()` 함수 하나로 모든 단계를 실행하도록 설계함.

의사코드 구조 예시 (Cursor가 실제 코드로 변환):

```python
def run_full_pipeline():
    # 1. 데이터 로드
    df = load_data(DATA_PATH)

    # 2. 라벨/비라벨 분리
    df_labeled, df_unlabeled = split_labeled_unlabeled(df, target_col="target")

    # 3. 전처리 학습 & 변환
    preprocessor = fit_preprocessor(df_labeled.drop(columns=["target"]))
    X_labeled = transform(preprocessor, df_labeled.drop(columns=["target"]))
    y_labeled = df_labeled["target"].values

    X_unlabeled = transform(preprocessor, df_unlabeled.drop(columns=["target"]))

    # 4. TabDDPM 학습
    tabddpm_model = train_tabddpm(X_labeled, y_labeled, config)

    # 5. 합성 데이터 생성
    classes = sorted(df_labeled["target"].unique())
    df_synth = generate_synthetic_data(tabddpm_model, preprocessor, n_per_class=CONFIG.N_SYNTH_PER_CLASS, classes=classes)

    # 6. Few-shot + 합성 데이터 기반 base classifier 학습
    #    (df_labeled + df_synth 사용)
    df_labeled_aug = concat_labeled_and_synthetic(df_labeled, df_synth)
    X_labeled_aug, y_labeled_aug = preprocess_for_classifier(df_labeled_aug)

    base_clf = train_base_classifier(X_labeled_aug, y_labeled_aug, config)

    # 7. unlabeled에 pseudo-label 부여
    pseudo_labels, confidences = pseudo_label_unlabeled(base_clf, X_unlabeled, threshold=CONFIG.CONFIDENCE_THRESHOLD)

    # 8. 고신뢰 샘플만 채택하여 최종 학습 데이터 구성
    df_train_final = build_autolabeled_dataset(df_labeled_aug, df_unlabeled, pseudo_labels, confidences, CONFIG.CONFIDENCE_THRESHOLD)
    df_train_final.to_csv(FINAL_TRAIN_PATH, index=False)

    # 9. 최종 classifier 학습 및 평가
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(...)
    final_clf = train_final_classifier(X_train_final, y_train_final, config)

    metrics = evaluate_classifier(final_clf, X_test_final, y_test_final)
    save_metrics(metrics, "reports/metrics.json")
```

### 6.2 `run_pipeline.py`

- 단순 실행 스크립트:
  - `if __name__ == "__main__": run_full_pipeline()` 형태로 구현.

---

## 7. 실험 설정 및 TODO

### 7.1 최소 구현 버전 (MVP)

1. TabDDPM 부분은 라이브러리 래퍼를 사용해도 됨
   - ex) `TabDDPMModel` 클래스를 **placeholder**로 두고, 나중에 실제 구현 대체 가능
2. 합성 데이터 생성 → XGBoost 단일 모델 학습 → pseudo-label → 최종 classifier 학습까지 한 번이라도 end-to-end로 돌아가게 하는 것이 1차 목표임.
3. 처음에는:
   - 모든 라벨이 있는 데이터셋을 사용해, 일부분만 라벨을 지운 버전을 만들어 **semi-supervised 상황을 시뮬레이션**해도 됨.

### 7.2 향후 확장 아이디어 (주석 수준으로만 명시)

- class imbalance가 심한 케이스에서, 소수 클래스에 더 많은 synthetic sample을 생성하도록 weight 조정
- confidence threshold를 여러 값(0.7, 0.8, 0.9)로 바꾸며 성능 비교
- TabDDPM vs CTGAN 성능 비교 모듈 추가
- Airflow DAG로 파이프라인 이식

---

## 8. 구현 스타일 가이드 (Cursor 참고용)

- 타입힌트 적극 사용 (Python 3.10+ 기준)
- 함수 단위는 **단일 책임 원칙**에 맞게 설계
- 로깅은 `logging` 모듈 사용 (print 최소화)
- 설정값은 `config.py` 또는 `.env`에서 관리
- 외부 의존 라이브러리는 `requirements.txt`에 정리

---

이 문서는 **"기획서 + 코딩 명세서" 역할**을 동시에 수행함.  
Cursor는 이 문서를 읽고:
- 디렉토리 구조 생성
- 각 파일 스켈레톤 코드 작성
- 주요 함수 시그니처/로직 골격 작성
- TODO 주석으로 세부 구현 포인트 표시
까지 자동 생성하도록 사용함.
