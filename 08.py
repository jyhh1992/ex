"""
빅데이터 분석기사 실기 - 시험 빈출 패턴 & 빠른 참조

[작업형 1유형] 데이터 처리 결과값 출력 (숫자 1~2개)
[작업형 2유형] 머신러닝 모델로 예측 후 CSV 제출
[작업형 3유형] 통계 검정 결과 해석
"""

import pandas as pd
import numpy as np
from scipy import stats

# ============================================================
# 작업형 1유형 - 자주 나오는 문제 유형
# ============================================================

df = pd.read_csv('data.csv')

# Q: 특정 조건을 만족하는 행의 특정 컬럼 통계값
# 예: 나이가 30 이상인 남성의 소득 중앙값
result = df[(df['age'] >= 30) & (df['gender'] == 'M')]['income'].median()
print(round(result, 2))

# Q: 그룹별 통계 후 특정 그룹 값
result = df.groupby('group')['value'].mean()
print(round(result['A'], 2))

# Q: 이상치 제거 후 평균
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
clean = df[(df['col'] >= Q1 - 1.5*IQR) & (df['col'] <= Q3 + 1.5*IQR)]
print(round(clean['col'].mean(), 2))

# Q: 결측치를 중앙값으로 대체 후 합계
df['col'] = df['col'].fillna(df['col'].median())
print(df['col'].sum())

# Q: 특정 컬럼 상위 N개의 합계
print(df['col'].nlargest(10).sum())

# Q: 비율 계산
ratio = (df['col'] == '특정값').sum() / len(df) * 100
print(round(ratio, 2))

# Q: 두 컬럼의 상관계수
corr = df['col1'].corr(df['col2'])
print(round(corr, 4))

# ============================================================
# 작업형 2유형 - 완전한 제출 템플릿
# ============================================================

def type2_template():
    """작업형 2유형 완전 템플릿"""
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    # 1. 데이터 로드
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')

    # 2. ID 저장 및 타겟 분리
    test_id = test['id'] if 'id' in test.columns else test.index
    target  = 'target'
    X       = train.drop(columns=[target])
    y       = train[target]

    # 3. train + test 합쳐서 전처리
    n_train = len(X)
    all_X   = pd.concat([X, test], axis=0, ignore_index=True)

    # 4. 결측치 처리
    num_cols = all_X.select_dtypes(include='number').columns
    cat_cols = all_X.select_dtypes(include='object').columns

    for col in num_cols:
        all_X[col].fillna(all_X[col].median(), inplace=True)
    for col in cat_cols:
        all_X[col].fillna(all_X[col].mode()[0], inplace=True)

    # 5. 범주형 인코딩
    for col in cat_cols:
        le = LabelEncoder()
        all_X[col] = le.fit_transform(all_X[col].astype(str))

    # 6. 다시 분리
    X_processed    = all_X.iloc[:n_train]
    test_processed = all_X.iloc[n_train:]

    # 7. train/val 분리
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # 8. 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 9. 평가
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred_proba)
    print(f"Validation AUC: {auc:.4f}")

    # 10. 제출
    test_pred = model.predict_proba(test_processed)[:, 1]
    submission = pd.DataFrame({'id': test_id, 'target': test_pred})
    submission.to_csv('submission.csv', index=False)
    print(submission.head())


# ============================================================
# 작업형 3유형 - 검정 선택 가이드
# ============================================================

"""
[검정 선택 가이드]

데이터 종류       │ 집단 수    │ 정규성   │ 검정 방법
─────────────────┼───────────┼─────────┼──────────────────────
연속형(모평균)   │ 1집단      │ O       │ 단일 표본 t-검정
연속형(모평균)   │ 2집단(독립)│ O       │ 독립 표본 t-검정
연속형(모평균)   │ 2집단(대응)│ O       │ 대응 표본 t-검정
연속형(모평균)   │ 3집단 이상 │ O       │ 일원 분산분석(ANOVA)
연속형           │ 2집단(독립)│ X       │ Mann-Whitney U
연속형           │ 2집단(대응)│ X       │ Wilcoxon 부호순위
연속형           │ 3집단 이상 │ X       │ Kruskal-Wallis
범주형           │ 빈도 비교  │ -       │ 카이제곱 적합도
범주형×범주형    │ 독립성     │ -       │ 카이제곱 독립성
"""

def select_and_run_test(data1, data2=None, data3=None, paired=False, categorical=False):
    """상황에 맞는 검정 자동 선택"""
    alpha = 0.05

    if categorical:
        # 카이제곱 독립성 검정
        stat, p, dof, expected = stats.chi2_contingency(data1)
        return {'test': 'chi2', 'stat': stat, 'p': p}

    # 정규성 검정
    _, p_norm1 = stats.shapiro(data1)
    normal = p_norm1 >= alpha
    if data2 is not None:
        _, p_norm2 = stats.shapiro(data2)
        normal = normal and (p_norm2 >= alpha)

    if data2 is None:
        # 단일 표본 t-검정
        stat, p = stats.ttest_1samp(data1, popmean=0)
        return {'test': 'one-sample-t', 'stat': stat, 'p': p}
    elif data3 is not None:
        if normal:
            stat, p = stats.f_oneway(data1, data2, data3)
            return {'test': 'ANOVA', 'stat': stat, 'p': p}
        else:
            stat, p = stats.kruskal(data1, data2, data3)
            return {'test': 'Kruskal-Wallis', 'stat': stat, 'p': p}
    elif paired:
        if normal:
            stat, p = stats.ttest_rel(data1, data2)
            return {'test': 'paired-t', 'stat': stat, 'p': p}
        else:
            stat, p = stats.wilcoxon(data1, data2)
            return {'test': 'Wilcoxon', 'stat': stat, 'p': p}
    else:
        if normal:
            _, p_lev = stats.levene(data1, data2)
            eq_var = p_lev >= alpha
            stat, p = stats.ttest_ind(data1, data2, equal_var=eq_var)
            return {'test': 'independent-t', 'stat': stat, 'p': p}
        else:
            stat, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            return {'test': 'Mann-Whitney', 'stat': stat, 'p': p}


# ============================================================
# 자주 나오는 pandas 조작 치트시트
# ============================================================

# 조건부 집계 (SQL의 HAVING)
result = df.groupby('group').filter(lambda x: len(x) > 10)

# 순위 매기기
df['rank'] = df['score'].rank(ascending=False, method='dense')

# 누적합 / 누적평균
df['cumsum']  = df['value'].cumsum()
df['cummean'] = df['value'].expanding().mean()

# 행/열 방향 합계 (여러 컬럼)
df['row_sum']  = df[['col1', 'col2', 'col3']].sum(axis=1)
df['row_mean'] = df[['col1', 'col2', 'col3']].mean(axis=1)

# 데이터 재구조화
df_wide = df.pivot(index='id', columns='variable', values='value')
df_long = df.melt(id_vars=['id'], value_vars=['col1', 'col2'])

# 문자열 분리 후 첫 번째 요소
df['first'] = df['name'].str.split(' ').str[0]

# 값 치환
df['col'] = df['col'].map({'A': 1, 'B': 2, 'C': 3})
df['col'] = df['col'].replace({'Yes': 1, 'No': 0})

# 특정 조건에서 값 대입
df.loc[df['age'] < 0, 'age'] = np.nan
df.loc[df['col'] > 100, 'col'] = 100

# 여러 컬럼 동시 결측치 확인
print(df[df.isnull().any(axis=1)])  # 하나라도 결측인 행

# 중복 행 상세 확인
print(df[df.duplicated(keep=False)])  # 중복 행 전체 표시

# 데이터 샘플링
df_sample = df.sample(frac=0.1, random_state=42)   # 10% 샘플
df_sample = df.sample(n=100, random_state=42)       # 100개 샘플

# ============================================================
# 시험 팁
# ============================================================
"""
[시험 주의사항]
1. 결과값 반올림: round(result, 2) 또는 round(result, 4)
2. 인덱스 리셋: reset_index(drop=True)
3. 경고 무시: import warnings; warnings.filterwarnings('ignore')
4. 제출 파일: to_csv('submission.csv', index=False)
5. 작업형 2유형 제출 형식: id 컬럼 + 예측값(확률 또는 클래스)
6. 분류 문제 기본 평가: ROC-AUC (이진), F1(다중)
7. 회귀 문제 기본 평가: RMSE
8. stratify=y 옵션으로 클래스 비율 유지
9. 결측치는 train 기준 통계로 test도 채우기
10. 스케일러는 train에만 fit, test는 transform만
"""
