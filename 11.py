"""
빅데이터 분석기사 실기 - 실전 예제 (기출 유사 문제)
각 유형별 처음부터 끝까지 완전한 예제
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# [작업형 1유형] 예제 1
# 문제: 나이가 30 이상인 고객 중 소득이 상위 25%에 해당하는
#       고객의 평균 지출액을 구하시오. (소수점 2자리)
# ============================================================

def type1_example1():
    # 데이터 생성 (시험에서는 read_csv)
    np.random.seed(42)
    df = pd.DataFrame({
        'age':    np.random.randint(20, 70, 500),
        'income': np.random.normal(5000, 1500, 500),
        'spend':  np.random.normal(2000, 500, 500)
    })

    # 풀이
    df_filtered = df[df['age'] >= 30]
    threshold   = df_filtered['income'].quantile(0.75)
    result      = df_filtered[df_filtered['income'] >= threshold]['spend'].mean()

    print(round(result, 2))


# ============================================================
# [작업형 1유형] 예제 2
# 문제: 결측치를 평균으로 채운 후 이상치(IQR)를 제거했을 때
#       'value' 컬럼의 중앙값을 구하시오.
# ============================================================

def type1_example2():
    np.random.seed(0)
    df = pd.DataFrame({'value': np.random.normal(50, 10, 200)})
    df.loc[np.random.choice(200, 20, replace=False), 'value'] = np.nan
    df.loc[np.random.choice(200, 5, replace=False), 'value'] = 200

    # 결측치 → 평균
    df['value'].fillna(df['value'].mean(), inplace=True)

    # 이상치 제거
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[(df['value'] >= Q1 - 1.5*IQR) & (df['value'] <= Q3 + 1.5*IQR)]

    result = df_clean['value'].median()
    print(round(result, 2))


# ============================================================
# [작업형 2유형] 예제 - 이진 분류 (타이타닉 스타일)
# ============================================================

def type2_classification_example():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    # 데이터 생성
    np.random.seed(42)
    n = 800
    train = pd.DataFrame({
        'age':      np.random.randint(1, 80, n),
        'fare':     np.random.exponential(30, n),
        'pclass':   np.random.choice([1, 2, 3], n),
        'sex':      np.random.choice(['male', 'female'], n),
        'embarked': np.random.choice(['S', 'C', 'Q', np.nan], n),
        'survived': np.random.choice([0, 1], n, p=[0.6, 0.4])
    })
    test = train.drop('survived', axis=1).sample(200, random_state=1).reset_index(drop=True)
    test.insert(0, 'id', range(len(test)))

    # ── 전처리 ──
    target = 'survived'
    X = train.drop(columns=[target])
    y = train[target]
    n_train = len(X)

    all_X = pd.concat([X, test.drop('id', axis=1)], ignore_index=True)

    # 결측치
    all_X['age'].fillna(all_X['age'].median(), inplace=True)
    all_X['fare'].fillna(all_X['fare'].median(), inplace=True)
    all_X['embarked'].fillna(all_X['embarked'].mode()[0], inplace=True)

    # 인코딩
    for col in ['sex', 'embarked']:
        le = LabelEncoder()
        all_X[col] = le.fit_transform(all_X[col].astype(str))

    X_proc = all_X.iloc[:n_train]
    test_proc = all_X.iloc[n_train:]

    # ── 모델 학습 ──
    X_train, X_val, y_train, y_val = train_test_split(
        X_proc, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    print(f"Validation AUC: {roc_auc_score(y_val, val_proba):.4f}")

    # ── 제출 ──
    test_proba = model.predict_proba(test_proc)[:, 1]
    submission = pd.DataFrame({'id': test['id'], 'survived': test_proba})
    submission.to_csv('submission.csv', index=False)
    print(submission.head())
    return submission


# ============================================================
# [작업형 2유형] 예제 - 회귀 (주택 가격 스타일)
# ============================================================

def type2_regression_example():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error
    import numpy as np

    np.random.seed(42)
    n = 1000
    train = pd.DataFrame({
        'area':    np.random.normal(100, 30, n),
        'rooms':   np.random.randint(1, 6, n),
        'floor':   np.random.randint(1, 20, n),
        'type':    np.random.choice(['아파트', '단독', '빌라'], n),
        'year':    np.random.randint(1990, 2023, n),
        'price':   np.random.normal(50000, 15000, n)
    })
    test = train.drop('price', axis=1).sample(200, random_state=1).reset_index(drop=True)
    test.insert(0, 'id', range(len(test)))

    target = 'price'
    X = train.drop(columns=[target])
    y = train[target]
    n_train = len(X)

    all_X = pd.concat([X, test.drop('id', axis=1)], ignore_index=True)

    # 파생 변수
    all_X['age'] = 2023 - all_X['year']

    # 인코딩
    le = LabelEncoder()
    all_X['type'] = le.fit_transform(all_X['type'])

    X_proc = all_X.iloc[:n_train]
    test_proc = all_X.iloc[n_train:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_proc, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(n_estimators=200, learning_rate=0.05,
                         max_depth=6, random_state=42)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {rmse:.2f}")

    test_pred = model.predict(test_proc)
    submission = pd.DataFrame({'id': test['id'], 'price': test_pred})
    submission.to_csv('submission.csv', index=False)
    print(submission.head())
    return submission


# ============================================================
# [작업형 3유형] 예제 1 - 독립 표본 t-검정
# 문제: A 그룹과 B 그룹의 평균 점수가 유의미하게 다른지 검정하시오.
#       (유의수준 5%, 등분산 여부 먼저 확인)
# ============================================================

def type3_ttest_example():
    from scipy import stats

    np.random.seed(42)
    group_A = np.random.normal(75, 10, 30)
    group_B = np.random.normal(80, 12, 35)

    print("=== 독립 표본 t-검정 ===")
    print(f"A 그룹: 평균={group_A.mean():.2f}, 표준편차={group_A.std():.2f}, n={len(group_A)}")
    print(f"B 그룹: 평균={group_B.mean():.2f}, 표준편차={group_B.std():.2f}, n={len(group_B)}")

    # 1단계: 등분산 검정 (Levene)
    lev_stat, lev_p = stats.levene(group_A, group_B)
    print(f"\n[등분산 검정] F={lev_stat:.4f}, p={lev_p:.4f}")
    equal_var = lev_p >= 0.05
    print(f"결론: {'등분산 가정 성립' if equal_var else '이분산'}")

    # 2단계: t-검정
    t_stat, p_value = stats.ttest_ind(group_A, group_B, equal_var=equal_var)
    print(f"\n[t-검정] t={t_stat:.4f}, p={p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"결론: p({p_value:.4f}) < α({alpha}) → 귀무가설 기각")
        print("      두 그룹 간 평균 점수에 통계적으로 유의한 차이가 있다")
    else:
        print(f"결론: p({p_value:.4f}) ≥ α({alpha}) → 귀무가설 채택")
        print("      두 그룹 간 평균 점수에 통계적으로 유의한 차이가 없다")


# ============================================================
# [작업형 3유형] 예제 2 - 카이제곱 독립성 검정
# 문제: 성별과 구매 여부가 독립적인지 검정하시오.
# ============================================================

def type3_chi2_example():
    from scipy import stats

    # 분할표
    contingency = pd.DataFrame({
        '구매O': [120, 80],
        '구매X': [60, 140]
    }, index=['남성', '여성'])

    print("=== 카이제곱 독립성 검정 ===")
    print("관측 빈도:\n", contingency)

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    print(f"\n카이제곱 통계량: {chi2:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"자유도: {dof}")
    print(f"\n기대 빈도:\n{pd.DataFrame(expected, index=['남성', '여성'], columns=['구매O', '구매X']).round(2)}")

    alpha = 0.05
    if p_value < alpha:
        print(f"\n결론: p({p_value:.4f}) < α({alpha}) → 귀무가설 기각")
        print("      성별과 구매 여부는 독립적이지 않다 (연관성 있음)")
    else:
        print(f"\n결론: p({p_value:.4f}) ≥ α({alpha}) → 귀무가설 채택")
        print("      성별과 구매 여부는 독립적이다")


# ============================================================
# 실행
# ============================================================
if __name__ == '__main__':
    print("=" * 50)
    print("[작업형 1유형] 예제 1")
    type1_example1()

    print("\n[작업형 1유형] 예제 2")
    type1_example2()

    print("\n[작업형 2유형] 분류 예제")
    type2_classification_example()

    print("\n[작업형 2유형] 회귀 예제")
    type2_regression_example()

    print("\n[작업형 3유형] t-검정 예제")
    type3_ttest_example()

    print("\n[작업형 3유형] 카이제곱 예제")
    type3_chi2_example()
