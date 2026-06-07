"""
빅데이터 분석기사 실기 - 피처 엔지니어링
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OrdinalEncoder, OneHotEncoder,
    PolynomialFeatures, PowerTransformer
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    mutual_info_classif, RFE, SelectFromModel
)

# ============================================================
# 1. 스케일링 (Scaling)
# ============================================================

X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [100, 200, 150, 300, 250]})

# StandardScaler: 평균=0, 표준편차=1
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# MinMaxScaler: 0~1 범위
scaler = MinMaxScaler()
X_mm = scaler.fit_transform(X)
# 특정 범위로 조정
scaler = MinMaxScaler(feature_range=(-1, 1))

# RobustScaler: 이상치에 강건 (중앙값, IQR 기반)
scaler = RobustScaler()
X_robust = scaler.fit_transform(X)

# ============================================================
# 2. 인코딩
# ============================================================

df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red'],
                   'size': ['S', 'M', 'L', 'XL'],
                   'score': [10, 20, 30, 40]})

# Label Encoding
le = LabelEncoder()
df['color_le'] = le.fit_transform(df['color'])
# 역변환
original = le.inverse_transform(df['color_le'])

# Ordinal Encoding (순서 지정)
oe = OrdinalEncoder(categories=[['S', 'M', 'L', 'XL']])
df['size_oe'] = oe.fit_transform(df[['size']])

# One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded = ohe.fit_transform(df[['color']])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out())

# pd.get_dummies (더 간편)
df_dummies = pd.get_dummies(df, columns=['color', 'size'], drop_first=True)

# Target Encoding (평균 인코딩)
target_mean = df.groupby('color')['score'].mean()
df['color_te'] = df['color'].map(target_mean)

# ============================================================
# 3. 변수 변환 (분포 정규화)
# ============================================================

# 로그 변환 (양의 왜도)
df['log_col'] = np.log1p(df['score'])           # log(1+x)
df['log_col'] = np.log(df['score'])             # log(x), x>0

# 제곱근 변환
df['sqrt_col'] = np.sqrt(df['score'])

# Box-Cox 변환 (양수 데이터만)
from scipy import stats
df['boxcox_col'], lambda_ = stats.boxcox(df['score'])

# Yeo-Johnson 변환 (음수 포함 가능)
pt = PowerTransformer(method='yeo-johnson')
df['yj_col'] = pt.fit_transform(df[['score']])

# ============================================================
# 4. 파생 변수 생성
# ============================================================

df = pd.DataFrame({
    'start_date': pd.to_datetime(['2020-01-01', '2021-06-15', '2019-03-20']),
    'end_date':   pd.to_datetime(['2023-01-01', '2023-06-15', '2023-03-20']),
    'col1': [10, 20, 30],
    'col2': [5, 15, 25],
})

# 날짜 파생
df['year']        = df['start_date'].dt.year
df['month']       = df['start_date'].dt.month
df['day']         = df['start_date'].dt.day
df['dayofweek']   = df['start_date'].dt.dayofweek   # 0=월
df['is_weekend']  = df['dayofweek'].isin([5, 6]).astype(int)
df['duration']    = (df['end_date'] - df['start_date']).dt.days
df['quarter']     = df['start_date'].dt.quarter

# 수치형 파생
df['ratio']       = df['col1'] / (df['col2'] + 1e-8)  # 0 나눔 방지
df['diff']        = df['col1'] - df['col2']
df['interaction'] = df['col1'] * df['col2']

# 다항 특성
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['col1', 'col2']])
poly_features = poly.get_feature_names_out()

# 롤링 통계 (시계열)
df = df.sort_values('start_date')
df['rolling_mean'] = df['col1'].rolling(window=3).mean()
df['rolling_std']  = df['col1'].rolling(window=3).std()
df['lag_1']        = df['col1'].shift(1)          # 1 시점 전 값
df['lag_2']        = df['col1'].shift(2)

# ============================================================
# 5. 변수 선택 (Feature Selection)
# ============================================================

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=20, random_state=42)
X = pd.DataFrame(X, columns=[f'f{i}' for i in range(20)])

# Filter Method - ANOVA F-test
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("선택된 변수:", selected_features.tolist())

# Filter Method - 상호정보량
selector = SelectKBest(mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Wrapper Method - RFE
from sklearn.linear_model import LogisticRegression
estimator = LogisticRegression(max_iter=1000)
rfe = RFE(estimator, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)
print("RFE 선택 변수:", X.columns[rfe.support_].tolist())

# Embedded Method - L1 규제 (Lasso)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01)
selector = SelectFromModel(lasso)
X_lasso = selector.fit_transform(X, y)

# Embedded Method - 트리 기반 중요도
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
selector = SelectFromModel(rf)
X_rf = selector.fit_transform(X, y)

# 분산 기반 제거 (분산이 너무 낮은 변수 제거)
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.01)
X_var = sel.fit_transform(X)

# ============================================================
# 6. 다중공선성 처리
# ============================================================

# VIF (Variance Inflation Factor) 확인
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

X_const = sm.add_constant(X)
vif_data = pd.DataFrame({
    'feature': X_const.columns,
    'VIF': [variance_inflation_factor(X_const.values, i)
            for i in range(X_const.shape[1])]
})
print(vif_data.sort_values('VIF', ascending=False))
# VIF > 10 이면 다중공선성 문제 있음

# 상관관계 높은 변수 제거
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.9)]
X_reduced = X.drop(columns=high_corr_cols)
print(f"제거된 변수: {high_corr_cols}")
