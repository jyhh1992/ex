"""
빅데이터 분석기사 실기 - 심화 기법
앙상블, 파이프라인, 시계열, 딥러닝 기초
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. 앙상블 기법
# ============================================================

from sklearn.ensemble import (
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=300, random_state=42)

# Voting (Hard: 다수결, Soft: 확률 평균)
voting_clf = VotingClassifier(
    estimators=[
        ('lr',  LogisticRegression(max_iter=1000)),
        ('dt',  DecisionTreeClassifier(random_state=42)),
        ('rf',  RandomForestClassifier(random_state=42)),
    ],
    voting='soft'
)
voting_clf.fit(X, y)

# Stacking
from sklearn.model_selection import cross_val_predict
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
stacking_clf.fit(X, y)

# AdaBoost
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
ada.fit(X, y)

# ============================================================
# 2. 파이프라인 (Pipeline)
# ============================================================

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# 분류 파이프라인
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_components=5)),
    ('model',  RandomForestClassifier(random_state=42))
])

pipeline.fit(X, y)
pred = pipeline.predict(X)

# make_pipeline (이름 자동 생성)
pipe = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

# 파이프라인 + GridSearch
from sklearn.model_selection import GridSearchCV
param_grid = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print("Best params:", grid.best_params_)

# ColumnTransformer (수치형 + 범주형 각각 처리)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_features = ['age', 'income']
cat_features = ['gender', 'region']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),                          num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'),   cat_features),
    ]
)

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# ============================================================
# 3. 시계열 분석
# ============================================================

import pandas as pd

# 시계열 데이터 생성
dates = pd.date_range('2020-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100).cumsum(), index=dates)

# 이동 평균
ts_ma7  = ts.rolling(window=7).mean()   # 7일 이동평균
ts_ma30 = ts.rolling(window=30).mean()  # 30일 이동평균

# 지수 이동 평균
ts_ema = ts.ewm(span=7, adjust=False).mean()

# 시차 특성
df_ts = pd.DataFrame({'value': ts.values}, index=ts.index)
df_ts['lag_1']  = df_ts['value'].shift(1)
df_ts['lag_7']  = df_ts['value'].shift(7)
df_ts['diff_1'] = df_ts['value'].diff(1)   # 1차 차분

# 정상성 검정 (ADF Test)
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(ts.dropna())
print(f"ADF 통계량: {adf_result[0]:.4f}")
print(f"p-value:    {adf_result[1]:.4f}")
if adf_result[1] < 0.05:
    print("정상 시계열 (단위근 없음)")
else:
    print("비정상 시계열 → 차분 필요")

# ARIMA 모델
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(ts, order=(1, 1, 1))  # (p, d, q)
result = model.fit()
print(result.summary())

forecast = result.forecast(steps=10)
print("예측값:", forecast)

# ============================================================
# 4. 회귀 분석 심화 (statsmodels)
# ============================================================

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_white, het_breuschpagan

# 다중 선형 회귀
df_reg = pd.DataFrame(np.random.randn(100, 4), columns=['y', 'x1', 'x2', 'x3'])

model = smf.ols('y ~ x1 + x2 + x3', data=df_reg).fit()
print(model.summary())

# 잔차 분석
residuals = model.resid
fitted    = model.fittedvalues

# 이분산성 검정 (Breusch-Pagan)
bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
print(f"Breusch-Pagan p: {bp_p:.4f}")

# 로지스틱 회귀 (statsmodels)
model_logit = smf.logit('y_binary ~ x1 + x2', data=df_reg).fit()
print(model_logit.summary())
print("오즈비:", np.exp(model_logit.params))

# ============================================================
# 5. 딥러닝 기초 (keras/tensorflow)
# ============================================================

try:
    from tensorflow import keras
    from tensorflow.keras import layers

    # 분류 모델
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')   # 이진 분류
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 학습
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # 다중 분류
    model_mc = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')   # 3클래스
    ])
    model_mc.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

    # 회귀 모델
    model_reg = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)   # 출력층 활성화 없음
    ])
    model_reg.compile(optimizer='adam', loss='mse', metrics=['mae'])

except ImportError:
    print("TensorFlow가 설치되지 않았습니다")

# ============================================================
# 6. 교차 검증 전략
# ============================================================

from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold,
    TimeSeriesSplit, cross_val_score
)

model = RandomForestClassifier(random_state=42)

# 일반 KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')
print(f"KFold AUC: {scores.mean():.4f}")

# Stratified KFold (클래스 비율 유지)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
print(f"Stratified KFold AUC: {scores.mean():.4f}")

# 시계열 KFold
tscv = TimeSeriesSplit(n_splits=5)
# 시계열 데이터에서 미래 데이터 누출 방지

# ============================================================
# 7. 모델 저장 및 불러오기
# ============================================================

import joblib

# 저장
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 불러오기
loaded_model  = joblib.load('model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

pred = loaded_model.predict(loaded_scaler.transform(X))

# pickle 방식
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
