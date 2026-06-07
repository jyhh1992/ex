"""
빅데이터 분석기사 실기 - 작업형 2유형 (회귀)
수치형 타겟 예측 모델 구축
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# 회귀 모델 기본 템플릿
# ============================================================

# 데이터 로드
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

X = train.drop(columns=['target'])
y = train['target']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 회귀 모델들
# ============================================================

# ── 선형 회귀 ──────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print("회귀 계수:", model.coef_)
print("절편:", model.intercept_)

# ── Ridge (L2 규제) ────────────────────────────────────────
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# ── Lasso (L1 규제, 변수 선택 효과) ───────────────────────
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01)
model.fit(X_train, y_train)

# ── ElasticNet (L1 + L2) ───────────────────────────────────
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.01, l1_ratio=0.5)
model.fit(X_train, y_train)

# ── 결정 트리 회귀 ─────────────────────────────────────────
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# ── 랜덤 포레스트 회귀 ─────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── XGBoost 회귀 ───────────────────────────────────────────
from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=False)

# ── LightGBM 회귀 ──────────────────────────────────────────
from lightgbm import LGBMRegressor
model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)])

# ── Gradient Boosting 회귀 ─────────────────────────────────
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# ── SVR ────────────────────────────────────────────────────
from sklearn.svm import SVR
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_train, y_train)  # 스케일링 필수

# ── KNN 회귀 ───────────────────────────────────────────────
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

# ============================================================
# 모델 평가 - 회귀
# ============================================================

pred = model.predict(X_val)

mse  = mean_squared_error(y_val, pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_val, pred)
r2   = r2_score(y_val, pred)

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_val - pred) / y_val)) * 100
print(f"MAPE: {mape:.2f}%")

# 교차 검증
cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=5,
                                    scoring='neg_mean_squared_error'))
print(f"CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# ============================================================
# 타겟 변수 변환 (로그 변환) - 왜도가 심할 때
# ============================================================

y_log = np.log1p(y)   # log(1 + y) → 0 포함 데이터에 안전

model.fit(X_train, np.log1p(y_train))
pred_log = model.predict(X_val)
pred_original = np.expm1(pred_log)  # 역변환: exp(x) - 1

rmse = np.sqrt(mean_squared_error(y_val, pred_original))

# ============================================================
# 제출 파일 생성
# ============================================================

test_pred = model.predict(test)

submission = pd.DataFrame({
    'id': test['id'],
    'target': test_pred
})
submission.to_csv('submission.csv', index=False)
print(submission.head())
