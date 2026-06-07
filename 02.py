"""
빅데이터 분석기사 실기 - 작업형 2유형 (분류)
이진/다중 분류 모델 구축 및 예측
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)

# ============================================================
# 작업형 2유형 기본 흐름 템플릿
# ============================================================

# 1. 데이터 불러오기
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

# 2. 타겟 변수 분리
X = train.drop(columns=['target'])
y = train['target']

# 3. 전처리 (train/test 동시 적용)
# - 결측치
X['col'].fillna(X['col'].median(), inplace=True)
# - 인코딩
# - 스케일링

# 4. train/validation 분리
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 모델 학습 → 평가 → 예측 → 제출

# ============================================================
# 전처리 - 인코딩
# ============================================================

# Label Encoding (순서 있는 범주형 or 트리 모델)
le = LabelEncoder()
df['col_encoded'] = le.fit_transform(df['col'])

# 여러 열 한번에 Label Encoding
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# OneHotEncoding (순서 없는 범주형)
df_encoded = pd.get_dummies(df, columns=['col1', 'col2'], drop_first=True)

# train과 test 동시에 처리하는 패턴
all_data = pd.concat([X, test], axis=0, ignore_index=True)
# 전처리 후 다시 분리
X_processed = all_data.iloc[:len(X)]
test_processed = all_data.iloc[len(X):]

# ============================================================
# 전처리 - 스케일링
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
test_scaled    = scaler.transform(test)

# MinMaxScaler (0~1 범위)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# ============================================================
# 분류 모델들
# ============================================================

# ── 로지스틱 회귀 ──────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_val)
pred_proba = model.predict_proba(X_val)[:, 1]  # 양성 클래스 확률

# ── 결정 트리 ──────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# ── 랜덤 포레스트 ──────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── XGBoost ────────────────────────────────────────────────
from xgboost import XGBClassifier
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    use_label_encoder=False
)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=False)

# ── LightGBM ───────────────────────────────────────────────
from lightgbm import LGBMClassifier
model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[])

# ── SVM ────────────────────────────────────────────────────
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
model.fit(X_train_scaled, y_train)

# ── KNN ────────────────────────────────────────────────────
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# ── Gradient Boosting ──────────────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# ── 나이브 베이즈 ──────────────────────────────────────────
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

# ============================================================
# 모델 평가 - 분류
# ============================================================

pred       = model.predict(X_val)
pred_proba = model.predict_proba(X_val)[:, 1]

print("정확도(Accuracy):", accuracy_score(y_val, pred))
print("정밀도(Precision):", precision_score(y_val, pred))
print("재현율(Recall):",    recall_score(y_val, pred))
print("F1 Score:",          f1_score(y_val, pred))
print("ROC-AUC:",           roc_auc_score(y_val, pred_proba))

# confusion matrix
cm = confusion_matrix(y_val, pred)
print("Confusion Matrix:\n", cm)
# [[TN, FP],
#  [FN, TP]]
TN, FP, FN, TP = cm.ravel()

# 분류 리포트
print(classification_report(y_val, pred))

# 다중 클래스인 경우
print("F1(macro):",   f1_score(y_val, pred, average='macro'))
print("F1(weighted):", f1_score(y_val, pred, average='weighted'))
print("ROC-AUC(ovr):", roc_auc_score(y_val, model.predict_proba(X_val), multi_class='ovr'))

# ============================================================
# 교차 검증
# ============================================================

# 기본 교차 검증
cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Stratified K-Fold (클래스 불균형 시)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

# ============================================================
# 하이퍼파라미터 튜닝
# ============================================================

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)
print("Best AUC:",    grid_search.best_score_)
best_model = grid_search.best_estimator_

# ============================================================
# 특성 중요도
# ============================================================

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10))

# ============================================================
# 제출 파일 생성 (작업형 2유형 마지막 단계)
# ============================================================

# 모델로 test 예측
test_pred = model.predict(test)
test_pred_proba = model.predict_proba(test)[:, 1]

# 제출 양식 생성
submission = pd.DataFrame({
    'id': test['id'],             # id 컬럼이 있는 경우
    'target': test_pred_proba     # 확률값 or 클래스
})
submission.to_csv('submission.csv', index=False)
print(submission.head())
