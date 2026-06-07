"""
빅데이터 분석기사 실기 - 모델 평가 지표 총정리
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    # 분류
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    log_loss, balanced_accuracy_score,
    # 회귀
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
)

# ============================================================
# 분류 평가 지표
# ============================================================

y_true  = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred  = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
y_proba = [0.9, 0.1, 0.8, 0.3, 0.2, 0.7, 0.6, 0.1, 0.85, 0.15]

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
# [[TN FP]
#  [FN TP]]
TN, FP, FN, TP = cm.ravel()

# 핵심 지표
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)   # TP / (TP + FP)
recall    = recall_score(y_true, y_pred)       # TP / (TP + FN) = 민감도
f1        = f1_score(y_true, y_pred)           # 2 * P * R / (P + R)
specificity = TN / (TN + FP)                  # TN / (TN + FP)
auc       = roc_auc_score(y_true, y_proba)

print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}  (민감도/재현율)")
print(f"Specificity: {specificity:.4f} (특이도)")
print(f"F1 Score:    {f1:.4f}")
print(f"ROC-AUC:     {auc:.4f}")

# 임계값(threshold) 조정
threshold = 0.4
y_pred_custom = (np.array(y_proba) >= threshold).astype(int)
print(f"\n임계값 {threshold} 적용 후:")
print(f"Precision: {precision_score(y_true, y_pred_custom):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred_custom):.4f}")

# 다중 클래스
y_true_mc = [0, 1, 2, 1, 0, 2]
y_pred_mc = [0, 1, 1, 1, 0, 2]
print("\n[다중 클래스]")
print(f"Accuracy:         {accuracy_score(y_true_mc, y_pred_mc):.4f}")
print(f"F1 (macro):       {f1_score(y_true_mc, y_pred_mc, average='macro'):.4f}")
print(f"F1 (weighted):    {f1_score(y_true_mc, y_pred_mc, average='weighted'):.4f}")
print(f"F1 (micro):       {f1_score(y_true_mc, y_pred_mc, average='micro'):.4f}")
print(classification_report(y_true_mc, y_pred_mc))

# ============================================================
# 회귀 평가 지표
# ============================================================

y_true_r = [3.0, 5.0, 2.5, 7.0, 4.5]
y_pred_r = [2.8, 5.2, 3.0, 6.5, 4.0]

mse  = mean_squared_error(y_true_r, y_pred_r)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true_r, y_pred_r)
r2   = r2_score(y_true_r, y_pred_r)
mape = mean_absolute_percentage_error(y_true_r, y_pred_r) * 100

print(f"\n[회귀 평가]")
print(f"MSE:   {mse:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"MAE:   {mae:.4f}")
print(f"R²:    {r2:.4f}")
print(f"MAPE:  {mape:.2f}%")

# 조정 R² (변수 수를 고려)
n = len(y_true_r)
k = 3  # 독립변수 수
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
print(f"조정 R²: {adj_r2:.4f}")

# ============================================================
# 교차 검증 상세
# ============================================================

from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold,
    KFold, LeaveOneOut, cross_val_predict
)

# 다양한 scoring
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, random_state=42)
model = RandomForestClassifier(random_state=42)

results = cross_validate(model, X, y, cv=5, scoring=scoring)
for metric in scoring:
    scores = results[f'test_{metric}']
    print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")

# ============================================================
# ROC 커브 및 최적 임계값
# ============================================================

model.fit(X, y)
y_proba_all = model.predict_proba(X)[:, 1]

fpr, tpr, thresholds = roc_curve(y, y_proba_all)

# Youden's J statistic으로 최적 임계값 찾기
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]
print(f"최적 임계값: {best_threshold:.4f}")

# ============================================================
# 불균형 데이터 처리
# ============================================================

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# SMOTE (오버샘플링)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"SMOTE 후 클래스 분포: {pd.Series(y_res).value_counts().to_dict()}")

# 랜덤 언더샘플링
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

# 클래스 가중치 적용 (모델 내에서)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced', random_state=42)
# XGBoost의 경우: scale_pos_weight = neg_count / pos_count
