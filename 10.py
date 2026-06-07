"""
빅데이터 분석기사 실기 - 데이터 시각화
matplotlib & seaborn 기본 패턴
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (시험 환경)
plt.rcParams['font.family'] = 'DejaVu Sans'
# 또는
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 분포 시각화
# ============================================================

data = np.random.normal(loc=50, scale=10, size=200)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 히스토그램
axes[0].hist(data, bins=20, edgecolor='black', color='steelblue', alpha=0.7)
axes[0].set_title('히스토그램')
axes[0].set_xlabel('값')
axes[0].set_ylabel('빈도')

# 박스플롯
axes[1].boxplot(data, vert=True)
axes[1].set_title('박스플롯')

# 커널밀도추정 (KDE)
sns.kdeplot(data, ax=axes[2], fill=True)
axes[2].set_title('KDE 플롯')

plt.tight_layout()
plt.savefig('distribution.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================
# 2. 범주형 데이터 시각화
# ============================================================

df = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'value': np.random.randn(100) * 10 + 50,
    'group': np.random.choice(['X', 'Y'], 100)
})

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 막대 그래프
freq = df['category'].value_counts()
axes[0, 0].bar(freq.index, freq.values, color='steelblue')
axes[0, 0].set_title('빈도 막대 그래프')

# 그룹별 박스플롯
df.boxplot(column='value', by='category', ax=axes[0, 1])
axes[0, 1].set_title('그룹별 박스플롯')

# seaborn 바이올린 플롯
sns.violinplot(x='category', y='value', data=df, ax=axes[1, 0])
axes[1, 0].set_title('바이올린 플롯')

# 누적 막대 그래프
pivot = df.groupby(['category', 'group']).size().unstack(fill_value=0)
pivot.plot(kind='bar', stacked=True, ax=axes[1, 1])
axes[1, 1].set_title('누적 막대 그래프')

plt.tight_layout()
plt.savefig('categorical.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================
# 3. 관계 시각화
# ============================================================

df_num = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 산점도
axes[0].scatter(df_num['A'], df_num['B'], alpha=0.6)
axes[0].set_xlabel('A')
axes[0].set_ylabel('B')
axes[0].set_title('산점도')

# 상관관계 히트맵
corr = df_num.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title('상관관계 히트맵')

# 페어 플롯 (seaborn)
# sns.pairplot(df_num)  # 독립적으로 실행

# 회귀선 포함 산점도
axes[2].scatter(df_num['A'], df_num['B'], alpha=0.6)
m, b = np.polyfit(df_num['A'], df_num['B'], 1)
x_line = np.linspace(df_num['A'].min(), df_num['A'].max(), 100)
axes[2].plot(x_line, m * x_line + b, 'r-', linewidth=2)
axes[2].set_title('회귀선 포함 산점도')

plt.tight_layout()
plt.savefig('relationship.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================
# 4. 시계열 시각화
# ============================================================

dates = pd.date_range('2022-01-01', periods=100, freq='D')
ts = pd.Series(np.cumsum(np.random.randn(100)), index=dates)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(ts.index, ts.values, label='원본')
ax.plot(ts.index, ts.rolling(7).mean(), label='7일 이동평균', linewidth=2)
ax.set_title('시계열 데이터')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('timeseries.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================
# 5. 모델 평가 시각화
# ============================================================

from sklearn.metrics import roc_curve, confusion_matrix

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR (1 - 특이도)')
    plt.ylabel('TPR (민감도)')
    plt.title('ROC 커브')
    plt.legend()
    plt.savefig('roc_curve.png', dpi=100)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('실제값')
    plt.xlabel('예측값')
    plt.title('혼동 행렬')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.show()

def plot_feature_importance(model, feature_names, top_n=15):
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(top_n)

    plt.figure(figsize=(8, 6))
    plt.barh(importance['feature'], importance['importance'])
    plt.xlabel('중요도')
    plt.title(f'상위 {top_n} 특성 중요도')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100)
    plt.show()

def plot_learning_curve(train_sizes, train_scores, val_scores):
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    val_mean   = np.mean(val_scores, axis=1)
    val_std    = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, label='훈련 점수')
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, label='검증 점수')
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    plt.xlabel('훈련 샘플 수')
    plt.ylabel('점수')
    plt.title('학습 곡선')
    plt.legend()
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=100)
    plt.show()

# ============================================================
# 6. EDA 시각화 자동화 함수
# ============================================================

def eda_report(df):
    """데이터프레임 EDA 시각화 자동 생성"""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # 수치형 분포
    if num_cols:
        n = len(num_cols)
        fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)

        for i, col in enumerate(num_cols):
            axes[0, i].hist(df[col].dropna(), bins=20, color='steelblue')
            axes[0, i].set_title(f'{col} 히스토그램')
            axes[1, i].boxplot(df[col].dropna())
            axes[1, i].set_title(f'{col} 박스플롯')

        plt.tight_layout()
        plt.savefig('eda_numeric.png', dpi=80)
        plt.show()

    # 범주형 빈도
    if cat_cols:
        n = len(cat_cols)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for i, col in enumerate(cat_cols):
            freq = df[col].value_counts()
            axes[i].bar(freq.index[:10], freq.values[:10])
            axes[i].set_title(f'{col} 빈도')
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('eda_categorical.png', dpi=80)
        plt.show()

    # 상관관계
    if len(num_cols) > 1:
        plt.figure(figsize=(max(6, len(num_cols)), max(5, len(num_cols) - 1)))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
                    cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('상관관계 히트맵')
        plt.tight_layout()
        plt.savefig('eda_correlation.png', dpi=80)
        plt.show()
