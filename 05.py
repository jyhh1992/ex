"""
빅데이터 분석기사 실기 - 군집 분석 & 차원 축소
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. K-Means 군집화
# ============================================================

from sklearn.cluster import KMeans

# 스케일링 필수
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 최적 K 찾기 - Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# 최적 K 찾기 - Silhouette Score
from sklearn.metrics import silhouette_score
sil_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

best_k = range(2, 11)[np.argmax(sil_scores)]
print(f"최적 K: {best_k}")

# K-Means 모델 학습
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

df['cluster'] = labels
print(df.groupby('cluster').mean())

# 군집 중심
centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("군집 중심:\n", pd.DataFrame(centers, columns=X.columns))

# ============================================================
# 2. 계층적 군집화 (Hierarchical Clustering)
# ============================================================

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 덴드로그램으로 군집 수 결정
Z = linkage(X_scaled, method='ward')

# 계층적 군집화
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(X_scaled)

# ============================================================
# 3. DBSCAN (밀도 기반)
# ============================================================

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# -1 = 노이즈(이상치)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = list(labels).count(-1)
print(f"군집 수: {n_clusters}, 노이즈 포인트: {n_noise}")

# ============================================================
# 4. 차원 축소 - PCA
# ============================================================

from sklearn.decomposition import PCA

# 주성분 분석
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 설명 분산 비율
print("설명 분산 비율:", pca.explained_variance_ratio_)
print("누적 설명 분산:", np.cumsum(pca.explained_variance_ratio_))

# 최적 주성분 수 결정 (누적 분산 95% 이상)
pca_full = PCA()
pca_full.fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cum_var >= 0.95) + 1
print(f"95% 설명을 위한 주성분 수: {n_components}")

pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X_scaled)

# ============================================================
# 5. t-SNE (시각화용)
# ============================================================

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# ============================================================
# 6. 군집 평가 지표
# ============================================================

from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)

sil  = silhouette_score(X_scaled, labels)     # 높을수록 좋음 (-1~1)
db   = davies_bouldin_score(X_scaled, labels) # 낮을수록 좋음
ch   = calinski_harabasz_score(X_scaled, labels) # 높을수록 좋음

print(f"Silhouette Score: {sil:.4f}")
print(f"Davies-Bouldin:   {db:.4f}")
print(f"Calinski-Harabasz: {ch:.4f}")
