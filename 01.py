"""
빅데이터 분석기사 실기 - 작업형 1유형
데이터 전처리 및 탐색적 데이터 분석 (EDA)
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. 데이터 불러오기
# ============================================================

# CSV 파일 불러오기
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', encoding='utf-8')
df = pd.read_csv('data.csv', encoding='cp949')

# 시험에서 자주 쓰는 기본 확인
print(df.shape)          # (행, 열)
print(df.dtypes)         # 각 컬럼 데이터 타입
print(df.info())         # 전체 정보
print(df.describe())     # 수치형 기술통계
print(df.head())
print(df.tail())
print(df.columns.tolist())

# ============================================================
# 2. 결측치 처리
# ============================================================

# 결측치 확인
print(df.isnull().sum())                    # 컬럼별 결측치 수
print(df.isnull().sum() / len(df) * 100)   # 결측치 비율(%)
print(df.isnull().any())                    # 결측치 존재 여부

# 결측치 제거
df_drop = df.dropna()                       # 결측치 있는 행 전체 제거
df_drop = df.dropna(axis=1)                 # 결측치 있는 열 전체 제거
df_drop = df.dropna(subset=['col1', 'col2'])  # 특정 열 기준으로 제거

# 결측치 대체 - 수치형
df['col'].fillna(df['col'].mean(), inplace=True)    # 평균
df['col'].fillna(df['col'].median(), inplace=True)  # 중앙값
df['col'].fillna(0, inplace=True)                   # 0으로
df['col'].fillna(method='ffill', inplace=True)      # 앞 값으로 채우기
df['col'].fillna(method='bfill', inplace=True)      # 뒤 값으로 채우기

# 결측치 대체 - 범주형
df['col'].fillna(df['col'].mode()[0], inplace=True) # 최빈값

# ============================================================
# 3. 이상치 처리
# ============================================================

# IQR 방법으로 이상치 탐지
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# 이상치 확인
outliers = df[(df['col'] < lower) | (df['col'] > upper)]
print(f"이상치 수: {len(outliers)}")

# 이상치 제거
df_clean = df[(df['col'] >= lower) & (df['col'] <= upper)]

# 이상치 대체 (클리핑)
df['col'] = df['col'].clip(lower=lower, upper=upper)

# Z-score 방법
from scipy import stats
z_scores = np.abs(stats.zscore(df[['col1', 'col2']]))
df_clean = df[(z_scores < 3).all(axis=1)]

# ============================================================
# 4. 데이터 타입 변환
# ============================================================

df['col'] = df['col'].astype(int)
df['col'] = df['col'].astype(float)
df['col'] = df['col'].astype(str)
df['col'] = pd.to_numeric(df['col'], errors='coerce')  # 변환 실패 시 NaN

# 날짜형 변환
df['date'] = pd.to_datetime(df['date'])
df['year']  = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day']   = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek  # 0=월요일

# ============================================================
# 5. 데이터 선택 및 필터링
# ============================================================

# 열 선택
df[['col1', 'col2']]
df.loc[:, 'col1':'col3']

# 조건 필터링
df[df['col'] > 10]
df[(df['col1'] > 10) & (df['col2'] == 'A')]
df[(df['col1'] > 10) | (df['col2'] == 'A')]
df[df['col'].isin(['A', 'B', 'C'])]
df[~df['col'].isin(['A', 'B'])]          # NOT IN

# loc, iloc
df.loc[0:5, ['col1', 'col2']]            # 라벨 기반
df.iloc[0:5, 0:3]                         # 인덱스 기반

# ============================================================
# 6. 데이터 집계 및 그룹화
# ============================================================

# 기본 집계
df['col'].sum()
df['col'].mean()
df['col'].median()
df['col'].std()
df['col'].var()
df['col'].min()
df['col'].max()
df['col'].count()
df['col'].nunique()        # 고유값 수
df['col'].value_counts()   # 값별 빈도

# groupby
df.groupby('group_col')['value_col'].mean()
df.groupby('group_col')['value_col'].agg(['mean', 'std', 'count'])
df.groupby(['col1', 'col2'])['value'].sum()

# 피벗 테이블
pivot = df.pivot_table(values='value', index='row_col', columns='col_col', aggfunc='mean')

# ============================================================
# 7. 데이터 변환
# ============================================================

# 열 추가 / 계산
df['new_col'] = df['col1'] + df['col2']
df['ratio'] = df['col1'] / df['col2']

# apply 활용
df['new_col'] = df['col'].apply(lambda x: x * 2)
df['category'] = df['col'].apply(lambda x: 'high' if x > 100 else 'low')

# cut - 연속 변수를 범주로 변환
df['grade'] = pd.cut(df['score'], bins=[0, 60, 80, 100], labels=['C', 'B', 'A'])
df['grade'] = pd.cut(df['score'], bins=5)  # 5개 구간으로 균등 분할

# qcut - 분위수 기반 변환
df['quartile'] = pd.qcut(df['score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# ============================================================
# 8. 정렬 및 중복 처리
# ============================================================

# 정렬
df.sort_values('col', ascending=True)
df.sort_values(['col1', 'col2'], ascending=[True, False])

# 중복 확인 및 제거
print(df.duplicated().sum())
df_unique = df.drop_duplicates()
df_unique = df.drop_duplicates(subset=['col1', 'col2'])
df_unique = df.drop_duplicates(keep='last')

# 인덱스 리셋
df = df.reset_index(drop=True)

# ============================================================
# 9. 데이터 합치기
# ============================================================

# concat - 단순 이어붙이기
df_merged = pd.concat([df1, df2], axis=0, ignore_index=True)  # 행 방향
df_merged = pd.concat([df1, df2], axis=1)                      # 열 방향

# merge - SQL 조인 방식
df_merged = pd.merge(df1, df2, on='key_col', how='inner')  # inner join
df_merged = pd.merge(df1, df2, on='key_col', how='left')   # left join
df_merged = pd.merge(df1, df2, on='key_col', how='right')  # right join
df_merged = pd.merge(df1, df2, on='key_col', how='outer')  # outer join
df_merged = pd.merge(df1, df2, left_on='key1', right_on='key2', how='left')

# ============================================================
# 10. 기술 통계 - 자주 나오는 문제 패턴
# ============================================================

# 특정 조건의 통계값 구하기
result = df[df['group'] == 'A']['value'].mean()
result = df.groupby('group')['value'].mean().reset_index()

# 상위/하위 N개
top5 = df.nlargest(5, 'col')
bottom5 = df.nsmallest(5, 'col')

# 백분위수
p25 = df['col'].quantile(0.25)
p75 = df['col'].quantile(0.75)
p90 = df['col'].quantile(0.90)

# 상관관계
correlation = df['col1'].corr(df['col2'])
corr_matrix = df[['col1', 'col2', 'col3']].corr()

# ============================================================
# 11. 문자열 처리
# ============================================================

df['col'].str.upper()
df['col'].str.lower()
df['col'].str.strip()
df['col'].str.replace('old', 'new')
df['col'].str.contains('pattern')
df['col'].str.startswith('prefix')
df['col'].str.endswith('suffix')
df['col'].str.split(',')                          # 분리
df['col'].str.split(',').str[0]                   # 분리 후 첫 번째 요소
df['col'].str.len()                               # 문자열 길이
df['col'].str.extract(r'(\d+)')                   # 정규식 추출

# ============================================================
# 12. 시험 자주 나오는 패턴 정리
# ============================================================

# 패턴 1: 특정 컬럼 기준으로 상위 N% 데이터 필터링
threshold = df['col'].quantile(0.9)
top10pct = df[df['col'] >= threshold]

# 패턴 2: 그룹별 최대/최소값을 가진 행 찾기
idx = df.groupby('group')['value'].idxmax()
df_max = df.loc[idx]

# 패턴 3: 조건에 따른 새 컬럼 생성
df['label'] = np.where(df['score'] >= 60, '합격', '불합격')

# 패턴 4: 결측치 비율이 특정 값 이상인 컬럼 삭제
missing_ratio = df.isnull().sum() / len(df)
df = df.drop(columns=missing_ratio[missing_ratio > 0.3].index)

# 패턴 5: 범주형 변수 빈도 테이블
freq_table = df['col'].value_counts(normalize=True) * 100

# 패턴 6: 특정 값으로 대체 후 집계
df['col'] = df['col'].replace({'Yes': 1, 'No': 0})

# 정답 출력 (작업형 1유형은 보통 숫자 하나 출력)
answer = round(result, 2)
print(answer)
