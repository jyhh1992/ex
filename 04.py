"""
빅데이터 분석기사 실기 - 작업형 3유형
통계 분석 및 가설 검정
"""

import pandas as pd
import numpy as np
from scipy import stats

# ============================================================
# 작업형 3유형 공통 구조
# 1. 가설 설정 (귀무/대립)
# 2. 유의수준 설정 (α = 0.05)
# 3. 검정 통계량 및 p-value 계산
# 4. 결론 도출 (p < α → 귀무가설 기각)
# ============================================================

# ============================================================
# 1. 단일 표본 T-검정 (One-sample t-test)
# 모집단 평균이 특정 값과 같은지 검정
# H0: μ = μ0  /  H1: μ ≠ μ0
# ============================================================

data = [23, 25, 28, 21, 30, 27, 24, 26]
mu0 = 25  # 귀무가설의 모평균

t_stat, p_value = stats.ttest_1samp(data, popmean=mu0)
print(f"t-통계량: {t_stat:.4f}")
print(f"p-value:  {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("귀무가설 기각: 평균이 μ0과 다르다")
else:
    print("귀무가설 채택: 평균이 μ0과 같다")

# 단측 검정 (H1: μ > μ0)
t_stat, p_value_two = stats.ttest_1samp(data, popmean=mu0)
p_value_one_right = p_value_two / 2 if t_stat > 0 else 1 - p_value_two / 2

# ============================================================
# 2. 독립 표본 T-검정 (Two-sample independent t-test)
# 두 집단의 평균이 같은지 검정
# H0: μ1 = μ2  /  H1: μ1 ≠ μ2
# ============================================================

group_A = [23, 25, 28, 30, 22, 27]
group_B = [30, 32, 35, 28, 33, 31]

# 등분산 검정 먼저 (Levene 검정)
levene_stat, levene_p = stats.levene(group_A, group_B)
print(f"Levene p-value: {levene_p:.4f}")
equal_var = levene_p >= 0.05  # p >= 0.05 이면 등분산 가정

t_stat, p_value = stats.ttest_ind(group_A, group_B, equal_var=equal_var)
print(f"t-통계량: {t_stat:.4f}")
print(f"p-value:  {p_value:.4f}")

# ============================================================
# 3. 대응 표본 T-검정 (Paired t-test)
# 동일 집단의 전/후 비교
# H0: μd = 0  /  H1: μd ≠ 0
# ============================================================

before = [72, 68, 75, 80, 65, 70]
after  = [68, 65, 70, 75, 62, 67]

t_stat, p_value = stats.ttest_rel(before, after)
print(f"t-통계량: {t_stat:.4f}")
print(f"p-value:  {p_value:.4f}")

# ============================================================
# 4. 카이제곱 검정 (Chi-square test)
# ============================================================

# 4-1. 적합도 검정 (관측값이 기대 분포를 따르는지)
# H0: 관측 분포 = 기대 분포
observed = [30, 20, 25, 25]
expected = [25, 25, 25, 25]  # 균등 분포 기대

chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
print(f"카이제곱 통계량: {chi2_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# 4-2. 독립성 검정 (두 범주형 변수 간 독립성)
# H0: 두 변수는 독립적이다
contingency_table = pd.DataFrame({
    '찬성': [40, 60],
    '반대': [30, 20]
}, index=['남성', '여성'])

chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"카이제곱 통계량: {chi2_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"자유도: {dof}")
print(f"기대 빈도:\n{expected}")

# ============================================================
# 5. 분산 분석 (ANOVA)
# 세 집단 이상의 평균 비교
# H0: μ1 = μ2 = μ3  /  H1: 적어도 하나는 다르다
# ============================================================

group1 = [23, 25, 28, 24, 26]
group2 = [30, 32, 35, 31, 33]
group3 = [20, 22, 19, 21, 23]

f_stat, p_value = stats.f_oneway(group1, group2, group3)
print(f"F-통계량: {f_stat:.4f}")
print(f"p-value:  {p_value:.4f}")

# 사후 검정 (Tukey HSD) - ANOVA 유의 시 어떤 집단이 다른지 확인
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

data_all    = group1 + group2 + group3
group_labels = ['G1'] * len(group1) + ['G2'] * len(group2) + ['G3'] * len(group3)

tukey = pairwise_tukeyhsd(data_all, group_labels, alpha=0.05)
print(tukey.summary())

# ============================================================
# 6. 정규성 검정
# ============================================================

# Shapiro-Wilk 검정 (소표본, n < 50 권장)
# H0: 정규분포를 따른다
stat, p_value = stats.shapiro(data)
print(f"Shapiro-Wilk: stat={stat:.4f}, p={p_value:.4f}")

# Kolmogorov-Smirnov 검정 (대표본)
stat, p_value = stats.kstest(data, 'norm',
                              args=(np.mean(data), np.std(data)))
print(f"KS test: stat={stat:.4f}, p={p_value:.4f}")

# Anderson-Darling 검정
result = stats.anderson(data, dist='norm')
print(f"Anderson-Darling 통계량: {result.statistic:.4f}")
for sig_level, crit_val in zip(result.significance_level, result.critical_values):
    print(f"  유의수준 {sig_level}%: 임계값={crit_val:.4f}",
          "→ 기각" if result.statistic > crit_val else "→ 채택")

# ============================================================
# 7. 비모수 검정 (정규성 위반 시)
# ============================================================

# Mann-Whitney U 검정 (독립 2표본 t-test 대안)
# H0: 두 집단의 중앙값이 같다
stat, p_value = stats.mannwhitneyu(group_A, group_B, alternative='two-sided')
print(f"Mann-Whitney U: stat={stat:.4f}, p={p_value:.4f}")

# Wilcoxon 부호순위 검정 (대응 t-test 대안)
stat, p_value = stats.wilcoxon(before, after)
print(f"Wilcoxon: stat={stat:.4f}, p={p_value:.4f}")

# Kruskal-Wallis 검정 (일원 ANOVA 대안)
stat, p_value = stats.kruskal(group1, group2, group3)
print(f"Kruskal-Wallis: stat={stat:.4f}, p={p_value:.4f}")

# ============================================================
# 8. 상관 분석
# ============================================================

x = [1, 2, 3, 4, 5, 6, 7]
y = [2, 4, 5, 4, 5, 7, 8]

# Pearson 상관계수 (선형 관계, 정규분포 가정)
r, p_value = stats.pearsonr(x, y)
print(f"Pearson r={r:.4f}, p={p_value:.4f}")

# Spearman 상관계수 (순위 기반, 비모수)
r, p_value = stats.spearmanr(x, y)
print(f"Spearman r={r:.4f}, p={p_value:.4f}")

# Kendall 상관계수 (순위 기반)
r, p_value = stats.kendalltau(x, y)
print(f"Kendall tau={r:.4f}, p={p_value:.4f}")

# 판다스 상관계수 행렬
df = pd.DataFrame({'x': x, 'y': y})
print(df.corr(method='pearson'))
print(df.corr(method='spearman'))

# ============================================================
# 9. 회귀 분석 (statsmodels)
# ============================================================

import statsmodels.api as sm
import statsmodels.formula.api as smf

# OLS 회귀
X_data = sm.add_constant(df[['col1', 'col2']])  # 상수항 추가
model  = sm.OLS(df['target'], X_data)
result = model.fit()
print(result.summary())

# 공식 기반
model  = smf.ols('target ~ col1 + col2 + col3', data=df)
result = model.fit()
print(result.summary())
print("R-squared:", result.rsquared)
print("계수:", result.params)
print("p-values:", result.pvalues)

# ============================================================
# 10. 구간 추정 (신뢰구간)
# ============================================================

data = np.array([23, 25, 28, 21, 30, 27, 24, 26])
n    = len(data)
mean = np.mean(data)
se   = stats.sem(data)   # 표준오차

# 95% 신뢰구간 (t분포)
ci = stats.t.interval(confidence=0.95, df=n-1, loc=mean, scale=se)
print(f"95% 신뢰구간: ({ci[0]:.4f}, {ci[1]:.4f})")

# 99% 신뢰구간
ci_99 = stats.t.interval(confidence=0.99, df=n-1, loc=mean, scale=se)
print(f"99% 신뢰구간: ({ci_99[0]:.4f}, {ci_99[1]:.4f})")

# ============================================================
# 11. 검정력 및 표본 크기 계산
# ============================================================

from statsmodels.stats.power import TTestIndPower, TTestPower

# 독립 t-검정 표본 크기 계산
analysis = TTestIndPower()
n = analysis.solve_power(effect_size=0.5, alpha=0.05, power=0.8)
print(f"필요 표본 크기 (집단당): {np.ceil(n):.0f}")

# 검정력 계산
power = analysis.power(effect_size=0.5, nobs1=30, alpha=0.05)
print(f"검정력: {power:.4f}")

# ============================================================
# 12. 결론 작성 템플릿
# ============================================================

def hypothesis_conclusion(p_value, alpha=0.05, test_name="검정"):
    print(f"\n[{test_name} 결론]")
    print(f"유의수준 α = {alpha}")
    print(f"p-value = {p_value:.4f}")
    if p_value < alpha:
        print(f"→ p({p_value:.4f}) < α({alpha}): 귀무가설 기각")
        print("→ 통계적으로 유의미한 차이가 있다")
    else:
        print(f"→ p({p_value:.4f}) ≥ α({alpha}): 귀무가설 채택")
        print("→ 통계적으로 유의미한 차이가 없다")

hypothesis_conclusion(0.032, alpha=0.05, test_name="독립표본 t-검정")
