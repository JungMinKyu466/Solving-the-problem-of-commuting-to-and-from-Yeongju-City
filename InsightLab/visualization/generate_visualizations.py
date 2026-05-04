"""
임금체불 예측 모델 결과 시각화 (v3 — Precision@K 헤드라인).

생성 그림:
  Fig 01. 모델별 Precision@10 ⭐ 헤드라인
  Fig 02. Precision@K 종합 (K=10, 50, 100)
  Fig 03. 모델별 다중 메트릭
  Fig 04. SHAP 글로벌 피처 중요도 Top 15
  Fig 05. SHAP 그룹별 영향력 비율
  Fig 06. 핵심 피처 양성/음성 분포
  Fig 07. 공개차수별 모델 성능
  Fig 08. OOF 예측 점수 분포
  Fig 09. 결측 패턴 — 진짜 위험 신호
  Fig 10. 산업 카테고리별 양성률
  Fig 11. 양성/음성 시계열 평균 패턴
  Fig 12. 핵심 피처 방사형 비교
"""
import os
import sys
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 한글 폰트 설정
# ============================================================
def setup_korean_font():
    system = platform.system()
    if system == 'Windows':
        for p in ['C:/Windows/Fonts/malgun.ttf', 'C:/Windows/Fonts/malgunbd.ttf']:
            if os.path.exists(p):
                fm.fontManager.addfont(p)
        font_name = 'Malgun Gothic'
    elif system == 'Darwin':
        font_name = 'AppleGothic'
    else:
        font_name = 'NanumGothic'
    plt.rcParams['font.family'] = font_name
    plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return font_name

sns.set_style('whitegrid')
font_used = setup_korean_font()
print(f'한글 폰트 설정: {font_used}')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================
# 경로
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'dataset')
MODEL_OUT = os.path.join(BASE, 'model', 'outputs')
OUT_DIR = os.path.join(BASE, 'visualization')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 데이터 로드
# ============================================================
print('데이터 로드 중...')
train_df = pd.read_csv(os.path.join(DATA_DIR, 'training_dataset.csv'), encoding='utf-8-sig')
cv_summary = pd.read_csv(os.path.join(MODEL_OUT, 'cv_results_summary.csv'), encoding='utf-8-sig')
oof = pd.read_csv(os.path.join(MODEL_OUT, 'oof_predictions.csv'), encoding='utf-8-sig')
shap_imp = pd.read_csv(os.path.join(MODEL_OUT, 'shap_importance.csv'), encoding='utf-8-sig')

# 한국어 라벨
MODEL_LABELS = {
    'logreg_full': '로지스틱 회귀',
    'lightgbm_full': 'LightGBM',
    'lightgbm_no_imp': 'LightGBM\n(축소 피처)',
    'catboost_full': 'CatBoost',
    'lightgbm_multi_seed': 'LightGBM\n(5-seed Bagging)',
    'voting_simple': 'Voting (3모델)',
    'voting_with_seeds': 'Voting + Bagging',
    'voting_trees': 'Voting (트리)',
    'voting_weighted': 'Voting (가중)',
}

FEATURE_LABELS = {
    'turnover_avg_12m': '12개월 평균 퇴사율',
    'turnover_avg_3m': '3개월 평균 퇴사율',
    'turnover_max_12m': '12개월 최대 퇴사율',
    'turnover_std_12m': '12개월 퇴사율 변동',
    'emp_change_3m': '3개월 가입자 변화율',
    'emp_change_6m': '6개월 가입자 변화율',
    'emp_change_12m': '12개월 가입자 변화율',
    'salary_avg_12m': '12개월 평균 추정임금',
    'salary_last': '마지막월 추정임금',
    'salary_change_6m': '6개월 임금 변화율',
    'salary_change_12m': '12개월 임금 변화율',
    'replacement_avg_12m': '12개월 평균 인력대체율',
    'replacement_avg_3m': '3개월 평균 인력대체율',
    'replacement_min_12m': '12개월 최저 인력대체율',
    'salary_drop_consecutive': '임금 연속 하락 월수',
    'turnover_momentum': '퇴사율 모멘텀',
    'zero_emp_months': '가입자 0인 월수',
    'emp_volatility': '가입자수 변동성',
    'log_emp_count': '사업장 규모',
    'firm_age_months': '사업장 연령',
    'sido_code': '시도',
    'industry_category': '산업 카테고리',
    'industry_death_rate_2023': '산업 소멸률',
    'imputed_months_count': '결측 월수',
    'imputed_ratio': '결측 비율',
    'has_missing_recent_3m': '최근 3개월 결측',
}

# 색상 — 두 우수 모델을 각각 다른 강조색으로
def get_model_color(m):
    if m == 'voting_weighted':
        return '#d62728'  # red - P@10 1위
    elif m == 'lightgbm_multi_seed':
        return '#2ca02c'  # green - 종합 메트릭 1위
    elif m.startswith('voting'):
        return '#9467bd'  # purple - 다른 voting
    elif 'lightgbm' in m or 'catboost' in m:
        return '#1f77b4'  # blue - 트리 모델
    else:
        return '#7f7f7f'  # gray - baseline

models_list = cv_summary['model'].tolist()
labels = [MODEL_LABELS.get(m, m) for m in models_list]
colors = [get_model_color(m) for m in models_list]

# ============================================================
# Fig 01. 헤드라인 — 모델별 Precision@10
# ============================================================
print('\n[01] Precision@10 헤드라인')
fig, ax = plt.subplots(figsize=(13, 7))
p10 = cv_summary['precision@10_mean'].values
p10_err = cv_summary['precision@10_std'].values

bars = ax.bar(labels, p10, yerr=p10_err, capsize=5, color=colors,
              edgecolor='black', alpha=0.88, linewidth=1.2)
for bar, v in zip(bars, p10):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f'{v:.2f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Precision@10 (상위 10개 위험 사업장 정확도)', fontsize=13)
ax.set_title('체불 위험 사업장 식별 정확도\n— 상위 10개 식별 시 모델별 정답률 —',
             fontsize=15, fontweight='bold', pad=15)
ax.set_ylim(0, 1.05)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, label='무작위 검사 기대치 (50%)')

# 두 우수 모델 범례 추가
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', label='Precision@10 1위 (Voting 가중)'),
    Patch(facecolor='#2ca02c', label='종합 메트릭 1위 (LightGBM 5-seed Bagging)'),
    Patch(facecolor='#9467bd', label='다른 Voting 변형'),
    Patch(facecolor='#1f77b4', label='단일 트리 모델'),
    Patch(facecolor='#7f7f7f', label='베이스라인'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.xticks(rotation=15, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig01_헤드라인_Precision10.png'))
plt.close()

# ============================================================
# Fig 02. Precision@K 종합 (K=10, 50, 100)
# ============================================================
print('[02] Precision@K 종합')
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(models_list))
width = 0.27
p10_v = cv_summary['precision@10_mean'].values
p50_v = cv_summary['precision@50_mean'].values
p100_v = cv_summary['precision@100_mean'].values

bars1 = ax.bar(x - width, p10_v, width, label='Precision@10 (상위 10개)',
               color='#d62728', edgecolor='black', alpha=0.88)
bars2 = ax.bar(x, p50_v, width, label='Precision@50 (상위 50개)',
               color='#ff7f0e', edgecolor='black', alpha=0.88)
bars3 = ax.bar(x + width, p100_v, width, label='Precision@100 (상위 100개)',
               color='#2ca02c', edgecolor='black', alpha=0.88)

for bars, vals in [(bars1, p10_v), (bars2, p50_v), (bars3, p100_v)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.2f}',
                ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
ax.set_ylabel('Precision (정확도)', fontsize=12)
ax.set_title('모델별 상위 K개 위험 사업장 식별 정확도', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.legend(loc='lower left', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig02_PrecisionK_종합.png'))
plt.close()

# ============================================================
# Fig 03. 모델별 다중 메트릭 (Precision 위주)
# ============================================================
print('[03] 모델별 다중 메트릭')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metric_specs = [
    ('precision@10_mean', 'precision@10_std', 'Precision@10\n(상위 10개 정확도)', (0, 1.05)),
    ('precision@50_mean', 'precision@50_std', 'Precision@50\n(상위 50개 정확도)', (0, 1.0)),
    ('roc_auc_mean', 'roc_auc_std', 'ROC-AUC\n(전체 분류 성능)', (0.5, 0.8)),
]
for ax, (mcol, scol, title, ylim) in zip(axes, metric_specs):
    vals = cv_summary[mcol].values
    errs = cv_summary[scol].values
    bars = ax.bar(labels, vals, yerr=errs, capsize=4, color=colors,
                  edgecolor='black', alpha=0.88)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(ylim)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('모델별 종합 성능 — Precision@K 중심', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig03_모델별_다중메트릭.png'))
plt.close()

# ============================================================
# Fig 04. SHAP 글로벌 피처 중요도 Top 15
# ============================================================
print('[04] SHAP 글로벌 피처 중요도')
fig, ax = plt.subplots(figsize=(11, 8))
top15 = shap_imp.head(15).iloc[::-1]
ko_labels = [FEATURE_LABELS.get(f, f) for f in top15['feature']]

def feat_color(f):
    if f in ['imputed_months_count', 'imputed_ratio', 'has_missing_recent_3m']:
        return '#e74c3c'
    elif f in ['sido_code', 'industry_category']:
        return '#9b59b6'
    elif f == 'industry_death_rate_2023':
        return '#f39c12'
    elif f in ['firm_age_months', 'log_emp_count']:
        return '#27ae60'
    else:
        return '#3498db'

bar_colors = [feat_color(f) for f in top15['feature']]
bars = ax.barh(ko_labels, top15['mean_abs_shap'], color=bar_colors,
               edgecolor='black', alpha=0.88)
for bar, v in zip(bars, top15['mean_abs_shap']):
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
            va='center', fontsize=10)

ax.set_xlabel('평균 절대 SHAP 값 (영향력)', fontsize=12)
ax.set_title('체불 예측에 기여하는 핵심 피처 Top 15\n— 시계열·정적·결측·산업 시그널의 복합 활용 —',
             fontsize=14, fontweight='bold', pad=12)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', label='시계열 피처'),
    Patch(facecolor='#27ae60', label='정적 (연령·규모)'),
    Patch(facecolor='#9b59b6', label='카테고리 (시도·산업)'),
    Patch(facecolor='#e74c3c', label='결측 패턴 신호'),
    Patch(facecolor='#f39c12', label='외부 데이터 (KOSIS)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig04_SHAP_피처중요도_Top15.png'))
plt.close()

# ============================================================
# Fig 05. SHAP 그룹별 영향력 비율
# ============================================================
print('[05] SHAP 그룹별 영향력')
ts_features = ['emp_change_3m', 'emp_change_6m', 'emp_change_12m',
               'turnover_avg_12m', 'turnover_avg_3m', 'turnover_max_12m',
               'turnover_std_12m', 'turnover_momentum',
               'salary_change_6m', 'salary_change_12m', 'salary_drop_consecutive',
               'replacement_avg_12m', 'replacement_avg_3m', 'replacement_min_12m',
               'zero_emp_months', 'emp_volatility', 'salary_avg_12m', 'salary_last']
cat_features = ['sido_code', 'industry_category']
imp_features = ['imputed_months_count', 'imputed_ratio', 'has_missing_recent_3m']
static_features = ['firm_age_months', 'log_emp_count']
external_features = ['industry_death_rate_2023']

ts_share = shap_imp[shap_imp.feature.isin(ts_features)]['mean_abs_shap'].sum()
cat_share = shap_imp[shap_imp.feature.isin(cat_features)]['mean_abs_shap'].sum()
imp_share = shap_imp[shap_imp.feature.isin(imp_features)]['mean_abs_shap'].sum()
static_share = shap_imp[shap_imp.feature.isin(static_features)]['mean_abs_shap'].sum()
ext_share = shap_imp[shap_imp.feature.isin(external_features)]['mean_abs_shap'].sum()

sizes = [ts_share, cat_share, imp_share, static_share, ext_share]
group_labels = [
    f'시계열 피처\n({len(ts_features)}개)',
    f'카테고리\n({len(cat_features)}개)',
    f'결측 패턴\n({len(imp_features)}개)',
    f'정적 피처\n({len(static_features)}개)',
    f'외부 데이터\n({len(external_features)}개)',
]
group_colors = ['#3498db', '#9b59b6', '#e74c3c', '#27ae60', '#f39c12']

fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(sizes, labels=group_labels, colors=group_colors,
                                    autopct='%1.1f%%', startangle=90,
                                    textprops={'fontsize': 11, 'fontweight': 'bold'},
                                    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
ax.set_title('모델이 활용하는 피처 그룹별 영향력\n— 시계열 시그널이 주된 판단 근거 —',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig05_SHAP_그룹별_영향력.png'))
plt.close()

# ============================================================
# Fig 06. 핵심 피처 양성/음성 분포 (Boxplot)
# ============================================================
print('[06] 핵심 피처 양성/음성 분포')
key_features = ['emp_change_12m', 'emp_change_6m', 'turnover_momentum',
                'emp_volatility', 'zero_emp_months', 'replacement_min_12m']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, feat in zip(axes.flatten(), key_features):
    data_pos = train_df[train_df.label == 1][feat]
    data_neg = train_df[train_df.label == 0][feat]
    bp = ax.boxplot([data_neg, data_pos],
                    labels=['음성 (정상)', '양성 (체불)'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red', linewidth=2))
    bp['boxes'][1].set_facecolor('lightcoral')
    pos_mean, neg_mean = data_pos.mean(), data_neg.mean()
    ax.set_title(f'{FEATURE_LABELS.get(feat, feat)}\n'
                 f'양성 평균 {pos_mean:.3f} vs 음성 평균 {neg_mean:.3f}',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('체불 사업장의 차별화된 시계열 패턴', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig06_핵심피처_양성음성_분포.png'))
plt.close()

# ============================================================
# Fig 07. 공개차수별 모델 성능 (Precision 기준)
# ============================================================
print('[07] 공개차수별 모델 성능')
key_models = ['lightgbm_multi_seed', 'voting_trees', 'lightgbm_full', 'logreg_full']

# 차수별 Precision@10 계산
def compute_pak_per_gubun(oof_df, model_col, k):
    results = {}
    for g in sorted(oof_df['gubun'].unique()):
        sub = oof_df[oof_df.gubun == g]
        if len(sub) < k:
            k_eff = len(sub)
        else:
            k_eff = k
        sorted_df = sub.sort_values(model_col, ascending=False)
        top_k = sorted_df.head(k_eff)
        results[g] = top_k['label'].mean() if len(top_k) > 0 else 0
    return results

fig, ax = plt.subplots(figsize=(13, 7))
gubuns = sorted(oof['gubun'].unique())
x = np.arange(len(gubuns))
width = 0.2
for i, m in enumerate(key_models):
    col = f'pred_{m}'
    if col not in oof.columns:
        continue
    pak_dict = compute_pak_per_gubun(oof, col, k=10)
    vals = [pak_dict.get(g, 0) for g in gubuns]
    ax.bar(x + i*width, vals, width, label=MODEL_LABELS.get(m, m).replace('\n', ' '),
           edgecolor='black', alpha=0.88)

ax.set_xlabel('체불 명단 공개차수', fontsize=12)
ax.set_ylabel('Precision@10 (상위 10개 정확도)', fontsize=12)
ax.set_title('공개차수별 모델 식별 정확도', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(gubuns)
ax.legend(loc='lower left', fontsize=10)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig07_공개차수별_식별정확도.png'))
plt.close()

# ============================================================
# Fig 08. OOF 예측 점수 분포 (양성/음성)
# ============================================================
print('[08] OOF 예측 점수 분포')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, m in zip(axes, ['lightgbm_full', 'lightgbm_multi_seed']):
    col = f'pred_{m}'
    pos = oof[oof.label == 1][col]
    neg = oof[oof.label == 0][col]
    ax.hist(neg, bins=30, alpha=0.65, label='음성 (정상)', color='steelblue', edgecolor='black')
    ax.hist(pos, bins=30, alpha=0.65, label='양성 (체불)', color='salmon', edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='임계값 0.5')
    ax.set_xlabel('예측 점수', fontsize=11)
    ax.set_ylabel('빈도', fontsize=11)
    ax.set_title(MODEL_LABELS.get(m, m).replace('\n', ' '), fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('OOF 예측 점수 분포 — 양성·음성 분리도', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig08_OOF_예측점수_분포.png'))
plt.close()

# ============================================================
# Fig 09. 결측 패턴 — 진짜 위험 신호 (긍정 reframe)
# ============================================================
print('[09] 결측 패턴 — 위험 신호')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: 결측 여부에 따른 양성률
ax = axes[0]
miss_groups = train_df.groupby('has_missing_recent_3m').agg(
    n=('label', 'size'),
    pos_rate=('label', lambda x: (x==1).mean()*100),
    emp_change=('emp_change_12m', lambda x: x.mean()*100),
)

bars = ax.bar(['최근 3개월\n결측 없음', '최근 3개월\n결측 있음'],
              miss_groups['pos_rate'].values,
              color=['steelblue', 'salmon'], edgecolor='black', alpha=0.88, width=0.5)
for bar, n, rate in zip(bars, miss_groups['n'], miss_groups['pos_rate']):
    ax.text(bar.get_x() + bar.get_width()/2, rate + 2, f'{rate:.1f}%\n(n={n})',
            ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('체불 양성률 (%)', fontsize=12)
ax.set_title('결측 패턴 = 운영 불안정 신호\n양성률 +33%pt 증가',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Right: 결측 여부에 따른 가입자수 변화
ax = axes[1]
emp_changes = miss_groups['emp_change'].values
bars = ax.bar(['최근 3개월\n결측 없음', '최근 3개월\n결측 있음'],
              emp_changes,
              color=['steelblue', 'salmon'], edgecolor='black', alpha=0.88, width=0.5)
for bar, v in zip(bars, emp_changes):
    offset = 0.5 if v > 0 else -1.5
    ax.text(bar.get_x() + bar.get_width()/2, v + offset, f'{v:+.1f}%',
            ha='center', fontsize=12, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('12개월 가입자수 변화율 (%)', fontsize=12)
ax.set_title('결측 사업장은 가입자도 동시 감소\n(보험료 미납·신고 누락 동반 패턴)',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

fig.suptitle('결측 패턴이 의미하는 것 — 운영 불안정의 객관적 지표',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig09_결측패턴_위험신호.png'))
plt.close()

# ============================================================
# Fig 10. 산업 카테고리별 양성률
# ============================================================
print('[10] 산업 카테고리별 양성률')
cat_rate = train_df.groupby('industry_category').agg(
    n=('label', 'size'),
    pos_rate=('label', 'mean')
).reset_index()
cat_rate = cat_rate[cat_rate['n'] >= 10].sort_values('pos_rate', ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
bar_colors = ['#e74c3c' if r > 0.6 else '#f39c12' if r > 0.4 else '#3498db'
              for r in cat_rate['pos_rate']]
bars = ax.barh(cat_rate['industry_category'], cat_rate['pos_rate'] * 100,
               color=bar_colors, edgecolor='black', alpha=0.88)
for bar, n, rate in zip(bars, cat_rate['n'], cat_rate['pos_rate']):
    ax.text(rate * 100 + 1, bar.get_y() + bar.get_height()/2,
            f'{rate*100:.1f}% (n={n})', va='center', fontsize=10)

ax.set_xlabel('체불 양성률 (%)', fontsize=12)
ax.set_title('산업 카테고리별 체불 위험도\n— 도메인 통계 일치 (건설업·부동산업 高) —',
             fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig10_산업카테고리별_양성률.png'))
plt.close()

# ============================================================
# Fig 11. 양성/음성 시계열 평균 패턴 (가입자수 변화)
# ============================================================
print('[11] 양성/음성 시계열 평균 패턴')
ts_metrics = {
    'emp_change_3m': '3개월',
    'emp_change_6m': '6개월',
    'emp_change_12m': '12개월',
}
fig, ax = plt.subplots(figsize=(11, 6))
periods = list(ts_metrics.values())
pos_means = [train_df[train_df.label==1][f].mean() * 100 for f in ts_metrics.keys()]
neg_means = [train_df[train_df.label==0][f].mean() * 100 for f in ts_metrics.keys()]

x = np.arange(len(periods))
width = 0.35
bars1 = ax.bar(x - width/2, neg_means, width, label='음성 (정상)',
               color='steelblue', edgecolor='black', alpha=0.88)
bars2 = ax.bar(x + width/2, pos_means, width, label='양성 (체불)',
               color='salmon', edgecolor='black', alpha=0.88)

for bars, vals in [(bars1, neg_means), (bars2, pos_means)]:
    for bar, v in zip(bars, vals):
        offset = 0.3 if v > 0 else -0.6
        ax.text(bar.get_x() + bar.get_width()/2, v + offset,
                f'{v:+.1f}%', ha='center', fontsize=11, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(periods)
ax.set_xlabel('변화율 측정 기간', fontsize=12)
ax.set_ylabel('가입자수 변화율 평균 (%)', fontsize=12)
ax.set_title('체불 사업장의 가입자 감소 패턴\n— 공개 6개월 전부터 명확한 인력 이탈 —',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig11_가입자_변화_패턴.png'))
plt.close()

# ============================================================
# Fig 12. 핵심 피처 방사형 비교
# ============================================================
print('[12] 핵심 피처 방사형 비교')
key_for_radar = ['emp_change_12m', 'emp_change_6m', 'turnover_momentum',
                 'emp_volatility', 'zero_emp_months', 'replacement_min_12m']
labels_radar = [FEATURE_LABELS[f].replace(' ', '\n') for f in key_for_radar]

pos_vals = []
neg_vals = []
for f in key_for_radar:
    p = train_df[train_df.label==1][f].mean()
    n = train_df[train_df.label==0][f].mean()
    max_val = max(abs(p), abs(n)) or 1
    pos_vals.append(abs(p) / max_val)
    neg_vals.append(abs(n) / max_val)

angles = np.linspace(0, 2*np.pi, len(labels_radar), endpoint=False).tolist()
pos_vals += pos_vals[:1]
neg_vals += neg_vals[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))
ax.plot(angles, pos_vals, color='salmon', linewidth=2.5, label='양성 (체불)')
ax.fill(angles, pos_vals, color='salmon', alpha=0.3)
ax.plot(angles, neg_vals, color='steelblue', linewidth=2.5, label='음성 (정상)')
ax.fill(angles, neg_vals, color='steelblue', alpha=0.3)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_radar, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('체불 사업장의 다차원 위험 패턴\n— 6개 핵심 시그널의 동시 활성화 —',
             fontsize=13, fontweight='bold', pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig12_핵심피처_방사형비교.png'))
plt.close()

# ============================================================
# 완료
# ============================================================
print(f'\n✅ 12개 그림 저장 완료: {OUT_DIR}')
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.png'):
        path = os.path.join(OUT_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f'  {f} ({size_kb:.0f} KB)')
