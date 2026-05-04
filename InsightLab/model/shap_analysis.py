"""
LightGBM 최종 모델 SHAP 분석.

검증 항목:
  1. 발견 8 (BIZ_NO_MISSING) — 카테고리별 SHAP 영향력
  2. 발견 9 (산업 의존성) — industry_category 단독 영향력
  3. F15~F17 진짜 시그널 vs 추출 차이 인공물 — 결측 피처 SHAP 패턴

출력:
  - model/outputs/shap_summary.png — 글로벌 피처 중요도
  - model/outputs/shap_dependence_*.png — 핵심 피처별 의존성 플롯
  - model/outputs/shap_values.csv — 샘플별 SHAP value (해석용)
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # 화면 출력 X (서버용)
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# 한글 폰트 (Windows 기본)
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_BASE = os.path.join(os.path.dirname(BASE_DIR), 'dataset')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
TRAIN_CSV = os.path.join(DATASET_BASE, 'training_dataset.csv')
MODEL_PKL = os.path.join(OUT_DIR, 'lgbm_final_model.pkl')

CATEGORICAL = ['sido_code', 'industry_category']

# ============================================================
# 모델·데이터 로드
# ============================================================
print('모델 로드 중...')
with open(MODEL_PKL, 'rb') as f:
    saved = pickle.load(f)
model = saved['model']
features = saved['features']
print(f'  피처 수: {len(features)}')

print('데이터 로드 중...')
df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
for c in CATEGORICAL:
    df[c] = df[c].astype('category')
X = df[features]
y = df['label'].astype(int)

# ============================================================
# SHAP value 계산
# ============================================================
print('SHAP value 계산 중...')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 이진 분류에서 LightGBM은 단일 array 반환
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_values = shap_values[1]

print(f'  SHAP shape: {shap_values.shape}')

# ============================================================
# 1. Global Feature Importance (SHAP)
# ============================================================
print('\n[1] 글로벌 피처 중요도')
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_imp_df = pd.DataFrame({
    'feature': features,
    'mean_abs_shap': mean_abs_shap,
}).sort_values('mean_abs_shap', ascending=False)
shap_imp_df['rank'] = range(1, len(shap_imp_df) + 1)
shap_imp_df.to_csv(os.path.join(OUT_DIR, 'shap_importance.csv'),
                   index=False, encoding='utf-8-sig')
print('  Top 15:')
for _, row in shap_imp_df.head(15).iterrows():
    print(f'    {row["rank"]:>2}. {row["feature"]:<35} {row["mean_abs_shap"]:>8.4f}')

# Summary plot 저장
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, plot_type='bar', show=False, max_display=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'shap_summary_bar.png'), dpi=120, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 10))
shap.summary_plot(shap_values, X, show=False, max_display=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'shap_summary.png'), dpi=120, bbox_inches='tight')
plt.close()
print(f'  저장: shap_summary.png, shap_summary_bar.png')

# ============================================================
# 2. 발견 8 검증 — BIZ_NO_MISSING 카테고리 영향력
# ============================================================
print('\n[2] 발견 8 검증: BIZ_NO_MISSING vs 매핑된 카테고리')
ic_idx = features.index('industry_category')
shap_ic = shap_values[:, ic_idx]
df['shap_industry'] = shap_ic
by_cat = df.groupby('industry_category', observed=True).agg(
    n=('label', 'size'),
    shap_mean=('shap_industry', 'mean'),
    shap_abs_mean=('shap_industry', lambda x: np.abs(x).mean()),
).sort_values('shap_abs_mean', ascending=False)
print(by_cat.to_string())

# ============================================================
# 3. 발견 9 검증 — 산업 의존성
# ============================================================
print('\n[3] 발견 9 검증: 카테고리별 양성률 vs SHAP 영향력')
total_shap_abs = np.abs(shap_values).sum()
ic_shap_abs = np.abs(shap_ic).sum()
print(f'  industry_category 절대 SHAP 합계 비율: {ic_shap_abs / total_shap_abs * 100:.1f}%')

sido_idx = features.index('sido_code')
sido_shap_abs = np.abs(shap_values[:, sido_idx]).sum()
print(f'  sido_code 절대 SHAP 합계 비율: {sido_shap_abs / total_shap_abs * 100:.1f}%')

# 시계열 피처 합계
ts_features = ['emp_change_3m', 'emp_change_6m', 'emp_change_12m',
               'turnover_avg_12m', 'turnover_avg_3m', 'turnover_max_12m',
               'turnover_std_12m', 'turnover_momentum',
               'salary_change_6m', 'salary_change_12m', 'salary_drop_consecutive',
               'replacement_avg_12m', 'replacement_avg_3m', 'replacement_min_12m',
               'zero_emp_months', 'emp_volatility']
ts_idx = [features.index(f) for f in ts_features if f in features]
ts_shap_abs = np.abs(shap_values[:, ts_idx]).sum()
print(f'  시계열 피처 합계 절대 SHAP 비율: {ts_shap_abs / total_shap_abs * 100:.1f}%')

imp_features = ['imputed_months_count', 'imputed_ratio', 'has_missing_recent_3m']
imp_idx = [features.index(f) for f in imp_features if f in features]
imp_shap_abs = np.abs(shap_values[:, imp_idx]).sum()
print(f'  결측 피처 합계 절대 SHAP 비율: {imp_shap_abs / total_shap_abs * 100:.1f}%')

# ============================================================
# 4. F15~F17 진짜 vs 인공 시그널 검증
# ============================================================
print('\n[4] F15~F17 결측 피처 분석')
for feat in imp_features:
    if feat in features:
        idx = features.index(feat)
        s = shap_values[:, idx]
        # 양성/음성 SHAP 평균
        s_pos = s[y == 1]
        s_neg = s[y == 0]
        print(f'  {feat}:')
        print(f'    양성 SHAP 평균: {s_pos.mean():>+.4f} (양성 분류 기여)')
        print(f'    음성 SHAP 평균: {s_neg.mean():>+.4f} (음성 분류 기여)')
        print(f'    절대 영향력: {np.abs(s).mean():.4f}')

# ============================================================
# 5. 핵심 피처 dependence plot
# ============================================================
print('\n[5] 핵심 피처 dependence plot 생성')
key_features_for_plot = ['has_missing_recent_3m', 'firm_age_months',
                         'emp_volatility', 'turnover_max_12m',
                         'emp_change_12m', 'salary_change_6m']
for feat in key_features_for_plot:
    if feat not in features:
        continue
    plt.figure(figsize=(8, 5))
    try:
        idx = features.index(feat)
        shap.dependence_plot(idx, shap_values, X, show=False, ax=plt.gca())
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'shap_dep_{feat}.png'), dpi=120,
                    bbox_inches='tight')
        plt.close()
        print(f'    저장: shap_dep_{feat}.png')
    except Exception as e:
        plt.close()
        print(f'    실패 {feat}: {e}')

# ============================================================
# 6. 샘플별 SHAP value 저장
# ============================================================
shap_df = pd.DataFrame(shap_values, columns=[f'shap_{c}' for c in features])
shap_df['business_id'] = df['business_id'].values
shap_df['label'] = df['label'].values
shap_df['source_business'] = df['source_business'].values
shap_df.to_csv(os.path.join(OUT_DIR, 'shap_values.csv'),
               index=False, encoding='utf-8-sig')
print(f'\n  샘플별 SHAP: shap_values.csv ({len(shap_df)} rows)')

print('\n완료')
