"""
임금체불 위험 예측 모델 학습 (LightGBM 메인 + Logistic Regression baseline + CatBoost 비교).

설계 결정:
  - GroupKFold by `source_business` (augmented 샘플 leakage 방지)
  - Imbalance 1:1 (사전 다운샘플링) → scale_pos_weight 사용 안 함
  - Categorical 직접 지원 (LightGBM/CatBoost), LogReg는 one-hot
  - 평가 메트릭: ROC-AUC, PR-AUC, Precision@K (K=10, 50)
  - F15~F17 결측 패턴 피처 A/B 비교 (with/without)
  - 5-fold × 5 repeats CV → 분산 측정

출력:
  - model/outputs/cv_results.csv (모델별 fold별 메트릭)
  - model/outputs/lgbm_model.pkl (최종 학습 모델)
  - model/outputs/feature_importance.csv (LightGBM 중요도)
  - model/outputs/predictions.csv (out-of-fold 예측)
"""
import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

import lightgbm as lgb

# CatBoost는 선택적 (학습 느림)
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 경로 / 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_BASE = os.path.join(os.path.dirname(BASE_DIR), 'dataset')
TRAIN_CSV = os.path.join(DATASET_BASE, 'training_dataset.csv')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_FOLDS = 5
N_REPEATS = 5
PRECISION_AT_K = [10, 50, 100]

# ============================================================
# 피처 정의
# ============================================================
META_COLS = ['business_id', 'source_business', 'label', 'gubun', 't_disclose',
             'shift', 'coverage', 'last_month', 'n_months_observed']

# F15~F17 결측 패턴 피처 (A/B 비교 대상)
IMPUTED_FEATURES = ['imputed_months_count', 'imputed_ratio', 'has_missing_recent_3m']

# Categorical 피처
CATEGORICAL = ['sido_code', 'industry_category']

# 시계열·정적 수치 피처
NUMERIC_FEATURES = [
    # F01 turnover (4)
    'turnover_avg_12m', 'turnover_avg_3m', 'turnover_max_12m', 'turnover_std_12m',
    # F02 emp_change (3)
    'emp_change_3m', 'emp_change_6m', 'emp_change_12m',
    # F03 salary (4)
    'salary_avg_12m', 'salary_last', 'salary_change_6m', 'salary_change_12m',
    # F04 replacement (3)
    'replacement_avg_12m', 'replacement_avg_3m', 'replacement_min_12m',
    # F05~F08, F10
    'salary_drop_consecutive', 'turnover_momentum', 'zero_emp_months',
    'emp_volatility', 'log_emp_count',
    # F09
    'firm_age_months',
    # F14
    'industry_death_rate_2023',
]


def get_feature_set(include_imputed=True):
    """피처 셋 반환. include_imputed=False면 F15~F17 제외 (A/B 비교)."""
    feats = list(NUMERIC_FEATURES) + list(CATEGORICAL)
    if include_imputed:
        feats += IMPUTED_FEATURES
    return feats


# ============================================================
# 데이터 로드
# ============================================================
print('=' * 60)
print('학습 데이터 로드')
print('=' * 60)
df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
print(f'  총 {len(df)}건 (양성 {(df["label"]==1).sum()}, 음성 {(df["label"]==0).sum()})')
print(f'  unique source_business: {df["source_business"].nunique()}개')

# Categorical 컬럼 변환
for c in CATEGORICAL:
    df[c] = df[c].astype('category')


# ============================================================
# 메트릭 헬퍼
# ============================================================
def compute_metrics(y_true, y_score):
    """주요 메트릭 계산."""
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_score),
        'pr_auc': average_precision_score(y_true, y_score),
    }
    # Precision@K: 상위 K개 예측의 정확도
    n_pos = int(y_true.sum())
    sorted_idx = np.argsort(-y_score)
    for k in PRECISION_AT_K:
        k_eff = min(k, len(y_true))
        top_k = sorted_idx[:k_eff]
        metrics[f'precision@{k}'] = y_true.iloc[top_k].mean() if hasattr(y_true, 'iloc') else y_true[top_k].mean()
    return metrics


# ============================================================
# 모델 학습 함수
# ============================================================
def train_lightgbm(X_train, y_train, X_val, y_val, cat_features, seed=SEED):
    """LightGBM 학습 + early stopping. seed 인자로 재현성·다양성 제어."""
    train_set = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
    val_set = lgb.Dataset(X_val, y_val, categorical_feature=cat_features, reference=train_set)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 15,
        'max_depth': 4,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 10,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'verbose': -1,
        'random_state': seed,
    }
    model = lgb.train(
        params, train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                   lgb.log_evaluation(period=0)],
    )
    return model


def train_logreg(X_train, y_train, X_val, y_val):
    """Logistic Regression baseline (one-hot encoded)."""
    # One-hot 카테고리
    X_train_oh = pd.get_dummies(X_train, columns=CATEGORICAL, drop_first=True)
    X_val_oh = pd.get_dummies(X_val, columns=CATEGORICAL, drop_first=True)
    # 컬럼 align
    X_val_oh = X_val_oh.reindex(columns=X_train_oh.columns, fill_value=0)
    # 결측 / inf 처리
    X_train_oh = X_train_oh.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_val_oh = X_val_oh.replace([np.inf, -np.inf], np.nan).fillna(0)
    # 표준화 (LR은 스케일 영향)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_oh)
    X_val_scaled = scaler.transform(X_val_oh)
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X_train_oh.columns


def train_catboost(X_train, y_train, X_val, y_val, cat_features):
    """CatBoost 학습."""
    cat_idx = [X_train.columns.get_loc(c) for c in cat_features]
    model = CatBoostClassifier(
        iterations=500,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=SEED,
        verbose=0,
        cat_features=cat_idx,
        early_stopping_rounds=30,
    )
    # CatBoost는 categorical을 string으로 받음
    X_train_cb = X_train.copy()
    X_val_cb = X_val.copy()
    for c in cat_features:
        X_train_cb[c] = X_train_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)
    model.fit(X_train_cb, y_train, eval_set=(X_val_cb, y_val))
    return model


# ============================================================
# Cross-Validation 학습 루프
# ============================================================
def run_cv(df, features, model_name, repeats=N_REPEATS):
    """GroupKFold × repeats 반복 CV.

    Returns:
      fold_metrics: list of dict (fold당 메트릭)
      oof_predictions: pd.Series (out-of-fold 예측, 인덱스 = df.index)
    """
    fold_metrics = []
    # OOF 예측 누적
    oof_pred_sum = np.zeros(len(df))
    oof_pred_count = np.zeros(len(df))

    cat_in_features = [c for c in CATEGORICAL if c in features]

    for rep in range(repeats):
        gkf = GroupKFold(n_splits=N_FOLDS)
        # GroupKFold는 random_state 미지원이라 데이터 셔플로 변형
        df_shuffled_idx = df.sample(frac=1, random_state=SEED + rep).index
        df_rep = df.loc[df_shuffled_idx].reset_index(drop=True)
        groups = df_rep['source_business']

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(df_rep, df_rep['label'], groups=groups)):
            X_tr = df_rep.iloc[tr_idx][features].copy()
            y_tr = df_rep.iloc[tr_idx]['label'].astype(int)
            X_va = df_rep.iloc[va_idx][features].copy()
            y_va = df_rep.iloc[va_idx]['label'].astype(int)

            if model_name == 'lightgbm':
                model = train_lightgbm(X_tr, y_tr, X_va, y_va, cat_in_features, seed=SEED)
                y_pred = model.predict(X_va, num_iteration=model.best_iteration)
            elif model_name == 'lightgbm_multi_seed':
                # 5-seed bagging: 같은 데이터에 다른 시드로 학습 후 평균
                preds_seeds = []
                for s in MULTI_SEEDS:
                    m = train_lightgbm(X_tr, y_tr, X_va, y_va, cat_in_features, seed=s)
                    preds_seeds.append(m.predict(X_va, num_iteration=m.best_iteration))
                y_pred = np.mean(preds_seeds, axis=0)
            elif model_name == 'logreg':
                model, scaler, cols = train_logreg(X_tr, y_tr, X_va, y_va)
                X_va_oh = pd.get_dummies(X_va, columns=CATEGORICAL, drop_first=True)
                X_va_oh = X_va_oh.reindex(columns=cols, fill_value=0)
                X_va_oh = X_va_oh.replace([np.inf, -np.inf], np.nan).fillna(0)
                X_va_scaled = scaler.transform(X_va_oh)
                y_pred = model.predict_proba(X_va_scaled)[:, 1]
            elif model_name == 'catboost' and HAS_CATBOOST:
                model = train_catboost(X_tr, y_tr, X_va, y_va, cat_in_features)
                X_va_cb = X_va.copy()
                for c in cat_in_features:
                    X_va_cb[c] = X_va_cb[c].astype(str)
                y_pred = model.predict_proba(X_va_cb)[:, 1]
            else:
                continue

            metrics = compute_metrics(y_va.values, y_pred)
            metrics['model'] = model_name
            metrics['rep'] = rep
            metrics['fold'] = fold
            metrics['n_train'] = len(tr_idx)
            metrics['n_val'] = len(va_idx)
            metrics['n_pos_val'] = int(y_va.sum())
            fold_metrics.append(metrics)

            # 원본 df 인덱스 기준 OOF 누적
            orig_idx = df_shuffled_idx[va_idx]
            oof_pred_sum[df.index.get_indexer(orig_idx)] += y_pred
            oof_pred_count[df.index.get_indexer(orig_idx)] += 1

    # 평균 OOF 예측
    oof_pred = np.divide(oof_pred_sum, oof_pred_count,
                         out=np.zeros_like(oof_pred_sum),
                         where=oof_pred_count > 0)
    return fold_metrics, oof_pred


def summarize_metrics(fold_metrics, name):
    """fold별 메트릭 → 평균 ± std 요약."""
    if not fold_metrics:
        return None
    df_m = pd.DataFrame(fold_metrics)
    summary = {'model': name, 'n_folds': len(df_m)}
    for k in ['roc_auc', 'pr_auc'] + [f'precision@{k}' for k in PRECISION_AT_K]:
        if k in df_m.columns:
            summary[f'{k}_mean'] = df_m[k].mean()
            summary[f'{k}_std'] = df_m[k].std()
    return summary


# ============================================================
# 실험 정의 — 4개 base 모델 (앙상블은 후처리)
# ============================================================
EXPERIMENTS = [
    ('logreg_full', 'logreg', True),         # baseline
    ('lightgbm_full', 'lightgbm', True),     # 메인 (F15~F17 포함)
    ('lightgbm_no_imp', 'lightgbm', False),  # A/B (F15~F17 제외)
]
if HAS_CATBOOST:
    EXPERIMENTS.append(('catboost_full', 'catboost', True))

# Multi-seed bagging seeds (LightGBM에 적용)
MULTI_SEEDS = [42, 123, 456, 789, 1000]

# Multi-seed 실험 추가
EXPERIMENTS.append(('lightgbm_multi_seed', 'lightgbm_multi_seed', True))


# ============================================================
# 실험 실행
# ============================================================
all_fold_metrics = []
all_summaries = []
oof_predictions = {}

for exp_name, model_name, include_imputed in EXPERIMENTS:
    print(f'\n{"="*60}')
    print(f'실험: {exp_name}')
    print(f'  모델: {model_name}, F15~F17 포함: {include_imputed}')
    print(f'{"="*60}')
    features = get_feature_set(include_imputed=include_imputed)
    print(f'  피처 수: {len(features)}')
    fold_metrics, oof_pred = run_cv(df, features, model_name, repeats=N_REPEATS)
    for m in fold_metrics:
        m['experiment'] = exp_name
    all_fold_metrics.extend(fold_metrics)
    summary = summarize_metrics(fold_metrics, exp_name)
    all_summaries.append(summary)
    oof_predictions[exp_name] = oof_pred
    print(f'  결과: ROC-AUC {summary["roc_auc_mean"]:.3f} ± {summary["roc_auc_std"]:.3f}')
    print(f'        PR-AUC  {summary["pr_auc_mean"]:.3f} ± {summary["pr_auc_std"]:.3f}')
    print(f'        Precision@10 {summary["precision@10_mean"]:.3f} ± {summary["precision@10_std"]:.3f}')


# ============================================================
# Voting Ensemble 계산 (사후 처리)
# ============================================================
print(f'\n{"="*60}')
print('Voting Ensemble 계산')
print(f'{"="*60}')


def compute_voting_metrics(oof_pred, df):
    """Voting OOF 예측에서 메트릭 계산 (전체 데이터 기준 단일 점수)."""
    y = df['label'].astype(int).values
    return compute_metrics(pd.Series(y), oof_pred)


def compute_voting_fold_metrics(oof_pred, df, name, base_fold_metrics):
    """기존 fold 분할을 그대로 사용해 voting의 fold별 메트릭 계산."""
    fold_metrics = []
    # base_fold_metrics에서 어떤 fold 인덱스가 사용됐는지는 직접 모름
    # → 동일 GroupKFold 셔플로 재현
    for rep in range(N_REPEATS):
        gkf = GroupKFold(n_splits=N_FOLDS)
        df_shuffled_idx = df.sample(frac=1, random_state=SEED + rep).index
        df_rep = df.loc[df_shuffled_idx].reset_index(drop=True)
        groups = df_rep['source_business']
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(df_rep, df_rep['label'], groups=groups)):
            orig_idx = df_shuffled_idx[va_idx]
            y_va = df.loc[orig_idx, 'label'].astype(int).values
            y_pred_va = oof_pred[df.index.get_indexer(orig_idx)]
            m = compute_metrics(pd.Series(y_va), y_pred_va)
            m['model'] = name
            m['rep'] = rep
            m['fold'] = fold
            m['n_train'] = len(tr_idx)
            m['n_val'] = len(va_idx)
            m['n_pos_val'] = int(y_va.sum())
            m['experiment'] = name
            fold_metrics.append(m)
    return fold_metrics


# voting_simple = (logreg + lightgbm_full + catboost) / 3
if all(k in oof_predictions for k in ['logreg_full', 'lightgbm_full', 'catboost_full']):
    voting_simple = (oof_predictions['logreg_full'] +
                     oof_predictions['lightgbm_full'] +
                     oof_predictions['catboost_full']) / 3
    oof_predictions['voting_simple'] = voting_simple
    fm = compute_voting_fold_metrics(voting_simple, df, 'voting_simple', all_fold_metrics)
    all_fold_metrics.extend(fm)
    summary = summarize_metrics(fm, 'voting_simple')
    all_summaries.append(summary)
    print(f'\nvoting_simple = (LogReg + LightGBM + CatBoost) / 3')
    print(f'  ROC-AUC: {summary["roc_auc_mean"]:.3f} ± {summary["roc_auc_std"]:.3f}')
    print(f'  PR-AUC:  {summary["pr_auc_mean"]:.3f} ± {summary["pr_auc_std"]:.3f}')
    print(f'  P@10:    {summary["precision@10_mean"]:.3f}')

# voting_with_seeds = (logreg + lightgbm_multi_seed + catboost) / 3
if all(k in oof_predictions for k in ['logreg_full', 'lightgbm_multi_seed', 'catboost_full']):
    voting_seeds = (oof_predictions['logreg_full'] +
                    oof_predictions['lightgbm_multi_seed'] +
                    oof_predictions['catboost_full']) / 3
    oof_predictions['voting_with_seeds'] = voting_seeds
    fm = compute_voting_fold_metrics(voting_seeds, df, 'voting_with_seeds', all_fold_metrics)
    all_fold_metrics.extend(fm)
    summary = summarize_metrics(fm, 'voting_with_seeds')
    all_summaries.append(summary)
    print(f'\nvoting_with_seeds = (LogReg + LightGBM_5seed + CatBoost) / 3')
    print(f'  ROC-AUC: {summary["roc_auc_mean"]:.3f} ± {summary["roc_auc_std"]:.3f}')
    print(f'  PR-AUC:  {summary["pr_auc_mean"]:.3f} ± {summary["pr_auc_std"]:.3f}')
    print(f'  P@10:    {summary["precision@10_mean"]:.3f}')

# voting_trees = (lightgbm_multi_seed + catboost) / 2 — LogReg 제외
if all(k in oof_predictions for k in ['lightgbm_multi_seed', 'catboost_full']):
    voting_trees = (oof_predictions['lightgbm_multi_seed'] +
                    oof_predictions['catboost_full']) / 2
    oof_predictions['voting_trees'] = voting_trees
    fm = compute_voting_fold_metrics(voting_trees, df, 'voting_trees', all_fold_metrics)
    all_fold_metrics.extend(fm)
    summary = summarize_metrics(fm, 'voting_trees')
    all_summaries.append(summary)
    print(f'\nvoting_trees = (LightGBM_5seed + CatBoost) / 2')
    print(f'  ROC-AUC: {summary["roc_auc_mean"]:.3f} ± {summary["roc_auc_std"]:.3f}')
    print(f'  PR-AUC:  {summary["pr_auc_mean"]:.3f} ± {summary["pr_auc_std"]:.3f}')
    print(f'  P@10:    {summary["precision@10_mean"]:.3f}')

# voting_weighted = LightGBM 0.5 + CatBoost 0.3 + LogReg 0.2 (가중)
if all(k in oof_predictions for k in ['logreg_full', 'lightgbm_multi_seed', 'catboost_full']):
    voting_weighted = (0.5 * oof_predictions['lightgbm_multi_seed'] +
                       0.3 * oof_predictions['catboost_full'] +
                       0.2 * oof_predictions['logreg_full'])
    oof_predictions['voting_weighted'] = voting_weighted
    fm = compute_voting_fold_metrics(voting_weighted, df, 'voting_weighted', all_fold_metrics)
    all_fold_metrics.extend(fm)
    summary = summarize_metrics(fm, 'voting_weighted')
    all_summaries.append(summary)
    print(f'\nvoting_weighted = 0.5×LGBM + 0.3×CatBoost + 0.2×LogReg')
    print(f'  ROC-AUC: {summary["roc_auc_mean"]:.3f} ± {summary["roc_auc_std"]:.3f}')
    print(f'  PR-AUC:  {summary["pr_auc_mean"]:.3f} ± {summary["pr_auc_std"]:.3f}')
    print(f'  P@10:    {summary["precision@10_mean"]:.3f}')


# ============================================================
# 결과 저장
# ============================================================
print(f'\n{"="*60}')
print('결과 저장')
print(f'{"="*60}')

# fold별 결과
fold_df = pd.DataFrame(all_fold_metrics)
fold_df.to_csv(os.path.join(OUT_DIR, 'cv_results_fold.csv'),
               index=False, encoding='utf-8-sig')
print(f'  fold별 메트릭: cv_results_fold.csv ({len(fold_df)} rows)')

# 요약
summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv(os.path.join(OUT_DIR, 'cv_results_summary.csv'),
                  index=False, encoding='utf-8-sig')
print(f'  요약: cv_results_summary.csv')

# OOF 예측
oof_df = df[['business_id', 'source_business', 'label', 'gubun', 'shift']].copy()
for exp_name, pred in oof_predictions.items():
    oof_df[f'pred_{exp_name}'] = pred
oof_df.to_csv(os.path.join(OUT_DIR, 'oof_predictions.csv'),
              index=False, encoding='utf-8-sig')
print(f'  OOF 예측: oof_predictions.csv')


# ============================================================
# 최종 모델 학습 (LightGBM 전체 데이터)
# ============================================================
print(f'\n{"="*60}')
print('최종 모델 학습 (전체 데이터, LightGBM)')
print(f'{"="*60}')
features_full = get_feature_set(include_imputed=True)
cat_in_features = [c for c in CATEGORICAL if c in features_full]
X_full = df[features_full]
y_full = df['label'].astype(int)
final_train_set = lgb.Dataset(X_full, y_full, categorical_feature=cat_in_features)
params = {
    'objective': 'binary', 'metric': 'auc',
    'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_data_in_leaf': 10, 'reg_alpha': 1.0, 'reg_lambda': 1.0,
    'verbose': -1, 'random_state': SEED,
}
# CV 평균 best_iteration 추정
mean_best_iter = int(np.mean([fm.get('best_iteration', 200)
                              for fm in all_fold_metrics
                              if fm['experiment'] == 'lightgbm_full'])) or 200
final_model = lgb.train(params, final_train_set, num_boost_round=200)

# 모델 저장
with open(os.path.join(OUT_DIR, 'lgbm_final_model.pkl'), 'wb') as f:
    pickle.dump({
        'model': final_model,
        'features': features_full,
        'categorical': cat_in_features,
        'params': params,
    }, f)
print(f'  최종 모델: lgbm_final_model.pkl')

# Feature importance
importance = final_model.feature_importance(importance_type='gain')
fi_df = pd.DataFrame({
    'feature': features_full,
    'importance_gain': importance,
}).sort_values('importance_gain', ascending=False)
fi_df['rank'] = range(1, len(fi_df) + 1)
fi_df.to_csv(os.path.join(OUT_DIR, 'feature_importance.csv'),
             index=False, encoding='utf-8-sig')
print(f'  피처 중요도: feature_importance.csv')
print('\n  Top 10 피처:')
for _, row in fi_df.head(10).iterrows():
    print(f'    {row["rank"]:>2}. {row["feature"]:<35} {row["importance_gain"]:>10.1f}')

# ============================================================
# 최종 요약 출력
# ============================================================
print(f'\n{"="*60}')
print('실험 결과 종합')
print(f'{"="*60}')
print(summary_df.to_string(index=False))
print(f'\n저장 위치: {OUT_DIR}')
