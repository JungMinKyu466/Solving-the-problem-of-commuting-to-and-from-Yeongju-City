import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

print('=== Review 2 ===')

oof = pd.read_csv('dataset/../model/outputs/oof_predictions.csv', encoding='utf-8-sig')

from sklearn.metrics import roc_auc_score
print('\n[a] 공개차수별 lightgbm_full OOF AUC')
for g in sorted(oof['gubun'].unique()):
    sub = oof[oof['gubun']==g]
    if sub['label'].nunique() > 1:
        auc = roc_auc_score(sub['label'], sub['pred_lightgbm_full'])
        print(f'  {g}: n={len(sub)} (양성 {sub.label.sum()}), AUC {auc:.3f}')

print('\n[b] shift별 lightgbm_full 양성 OOF 평균 점수')
for s in sorted(oof[oof.label==1]['shift'].unique(), key=int):
    sub = oof[(oof.label==1) & (oof['shift']==s)]
    print(f'  shift {s}: n={len(sub)}, 양성 평균 점수 {sub.pred_lightgbm_full.mean():.3f}')

print('\n[c] 같은 사업장 augmented 샘플 점수 일관성')
src_pred = oof[oof.label==1].groupby('source_business')['pred_lightgbm_full'].agg(['mean', 'std', 'count'])
src_pred = src_pred[src_pred['count'] >= 3]
print(f'  {len(src_pred)}개 사업장 (3+ shifts)')
print(f'  사업장 내 점수 std 평균: {src_pred["std"].mean():.4f}')
print(f'  사업장 내 점수 std 최대: {src_pred["std"].max():.4f}')

print('\n[d] 음성 false-positive (점수 > 0.5)')
fp_neg = oof[(oof.label==0) & (oof['pred_lightgbm_full'] > 0.5)]
print(f'  음성 중 0.5 초과: {len(fp_neg)}/{(oof.label==0).sum()} ({len(fp_neg)/(oof.label==0).sum()*100:.1f}%)')
print(f'  최고 점수 음성 5개:')
print(fp_neg.nlargest(5, 'pred_lightgbm_full')[['business_id','pred_lightgbm_full']].to_string(index=False))

train_df = pd.read_csv('dataset/training_dataset.csv', encoding='utf-8-sig')
print('\n[e] has_missing_recent_3m 값별 분포')
for v in [0, 1]:
    sub = train_df[train_df.has_missing_recent_3m == v]
    if len(sub) > 0:
        pos_pct = (sub.label==1).sum() / len(sub) * 100
        print(f'  값 {v}: n={len(sub)}, 양성 비율 {pos_pct:.1f}%')

print('\n[f] 카테고리 × has_missing_recent_3m 교차분석')
for cat in ['BIZ_NO_MISSING', '건설업', '부동산업', '제조업', 'UNKNOWN']:
    sub = train_df[train_df.industry_category == cat]
    if len(sub) > 0:
        miss_sub = sub[sub.has_missing_recent_3m == 1]
        ratio = len(miss_sub) / len(sub) * 100
        pos_in_miss = ((miss_sub.label==1).sum() / len(miss_sub) * 100) if len(miss_sub) > 0 else 0
        nonmiss_pos = ((sub[sub.has_missing_recent_3m==0].label==1).sum() / max(1,len(sub)-len(miss_sub)) * 100)
        print(f'  {cat}: n={len(sub)}, 결측3m {len(miss_sub)} (양성 {pos_in_miss:.0f}%), 결측없음 양성 {nonmiss_pos:.0f}%')
