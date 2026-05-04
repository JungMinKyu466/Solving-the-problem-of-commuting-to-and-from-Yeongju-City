"""임금체불 사업장 단위로 국민연금 데이터 보유 기간 집계"""
import csv, os, sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
TREND = os.path.join(BASE, 'pension_trend.csv')

# 임금체불 사업장명 단위로 고유 기준월 집계
months_by_unpaid = defaultdict(set)
biz_by_unpaid = defaultdict(set)
with open(TREND, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        unpaid_name = row['매칭_임금체불_사업장명']
        months_by_unpaid[unpaid_name].add(row['기준월'])
        biz_by_unpaid[unpaid_name].add((row['사업장명'], row['사업자등록번호']))

total = len(months_by_unpaid)
buckets = defaultdict(int)
for name, months in months_by_unpaid.items():
    n = len(months)
    if n >= 36: b = '36개월+'
    elif n >= 24: b = '24~35개월'
    elif n >= 12: b = '12~23개월'
    elif n >= 6: b = '6~11개월'
    elif n >= 3: b = '3~5개월'
    else: b = '1~2개월'
    buckets[b] += 1

print(f"=== 임금체불 사업장 {total}개 기준 - 보유 기준월 수 ===")
print("(같은 임금체불 사업장에 여러 국민연금 사업장이 매칭된 경우 기준월의 합집합으로 계산)\n")
for b in ['1~2개월','3~5개월','6~11개월','12~23개월','24~35개월','36개월+']:
    c = buckets.get(b, 0)
    print(f"  {b}: {c}개 ({c/total*100:.1f}%)")

ge12 = sum(1 for m in months_by_unpaid.values() if len(m) >= 12)
ge6 = sum(1 for m in months_by_unpaid.values() if len(m) >= 6)
print(f"\n>> 12개월 이상: {ge12}개 ({ge12/total*100:.1f}%)")
print(f">> 6개월 이상: {ge6}개 ({ge6/total*100:.1f}%)")
