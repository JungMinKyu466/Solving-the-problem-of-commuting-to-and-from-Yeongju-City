"""사업장당 스냅샷 수가 적은 케이스 진단"""
import csv, os, sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
TREND = os.path.join(BASE, 'pension_trend.csv')

rows_by_biz = defaultdict(list)
with open(TREND, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        key = (row['사업장명'], row['사업자등록번호'])
        rows_by_biz[key].append(row)

# 스냅샷 수 분포
buckets = defaultdict(int)
for key, rows in rows_by_biz.items():
    n = len(rows)
    if n == 1: bucket = '1개월'
    elif n <= 3: bucket = '2~3개월'
    elif n <= 6: bucket = '4~6개월'
    elif n <= 12: bucket = '7~12개월'
    elif n <= 24: bucket = '13~24개월'
    elif n <= 40: bucket = '25~40개월'
    else: bucket = '41개월+'
    buckets[bucket] += 1

print("=== 사업장당 스냅샷 수 분포 ===")
for b in ['1개월','2~3개월','4~6개월','7~12개월','13~24개월','25~40개월','41개월+']:
    print(f"  {b}: {buckets.get(b,0)}개")

# 스냅샷 적은 사업장의 특성 (1~3개월짜리)
print("\n=== 스냅샷 1~3개월짜리 사업장 진단 (상위 30건) ===")
short = [(k,v) for k,v in rows_by_biz.items() if len(v) <= 3]
print(f"총 {len(short)}개\n")

reasons = defaultdict(int)
for key, rows in short:
    rows.sort(key=lambda r: r['기준월'])
    first, last = rows[0], rows[-1]
    stage = first['매칭_단계']
    has_taegoe = any(r['탈퇴일자'] for r in rows)
    state_codes = set(r['가입상태코드'] for r in rows)

    # 분류
    if stage.startswith('5단계'):
        reason = 'A. 5단계 매칭 (false-positive 가능)'
    elif stage.startswith('3단계') or stage.startswith('4단계'):
        reason = 'B. 3~4단계 부분포함 매칭'
    elif has_taegoe:
        reason = 'C. 탈퇴(폐업) — 탈퇴일자 있음'
    elif '2' in state_codes:
        reason = 'D. 가입상태=탈퇴'
    elif first['기준월'] >= '2025':
        reason = 'E. 최근 신규 등록'
    else:
        reason = 'F. 기타'
    reasons[reason] += 1

print("분류별 카운트:")
for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
    print(f"  {r}: {c}개")

print("\n샘플 (최대 30건):")
for key, rows in short[:30]:
    rows.sort(key=lambda r: r['기준월'])
    months = [r['기준월'] for r in rows]
    stage = rows[0]['매칭_단계']
    state = rows[-1]['가입상태코드']
    taegoe = rows[-1]['탈퇴일자']
    gubun = rows[0]['체불공개구분']
    print(f"  {key[0]} | biz={key[1]} | {stage} | 월={months} | 상태={state} | 탈퇴={taegoe} | 공개={gubun}")
