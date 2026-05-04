"""
체불-국민연금 매칭 데이터셋 종합 EDA (8가지 점검).
ML 학습 데이터셋 구축 전 데이터 품질·편향·시점 정합성 확인.
"""
import csv, os, sys, re, random
from collections import defaultdict, Counter
from statistics import mean, median

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(10**7)

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
TREND = os.path.join(BASE, 'pension_trend.csv')
MATCHED = os.path.join(BASE, 'matched_pension_unpaid.csv')
UNPAID = os.path.join(BASE, 'unpaid_employers.csv')

print('데이터 로드 중...')
with open(TREND, encoding='utf-8-sig') as f:
    trend_rows = list(csv.DictReader(f))
with open(MATCHED, encoding='utf-8-sig') as f:
    matched_rows = list(csv.DictReader(f))
with open(UNPAID, encoding='utf-8-sig') as f:
    unpaid_rows = list(csv.DictReader(f))
print(f'  trend: {len(trend_rows)} | matched: {len(matched_rows)} | unpaid: {len(unpaid_rows)}\n')

GUBUN_DATE = {
    '2023년 1차': '2023-08', '2023년 2차': '2023-12',
    '2024년 1차': '2024-08', '2024년 2차': '2024-12',
    '2025년 1차': '2025-08', '2025년 2차': '2025-12',
}

def add_months(yyyymm, delta):
    y, m = map(int, yyyymm.split('-'))
    total = y * 12 + (m - 1) + delta
    return f'{total // 12:04d}-{total % 12 + 1:02d}'

biz_keys_by_unpaid = defaultdict(set)
for r in matched_rows:
    biz_keys_by_unpaid[r['매칭_임금체불_사업장명']].add((r['사업장명'], r['사업자등록번호']))

months_by_unpaid = defaultdict(set)
for r in trend_rows:
    months_by_unpaid[r['매칭_임금체불_사업장명']].add(r['기준월'])

first_gubun_by_name = {}
for r in unpaid_rows:
    n = r['사업장명'].strip()
    g = r['구분'].strip()
    if n not in first_gubun_by_name or g < first_gubun_by_name[n]:
        first_gubun_by_name[n] = g

# [1] 양성 입력구간 커버리지
print('=' * 60)
print('[1] 양성 입력구간 커버리지 (lead 18개월, 입력 12개월)')
print('=' * 60)
print('  필요 구간: [공개월-30, 공개월-18]\n')
cover = defaultdict(lambda: {'total':0,'full':0,'partial':0,'none':0})
for unpaid_name, months in months_by_unpaid.items():
    g = first_gubun_by_name.get(unpaid_name)
    if not g or g not in GUBUN_DATE: continue
    disclose = GUBUN_DATE[g]
    needed = []
    cur = add_months(disclose, -30)
    end = add_months(disclose, -18)
    while cur <= end:
        needed.append(cur); cur = add_months(cur, 1)
    have = sum(1 for m in needed if m in months)
    cover[g]['total'] += 1
    if have >= len(needed) * 0.7: cover[g]['full'] += 1
    elif have > 0: cover[g]['partial'] += 1
    else: cover[g]['none'] += 1
print(f"  {'공개차수':<12} {'전체':>5} {'70%+':>6} {'일부':>6} {'없음':>6}")
for g in sorted(cover):
    s = cover[g]
    print(f"  {g:<12} {s['total']:>5} {s['full']:>6} {s['partial']:>6} {s['none']:>6}")
tot_full = sum(s['full'] for s in cover.values())
tot_any = sum(s['total'] for s in cover.values())
print(f"\n  >> 학습 가능(70%+ 커버): {tot_full}/{tot_any} ({tot_full/tot_any*100:.1f}%)")

# [2] 공개 차수별 매칭률
print('\n' + '=' * 60)
print('[2] 공개 차수별 매칭률')
print('=' * 60)
ub = defaultdict(set)
for r in unpaid_rows:
    ub[r['구분']].add(r['사업장명'].strip())
matched_names = set(biz_keys_by_unpaid.keys())
print(f"  {'차수':<12} {'전체':>5} {'매칭':>5} {'매칭률':>7}")
for g in sorted(ub):
    a = ub[g]; m = a & matched_names
    print(f"  {g:<12} {len(a):>5} {len(m):>5} {len(m)/len(a)*100:>6.1f}%")

# [3] 1단계 매칭 무작위 샘플
print('\n' + '=' * 60)
print('[3] 1단계 매칭 샘플 15건 (수동 검증)')
print('=' * 60)
random.seed(42)
stage1 = [r for r in matched_rows if r['매칭_단계'].startswith('1단계')]
samples = random.sample(stage1, min(15, len(stage1)))
print(f"  체불사업장명 | 국민연금사업장명 | 주소")
for r in samples:
    a1 = (r.get('사업장지번상세주소','') or r.get('사업장도로명상세주소',''))[:35]
    print(f"  {r['매칭_임금체불_사업장명'][:25]} | {r['사업장명'][:25]} | {a1}")

# [4] 시계열 결측
print('\n' + '=' * 60)
print('[4] 시계열 결측 분석')
print('=' * 60)
gaps = []
for unpaid_name, months in months_by_unpaid.items():
    if len(months) < 2: continue
    sm = sorted(months)
    cur = sm[0]
    while cur < sm[-1]:
        nxt = add_months(cur, 1)
        if nxt not in months and nxt <= sm[-1]:
            gaps.append((unpaid_name, cur, nxt))
        cur = nxt
n_with_gap = len(set(g[0] for g in gaps))
print(f"  결측 발생 건수: {len(gaps)}")
print(f"  결측 1+ 사업장: {n_with_gap}개 (전체 {len(months_by_unpaid)} 중)")
print(f"  사업장당 평균 결측: {len(gaps)/max(1,n_with_gap):.1f}개월")
print(f"  결측 없는 사업장: {len(months_by_unpaid)-n_with_gap}개")

# [5] 가입자수 분포
print('\n' + '=' * 60)
print('[5] 가입자수 분포')
print('=' * 60)
n_vals = [int(r['가입자수']) for r in trend_rows if r['가입자수'].isdigit()]
zero = sum(1 for v in n_vals if v == 0)
n_pos = [v for v in n_vals if v > 0]
print(f"  전체 row {len(n_vals)}건")
print(f"    0명: {zero}건 ({zero/len(n_vals)*100:.1f}%)")
print(f"    1~10명: {sum(1 for v in n_pos if v<=10)}건")
print(f"    11~50명: {sum(1 for v in n_pos if 10<v<=50)}건")
print(f"    51~200명: {sum(1 for v in n_pos if 50<v<=200)}건")
print(f"    200+명: {sum(1 for v in n_pos if v>200)}건")
if n_pos:
    print(f"    중앙값:{median(n_pos)} 평균:{mean(n_pos):.1f} 최대:{max(n_pos)}")

# [6] 사업장형태/업종
print('\n' + '=' * 60)
print('[6] 사업장형태 / 업종 분포 (matched_rows 기준)')
print('=' * 60)
fc = Counter()
for r in matched_rows:
    fc[r.get('사업장형태구분코드 1 법인 2 개인','').strip()] += 1
print(f"  법인(1): {fc.get('1',0)}  개인(2): {fc.get('2',0)}  미상: {fc.get('',0)}")
ic = Counter()
for r in matched_rows:
    ind = r.get('사업장업종코드명','').strip()
    if ind: ic[ind] += 1
print(f"\n  상위 업종 10:")
for ind, c in ic.most_common(10):
    print(f"    {c:>4} | {ind[:50]}")

# [7] 동일 사업자번호 다중 사업장명
print('\n' + '=' * 60)
print('[7] 동일 사업자번호 다중 사업장명')
print('=' * 60)
b2n = defaultdict(set)
for r in matched_rows:
    b2n[r['사업자등록번호']].add(r['사업장명'])
multi = {b:ns for b,ns in b2n.items() if len(ns)>1}
print(f"  사업자번호 {len(b2n)}개 중 다중명: {len(multi)}개 ({len(multi)/len(b2n)*100:.1f}%)")
print(f"  관련 row: {sum(len(ns) for ns in multi.values())}건")
print(f"  최다 분리 상위 5:")
for b,ns in sorted(multi.items(), key=lambda x:-len(x[1]))[:5]:
    print(f"    biz={b}: {len(ns)}개 — {sorted(ns)[0][:40]}...")

# [8] 탈퇴 시점 vs 공개 시점
print('\n' + '=' * 60)
print('[8] 탈퇴 시점 vs 체불공개 시점')
print('=' * 60)
buckets = Counter()
for r in matched_rows:
    t = r.get('탈퇴일자','').strip()
    if not t:
        buckets['미탈퇴(운영중)'] += 1; continue
    g = first_gubun_by_name.get(r['매칭_임금체불_사업장명'])
    if not g or g not in GUBUN_DATE: continue
    try:
        ty, tm = int(t[:4]), int(t[5:7])
        dy, dm = map(int, GUBUN_DATE[g].split('-'))
        diff = (ty-dy)*12 + (tm-dm)
    except: continue
    if diff < -24: buckets['공개 24개월+ 전 탈퇴'] += 1
    elif diff < -12: buckets['공개 12~24개월 전 탈퇴'] += 1
    elif diff < 0: buckets['공개 0~12개월 전 탈퇴'] += 1
    elif diff < 12: buckets['공개 후 0~12개월 탈퇴'] += 1
    else: buckets['공개 12개월+ 후 탈퇴'] += 1
for b in ['미탈퇴(운영중)','공개 24개월+ 전 탈퇴','공개 12~24개월 전 탈퇴','공개 0~12개월 전 탈퇴',
          '공개 후 0~12개월 탈퇴','공개 12개월+ 후 탈퇴']:
    print(f"  {b}: {buckets.get(b,0)}건")
print('\n완료')
