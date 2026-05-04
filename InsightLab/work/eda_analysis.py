"""EDA 데이터 수집 스크립트"""
import sys, csv, re, os, glob as globmod
from collections import defaultdict, Counter
from statistics import mean, median, stdev

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(10**7)

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')

# ========== 1. 임금체불 명단 분석 ==========
print('=' * 70)
print('1. 임금체불 명단 분석')
print('=' * 70)

unpaid = []
with open(os.path.join(BASE, 'unpaid_employers.csv'), encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        try:
            amount = int(row['체불액(원)'].replace(',', ''))
        except:
            amount = 0
        try:
            age = int(row['나이(대표자)'])
        except:
            age = None
        unpaid.append({
            'period': row['구분'],
            'name': row['사업장명'],
            'rep': row['성명(상호)'],
            'age': age,
            'industry': row['업종'],
            'addr1': row['주소지'],
            'addr2': row.get('소재지', ''),
            'amount': amount,
        })

print(f'총 {len(unpaid)}건')

# 1.1 구분별 분포
print('\n[구분별 분포]')
period_cnt = Counter(u['period'] for u in unpaid)
for k, v in sorted(period_cnt.items()):
    print(f'  {k}: {v}건')

# 1.2 업종별 분포
print('\n[업종별 분포]')
ind_cnt = Counter(u['industry'] for u in unpaid)
for k, v in ind_cnt.most_common():
    pct = v / len(unpaid) * 100
    print(f'  {k}: {v}건 ({pct:.1f}%)')

# 1.3 체불액 통계
amounts = [u['amount'] for u in unpaid if u['amount'] > 0]
print(f'\n[체불액 통계 (원)]')
print(f'  유효 데이터: {len(amounts)}건')
print(f'  총합: {sum(amounts):,}원')
print(f'  평균: {mean(amounts):,.0f}원')
print(f'  중앙값: {median(amounts):,.0f}원')
print(f'  최솟값: {min(amounts):,}원')
print(f'  최댓값: {max(amounts):,}원')
print(f'  표준편차: {stdev(amounts):,.0f}원')

# 체불액 분위수
sorted_amts = sorted(amounts)
print(f'  Q1 (25%): {sorted_amts[len(sorted_amts)//4]:,}원')
print(f'  Q3 (75%): {sorted_amts[len(sorted_amts)*3//4]:,}원')

# 1.4 체불액 구간별
print(f'\n[체불액 구간별 분포]')
bins = [(0, 5e7, '5천만원 미만'), (5e7, 1e8, '5천만~1억'),
        (1e8, 3e8, '1억~3억'), (3e8, 5e8, '3억~5억'),
        (5e8, 1e9, '5억~10억'), (1e9, float('inf'), '10억 이상')]
for lo, hi, label in bins:
    cnt = sum(1 for a in amounts if lo <= a < hi)
    pct = cnt / len(amounts) * 100
    print(f'  {label}: {cnt}건 ({pct:.1f}%)')

# 1.5 업종별 체불액 평균
print(f'\n[업종별 평균 체불액 (상위 10)]')
ind_amts = defaultdict(list)
for u in unpaid:
    if u['amount'] > 0:
        ind_amts[u['industry']].append(u['amount'])
ind_avg = [(k, mean(v), len(v), sum(v)) for k, v in ind_amts.items() if len(v) >= 5]
for k, avg, cnt, total in sorted(ind_avg, key=lambda x: -x[1])[:10]:
    print(f'  {k}: 평균 {avg:,.0f}원 (n={cnt}, 총 {total:,.0f}원)')

# 1.6 지역별 분포
def get_sido(a):
    ABBR = {'서울':'서울특별시','서울시':'서울특별시','부산':'부산광역시','부산시':'부산광역시',
        '대구':'대구광역시','대구시':'대구광역시','인천':'인천광역시','인천시':'인천광역시',
        '광주':'광주광역시','광주시':'광주광역시','대전':'대전광역시','대전시':'대전광역시',
        '울산':'울산광역시','울산시':'울산광역시','세종':'세종특별자치시','세종시':'세종특별자치시',
        '세종특별자치시':'세종특별자치시','경기':'경기도','강원':'강원특별자치도','강원도':'강원특별자치도',
        '충북':'충청북도','충남':'충청남도','전북':'전북특별자치도','전라북도':'전북특별자치도',
        '전남':'전라남도','경북':'경상북도','경남':'경상남도','제주':'제주특별자치도'}
    if not a: return ''
    p = a.strip().split()
    return ABBR.get(p[0], p[0]) if p else ''

print(f'\n[주소지 시도별 분포]')
sido_cnt = Counter(get_sido(u['addr1']) for u in unpaid)
for k, v in sido_cnt.most_common():
    pct = v / len(unpaid) * 100
    print(f'  {k}: {v}건 ({pct:.1f}%)')

# 1.7 대표자 나이 분포
ages = [u['age'] for u in unpaid if u['age'] is not None]
print(f'\n[대표자 나이 통계]')
print(f'  유효 데이터: {len(ages)}건')
print(f'  평균: {mean(ages):.1f}세')
print(f'  중앙값: {median(ages):.0f}세')
print(f'  최저: {min(ages)}세, 최고: {max(ages)}세')

age_bins = [(20, 30, '20대'), (30, 40, '30대'), (40, 50, '40대'),
            (50, 60, '50대'), (60, 70, '60대'), (70, 100, '70대 이상')]
for lo, hi, label in age_bins:
    cnt = sum(1 for a in ages if lo <= a < hi)
    pct = cnt / len(ages) * 100
    print(f'  {label}: {cnt}건 ({pct:.1f}%)')

# 1.8 구분별 평균 체불액 (트렌드)
print(f'\n[구분별 평균 체불액 추이]')
period_amts = defaultdict(list)
for u in unpaid:
    if u['amount'] > 0:
        period_amts[u['period']].append(u['amount'])
for k in sorted(period_amts.keys()):
    v = period_amts[k]
    print(f'  {k}: 평균 {mean(v):,.0f}원, 중앙 {median(v):,.0f}원, n={len(v)}')

# ========== 2. 매칭 결과 분석 ==========
print('\n' + '=' * 70)
print('2. 매칭 결과 분석')
print('=' * 70)

matched = []
with open(os.path.join(BASE, 'matched_pension_unpaid.csv'), encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        matched.append(row)

print(f'매칭된 국민연금 행: {len(matched)}')

# 고유 임금체불 사업장 수
unique_unpaid_matched = set(r['매칭_임금체불_사업장명'] for r in matched)
print(f'고유 임금체불 사업장 수: {len(unique_unpaid_matched)}')

# 단계별 + 가입상태별
print(f'\n[단계별 매칭]')
stage_cnt = Counter(r['매칭_단계'] for r in matched)
for k, v in sorted(stage_cnt.items()):
    print(f'  {k}: {v}건')

print(f'\n[가입상태]')
status_cnt = Counter(r.get('사업장가입상태코드 1 등록 2 탈퇴', '') for r in matched)
for k, v in status_cnt.items():
    label = '등록' if k == '1' else '탈퇴' if k == '2' else f'기타({k})'
    pct = v / len(matched) * 100
    print(f'  {label}: {v}건 ({pct:.1f}%)')

# 매칭된 사업장의 가입자수 통계
def safe_int(s):
    try: return int(s)
    except: return 0

subs = [safe_int(r.get('가입자수', 0)) for r in matched]
subs = [s for s in subs if s > 0]
print(f'\n[매칭 사업장의 가입자수 통계]')
print(f'  유효 데이터: {len(subs)}건')
print(f'  평균: {mean(subs):.1f}명')
print(f'  중앙값: {median(subs):.0f}명')
print(f'  최소: {min(subs)}, 최대: {max(subs)}')

sub_bins = [(1, 5, '1~4인'), (5, 10, '5~9인'), (10, 30, '10~29인'),
            (30, 50, '30~49인'), (50, 100, '50~99인'),
            (100, 300, '100~299인'), (300, 10000, '300인 이상')]
print(f'\n[가입자수 구간 분포]')
for lo, hi, label in sub_bins:
    cnt = sum(1 for s in subs if lo <= s < hi)
    pct = cnt / len(subs) * 100
    print(f'  {label}: {cnt}건 ({pct:.1f}%)')

# 당월고지금액 통계
amts = [safe_int(r.get('당월고지금액', 0)) for r in matched]
amts = [a for a in amts if a > 0]
print(f'\n[매칭 사업장의 당월고지금액 통계]')
print(f'  유효 데이터: {len(amts)}건')
print(f'  평균: {mean(amts):,.0f}원')
print(f'  중앙값: {median(amts):,.0f}원')

# 1인당 고지금액 (대략적인 평균 임금 추정)
per_cap = []
for r in matched:
    s = safe_int(r.get('가입자수', 0))
    a = safe_int(r.get('당월고지금액', 0))
    if s > 0 and a > 0:
        per_cap.append(a / s)
if per_cap:
    print(f'\n[1인당 고지금액 (월) - 임금 수준 추정]')
    print(f'  유효 데이터: {len(per_cap)}건')
    print(f'  평균: {mean(per_cap):,.0f}원')
    print(f'  중앙값: {median(per_cap):,.0f}원')

# 업종별 매칭률
print(f'\n[업종별 매칭률]')
unpaid_by_ind = Counter(u['industry'] for u in unpaid)
matched_names = set(r['매칭_임금체불_사업장명'] for r in matched)
matched_by_ind = Counter()
for u in unpaid:
    if u['name'] in matched_names:
        matched_by_ind[u['industry']] += 1
for ind in sorted(unpaid_by_ind.keys(), key=lambda x: -unpaid_by_ind[x]):
    total = unpaid_by_ind[ind]
    m = matched_by_ind[ind]
    pct = m / total * 100 if total else 0
    print(f'  {ind}: {m}/{total} ({pct:.1f}%)')

# 업종별 매칭 사업장 평균 가입자수
print(f'\n[업종별 매칭 사업장 평균 가입자수]')
ind_subs = defaultdict(list)
for r in matched:
    ind = r.get('사업장업종코드명', '')
    s = safe_int(r.get('가입자수', 0))
    if s > 0 and ind:
        ind_subs[ind].append(s)
ind_avg_subs = [(k, mean(v), len(v)) for k, v in ind_subs.items() if len(v) >= 3]
for k, avg, cnt in sorted(ind_avg_subs, key=lambda x: -x[2])[:15]:
    print(f'  {k}: 평균 {avg:.1f}명 (n={cnt})')

# ========== 3. 미매칭 분석 ==========
print('\n' + '=' * 70)
print('3. 미매칭 분석')
print('=' * 70)

unmatched = []
with open(os.path.join(BASE, 'unmatched_unpaid.csv'), encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        unmatched.append(row)

print(f'미매칭: {len(unmatched)}건')

# 미매칭 시도별
unmatched_sido = Counter(u['시도'].split('|')[0] if u['시도'] else '' for u in unmatched)
print(f'\n[미매칭 시도별]')
for k, v in unmatched_sido.most_common():
    print(f'  {k}: {v}건')

# 매칭/미매칭 vs 체불액 비교
matched_amts = [u['amount'] for u in unpaid if u['name'] in matched_names and u['amount'] > 0]
unmatched_amts = [u['amount'] for u in unpaid if u['name'] not in matched_names and u['amount'] > 0]
print(f'\n[매칭 vs 미매칭 체불액 비교]')
print(f'  매칭 사업장 평균 체불액: {mean(matched_amts):,.0f}원 (n={len(matched_amts)})')
print(f'  미매칭 사업장 평균 체불액: {mean(unmatched_amts):,.0f}원 (n={len(unmatched_amts)})')
print(f'  매칭 중앙값: {median(matched_amts):,.0f}원')
print(f'  미매칭 중앙값: {median(unmatched_amts):,.0f}원')

# ========== 4. 가입자수와 체불액 관계 ==========
print('\n' + '=' * 70)
print('4. 가입자수 vs 체불액 관계 (매칭 사업장)')
print('=' * 70)

# 매칭된 사업장명 -> 가입자수 (가장 큰 값 기준)
name_to_subs = {}
name_to_status = {}
for r in matched:
    n = r['매칭_임금체불_사업장명']
    s = safe_int(r.get('가입자수', 0))
    if n not in name_to_subs or s > name_to_subs[n]:
        name_to_subs[n] = s
    st = r.get('사업장가입상태코드 1 등록 2 탈퇴', '')
    if n not in name_to_status:
        name_to_status[n] = st

# 임금체불 매칭 후 가입자수와 체불액
sub_amt_pairs = []
for u in unpaid:
    if u['name'] in name_to_subs and name_to_subs[u['name']] > 0 and u['amount'] > 0:
        sub_amt_pairs.append((name_to_subs[u['name']], u['amount']))

print(f'분석 대상: {len(sub_amt_pairs)}건')

for lo, hi, label in sub_bins:
    pairs = [(s, a) for s, a in sub_amt_pairs if lo <= s < hi]
    if pairs:
        avg_amt = mean([a for _, a in pairs])
        per_emp = mean([a/s for s, a in pairs])
        print(f'  {label}: 평균 체불액 {avg_amt:,.0f}원, 1인당 {per_emp:,.0f}원 (n={len(pairs)})')

# 가입상태별 체불액
print(f'\n[가입상태별 체불액 비교]')
reg_amts = [u['amount'] for u in unpaid if u['name'] in name_to_status and name_to_status[u['name']] == '1' and u['amount'] > 0]
out_amts = [u['amount'] for u in unpaid if u['name'] in name_to_status and name_to_status[u['name']] == '2' and u['amount'] > 0]
print(f'  현재 등록 사업장 평균 체불액: {mean(reg_amts):,.0f}원 (n={len(reg_amts)})')
print(f'  탈퇴 사업장 평균 체불액: {mean(out_amts):,.0f}원 (n={len(out_amts)})')
print(f'  현재 등록 중앙값: {median(reg_amts):,.0f}원')
print(f'  탈퇴 중앙값: {median(out_amts):,.0f}원')

# ========== 5. 국민연금 사업장 시계열 (매칭 대상만) ==========
print('\n' + '=' * 70)
print('5. 매칭 사업장 시계열 등장 패턴')
print('=' * 70)

# 매칭된 사업장 전체 이력 추적 (사업장명 기준)
files = sorted(globmod.glob(os.path.join(BASE, '국민연금 공공데이터포털', '*.csv')))
matched_target_names = set()  # 매칭된 국민연금 사업장명
for r in matched:
    matched_target_names.add(r.get('사업장명', ''))

# 각 사업장이 몇 개월에 등장했는지
appearance = defaultdict(set)  # name -> set of file dates
for fp in files:
    fname = os.path.basename(fp).replace('국민연금 가입 사업장 내역_', '').replace('.csv', '')
    try:
        with open(fp, encoding='cp949', errors='replace', newline='') as f:
            for row in csv.DictReader(f):
                n = row.get('사업장명', '').strip()
                if n in matched_target_names:
                    appearance[n].add(fname)
    except:
        pass

months_present = [len(v) for v in appearance.values()]
if months_present:
    print(f'매칭 사업장 추적 가능: {len(appearance)}개')
    print(f'  평균 등장 개월수: {mean(months_present):.1f}개월 (전체 {len(files)}개월 중)')
    print(f'  중앙값: {median(months_present):.0f}개월')
    print(f'  최소 1개월: {sum(1 for m in months_present if m == 1)}개')
    print(f'  전체 기간 등장: {sum(1 for m in months_present if m == len(files))}개')

print('\n완료')
