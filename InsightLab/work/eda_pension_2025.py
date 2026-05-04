"""2025년 국민연금 데이터 EDA"""
import sys, csv, os, glob as globmod
from collections import defaultdict, Counter
from statistics import mean, median, stdev

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(10**7)

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
PENSION_DIR = os.path.join(BASE, '국민연금 공공데이터포털')

# 2025년 파일만
files_2025 = sorted(globmod.glob(os.path.join(PENSION_DIR, '*_2025*.csv')))
print(f'2025년 파일 수: {len(files_2025)}')
for f in files_2025:
    print(f'  {os.path.basename(f)}')

# 가장 최신 2025 파일로 단면 분석 (12월)
latest_2025 = files_2025[-1]
print(f'\n단면 분석 기준 파일: {os.path.basename(latest_2025)}')

def safe_int(s):
    try: return int(s)
    except: return 0

def get_sido(addr):
    ABBR = {'서울':'서울특별시','서울시':'서울특별시','부산':'부산광역시','부산시':'부산광역시',
        '대구':'대구광역시','인천':'인천광역시','광주':'광주광역시','대전':'대전광역시',
        '울산':'울산광역시','세종':'세종특별자치시','세종특별자치시':'세종특별자치시',
        '경기':'경기도','강원':'강원특별자치도','강원도':'강원특별자치도',
        '충북':'충청북도','충남':'충청남도','전북':'전북특별자치도','전라북도':'전북특별자치도',
        '전남':'전라남도','경북':'경상북도','경남':'경상남도','제주':'제주특별자치도'}
    if not addr: return ''
    p = addr.strip().split()
    return ABBR.get(p[0], p[0]) if p else ''

# ========== 1. 단면 분석 (2025년 12월 스냅샷) ==========
print('\n' + '=' * 70)
print('1. 2025년 12월 스냅샷 단면 분석')
print('=' * 70)

total = 0
status_cnt = Counter()
form_cnt = Counter()  # 법인/개인
sido_cnt = Counter()
industry_cnt = Counter()
subs_list = []
amt_list = []
per_cap_list = []
new_cnt_list = []
out_cnt_list = []
apply_years = Counter()
withdraw_years = Counter()

with open(latest_2025, encoding='cp949', errors='replace', newline='') as f:
    for row in csv.DictReader(f):
        total += 1
        status = row.get('사업장가입상태코드 1 등록 2 탈퇴', '')
        form = row.get('사업장형태구분코드 1 법인 2 개인', '')
        status_cnt[status] += 1
        form_cnt[form] += 1

        # 주소 - 도로명 우선
        addr = row.get('사업장도로명상세주소','').strip() or row.get('사업장지번상세주소','').strip()
        sido = get_sido(addr)
        if sido: sido_cnt[sido] += 1

        ind = row.get('사업장업종코드명','').strip()
        if ind: industry_cnt[ind] += 1

        subs = safe_int(row.get('가입자수',0))
        amt = safe_int(row.get('당월고지금액',0))
        new_c = safe_int(row.get('신규취득자수',0))
        out_c = safe_int(row.get('상실가입자수',0))
        if subs > 0:
            subs_list.append(subs)
            if amt > 0:
                per_cap_list.append(amt / subs)
        if amt > 0:
            amt_list.append(amt)
        new_cnt_list.append(new_c)
        out_cnt_list.append(out_c)

        # 적용일자/탈퇴일자
        apply_date = row.get('적용일자','').strip()
        if apply_date and len(apply_date) >= 4:
            apply_years[apply_date[:4]] += 1
        withdraw_date = row.get('탈퇴일자','').strip()
        if withdraw_date and len(withdraw_date) >= 4:
            withdraw_years[withdraw_date[:4]] += 1

print(f'\n총 사업장 수: {total:,}')

print(f'\n[가입상태]')
for k, v in sorted(status_cnt.items()):
    label = '등록(운영중)' if k == '1' else '탈퇴' if k == '2' else f'기타({k})'
    pct = v / total * 100
    print(f'  {label}: {v:,}건 ({pct:.1f}%)')

print(f'\n[사업장 형태]')
for k, v in sorted(form_cnt.items()):
    label = '법인' if k == '1' else '개인' if k == '2' else f'기타({k})'
    pct = v / total * 100
    print(f'  {label}: {v:,}건 ({pct:.1f}%)')

print(f'\n[시도별 분포 상위 10]')
for k, v in sido_cnt.most_common(10):
    pct = v / total * 100
    print(f'  {k}: {v:,}건 ({pct:.1f}%)')

print(f'\n[업종 상위 15]')
for k, v in industry_cnt.most_common(15):
    pct = v / total * 100
    print(f'  {k}: {v:,}건 ({pct:.1f}%)')

print(f'\n[가입자수 통계]')
print(f'  유효: {len(subs_list):,}건')
print(f'  평균: {mean(subs_list):.1f}명')
print(f'  중앙값: {median(subs_list):.0f}명')
print(f'  최소: {min(subs_list)}, 최대: {max(subs_list)}')
print(f'  표준편차: {stdev(subs_list):.1f}')

# 가입자수 구간
bins = [(1,5,'1~4인'),(5,10,'5~9인'),(10,30,'10~29인'),(30,50,'30~49인'),
        (50,100,'50~99인'),(100,300,'100~299인'),(300,1000,'300~999인'),(1000,999999,'1000인 이상')]
print(f'\n[가입자수 구간별]')
for lo, hi, label in bins:
    cnt = sum(1 for s in subs_list if lo <= s < hi)
    pct = cnt / len(subs_list) * 100
    print(f'  {label}: {cnt:,}건 ({pct:.1f}%)')

print(f'\n[당월고지금액 통계]')
print(f'  평균: {mean(amt_list):,.0f}원')
print(f'  중앙값: {median(amt_list):,.0f}원')
print(f'  총합: {sum(amt_list):,}원')

print(f'\n[1인당 고지금액 - 임금 수준 추정]')
print(f'  평균: {mean(per_cap_list):,.0f}원')
print(f'  중앙값: {median(per_cap_list):,.0f}원')

print(f'\n[월 신규취득자수 통계]')
active_new = [n for n in new_cnt_list if n > 0]
print(f'  신규 발생 사업장: {len(active_new):,}건 ({len(active_new)/total*100:.1f}%)')
if active_new:
    print(f'  평균: {mean(active_new):.1f}명')
    print(f'  중앙값: {median(active_new):.0f}명')
    print(f'  전체 신규 가입자: {sum(new_cnt_list):,}명')

print(f'\n[월 상실가입자수 통계]')
active_out = [n for n in out_cnt_list if n > 0]
print(f'  상실 발생 사업장: {len(active_out):,}건 ({len(active_out)/total*100:.1f}%)')
if active_out:
    print(f'  평균: {mean(active_out):.1f}명')
    print(f'  중앙값: {median(active_out):.0f}명')
    print(f'  전체 상실 가입자: {sum(out_cnt_list):,}명')

net_flow = sum(new_cnt_list) - sum(out_cnt_list)
print(f'\n[순유입 (신규 - 상실)]')
print(f'  {net_flow:+,}명')

print(f'\n[적용일자(사업장 등록연도) 상위 10]')
for k, v in sorted(apply_years.items(), key=lambda x: -x[1])[:10]:
    print(f'  {k}년: {v:,}건')

print(f'\n[등록연도 분포]')
decades = defaultdict(int)
for y, c in apply_years.items():
    try:
        yi = int(y)
        if yi < 1990: decades['~1989'] += c
        elif yi < 2000: decades['1990~1999'] += c
        elif yi < 2010: decades['2000~2009'] += c
        elif yi < 2020: decades['2010~2019'] += c
        else: decades['2020~'] += c
    except: pass
for k in ['~1989','1990~1999','2000~2009','2010~2019','2020~']:
    v = decades[k]
    pct = v/total*100
    print(f'  {k}: {v:,}건 ({pct:.1f}%)')

# ========== 2. 시계열 집계 (2025년 12개월) ==========
print('\n' + '=' * 70)
print('2. 2025년 12개월 시계열 추이')
print('=' * 70)

print(f'\n{"월":<10} {"총사업장":>10} {"등록":>10} {"탈퇴":>10} {"총가입자":>12} {"신규":>8} {"상실":>8} {"순유입":>10}')
print('-' * 90)

monthly_stats = []
for fp in files_2025:
    fname = os.path.basename(fp).replace('국민연금 가입 사업장 내역_','').replace('.csv','')
    total_c = 0
    active_c = 0
    withdrawn_c = 0
    total_subs = 0
    total_new = 0
    total_out = 0
    try:
        with open(fp, encoding='cp949', errors='replace', newline='') as f:
            for row in csv.DictReader(f):
                total_c += 1
                status = row.get('사업장가입상태코드 1 등록 2 탈퇴', '')
                if status == '1': active_c += 1
                elif status == '2': withdrawn_c += 1
                total_subs += safe_int(row.get('가입자수',0))
                total_new += safe_int(row.get('신규취득자수',0))
                total_out += safe_int(row.get('상실가입자수',0))
    except Exception as e:
        print(f'  skip {fname}: {e}')
        continue
    net = total_new - total_out
    monthly_stats.append((fname, total_c, active_c, withdrawn_c, total_subs, total_new, total_out, net))
    print(f'{fname:<10} {total_c:>10,} {active_c:>10,} {withdrawn_c:>10,} {total_subs:>12,} {total_new:>8,} {total_out:>8,} {net:>+10,}')

# 요약
if monthly_stats:
    print(f'\n[2025년 연간 요약]')
    year_new = sum(s[5] for s in monthly_stats)
    year_out = sum(s[6] for s in monthly_stats)
    print(f'  연간 총 신규취득: {year_new:,}명')
    print(f'  연간 총 상실: {year_out:,}명')
    print(f'  연간 순유입: {year_new - year_out:+,}명')
    print(f'  첫달 사업장 수: {monthly_stats[0][1]:,}')
    print(f'  마지막달 사업장 수: {monthly_stats[-1][1]:,}')
    print(f'  사업장 증감: {monthly_stats[-1][1] - monthly_stats[0][1]:+,}')
    print(f'  첫달 총가입자: {monthly_stats[0][4]:,}')
    print(f'  마지막달 총가입자: {monthly_stats[-1][4]:,}')
    print(f'  가입자 증감: {monthly_stats[-1][4] - monthly_stats[0][4]:+,}')

# ========== 3. 신규/탈퇴 사업장 분석 (2025년 연간) ==========
print('\n' + '=' * 70)
print('3. 2025년 신규 등록 사업장 분석')
print('=' * 70)

# 2025년에 최초 등장한 사업장 (사업자등록번호 기준)
first_file_ids = set()
with open(files_2025[0], encoding='cp949', errors='replace', newline='') as f:
    for row in csv.DictReader(f):
        bid = row.get('사업자등록번호','').strip()
        name = row.get('사업장명','').strip()
        key = f'{name}|{bid}'
        first_file_ids.add(key)

last_file_ids = set()
new_in_2025 = []
with open(files_2025[-1], encoding='cp949', errors='replace', newline='') as f:
    for row in csv.DictReader(f):
        bid = row.get('사업자등록번호','').strip()
        name = row.get('사업장명','').strip()
        key = f'{name}|{bid}'
        last_file_ids.add(key)
        if key not in first_file_ids:
            new_in_2025.append(row)

disappeared = first_file_ids - last_file_ids
print(f'2025년 1월 사업장: {len(first_file_ids):,}')
print(f'2025년 12월 사업장: {len(last_file_ids):,}')
print(f'2025년 중 신규 등장: {len(new_in_2025):,}')
print(f'2025년 중 사라짐: {len(disappeared):,}')

# 신규 사업장 업종 분포
new_ind = Counter(r.get('사업장업종코드명','').strip() for r in new_in_2025)
print(f'\n[2025년 신규 사업장 업종 상위 10]')
for k, v in new_ind.most_common(10):
    if k:
        print(f'  {k}: {v:,}건')

# 신규 사업장 가입자수
new_subs = [safe_int(r.get('가입자수',0)) for r in new_in_2025]
new_subs = [s for s in new_subs if s > 0]
if new_subs:
    print(f'\n[2025년 신규 사업장 가입자수]')
    print(f'  평균: {mean(new_subs):.1f}명, 중앙값: {median(new_subs):.0f}명')

# 신규 사업장 법인/개인
new_form = Counter(r.get('사업장형태구분코드 1 법인 2 개인','') for r in new_in_2025)
print(f'\n[2025년 신규 사업장 형태]')
for k, v in sorted(new_form.items()):
    label = '법인' if k == '1' else '개인' if k == '2' else f'기타'
    print(f'  {label}: {v:,}건')

print('\n완료')
