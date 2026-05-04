"""
음성 샘플 (정상 사업장) 시계열 추출.

전략:
- 양성과 동일한 lead time / 입력 구간 사용 (t-18 ~ t-6)
- 양성 사업장의 공개 차수(t) 분포에 비례하여 음성에 가상의 t 부여
- 매칭 풀에 없는 사업장만 추출
- 가입자수 ≥ 3명 필터 (노이즈 제거)
- 1:5 비율 (양성 대비 5배)

출력: dataset/pension_trend_negatives.csv
  같은 long format, 동일 컬럼 + 가상의 '체불공개구분' (t 식별용)
"""
import csv, os, re, sys, random, glob as globmod
from collections import defaultdict
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(10**7)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_BASE = os.path.join(os.path.dirname(BASE_DIR), 'dataset')
DATASET_DIR = os.path.join(DATASET_BASE, '국민연금 공공데이터포털')
MATCHED_CSV = os.path.join(DATASET_BASE, 'matched_pension_unpaid.csv')
UNPAID_CSV = os.path.join(DATASET_BASE, 'unpaid_employers.csv')
OUTPUT_CSV = os.path.join(DATASET_BASE, 'pension_trend_negatives.csv')

# 설정
# v2: 양성/음성 추출 기준 통일 (F17 결측 인공물 제거 목적)
NEG_RATIO = 5            # 양성:음성 = 1:5
MIN_EMP = 0              # 양성과 동일하게 가입자수 필터 없음
COVERAGE_THRESHOLD = 0.5 # 양성과 동일하게 50%+ 커버리지
SEED = 42

# 양성 공개 차수별 t (lead 6 → 입력 [t-18, t-6])
GUBUN_DATE = {
    '2023년 1차': '2023-08',
    '2023년 2차': '2023-12',
    '2024년 1차': '2024-08',
    '2024년 2차': '2024-12',
    '2025년 1차': '2025-08',
}

random.seed(SEED)


def add_months(yyyymm, delta):
    y, m = map(int, yyyymm.split('-'))
    total = y * 12 + (m - 1) + delta
    return f'{total // 12:04d}-{total % 12 + 1:02d}'


def _norm_col(c):
    c = c.strip()
    i = c.find('(')
    return c[:i].strip() if i > 0 else c


# ============================================================
# 1. 매칭된 사업장 키 로드 (음성에서 제외)
# ============================================================
print('매칭 사업장 키 로드 중...')
matched_keys = set()
with open(MATCHED_CSV, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        matched_keys.add((r['사업장명'].strip(), r['사업자등록번호'].strip()))
print(f'  매칭 키: {len(matched_keys)}개')

# ============================================================
# 2. 양성 공개 차수별 카운트 → 음성 샘플 수 결정
# ============================================================
print('\n양성 공개 차수별 카운트 로드 중...')
unpaid_by_gubun = defaultdict(set)
with open(UNPAID_CSV, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        unpaid_by_gubun[r['구분'].strip()].add(r['사업장명'].strip())

# 매칭된 양성만 카운트
import itertools
matched_unpaid_names = set()
with open(MATCHED_CSV, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        matched_unpaid_names.add(r['매칭_임금체불_사업장명'])

pos_count_by_gubun = {}
for g, names in unpaid_by_gubun.items():
    pos_count_by_gubun[g] = len(names & matched_unpaid_names)

print('  공개차수별 매칭 양성:')
for g, c in sorted(pos_count_by_gubun.items()):
    print(f'    {g}: {c}개 → 음성 {c * NEG_RATIO}개')

neg_target_by_gubun = {g: c * NEG_RATIO for g, c in pos_count_by_gubun.items()}

# ============================================================
# 3. 각 공개 차수별 입력 구간 12개월 파일 로드 + 음성 샘플 추출
# ============================================================
DATE_RE = re.compile(r'(\d{4})(\d{2})(\d{2})\.csv$')

def extract_month(fpath):
    m = DATE_RE.search(os.path.basename(fpath))
    return f'{m.group(1)}-{m.group(2)}' if m else ''

# 모든 파일 인덱싱
all_files = sorted(globmod.glob(os.path.join(DATASET_DIR, '*.csv')))
file_by_month = {extract_month(fp): fp for fp in all_files}
print(f'\n전체 파일 인덱싱: {len(file_by_month)}개월')

# 사업자번호별 12개월 데이터 캐시
# {(name, biz_no): {month: row_dict}}
def load_window_index(disclose_t):
    """disclose_t에 대한 [t-18, t-6] 12개월 사업장 인덱스 구축"""
    start = add_months(disclose_t, -18)
    end = add_months(disclose_t, -6)
    needed_months = []
    cur = start
    while cur <= end:
        needed_months.append(cur)
        cur = add_months(cur, 1)

    biz_data = defaultdict(dict)  # (name, biz_no) -> {month: row}
    months_loaded = []
    for m in needed_months:
        fp = file_by_month.get(m)
        if not fp:
            print(f'    {m}: 파일 없음 (skip)')
            continue
        with open(fp, encoding='cp949', errors='replace') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [_norm_col(c) for c in reader.fieldnames]
            for row in reader:
                name = row.get('사업장명', '').strip()
                biz_no = row.get('사업자등록번호', '').strip()
                if not name or not biz_no:
                    continue
                biz_data[(name, biz_no)][m] = row
        months_loaded.append(m)
    return biz_data, months_loaded


# 음성 추출: 차수별로 처리
all_neg_snapshots = []
for gubun, neg_target in neg_target_by_gubun.items():
    if neg_target == 0:
        continue
    disclose = GUBUN_DATE[gubun]
    print(f'\n--- {gubun} (t={disclose}) — 음성 {neg_target}개 추출 ---')
    biz_data, months_loaded = load_window_index(disclose)
    print(f'  {len(months_loaded)}개월 로드, 후보 사업장 {len(biz_data)}개')

    # 음성 후보 필터 (v2: 양성과 동일 기준):
    # 1. 매칭 사업장 제외
    # 2. 12개월 중 50% 이상 데이터 보유 (COVERAGE_THRESHOLD)
    # 3. 가입자수 필터 없음 (MIN_EMP=0)
    last_month = months_loaded[-1] if months_loaded else None
    candidates = []
    for key, months_dict in biz_data.items():
        if key in matched_keys:
            continue
        if len(months_dict) < len(months_loaded) * COVERAGE_THRESHOLD:
            continue
        last_row = months_dict.get(last_month)
        if not last_row:
            # 마지막 월에 데이터 없어도 통과 (양성과 동일)
            # 다른 월의 데이터로 학습 가능
            pass
        else:
            try:
                emp = int(last_row.get('가입자수', '0'))
                if emp < MIN_EMP:
                    continue
            except ValueError:
                pass
        candidates.append((key, months_dict))
    print(f'  필터링 후 후보: {len(candidates)}개')

    # 무작위 샘플
    sample_n = min(neg_target, len(candidates))
    sampled = random.sample(candidates, sample_n)
    print(f'  샘플링: {sample_n}개')

    # 시계열 row 생성
    for (name, biz_no), months_dict in sampled:
        for m, row in months_dict.items():
            all_neg_snapshots.append({
                '기준월': m,
                '사업장명': name,
                '사업자등록번호': biz_no,
                '가입상태코드': row.get('사업장가입상태코드 1 등록 2 탈퇴', '').strip(),
                '가입자수': row.get('가입자수', '').strip(),
                '신규취득자수': row.get('신규취득자수', '').strip(),
                '상실가입자수': row.get('상실가입자수', '').strip(),
                '당월고지금액': row.get('당월고지금액', '').strip(),
                '적용일자': row.get('적용일자', '').strip(),
                '재등록일자': row.get('재등록일자', '').strip(),
                '탈퇴일자': row.get('탈퇴일자', '').strip(),
                '사업장업종코드명': row.get('사업장업종코드명', '').strip(),
                '사업장업종코드': row.get('사업장업종코드', '').strip(),
                '사업장형태구분코드': row.get('사업장형태구분코드 1 법인 2 개인', '').strip(),
                '시도코드': row.get('법정동주소광역시도코드', '').strip(),
                '가상_공개차수': gubun,
                '가상_t': disclose,
            })

print(f'\n총 음성 스냅샷: {len(all_neg_snapshots)}건')

# ============================================================
# 4. 저장
# ============================================================
all_neg_snapshots.sort(key=lambda r: (r['사업자등록번호'], r['사업장명'], r['기준월']))
fieldnames = [
    '기준월', '사업장명', '사업자등록번호', '가입상태코드',
    '가입자수', '신규취득자수', '상실가입자수', '당월고지금액',
    '적용일자', '재등록일자', '탈퇴일자',
    '사업장업종코드명', '사업장업종코드', '사업장형태구분코드', '시도코드',
    '가상_공개차수', '가상_t',
]
with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_neg_snapshots)

# 요약
unique_neg = len(set((r['사업장명'], r['사업자등록번호']) for r in all_neg_snapshots))
print(f'\n저장 완료: {OUTPUT_CSV}')
print(f'  총 row: {len(all_neg_snapshots)}')
print(f'  고유 음성 사업장: {unique_neg}')
print(f'  공개차수별 음성:')
gc = defaultdict(set)
for r in all_neg_snapshots:
    gc[r['가상_공개차수']].add((r['사업장명'], r['사업자등록번호']))
for g, s in sorted(gc.items()):
    print(f'    {g}: {len(s)}개')
