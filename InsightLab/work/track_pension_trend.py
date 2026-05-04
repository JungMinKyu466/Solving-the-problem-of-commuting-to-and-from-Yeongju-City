"""
체불업체의 국민연금 가입자수 시계열 추출.

매칭된 (사업장명, 사업자등록번호) 쌍에 대해 67개 월별 스냅샷을 모두 모아
long format으로 저장. 체불 공개 시점(구분)도 함께 join하여 전후 비교 가능.

출력: dataset/pension_trend.csv
  기준월, 사업장명, 사업자등록번호, 가입상태코드, 가입자수, 신규취득자수,
  상실가입자수, 당월고지금액, 적용일자, 재등록일자, 탈퇴일자,
  매칭_단계, 체불공개구분, 체불액
"""

import csv
import os
import re
import sys
import glob as globmod
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(10**7)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_BASE = os.path.join(os.path.dirname(BASE_DIR), "dataset")
DATASET_DIR = os.path.join(DATASET_BASE, "국민연금 공공데이터포털")
MATCHED_CSV = os.path.join(DATASET_BASE, "matched_pension_unpaid.csv")
UNPAID_CSV = os.path.join(DATASET_BASE, "unpaid_employers.csv")
OUTPUT_CSV = os.path.join(DATASET_BASE, "pension_trend.csv")

# ============================================================
# 1. 매칭된 (사업장명, 사업자등록번호) 쌍 로드
# ============================================================
print("매칭 결과 로드 중...")
target_keys = {}  # (name, biz_no) -> {매칭_임금체불_사업장명, 매칭_단계}
with open(MATCHED_CSV, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        name = row['사업장명'].strip()
        biz_no = row['사업자등록번호'].strip()
        target_keys[(name, biz_no)] = {
            '매칭_임금체불_사업장명': row['매칭_임금체불_사업장명'],
            '매칭_단계': row['매칭_단계'],
        }
print(f"  대상 사업장: {len(target_keys)}개")

# ============================================================
# 2. 체불 정보 join용 dict (사업장명 -> [(구분, 체불액), ...])
# ============================================================
unpaid_by_name = defaultdict(list)
with open(UNPAID_CSV, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        unpaid_by_name[row['사업장명'].strip()].append({
            '구분': row['구분'].strip(),
            '체불액': row['체불액(원)'].strip(),
        })

def get_unpaid_info(matched_name):
    """매칭된 임금체불 사업장명으로 구분/체불액 조회. 여러 건이면 최초 공개 기준."""
    records = unpaid_by_name.get(matched_name.strip(), [])
    if not records:
        return '', ''
    records_sorted = sorted(records, key=lambda r: r['구분'])
    first = records_sorted[0]
    all_gubun = '|'.join(r['구분'] for r in records_sorted)
    return all_gubun, first['체불액']

# ============================================================
# 3. 67개 파일 전체 스캔 → 대상 사업장의 모든 월별 스냅샷 수집
# ============================================================
pension_files = sorted(globmod.glob(os.path.join(DATASET_DIR, "*.csv")))
print(f"\n국민연금 파일 {len(pension_files)}개 스캔 중...")

DATE_RE = re.compile(r'(\d{4})(\d{2})(\d{2})\.csv$')

def extract_month(fpath):
    m = DATE_RE.search(os.path.basename(fpath))
    return f"{m.group(1)}-{m.group(2)}" if m else ''

snapshots = []
for fpath in pension_files:
    month = extract_month(fpath)
    count = 0
    with open(fpath, encoding='cp949', errors='replace') as f:
        reader = csv.DictReader(f)
        def _norm_col(c):
            c = c.strip()
            i = c.find('(')
            return c[:i].strip() if i > 0 else c
        reader.fieldnames = [_norm_col(c) for c in reader.fieldnames]
        for row in reader:
            name = row.get('사업장명', '').strip()
            biz_no = row.get('사업자등록번호', '').strip()
            key = (name, biz_no)
            if key not in target_keys:
                continue
            meta = target_keys[key]
            gubun, amount = get_unpaid_info(meta['매칭_임금체불_사업장명'])
            snapshots.append({
                '기준월': month,
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
                '매칭_임금체불_사업장명': meta['매칭_임금체불_사업장명'],
                '매칭_단계': meta['매칭_단계'],
                '체불공개구분': gubun,
                '체불액': amount,
            })
            count += 1
    print(f"  {month}: {count}건")

print(f"\n총 스냅샷: {len(snapshots)}건")

# ============================================================
# 4. 저장 (사업장별 시계열 순으로 정렬)
# ============================================================
snapshots.sort(key=lambda r: (r['사업자등록번호'], r['사업장명'], r['기준월']))

fieldnames = [
    '기준월', '사업장명', '사업자등록번호', '가입상태코드',
    '가입자수', '신규취득자수', '상실가입자수', '당월고지금액',
    '적용일자', '재등록일자', '탈퇴일자',
    '사업장업종코드명', '사업장업종코드', '사업장형태구분코드', '시도코드',
    '매칭_임금체불_사업장명', '매칭_단계', '체불공개구분', '체불액',
]
with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(snapshots)

# ============================================================
# 5. 요약 통계
# ============================================================
by_biz = defaultdict(int)
for s in snapshots:
    by_biz[(s['사업장명'], s['사업자등록번호'])] += 1
months_per_biz = list(by_biz.values())

print(f"\n{'='*60}")
print(f"저장 완료: {OUTPUT_CSV}")
print(f"  총 row: {len(snapshots)}건")
print(f"  고유 사업장: {len(by_biz)}개")
if months_per_biz:
    print(f"  사업장당 평균 스냅샷: {sum(months_per_biz)/len(months_per_biz):.1f}개월")
    print(f"  최대: {max(months_per_biz)}개월, 최소: {min(months_per_biz)}개월")
