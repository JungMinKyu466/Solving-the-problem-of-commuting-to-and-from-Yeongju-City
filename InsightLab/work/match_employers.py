"""
임금체불 사업장을 국민연금 가입 사업장 내역에서 매칭하는 스크립트.

매칭 전략:
1. 사업장명 정규화: 법인 표기 통일, 공백/특수문자 제거
2. 주소: 임금체불의 주소지+소재지 양쪽 vs 국민연금의 지번+도로명 양쪽 교차비교
3. 전체 국민연금 파일(38개) 통합 검색 (탈퇴 사업장 포함)
4. 5단계 매칭:
   - 1단계: 이름 일치 + 시/군/구 일치
   - 2단계: 이름 일치 + 시/도 일치
   - 3단계: 이름 포함(4자+) + 시/군/구 일치
   - 4단계: 이름 포함(4자+) + 시/도 일치
   - 5단계: 이름 일치만 (5자+, 후보 3건 이하)
"""

import csv
import re
import sys
import os
import glob as globmod
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_BASE = os.path.join(os.path.dirname(BASE_DIR), "dataset")
DATASET_DIR = os.path.join(DATASET_BASE, "국민연금 공공데이터포털")
UNPAID_CSV = os.path.join(DATASET_BASE, "unpaid_employers.csv")
OUTPUT_CSV = os.path.join(DATASET_BASE, "matched_pension_unpaid.csv")
UNMATCHED_CSV = os.path.join(DATASET_BASE, "unmatched_unpaid.csv")

# ============================================================
# 사업장명 정규화
# ============================================================
CORP_REMOVALS = [
    '주식회사', '(주)', '(유)', '유한회사', '유한책임회사',
    '합자회사', '합명회사', '사회적협동조합', '협동조합',
    '영농조합법인', '농업회사법인', '어업회사법인',
    '재단법인', '사단법인', '학교법인', '의료법인', '사회복지법인',
]

def normalize_name(name):
    if not name:
        return ''
    s = name.strip()
    if s.startswith('개인건설업자') or s.startswith('개인사업자'):
        return ''
    for token in CORP_REMOVALS:
        s = s.replace(token, '')
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[\s\-_·.,/\\]', '', s)
    return s

# ============================================================
# 주소 정규화
# ============================================================
ADDR_ABBR = {
    '서울': '서울특별시', '서울시': '서울특별시',
    '부산': '부산광역시', '부산시': '부산광역시',
    '대구': '대구광역시', '대구시': '대구광역시',
    '인천': '인천광역시', '인천시': '인천광역시',
    '광주': '광주광역시', '광주시': '광주광역시',
    '대전': '대전광역시', '대전시': '대전광역시',
    '울산': '울산광역시', '울산시': '울산광역시',
    '세종': '세종특별자치시', '세종시': '세종특별자치시',
    '세종특별자치시': '세종특별자치시',
    '경기': '경기도', '강원': '강원특별자치도', '강원도': '강원특별자치도',
    '충북': '충청북도', '충남': '충청남도',
    '전북': '전북특별자치도', '전라북도': '전북특별자치도',
    '전남': '전라남도', '경북': '경상북도', '경남': '경상남도',
    '제주': '제주특별자치도',
}

def get_sido(addr):
    if not addr:
        return ''
    parts = addr.strip().split()
    return ADDR_ABBR.get(parts[0], parts[0]) if parts else ''

def get_sigungu(addr):
    if not addr:
        return ''
    parts = addr.strip().split()
    if len(parts) < 2:
        return get_sido(addr)
    sido = ADDR_ABBR.get(parts[0], parts[0])
    return f"{sido} {parts[1]}"


# ============================================================
# 임금체불 데이터 로드
# ============================================================
print("=" * 60)
print("임금체불 명단 로드 중...")
unpaid_list = []
with open(UNPAID_CSV, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row['사업장명'].strip()
        addr1 = row['주소지'].strip()  # 사업장 주소
        addr2 = row.get('소재지', '').strip()  # 소재지 (두 번째 주소)
        norm = normalize_name(name)

        # 두 주소 모두에서 시도/시군구 추출
        sidos = set(filter(None, [get_sido(addr1), get_sido(addr2)]))
        sigungus = set(filter(None, [get_sigungu(addr1), get_sigungu(addr2)]))

        unpaid_list.append({
            'name': name, 'addr': addr1, 'addr2': addr2,
            'norm': norm, 'sidos': sidos, 'sigungus': sigungus,
            'row': row,
        })

valid_unpaid = [u for u in unpaid_list if u['norm']]
skipped = [u for u in unpaid_list if not u['norm']]
print(f"  총 {len(unpaid_list)}건, 유효 {len(valid_unpaid)}건 (개인 {len(skipped)}건 제외)")

# ============================================================
# 국민연금 데이터 로드 (모든 파일 통합)
# ============================================================
print("\n국민연금 가입 사업장 내역 통합 로드 중...")
pension_files = sorted(globmod.glob(os.path.join(DATASET_DIR, "*.csv")))
print(f"  {len(pension_files)}개 파일")

# 사업자등록번호 기준 중복 제거, 가장 최근 행을 보관
# 하지만 사업자등록번호가 짧으므로 (사업장명+사업자등록번호) 조합으로 dedup
pension_by_name = defaultdict(list)  # norm_name -> list of entries
all_pension_rows = []  # 전체 row 보관
seen = set()  # dedup key

for fpath in reversed(pension_files):  # 최신 파일 우선
    fname = os.path.basename(fpath)
    with open(fpath, encoding='cp949', errors='replace') as f:
        reader = csv.DictReader(f)
        # 옛날 파일은 컬럼명에 앞 공백 + 괄호 설명 포함되어 있어 정규화
        def _norm_col(c):
            c = c.strip()
            i = c.find('(')
            return c[:i].strip() if i > 0 else c
        reader.fieldnames = [_norm_col(c) for c in reader.fieldnames]
        for row in reader:
            name = row.get('사업장명', '').strip()
            biz_no = row.get('사업자등록번호', '').strip()
            addr_jibun = row.get('사업장지번상세주소', '').strip()
            addr_road = row.get('사업장도로명상세주소', '').strip()
            norm = normalize_name(name)
            if not norm:
                continue

            # dedup: 사업장명 + 사업자등록번호
            dedup_key = f"{name}|{biz_no}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            sidos = set(filter(None, [get_sido(addr_road), get_sido(addr_jibun)]))
            sigungus = set(filter(None, [get_sigungu(addr_road), get_sigungu(addr_jibun)]))

            idx = len(all_pension_rows)
            all_pension_rows.append(row)
            pension_by_name[norm].append({
                'idx': idx, 'norm': norm,
                'sidos': sidos, 'sigungus': sigungus,
            })

print(f"  고유 사업장: {len(all_pension_rows)}건, 고유 이름: {len(pension_by_name)}개")

# ============================================================
# 매칭
# ============================================================
print("\n매칭 시작...")
matched_indices = {}
matched_unpaid = set()

for ui, u in enumerate(valid_unpaid):
    found = False
    candidates = pension_by_name.get(u['norm'], [])

    # 1단계: 이름 일치 + 시군구 일치
    for c in candidates:
        if u['sigungus'] & c['sigungus']:
            matched_indices[c['idx']] = (u['name'], '1단계_이름일치_시군구일치')
            found = True
    if found:
        matched_unpaid.add(ui)
        continue

    # 2단계: 이름 일치 + 시도 일치
    for c in candidates:
        if u['sidos'] & c['sidos']:
            matched_indices[c['idx']] = (u['name'], '2단계_이름일치_시도일치')
            found = True
    if found:
        matched_unpaid.add(ui)
        continue

    # 3단계: 이름 포함(4자+) + 시군구 일치
    if len(u['norm']) >= 4:
        for norm, entries in pension_by_name.items():
            if len(norm) < 4:
                continue
            if u['norm'] in norm or norm in u['norm']:
                for c in entries:
                    if u['sigungus'] & c['sigungus']:
                        matched_indices[c['idx']] = (u['name'], '3단계_이름포함_시군구일치')
                        found = True
    if found:
        matched_unpaid.add(ui)
        continue

    # 4단계: 이름 포함(4자+) + 시도 일치
    if len(u['norm']) >= 4:
        for norm, entries in pension_by_name.items():
            if len(norm) < 4:
                continue
            if u['norm'] in norm or norm in u['norm']:
                for c in entries:
                    if u['sidos'] & c['sidos']:
                        matched_indices[c['idx']] = (u['name'], '4단계_이름포함_시도일치')
                        found = True
    if found:
        matched_unpaid.add(ui)
        continue

    # 5단계: 이름 일치만 (5자+, 후보 3건 이하)
    if len(u['norm']) >= 5 and 0 < len(candidates) <= 3:
        for c in candidates:
            matched_indices[c['idx']] = (u['name'], '5단계_이름일치_주소무관')
        matched_unpaid.add(ui)

# ============================================================
# 결과
# ============================================================
unmatched = [u for i, u in enumerate(valid_unpaid) if i not in matched_unpaid]

print(f"\n{'='*60}")
print(f"매칭 결과:")
print(f"  총 임금체불: {len(unpaid_list)}건")
print(f"  유효 (개인 제외): {len(valid_unpaid)}건")
print(f"  매칭 성공: {len(matched_unpaid)}건 ({len(matched_unpaid)/len(valid_unpaid)*100:.1f}%)")
print(f"  매칭된 국민연금 행: {len(matched_indices)}건")
print(f"  매칭 실패: {len(unmatched)}건")

stage_counts = defaultdict(int)
for idx, (name, stage) in matched_indices.items():
    stage_counts[stage] += 1
print(f"\n단계별:")
for stage, count in sorted(stage_counts.items()):
    print(f"  {stage}: {count}건")

if unmatched:
    print(f"\n매칭 실패 (상위 20건):")
    for u in unmatched[:20]:
        print(f"  {u['name']} | {u['addr']} | {u['addr2']}")
    if len(unmatched) > 20:
        print(f"  ... 외 {len(unmatched)-20}건")

# 저장
print(f"\n결과 저장...")
fieldnames = list(all_pension_rows[0].keys()) + ['매칭_임금체불_사업장명', '매칭_단계']
with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for idx in sorted(matched_indices.keys()):
        row = dict(all_pension_rows[idx])
        name, stage = matched_indices[idx]
        row['매칭_임금체불_사업장명'] = name
        row['매칭_단계'] = stage
        writer.writerow(row)
print(f"  매칭 결과: {OUTPUT_CSV} ({len(matched_indices)}건)")

with open(UNMATCHED_CSV, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['사업장명', '주소지', '소재지', '정규화이름', '시도', '시군구'])
    for u in unmatched + skipped:
        writer.writerow([u['name'], u['addr'], u.get('addr2',''), u['norm'],
                        '|'.join(u['sidos']), '|'.join(u['sigungus'])])
print(f"  매칭 실패: {UNMATCHED_CSV} ({len(unmatched)+len(skipped)}건)")
print("완료!")
