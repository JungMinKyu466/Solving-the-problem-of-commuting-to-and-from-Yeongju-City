"""
사업장업종코드(KSIC 9차/8차) → KOSIS 산업 대분류 매핑 v2.

3단계 하이브리드 매핑:
  1. KSIC 9차/8차 prefix 2자리 매핑 (포괄적 dict)
  2. 사업장업종코드명 텍스트 키워드 fallback
  3. Manual override dict (직접 검증 결과)

출력:
  - dataset/ksic_industry_mapping.csv (prefix 매핑, 기존 호환용)
  - dataset/ksic_code_mapping.csv (코드 단위 매핑, 정밀)
"""
import csv, os, sys
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(10**7)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_BASE = os.path.join(os.path.dirname(BASE_DIR), 'dataset')
KOSIS_CSV = os.path.join(DATASET_BASE, '산업별_기업수_활동_신생_소멸__20260503004855.csv')
TREND_POS = os.path.join(DATASET_BASE, 'pension_trend.csv')
TREND_NEG = os.path.join(DATASET_BASE, 'pension_trend_negatives.csv')
PREFIX_OUT = os.path.join(DATASET_BASE, 'ksic_industry_mapping.csv')
CODE_OUT = os.path.join(DATASET_BASE, 'ksic_code_mapping.csv')

# ============================================================
# Step 1: KSIC 8차/9차 prefix 매핑 (포괄적)
# ============================================================
PREFIX_TO_CATEGORY = {}

# A. 농업, 임업 및 어업 (01-03)
for p in ['01', '02', '03']: PREFIX_TO_CATEGORY[p] = '농업, 임업 및 어업'

# B. 광업 (05-08)
for p in ['05', '06', '07', '08']: PREFIX_TO_CATEGORY[p] = '광업'

# C. 제조업 — KSIC 9차/8차 모두 포함 (10-37)
for i in range(10, 38):
    PREFIX_TO_CATEGORY[f'{i:02d}'] = '제조업'

# D. 전기, 가스, 증기 및 공기조절 공급업 (35 in KSIC 10차, 40 in 8차)
PREFIX_TO_CATEGORY['35'] = '전기, 가스, 증기 및 공기조절 공급업'
PREFIX_TO_CATEGORY['40'] = '전기, 가스, 증기 및 공기조절 공급업'

# E. 수도, 하수, 폐기물 (36-39 in KSIC 10차, 41 in 8차)
PREFIX_TO_CATEGORY['36'] = '수도, 하수 및 폐기물처리, 원료재생업'
PREFIX_TO_CATEGORY['37'] = '수도, 하수 및 폐기물처리, 원료재생업'
PREFIX_TO_CATEGORY['38'] = '수도, 하수 및 폐기물처리, 원료재생업'
PREFIX_TO_CATEGORY['39'] = '수도, 하수 및 폐기물처리, 원료재생업'
PREFIX_TO_CATEGORY['41'] = '수도, 하수 및 폐기물처리, 원료재생업'  # KSIC 8차

# F. 건설업 (41-42 in KSIC 10차, 45 in 8차/9차)
PREFIX_TO_CATEGORY['42'] = '건설업'
PREFIX_TO_CATEGORY['45'] = '건설업'  # KSIC 8차/9차

# G. 도매 및 소매업 (45-47 in 10차, 50-52 in 9차)
PREFIX_TO_CATEGORY['46'] = '도매 및 소매업'
PREFIX_TO_CATEGORY['47'] = '도매 및 소매업'
PREFIX_TO_CATEGORY['50'] = '도매 및 소매업'  # 자동차 및 부품 판매 (8차/9차)
PREFIX_TO_CATEGORY['51'] = '도매 및 소매업'  # 도매 및 상품 중개업
PREFIX_TO_CATEGORY['52'] = '도매 및 소매업'  # 소매업

# H. 운수 및 창고업 (49-52 in 10차, 60-63 in 9차)
PREFIX_TO_CATEGORY['49'] = '운수 및 창고업'
PREFIX_TO_CATEGORY['60'] = '운수 및 창고업'  # 육상운송 (8차/9차)
PREFIX_TO_CATEGORY['61'] = '운수 및 창고업'  # 수상운송
PREFIX_TO_CATEGORY['62'] = '운수 및 창고업'  # 항공운송
PREFIX_TO_CATEGORY['63'] = '운수 및 창고업'  # 창고 및 운송관련

# I. 숙박 및 음식점업 (55-56 in 10차, 55 in 9차)
PREFIX_TO_CATEGORY['55'] = '숙박 및 음식점업'
PREFIX_TO_CATEGORY['56'] = '숙박 및 음식점업'

# J. 정보통신업 (58-63 in 10차, 22, 64, 92 일부 in 8차)
PREFIX_TO_CATEGORY['58'] = '정보통신업'
PREFIX_TO_CATEGORY['59'] = '정보통신업'
# 60, 61, 62, 63 이미 운수로 매핑됨 — 9차 데이터에서는 운수임

# K. 금융 및 보험업 (64-66 in 10차, 65-67 in 9차)
PREFIX_TO_CATEGORY['64'] = '금융 및 보험업'  # 9차 8차 모두 금융업
PREFIX_TO_CATEGORY['65'] = '금융 및 보험업'  # 보험·연금
PREFIX_TO_CATEGORY['66'] = '금융 및 보험업'  # 금융보조 (10차)
PREFIX_TO_CATEGORY['67'] = '금융 및 보험업'  # 금융보조 (9차)

# L. 부동산업 (68 in 10차, 70 in 9차)
PREFIX_TO_CATEGORY['68'] = '부동산업'
PREFIX_TO_CATEGORY['70'] = '부동산업'  # 부동산업 및 임대업 (9차)

# M. 전문, 과학 및 기술 서비스업 (70-73 in 10차, 71-73 in 9차)
PREFIX_TO_CATEGORY['71'] = '전문과학기술서비스업'
PREFIX_TO_CATEGORY['72'] = '전문과학기술서비스업'
PREFIX_TO_CATEGORY['73'] = '전문과학기술서비스업'

# N. 사업시설관리, 사업지원 및 임대 서비스업 (74-76 in 10차/9차)
PREFIX_TO_CATEGORY['74'] = '사업시설관리, 사업지원 및 임대 서비스업'
PREFIX_TO_CATEGORY['75'] = '사업시설관리, 사업지원 및 임대 서비스업'
PREFIX_TO_CATEGORY['76'] = '사업시설관리, 사업지원 및 임대 서비스업'

# O. 공공행정 (84) → 협회·단체로 fallback (KOSIS에 별도 카테고리 없음)
PREFIX_TO_CATEGORY['84'] = '협회 및 단체, 수리 및 기타 개인서비스업'

# P. 교육서비스업 (85 in 10차, 80 in 8차/9차 일부)
PREFIX_TO_CATEGORY['85'] = '교육서비스업'
PREFIX_TO_CATEGORY['80'] = '교육서비스업'  # 8차 교육

# Q. 보건업 및 사회복지 서비스업 (86-87)
PREFIX_TO_CATEGORY['86'] = '보건업 및 사회복지 서비스업'
PREFIX_TO_CATEGORY['87'] = '보건업 및 사회복지 서비스업'

# R. 예술, 스포츠 및 여가관련 서비스업 (90-91 in 10차)
PREFIX_TO_CATEGORY['90'] = '예술, 스포츠 및 여가관련 서비스업'
PREFIX_TO_CATEGORY['91'] = '예술, 스포츠 및 여가관련 서비스업'

# S. 협회 및 단체, 수리 및 기타 개인서비스업 (94-96 in 10차, 92-93 in 8차/9차)
PREFIX_TO_CATEGORY['92'] = '협회 및 단체, 수리 및 기타 개인서비스업'  # 8차: 영화·방송 → 정보통신이지만 우리 데이터에선 자동차수리 多, 일관성 위해 협회기타
PREFIX_TO_CATEGORY['93'] = '협회 및 단체, 수리 및 기타 개인서비스업'  # 8차/9차: 기타 서비스
PREFIX_TO_CATEGORY['94'] = '협회 및 단체, 수리 및 기타 개인서비스업'
PREFIX_TO_CATEGORY['95'] = '협회 및 단체, 수리 및 기타 개인서비스업'
PREFIX_TO_CATEGORY['96'] = '협회 및 단체, 수리 및 기타 개인서비스업'

# T, U: 가구내·국제기구 — 매핑 안 함 (UNKNOWN 유지)
# '99', '00', 'EMPTY' → UNKNOWN

print(f'Step 1: prefix 매핑 dict {len(PREFIX_TO_CATEGORY)}개 정의')

# ============================================================
# Step 2: 사업장업종코드명 키워드 fallback
# ============================================================
# 우선순위 순 (먼저 매칭되면 그 카테고리)
KEYWORD_RULES = [
    # === 우선순위 높은 복합 키워드 ===
    # 사업시설관리 우선 매칭 (74에서 청소 등 잘못 잡히지 않게)
    (['건축물 일반 청소', '건축물 청소', '경비', '인력 공급', '고용 알선',
      '콜센터', '텔레마케팅', '문서 작성', '포장 및 충전', '통관 대리',
      '전시', '컨벤션', '행사 대행'],
     '사업시설관리, 사업지원 및 임대 서비스업'),

    # 전문과학기술 (74 prefix 충돌 해소)
    (['건축 설계', '건축설계', '엔지니어링 서비스', '광고 대행', '경영 컨설팅',
      '경영컨설팅', '세무사', '회계', '연구개발업', '연구 개발',
      '법무', '법률', '특허', '감정평가', '시장 조사', '시장조사',
      '변호사', '변리사', '측량', '제도업', '시험 검사', '분석업',
      '시각 디자인', '제품 디자인', '인테리어 디자인', '공공관계',
      '여론 조사', '경영 자문', '경영자문'],
     '전문과학기술서비스업'),

    # 정보통신 (92 prefix 충돌 해소)
    (['영화', '비디오물', '방송 프로그램', '비디오 제작', '사진 촬영', '영상 촬영'],
     '정보통신업'),

    # 예술·스포츠 (92 prefix 충돌 해소)
    (['골프장', '골프 연습', '스키장', '스포츠 서비스', '경기장',
      '여가', '오락', '공연', '미술', '박물관', '도서관', '체력 단련'],
     '예술, 스포츠 및 여가관련 서비스업'),

    # 교육 (직원 훈련기관 등)
    (['직원 훈련', '훈련기관'],
     '교육서비스업'),

    # === 일반 키워드 (덜 구체적) ===
    (['건설', '시공', '건축', '토목', '도장', '도배', '미장', '방수', '창호', '석공',
      '철근', '콘크리트', '굴착', '비계', '공사업'],
     '건설업'),

    (['제조', '생산', '가공', '주물', '주조', '금형', '제작', '제분', '제빵',
      '제과', '봉제', '의류 생산', '식품 제조'],
     '제조업'),

    (['도매', '소매', '판매', '매장', '백화점', '마트', '편의점', '쇼핑'],
     '도매 및 소매업'),

    (['숙박', '호텔', '모텔', '여관', '음식점', '식당', '카페', '주점', '뷔페',
      '레스토랑', '패스트푸드'],
     '숙박 및 음식점업'),

    (['운수', '운송', '물류', '창고', '택배', '배송', '화물', '여객'],
     '운수 및 창고업'),

    (['교육', '학원', '강사', '교습', '학습', '교과', '입시', '평생교육'],
     '교육서비스업'),

    (['금융', '보험', '대출', '신용', '증권', '카드', '은행', '예금', '연금'],
     '금융 및 보험업'),

    (['부동산', '임대', '중개', '주거 시설', '비주거용', '아파트 관리'],
     '부동산업'),

    (['병원', '의원', '진료', '의료', '치과', '한의원', '약국', '복지', '요양',
      '간병', '노인', '장애'],
     '보건업 및 사회복지 서비스업'),

    (['예술', '스포츠', '여가', '골프', '영화', '공연', '체육', '관광',
      '오락', '스파', '미술', '음악'],
     '예술, 스포츠 및 여가관련 서비스업'),

    (['IT', '소프트웨어', '정보', '통신', '방송', '컴퓨터', '인터넷', '웹',
      '데이터', '시스템', '플랫폼', '미디어'],
     '정보통신업'),

    (['연구', '과학', '엔지니어링', '컨설팅', '회계', '법률', '광고',
      '디자인', '번역', '특허', '감정'],
     '전문과학기술서비스업'),

    (['청소', '경비', '보안', '인력', '관리 서비스', '지원 서비스'],
     '사업시설관리, 사업지원 및 임대 서비스업'),

    (['수리', '정비', '미용', '이용', '세탁', '장묘', '애완', '개인 서비스'],
     '협회 및 단체, 수리 및 기타 개인서비스업'),

    (['농업', '재배', '축산', '양식', '임업'],
     '농업, 임업 및 어업'),

    (['광물', '광업', '석탄', '원유'],
     '광업'),

    (['전기', '가스', '증기'],
     '전기, 가스, 증기 및 공기조절 공급업'),

    (['수도', '하수', '폐기물', '재활용'],
     '수도, 하수 및 폐기물처리, 원료재생업'),
]


def map_by_prefix(code):
    if not code or len(code) < 2:
        return None
    return PREFIX_TO_CATEGORY.get(code[:2])


def map_by_keyword(codename):
    if not codename:
        return None
    for keywords, category in KEYWORD_RULES:
        if any(kw in codename for kw in keywords):
            return category
    return None


# ============================================================
# Step 3: Manual override (수동 검증 결과)
# ============================================================
# unique 코드 검토 후 잘못된 매핑 발견 시 여기에 추가
MANUAL_OVERRIDES = {
    # 코드 6자리: 카테고리 (수동 검증 결과)
    '749942': '전문과학기술서비스업',  # 기타 전문 서비스업
    # UNKNOWN을 의미적 명시 카테고리로 분리 (발견 8 보완)
    '999999': 'BIZ_NO_MISSING',  # 사업자번호 미존재 — 행정기관·임시 프로젝트·청산 사업장 등
    '000000': 'NA',              # 해당없음 — 분류 불가
}


# 충돌 prefix: prefix 매핑이 부정확한 경우, 키워드 매핑 우선
CONFLICT_PREFIXES = {'74', '92', '93'}


def map_industry(code, codename):
    """3단계 하이브리드 매핑.
    - 충돌 prefix(74, 92, 93)는 코드명 키워드 우선
    - 그 외는 prefix 우선
    """
    # 1. Manual override 우선
    if code in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[code], 'manual'

    # 2. 충돌 prefix는 키워드 매핑 우선 시도
    if code and len(code) >= 2 and code[:2] in CONFLICT_PREFIXES:
        cat = map_by_keyword(codename)
        if cat:
            return cat, 'keyword(conflict)'
        # 키워드 매칭 실패 시 prefix로 fallback
        cat = map_by_prefix(code)
        if cat:
            return cat, 'prefix(conflict-fallback)'

    # 3. 일반 prefix 매핑
    cat = map_by_prefix(code)
    if cat:
        return cat, 'prefix'

    # 4. 키워드 fallback
    cat = map_by_keyword(codename)
    if cat:
        return cat, 'keyword'

    return 'UNKNOWN', 'unmapped'


# ============================================================
# KOSIS 데이터 (소멸률, 신생률) 로드
# ============================================================
kosis_data = {}
with open(KOSIS_CSV, encoding='cp949') as f:
    reader = csv.reader(f)
    next(reader)
    next(reader)
    for row in reader:
        if not row or len(row) < 6:
            continue
        cat = row[0].strip().strip('"')
        if not cat or cat == '전체':
            continue
        try:
            kosis_data[cat] = {
                '활동_2023': int(row[1]),
                '신생_2023': int(row[2]),
                '소멸_2023': int(row[3]),
                '신생률_2023': float(row[4]),
                '소멸률_2023': float(row[5]),
            }
        except (ValueError, IndexError):
            continue
print(f'KOSIS 카테고리: {len(kosis_data)}개')

# ============================================================
# 모든 unique 사업장업종코드 수집 + 매핑
# ============================================================
print('\n양성/음성 trend에서 unique 코드 수집 중...')
codes = {}  # code -> (codename, count)
for fn in [TREND_POS, TREND_NEG]:
    if not os.path.exists(fn):
        continue
    with open(fn, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            c = r.get('사업장업종코드', '').strip()
            n = r.get('사업장업종코드명', '').strip()
            if not c:
                continue
            if c not in codes:
                codes[c] = [n, 0]
            codes[c][1] += 1

print(f'  unique 코드: {len(codes)}개')

# 매핑 적용
mapped = []
method_count = Counter()
for code, (name, cnt) in codes.items():
    cat, method = map_industry(code, name)
    info = kosis_data.get(cat, {})
    mapped.append({
        'code': code,
        'codename': name,
        'count': cnt,
        'category': cat,
        'method': method,
        'death_rate_2023': info.get('소멸률_2023', 0.0),
        'birth_rate_2023': info.get('신생률_2023', 0.0),
    })
    method_count[method] += 1

print(f'\n매핑 방법별 카운트:')
for m, c in method_count.most_common():
    print(f'  {m}: {c}개')

# UNKNOWN 케이스 확인
unmapped = [m for m in mapped if m['category'] == 'UNKNOWN']
print(f'\nUNKNOWN {len(unmapped)}개 (수동 검증 대상):')
for m in sorted(unmapped, key=lambda x: -x['count'])[:30]:
    print(f"  {m['code']} ({m['count']:>4}건) | {m['codename'][:50]}")
if len(unmapped) > 30:
    print(f'  ... 외 {len(unmapped)-30}개')

# 저장: 코드별 정밀 매핑
with open(CODE_OUT, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['code', 'codename', 'count', 'category',
                                            'method', 'death_rate_2023', 'birth_rate_2023'])
    writer.writeheader()
    writer.writerows(sorted(mapped, key=lambda x: -x['count']))
print(f'\n코드 매핑 저장: {CODE_OUT}')

# 저장: prefix 매핑 (기존 호환)
with open(PREFIX_OUT, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['ksic_prefix', 'industry_category',
                                            'activity_2023', 'birth_2023', 'death_2023',
                                            'birth_rate_2023', 'death_rate_2023'])
    writer.writeheader()
    for prefix, cat in sorted(PREFIX_TO_CATEGORY.items()):
        d = kosis_data.get(cat, {})
        writer.writerow({
            'ksic_prefix': prefix,
            'industry_category': cat,
            'activity_2023': d.get('활동_2023', ''),
            'birth_2023': d.get('신생_2023', ''),
            'death_2023': d.get('소멸_2023', ''),
            'birth_rate_2023': d.get('신생률_2023', ''),
            'death_rate_2023': d.get('소멸률_2023', ''),
        })
print(f'Prefix 매핑 저장: {PREFIX_OUT}')

# 카테고리별 분포
print(f'\n카테고리별 unique 코드 수:')
cat_count = Counter(m['category'] for m in mapped)
for cat, c in cat_count.most_common():
    print(f'  {c:>4} | {cat}')
