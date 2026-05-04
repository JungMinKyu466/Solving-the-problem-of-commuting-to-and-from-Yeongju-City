"""
LightGBM 학습용 데이터셋 빌드.

입력:
  - dataset/pension_trend.csv (양성 시계열, 풀히스토리)
  - dataset/pension_trend_negatives.csv (음성 시계열, [t-18, t-6]만)
  - dataset/ksic_industry_mapping.csv (F14 매핑)

처리 (3가지 보정 적용):
  1. 옵션 D 보간: 가입자수·고지금액은 forward-fill, 신규·상실은 0,
                  결측 월수/비율을 별도 피처로 추가
  2. 시점 augmentation: 양성 1개당 5개 shift 윈도우 (-2, -1, 0, +1, +2)
  3. 음성 다운샘플링: 양성 augmented 샘플 수에 맞춰 1:1

출력: dataset/training_dataset.csv
  business_id, source_business (Group ID), label, gubun, t_disclose,
  + 14개 컨셉 피처의 압축 컬럼 + 결측 패턴 피처 = 약 33개
"""
import csv, os, sys, math, random
from collections import defaultdict
from statistics import mean, stdev

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(10**7)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_BASE = os.path.join(os.path.dirname(BASE_DIR), 'dataset')
TREND_POS = os.path.join(DATASET_BASE, 'pension_trend.csv')
TREND_NEG = os.path.join(DATASET_BASE, 'pension_trend_negatives.csv')
KSIC_MAP = os.path.join(DATASET_BASE, 'ksic_industry_mapping.csv')
OUTPUT_CSV = os.path.join(DATASET_BASE, 'training_dataset.csv')

# 양성 매칭 단계 필터 (4단계 제외 — false-positive 위험 高)
USE_STAGES = ('1단계', '2단계', '3단계', '5단계')

# 입력 구간 12개월 중 최소 보유 비율
MIN_COVERAGE = 6 / 12  # 50%

# 시점 augmentation shift 범위 (개월)
AUG_SHIFTS = [-2, -1, 0, 1, 2]

# 음성:양성 비율 (양성 augmented 기준)
NEG_RATIO = 1  # 1:1

SEED = 42
random.seed(SEED)

# 공개 차수 → t (lead 6 → 입력 [t-18, t-6])
GUBUN_DATE = {
    '2023년 1차': '2023-08',
    '2023년 2차': '2023-12',
    '2024년 1차': '2024-08',
    '2024년 2차': '2024-12',
    '2025년 1차': '2025-08',
}


def add_months(yyyymm, delta):
    y, m = map(int, yyyymm.split('-'))
    total = y * 12 + (m - 1) + delta
    return f'{total // 12:04d}-{total % 12 + 1:02d}'


def safe_int(x, default=0):
    try:
        return int(x) if x and x.strip() else default
    except (ValueError, TypeError):
        return default


def safe_float(x, default=0.0):
    try:
        return float(x) if x and x.strip() else default
    except (ValueError, TypeError):
        return default


def parse_yyyymmdd(s):
    if not s or len(s) < 7:
        return None
    s = s.strip()
    if '-' in s:
        return s[:7]
    return f'{s[:4]}-{s[4:6]}'


def months_between(start_yyyymm, end_yyyymm):
    sy, sm = map(int, start_yyyymm.split('-'))
    ey, em = map(int, end_yyyymm.split('-'))
    return (ey - sy) * 12 + (em - sm)


# ============================================================
# KSIC 매핑 로드 (코드 단위 정밀 매핑)
# ============================================================
KSIC_CODE_MAP_CSV = os.path.join(DATASET_BASE, 'ksic_code_mapping.csv')
KSIC_PREFIX_MAP_CSV = KSIC_MAP  # 기존 prefix 매핑 (fallback용)

# 코드 단위 매핑 (정밀)
code_map = {}
with open(KSIC_CODE_MAP_CSV, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        code_map[r['code']] = {
            'industry_category': r['category'],
            'death_rate_2023': safe_float(r['death_rate_2023']),
            'birth_rate_2023': safe_float(r['birth_rate_2023']),
        }

# Prefix 매핑 (코드에 없는 경우의 fallback)
prefix_map = {}
with open(KSIC_PREFIX_MAP_CSV, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        prefix_map[r['ksic_prefix']] = {
            'industry_category': r['industry_category'],
            'death_rate_2023': safe_float(r['death_rate_2023']),
            'birth_rate_2023': safe_float(r['birth_rate_2023']),
        }


def map_industry(industry_code, industry_codename=''):
    """1차: 코드 단위 매핑, 2차: prefix fallback."""
    if not industry_code:
        return {'industry_category': 'UNKNOWN', 'death_rate_2023': 0.0, 'birth_rate_2023': 0.0}
    if industry_code in code_map:
        return code_map[industry_code]
    if len(industry_code) >= 2:
        return prefix_map.get(industry_code[:2], {
            'industry_category': 'UNKNOWN',
            'death_rate_2023': 0.0, 'birth_rate_2023': 0.0,
        })
    return {'industry_category': 'UNKNOWN', 'death_rate_2023': 0.0, 'birth_rate_2023': 0.0}


# ============================================================
# 양성/음성 시계열 로드
# ============================================================
print('양성 시계열 로드 중...')
pos_groups = defaultdict(list)
with open(TREND_POS, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        if not any(r['매칭_단계'].startswith(s) for s in USE_STAGES):
            continue
        gubun = r['체불공개구분'].split('|')[0].strip() if r['체불공개구분'] else ''
        if gubun not in GUBUN_DATE:
            continue
        key = (r['매칭_임금체불_사업장명'], gubun)
        pos_groups[key].append(r)
print(f'  양성 그룹: {len(pos_groups)}개')

print('음성 시계열 로드 중...')
neg_groups = defaultdict(list)
with open(TREND_NEG, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        key = (r['사업자등록번호'], r['가상_공개차수'])
        neg_groups[key].append(r)
print(f'  음성 그룹: {len(neg_groups)}개')


# ============================================================
# 슬라이스 + 사업자번호 합산
# ============================================================
def slice_window(rows, t):
    """[t-18, t-6] 12개월 슬라이스. 동월 row는 합산."""
    start = add_months(t, -18)
    end = add_months(t, -6)
    by_month = defaultdict(list)
    for r in rows:
        m = r['기준월']
        if start <= m <= end:
            by_month[m].append(r)
    aggregated = {}
    for m, rs in by_month.items():
        emp = sum(safe_int(r.get('가입자수', '0')) for r in rs)
        new = sum(safe_int(r.get('신규취득자수', '0')) for r in rs)
        loss = sum(safe_int(r.get('상실가입자수', '0')) for r in rs)
        billing = sum(safe_int(r.get('당월고지금액', '0')) for r in rs)
        first = rs[0]
        aggregated[m] = {
            '가입자수': emp,
            '신규취득자수': new,
            '상실가입자수': loss,
            '당월고지금액': billing,
            '적용일자': first.get('적용일자', ''),
            '시도코드': first.get('시도코드', ''),
            '사업장업종코드': first.get('사업장업종코드', ''),
            '사업장업종코드명': first.get('사업장업종코드명', ''),
        }
    return aggregated


# ============================================================
# 옵션 D: 필드별 차등 보간 + 결측 플래그
# ============================================================
def impute_window(window, t):
    """
    [t-18, t-6] 12개월에 대해:
    - 가입자수, 고지금액: forward-fill (직전 관측값 복사)
    - 신규, 상실: 0으로
    - 정적 필드: 직전 관측값 복사
    추가로: 월별 보간 여부 플래그(is_imputed) 부여.
    """
    start = add_months(t, -18)
    end = add_months(t, -6)
    months_full = []
    cur = start
    while cur <= end:
        months_full.append(cur)
        cur = add_months(cur, 1)

    filled = {}
    last_state = None  # 가입자수, 고지금액 등 상태 필드용
    static_last = None
    is_imputed_flags = {}

    for m in months_full:
        if m in window:
            filled[m] = dict(window[m])
            last_state = window[m]
            static_last = window[m]
            is_imputed_flags[m] = 0
        else:
            is_imputed_flags[m] = 1
            base = {
                '가입자수': last_state['가입자수'] if last_state else 0,
                '신규취득자수': 0,  # 이벤트성 — 0
                '상실가입자수': 0,  # 이벤트성 — 0
                '당월고지금액': last_state['당월고지금액'] if last_state else 0,
                '적용일자': static_last.get('적용일자', '') if static_last else '',
                '시도코드': static_last.get('시도코드', '') if static_last else '',
                '사업장업종코드': static_last.get('사업장업종코드', '') if static_last else '',
                '사업장업종코드명': static_last.get('사업장업종코드명', '') if static_last else '',
            }
            filled[m] = base

    return filled, is_imputed_flags, months_full


# ============================================================
# 14개 피처 + 결측 플래그 derive
# ============================================================
def derive_features(window, t):
    if not window:
        return None
    coverage = len(window) / 12
    if coverage < MIN_COVERAGE:
        return None

    # 보간 적용
    filled, imputed_flags, months_full = impute_window(window, t)

    # 시계열 추출
    emp_series = [filled[m]['가입자수'] for m in months_full]
    new_series = [filled[m]['신규취득자수'] for m in months_full]
    loss_series = [filled[m]['상실가입자수'] for m in months_full]
    bill_series = [filled[m]['당월고지금액'] for m in months_full]

    # 파생 시계열
    turnover_series = [(l / e if e > 0 else 0.0) for e, l in zip(emp_series, loss_series)]
    salary_series = [(b / e / 0.09 if e > 0 else 0.0) for e, b in zip(emp_series, bill_series)]
    replace_series = [(n + 1) / (l + 1) for n, l in zip(new_series, loss_series)]

    # 통계 헬퍼
    def safe_mean(arr): return mean(arr) if arr else 0.0
    def safe_std(arr): return stdev(arr) if len(arr) > 1 else 0.0
    def safe_max(arr): return max(arr) if arr else 0.0
    def safe_min(arr): return min(arr) if arr else 0.0

    def change_rate(arr, n):
        if len(arr) < n + 1: return 0.0
        denom = arr[-(n + 1)]
        return (arr[-1] - denom) / denom if denom > 0 else 0.0

    # F05: salary_drop_consecutive
    max_streak = cur_streak = 0
    for i in range(1, len(salary_series)):
        if salary_series[i] < salary_series[i - 1]:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0

    # F06: turnover_momentum
    if len(turnover_series) >= 12:
        recent = mean(turnover_series[-3:])
        prev = mean(turnover_series[:-3])
        momentum = recent / prev if prev > 0 else 0.0
    else:
        momentum = 0.0

    # F07: zero_emp_months
    zero_emp_months = sum(1 for e in emp_series if e == 0)

    # F08: emp_volatility
    emp_avg = safe_mean(emp_series)
    emp_volatility = safe_std(emp_series) / emp_avg if emp_avg > 0 else 0.0

    # F09: firm_age_months
    last = filled[months_full[-1]]
    apply_yyyymm = parse_yyyymmdd(last.get('적용일자', ''))
    end_month = months_full[-1]
    if apply_yyyymm:
        try:
            firm_age = max(0, months_between(apply_yyyymm, end_month))
        except (ValueError, AttributeError):
            firm_age = 0
    else:
        firm_age = 0

    # F10: log_emp_count
    log_emp = math.log(emp_series[-1] + 1)

    # F12~F14 (F11 firm_type 제거: 데이터에 개인사업자 0건이라 무의미)
    sido = last.get('시도코드', '').strip() or 'UNKNOWN'
    industry_code = last.get('사업장업종코드', '').strip()
    industry_codename = last.get('사업장업종코드명', '').strip()
    industry_info = map_industry(industry_code, industry_codename)

    # F15~F17: 결측 패턴 피처
    imputed_count = sum(imputed_flags.values())
    imputed_ratio = imputed_count / 12
    recent_3m_imputed = sum(imputed_flags[m] for m in months_full[-3:])
    has_missing_recent_3m = 1 if recent_3m_imputed > 0 else 0

    return {
        # 메타
        'coverage': round(coverage, 2),
        'last_month': months_full[-1],
        'n_months_observed': len(window),

        # F01 turnover (4)
        'turnover_avg_12m': round(safe_mean(turnover_series), 4),
        'turnover_avg_3m': round(safe_mean(turnover_series[-3:]), 4),
        'turnover_max_12m': round(safe_max(turnover_series), 4),
        'turnover_std_12m': round(safe_std(turnover_series), 4),

        # F02 emp_change (3)
        'emp_change_3m': round(change_rate(emp_series, 3), 4),
        'emp_change_6m': round(change_rate(emp_series, 6), 4),
        'emp_change_12m': round(change_rate(emp_series, len(emp_series) - 1), 4),

        # F03 salary (4)
        'salary_avg_12m': round(safe_mean(salary_series), 0),
        'salary_last': round(salary_series[-1], 0),
        'salary_change_6m': round(change_rate(salary_series, 6), 4),
        'salary_change_12m': round(change_rate(salary_series, len(salary_series) - 1), 4),

        # F04 replacement (3)
        'replacement_avg_12m': round(safe_mean(replace_series), 4),
        'replacement_avg_3m': round(safe_mean(replace_series[-3:]), 4),
        'replacement_min_12m': round(safe_min(replace_series), 4),

        # F05~F08, F10
        'salary_drop_consecutive': max_streak,
        'turnover_momentum': round(momentum, 4),
        'zero_emp_months': zero_emp_months,
        'emp_volatility': round(emp_volatility, 4),
        'log_emp_count': round(log_emp, 4),

        # F09
        'firm_age_months': firm_age,

        # F12~F13 categorical (F11 firm_type 제거)
        'sido_code': sido,
        'industry_category': industry_info['industry_category'],

        # F14
        'industry_death_rate_2023': industry_info['death_rate_2023'],

        # F15~F17 결측 패턴
        'imputed_months_count': imputed_count,
        'imputed_ratio': round(imputed_ratio, 3),
        'has_missing_recent_3m': has_missing_recent_3m,
    }


# ============================================================
# 양성 샘플 빌드 + Augmentation
# ============================================================
print('\n양성 샘플 빌드 (시점 augmentation 적용) 중...')
samples = []
pos_aug_count = defaultdict(int)  # 사업장별 augmented 샘플 카운트
pos_total_attempts = 0
pos_skipped = 0

for (unpaid_name, gubun), rows in pos_groups.items():
    base_t = GUBUN_DATE[gubun]
    source_id = f'POS_{unpaid_name}_{gubun}'
    for shift in AUG_SHIFTS:
        pos_total_attempts += 1
        shifted_t = add_months(base_t, shift)
        window = slice_window(rows, shifted_t)
        feats = derive_features(window, shifted_t)
        if feats is None:
            pos_skipped += 1
            continue
        sample = {
            'business_id': f'{source_id}_shift{shift:+d}',
            'source_business': source_id,
            'label': 1,
            'gubun': gubun,
            't_disclose': base_t,
            'shift': shift,
            **feats,
        }
        samples.append(sample)
        pos_aug_count[source_id] += 1

n_pos_samples = len(samples)
n_pos_unique = len(pos_aug_count)
print(f'  양성 augmented 샘플: {n_pos_samples}건 (원본 사업장 {n_pos_unique}개)')
print(f'  사업장당 평균 augmented: {n_pos_samples / n_pos_unique:.1f}개')
print(f'  전체 시도 {pos_total_attempts}회 중 skip {pos_skipped}회 (커버리지 부족)')

# ============================================================
# 음성 샘플 빌드 + 다운샘플링
# ============================================================
print('\n음성 샘플 빌드 중...')
neg_samples_all = []
neg_skipped = 0
for (biz_no, gubun), rows in neg_groups.items():
    t = GUBUN_DATE.get(gubun)
    if not t:
        continue
    window = slice_window(rows, t)
    feats = derive_features(window, t)
    if feats is None:
        neg_skipped += 1
        continue
    sample = {
        'business_id': f'NEG_{biz_no}_{gubun}',
        'source_business': f'NEG_{biz_no}',
        'label': 0,
        'gubun': gubun,
        't_disclose': t,
        'shift': 0,
        **feats,
    }
    neg_samples_all.append(sample)
print(f'  음성 후보: {len(neg_samples_all)}개 (skip {neg_skipped})')

# 다운샘플링: 양성 augmented 샘플 수와 동일하게 (1:1)
neg_target = n_pos_samples * NEG_RATIO
neg_target = min(neg_target, len(neg_samples_all))
neg_sampled = random.sample(neg_samples_all, neg_target)
samples.extend(neg_sampled)
print(f'  다운샘플링 후 음성: {len(neg_sampled)}개 (목표 1:{NEG_RATIO})')

# ============================================================
# 저장
# ============================================================
print('\n=== 학습 데이터셋 통계 ===')
pos_n = sum(1 for s in samples if s['label'] == 1)
neg_n = sum(1 for s in samples if s['label'] == 0)
print(f'  총 샘플: {len(samples)}건')
print(f'  양성: {pos_n}개 (원본 사업장 {n_pos_unique}개)')
print(f'  음성: {neg_n}개')
print(f'  비율: 1:{neg_n / pos_n:.2f}' if pos_n > 0 else '')

fieldnames = list(samples[0].keys())
with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(samples)

print(f'\n저장 완료: {OUTPUT_CSV}')
print(f'  컬럼 수: {len(fieldnames)}')
print(f'  컬럼:\n    {", ".join(fieldnames)}')
