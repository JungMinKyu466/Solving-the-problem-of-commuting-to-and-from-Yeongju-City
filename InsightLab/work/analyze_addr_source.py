"""주소지 vs 소재지 매칭 기여도 분석"""
import sys, csv, re, os, glob as globmod
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(10**7)

CORP_REMOVALS = ['주식회사','(주)','(유)','유한회사','유한책임회사','합자회사','합명회사',
    '사회적협동조합','협동조합','영농조합법인','농업회사법인','어업회사법인',
    '재단법인','사단법인','학교법인','의료법인','사회복지법인']
ADDR_ABBR = {'서울':'서울특별시','서울시':'서울특별시','부산':'부산광역시','부산시':'부산광역시',
    '대구':'대구광역시','대구시':'대구광역시','인천':'인천광역시','인천시':'인천광역시',
    '광주':'광주광역시','광주시':'광주광역시','대전':'대전광역시','대전시':'대전광역시',
    '울산':'울산광역시','울산시':'울산광역시','세종':'세종특별자치시','세종시':'세종특별자치시',
    '세종특별자치시':'세종특별자치시','경기':'경기도','강원':'강원특별자치도','강원도':'강원특별자치도',
    '충북':'충청북도','충남':'충청남도','전북':'전북특별자치도','전라북도':'전북특별자치도',
    '전남':'전라남도','경북':'경상북도','경남':'경상남도','제주':'제주특별자치도'}

def norm_name(n):
    if not n: return ''
    s = n.strip()
    if s.startswith('개인건설업자') or s.startswith('개인사업자'): return ''
    for t in CORP_REMOVALS: s = s.replace(t,'')
    s = re.sub(r'\(.*?\)','',s)
    s = re.sub(r'[\s\-_\u00b7.,/\\\\]','',s)
    return s

def get_sido(a):
    if not a: return ''
    p = a.strip().split()
    return ADDR_ABBR.get(p[0],p[0]) if p else ''

def get_sigungu(a):
    if not a: return ''
    p = a.strip().split()
    if len(p)<2: return get_sido(a)
    return f"{ADDR_ABBR.get(p[0],p[0])} {p[1]}"

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')

# 임금체불 로드
unpaid = []
with open(os.path.join(BASE,'unpaid_employers.csv'),encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        n = norm_name(row['사업장명'])
        if not n: continue
        a1, a2 = row['주소지'].strip(), row.get('소재지','').strip()
        unpaid.append({
            'name': row['사업장명'], 'norm': n,
            'sido_addr': get_sido(a1), 'sigungu_addr': get_sigungu(a1),
            'sido_src': get_sido(a2), 'sigungu_src': get_sigungu(a2),
        })

# 국민연금 통합
pension_by_name = defaultdict(list)
seen = set()
files = sorted(globmod.glob(os.path.join(BASE,'국민연금 공공데이터포털','*.csv')))
print(f'파일 {len(files)}개 로드 중...')
for fp in reversed(files):
    try:
        with open(fp,encoding='cp949',errors='replace',newline='') as f:
            for row in csv.DictReader(f):
                name = row.get('사업장명','').strip()
                biz = row.get('사업자등록번호','').strip()
                key = f"{name}|{biz}"
                if key in seen: continue
                seen.add(key)
                n = norm_name(name)
                if not n: continue
                aj = row.get('사업장지번상세주소','').strip()
                ar = row.get('사업장도로명상세주소','').strip()
                sidos = frozenset(filter(None,[get_sido(aj),get_sido(ar)]))
                sigungus = frozenset(filter(None,[get_sigungu(aj),get_sigungu(ar)]))
                pension_by_name[n].append((sidos,sigungus))
    except Exception as e:
        print(f'  skip {os.path.basename(fp)}: {e}')

print(f'고유 사업장: {sum(len(v) for v in pension_by_name.values())}, 고유 이름: {len(pension_by_name)}')

# 분석
only_addr = 0
only_src = 0
both = 0
neither = 0
no_name = 0

for u in unpaid:
    cands = list(pension_by_name.get(u['norm'], []))
    # 포함관계 후보도 추가
    if len(u['norm']) >= 4:
        for nm, entries in pension_by_name.items():
            if len(nm) >= 4 and nm != u['norm'] and (u['norm'] in nm or nm in u['norm']):
                cands.extend(entries)

    if not cands:
        no_name += 1
        continue

    hit_addr = False
    hit_src = False
    for sidos, sigungus in cands:
        if u['sigungu_addr'] and u['sigungu_addr'] in sigungus: hit_addr = True
        elif u['sido_addr'] and u['sido_addr'] in sidos: hit_addr = True
        if u['sigungu_src'] and u['sigungu_src'] in sigungus: hit_src = True
        elif u['sido_src'] and u['sido_src'] in sidos: hit_src = True
        if hit_addr and hit_src: break

    if hit_addr and hit_src: both += 1
    elif hit_addr: only_addr += 1
    elif hit_src: only_src += 1
    else: neither += 1

print(f'\n=== 주소지 vs 소재지 매칭 기여 분석 ===')
print(f'유효 임금체불 사업장: {len(unpaid)}건 (개인 제외)')
print(f'')
print(f'  주소지로만 매칭:              {only_addr}건')
print(f'  소재지로만 매칭:              {only_src}건')
print(f'  둘 다 매칭 (양쪽 모두 일치):  {both}건')
print(f'  ------------------------------')
print(f'  이름은 있으나 주소 둘다 불일치: {neither}건')
print(f'  이름 자체가 국민연금에 없음:    {no_name}건')
print(f'')
print(f'  >> 주소지가 기여한 총 매칭: {only_addr + both}건')
print(f'  >> 소재지가 기여한 총 매칭: {only_src + both}건')
print(f'  >> 총 매칭 성공:            {only_addr + only_src + both}건')
