import sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib.font_manager as fm
import os

# 한글 폰트 후보들
candidates = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'AppleGothic',
              'NanumSquare', 'Gulim', 'Dotum', 'Batang']

print('=== 시스템 한글 폰트 검색 ===')
all_fonts = {f.name for f in fm.fontManager.ttflist}
for c in candidates:
    found = c in all_fonts
    print(f'  {c}: {"✓ 사용 가능" if found else "✗ 없음"}')

print('\n=== Malgun Gothic 경로 확인 ===')
malgun_paths = [f.fname for f in fm.fontManager.ttflist if 'malgun' in f.fname.lower()]
for p in malgun_paths[:5]:
    print(f'  {p}')

print('\n=== matplotlib font cache 위치 ===')
print(f'  {fm.get_cachedir()}')
