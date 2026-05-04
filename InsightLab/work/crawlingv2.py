"""
임금 체불 사업장 명단 추출 크롤링 프로그램. ver.2
동적 페이지에서 모든 명단을 추출할 수 있게 됨.

결과물 파일: ./dataset/unpaid_emplyers.csv
"""

import re
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

URL = "https://www.moel.go.kr/info/defaulter/defaulterList.do"
OUTPUT_FILE = "unpaid_employers.csv"
COLUMNS = ['구분', '성명(상호)', '나이(대표자)', '사업장명', '업종', '주소지', '소재지', '체불액(원)']


def init_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def wait_for_table(driver):
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "table.arr_list tbody tr"))
    )


def get_table_rows(driver):
    rows = driver.find_elements(By.CSS_SELECTOR, "table.arr_list tbody tr")
    data = []
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        if not cols:
            continue
        row_data = [col.text.strip() for col in cols]
        if row_data:
            data.append(row_data[:len(COLUMNS)])
    return data


def get_total_pages(driver):
    """마지막 페이지 번호를 onclick 속성에서 추출."""
    try:
        # 마지막 페이지 버튼 (class='page_last')
        last_btn = driver.find_element(By.CSS_SELECTOR, "a.page_last")
        onclick = last_btn.get_attribute("onclick")
        m = re.search(r'fnSelectInfs\((\d+)\)', onclick)
        if m:
            return int(m.group(1))
    except Exception:
        pass

    # fallback: 모든 onclick에서 가장 큰 페이지 번호
    try:
        links = driver.find_elements(By.CSS_SELECTOR, "a[onclick*='fnSelectInfs']")
        nums = []
        for link in links:
            onclick = link.get_attribute("onclick") or ""
            m = re.search(r'fnSelectInfs\((\d+)\)', onclick)
            if m:
                nums.append(int(m.group(1)))
        if nums:
            return max(nums)
    except Exception:
        pass

    return 1


def go_to_page(driver, page_num):
    driver.execute_script(f"fnSelectInfs({page_num});")
    time.sleep(1.5)
    wait_for_table(driver)


def main():
    print("체불사업주 명단 크롤링 시작...")
    driver = init_driver()
    all_data = []

    try:
        driver.get(URL)
        time.sleep(2)
        wait_for_table(driver)

        total_pages = get_total_pages(driver)
        print(f"전체 페이지 수: {total_pages}")

        for page in range(1, total_pages + 1):
            if page > 1:
                go_to_page(driver, page)

            rows = get_table_rows(driver)
            all_data.extend(rows)
            print(f"  페이지 {page}/{total_pages} -> {len(rows)}행 (누적: {len(all_data)}행)")

    finally:
        driver.quit()

    if not all_data:
        print("수집된 데이터가 없습니다.")
        return

    df = pd.DataFrame(all_data)
    if len(df.columns) == len(COLUMNS):
        df.columns = COLUMNS
    else:
        print(f"열 수 불일치: 예상={len(COLUMNS)}, 실제={len(df.columns)} — 컬럼명 없이 저장합니다.")

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n완료! 총 {len(df)}개 업체 -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
