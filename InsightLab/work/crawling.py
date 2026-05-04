"""
임금 체불 사업장 명단 추출 크롤링 프로그램. ver.1
"""

import requests
import pandas as pd 
from bs4 import BeautifulSoup

url = "https://www.moel.go.kr/info/defaulter/defaulterList.do"

response = requests.get(url, timeout=30)
response.raise_for_status() # Exception when error
soup = BeautifulSoup(response.text, "lxml")

dataframe = pd.DataFrame(columns=['구분','성명','나이','사업장명','업종','주소지','소재지','체불액(원)'])
print('dataframe')
print(dataframe)


elements = soup.find("table", attrs={"class": "arr_list"})

data_rows = elements.find('tbody').find_all('tr')

insert_row = []

for i, row in enumerate(data_rows):
    data_columns = row.find_all('td')
    for col in data_columns:
        insert_row.append(repr(col.get_text().strip()))

    dataframe.loc[len(dataframe)] = insert_row
    insert_row = []

dataframe.to_csv('data.csv', encoding='utf-8')



# for i, e in enumerate(elements):
#     if i == 0:
#         continue

#     e_text = e.get_text().replace('\t', '')

#     if e_text == '\n':
#         continue
#     print(e_text.strip())

