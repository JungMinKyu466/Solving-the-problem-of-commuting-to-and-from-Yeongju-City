# 📹 돈워리💸😟: 체불 발생 이전 고용 변동 데이터를 활용한 임금체불 위험 사업장 사전 탐지 모델

**InsightLab**은 근로자의 임금 체불 문제를 시계열 데이터 기반으로 분석하고, 노동자의 권익보호와 예산의 효율적 운용을 제안하는 프로젝트입니다.

<img width="1500" height="600" alt="logoimage" src="https://github.com/user-attachments/assets/bf3f9ac3-ed2c-4132-b7cb-172b088a973c" />


## 👀 프로젝트 선정 이유
- 매년마다 계속 증가하는 국내 총 체불액
- 근로자가 신고한 이후에 조사가 시작되는 현행 대응 방식의 한계점
- 피해자 피해 회복이 사실상 불가능

---

## 🎯 프로젝트 목표

저희 InsightLab 팀은 다음과 같은 목표를 바탕으로 프로젝트를 진행하였습니다.

- 임금체불 발생 이전의 고용 변화 신호를 기반으로 위험 사업장 조기 탐지
- 머신러닝 기반 모델을 활용한 체불 발생 확률 예측 시스템 구축
- 정책 및 자원의 선제적 개입을 통한 노동자 권익 보호 및 예산 효율화

---

## 📚 데이터 및 분석 방법론

저희는 총 **3종의 데이터를 결합한 시계열 기반 데이터 파이프라인을 구축**를 수집 및 전처리하여 분석에 활용하였습니다.

- 국민연금 신고 데이터
- 고용노동부 체불 사업 명단
- 외부 통계 데이터

이후 시계열 데이터를 기반으로 14개 핵심 파생 변수를 생성해 다음과 같은 기법을 적용하였습니다.

- 로지스틱 회귀 Logistic Regression
- 랜덤 포레스트 Random Forest
- SHAP 분석
- Python, Pandas, Scikit-learn

---

## 📊 주요 분석 결과

- SHAP_피처중요도_Top15
  <img width="800" height="600" alt="fig04_SHAP_피처중요도_Top15" src="https://github.com/user-attachments/assets/42416db7-cf5b-4954-9142-f4aafbce53d6" />

- 산업카테고리별_양성률
  <img width="800" height="500" alt="fig10_산업카테고리별_양성률" src="https://github.com/user-attachments/assets/dcaabacd-52b6-4653-889d-114048fe0143" />

- Precision10
  <img width="1934" height="1035" alt="fig01_헤드라인_Precision10" src="https://github.com/user-attachments/assets/08879cd8-4473-4ab4-80d6-c31a632ed5e9" />


- PrecisionK_종합
  <img width="1000" height="500" alt="fig02_PrecisionK_종합" src="https://github.com/user-attachments/assets/f2fb9985-d4bb-40c2-84c9-c24ccef512d0" />


- 공개차수별_식별정확도
  <img width="800" height="500" alt="fig07_공개차수별_식별정확도" src="https://github.com/user-attachments/assets/4c39df1f-81d5-4a14-8e6f-4c05db19cf66" />

  
---

## 🛠️ 정책 제안 / 서비스 제안

저희 InsightLab 팀은 단순한 사후 적발 중심의 기존 접근 방식이 아닌, **사전 예방 중심의 데이터 기반 대응 체계** 중심의 개선 방안을 제안합니다.

1. **🚨 체불 위험 사업장 조기 경보 시스템**  
- 일정 임계치 이상 위험도 발생 시 → 노동부 자동 알림

2. **📊 사업장 위험도 대시보드 구축**  
- 사업장별 위험 점수 시각화

3. **타겟 기반 근로감독 자원 배분**  
- 고위험군 사업장 집중 분석

---

## 🚩 결론 및 향후 계획

- 고용 변동 데이터만으로도 임금체불 위험을 사전에 탐지할 수 있음을 확인
- 단순 통계가 아닌 머신러닝 기반 예측 모델의 실효성 검증
  
- 다양한 데이터 결합
- 딥러닝 기반 시계열 모델 적용
- 실제 정책 시스템과 연계 가능한 프로토타입 개발

---

## 🚀 실행 방법

본 프로젝트는 `Anaconda` 기반의 Python 환경에서 실행됩니다.

1️⃣ 가상환경 생성 및 활성화

```
bash
conda create -n money_worry python=3.14 -y
conda activate money_worry
```
2️⃣ 필수 라이브러리 설치

```
pip install scikit-learn lightgbm xgboost catboost shap matplotlib seaborn pandas numpy
```

3️⃣ 프로젝트 실행
```
python wage_detect.py
```

## 🧑‍💻 팀 소개

- **팀명:** InsightLAb  
- **소속:** 인천대학교 데이터사이언스 동아리 SeD (Shall we Data?)  
- **참여:** 본 프로젝트는 제5회 고용노동 공공데잍/AI 활용 공모전의 결과물입니다.




