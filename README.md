# AI-Based HAZOP Safety Analysis Tool

## 프로젝트 개요
본 프로젝트는 공정 위험성 분석 기법인 HAZOP(Hazard and Operability Study)을 기반으로  
위험요소를 분석하고 개선 권고사항을 도출하는 AI 기반 안전 분석 웹 애플리케이션입니다.

Python과 Streamlit을 활용하여 HAZOP 분석 절차를 웹 환경에서 구현하였으며,  
AI를 활용하여 잠재 위험에 대한 개선 권고사항(Safeguard Recommendation)을 제시하도록 설계되었습니다.

---

## 주요 기능

### 1. 단일 편차 HAZOP 분석
- 공정 변수 편차(Deviation) 기반 위험 분석
- 원인(Cause) / 결과(Consequence) / 기존 안전조치 분석

### 2. 위험도 평가
- 발생빈도(Frequency)와 발생강도(Severity)를 기반으로 위험도 산정
- 위험도 매트릭스 기반 등급 판정

### 3. AI 기반 개선 권고사항 도출
- OpenAI API 활용
- 위험 상황에 대한 추가적인 안전조치 제안

### 4. 복합 편차 분석
- 다중 편차 상황에서의 위험 시나리오 분석

---

## 사용 기술

- Python
- Streamlit
- OpenAI API
---

## 실행 방법

필요 패키지 설치
pip install -r requirements.txt


## 프로그램 실행
streamlit run hazop_app.py
---

## 기대 효과

본 시스템을 통해 기존 HAZOP 분석 과정에서의  
- 위험요소 식별
- 위험도 평가
- 개선 권고사항 도출

과정을 보다 효율적이고 체계적으로 수행할 수 있습니다.

또한 AI 기반 분석을 통해 안전관리자의 의사결정을 보조하는  
스마트 안전관리 도구로 활용될 수 있습니다.
