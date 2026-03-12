import streamlit as st
import pandas as pd
import faiss
import pickle
import numpy as np
from openai import OpenAI
from pathlib import Path

st.set_page_config(page_title="HAZOP AI Program", layout="wide")

def is_openai_available():
    try:
        return "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"].startswith("sk-")
    except Exception:
        return False
    
if is_openai_available():
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    client = None

# ✅ hazop_db 정의를 먼저 해야 오류 방지 가능
hazop_db = {
    "Node1": {
        "More Flow": {
            "Cause": "충전밸브 과개방, 절차 미준수",
            "Consequence": "탱크 과충전 → 넘침, 누출 및 폭발",
            "Existing": "유량계, SOP 기본 준수",
            "Recommended": "Fail-Safe 밸브 추가 설치"
        },
        "Less Flow": {
            "Cause": "충전라인 압력 부족",
            "Consequence": "충전 지연, 생산 차질",
            "Existing": "유량계 점검",
            "Recommended": "배관 및 밸브 점검 주기 단축"
        },
        "No/None Flow": {
            "Cause": "충전밸브 완전 폐쇄, 전원 차단",
            "Consequence": "충전 불능 → 공정 중단",
            "Existing": "비상중단 매뉴얼",
            "Recommended": "자동 밸브 차단 시스템 설치"
        },
        "Reverse Flow": {
            "Cause": "체크밸브 불량",
            "Consequence": "저장탱크로 역류",
            "Existing": "체크밸브 설치",
            "Recommended": "체크밸브 이중화 및 주기점검"
        },
        "More Pressure": {
            "Cause": "Relief Valve 고장, 과압",
            "Consequence": "탱크 과압 → 파손 및 폭발 가능성",
            "Existing": "Relief Valve, 고압 알람",
            "Recommended": "Relief Valve 이중 설치 및 정기점검"
        },
        "Less Pressure": {
            "Cause": "배관 연결부 손상, 펌프 출력 부족",
            "Consequence": "압력 손실 및 에틸렌 공급 불능",
            "Existing": "압력계",
            "Recommended": "압력 손실 자동 탐지기 설치"
        },
        "No/None Pressure": {
            "Cause": "Relief Valve 손상, 탱크 대형 누출",
            "Consequence": "공정 중단 및 공급 차질",
            "Existing": "정기점검 및 유지보수",
            "Recommended": "누설 차단 장치 강화"
        },
        "More Temperature": {
            "Cause": "냉각장치 고장",
            "Consequence": "탱크 온도 상승 → 압력 증가",
            "Existing": "온도계, 냉각시스템",
            "Recommended": "온도 알람 및 냉각장치 이중화"
        },
        "Less Temperature": {
            "Cause": "냉각장치 과도 가동",
            "Consequence": "내용물 결빙, 배관 동결",
            "Existing": "온도 모니터링",
            "Recommended": "냉각 제어 자동화"
        },
        "No/None Temperature": {
            "Cause": "온도센서 고장",
            "Consequence": "온도 변화 감지 불가 → 대응 지연",
            "Existing": "온도계 기본 점검",
            "Recommended": "온도센서 이중화"
        },
        "More Level": {
            "Cause": "충전밸브 과개방, Level Gauge 오작동",
            "Consequence": "탱크 과충전 → 넘침, 누출 및 폭발 가능성",
            "Existing": "Level Gauge",
            "Recommended": "High Level Alarm 추가"
        },
        "Less Level": {
            "Cause": "충전 부족, 배관 누설",
            "Consequence": "수위 부족 → 펌프 공회전 위험",
            "Existing": "Level Gauge, Low Level Alarm",
            "Recommended": "누설감지 센서 추가"
        },
        "No/None Level": {
            "Cause": "Level Gauge 고장",
            "Consequence": "레벨 측정 불능 → 비상 대응 지연",
            "Existing": "비상 점검 체계",
            "Recommended": "이중 측정 시스템 도입"
        }
    },

    "Node2": {
        "More Flow": {
            "Cause": "펌프 과송출, 운전 절차 미준수",
            "Consequence": "압력 급상승 → 배관 손상 가능성",
            "Existing": "Relief Valve",
            "Recommended": "운전 SOP 준수 교육 및 유량 제한 설정"
        },
        "Less Flow": {
            "Cause": "펌프 출력 감소, 라인 일부 막힘",
            "Consequence": "유량 부족 → 공급 불안정",
            "Existing": "예비 펌프",
            "Recommended": "필터 및 배관 차압 모니터링"
        },
        "No/None Flow": {
            "Cause": "펌프 전원 차단, 밸브 폐쇄",
            "Consequence": "공급 중단 → 공정 차질",
            "Existing": "수동 Bypass",
            "Recommended": "비상전원(UPS) 및 인터록 점검"
        },
        "More Pressure": {
            "Cause": "과도한 유량, Relief Valve 설정 오류",
            "Consequence": "배관 손상 → 누출 및 Jet Fire 가능성",
            "Existing": "Relief Valve, 압력계",
            "Recommended": "과압 차단 시스템 설치"
        },
        "Less Pressure": {
            "Cause": "펌프 출력 부족, 배관 누설",
            "Consequence": "공급 압력 저하 → 설비 운전 불안정",
            "Existing": "압력 센서",
            "Recommended": "배관 기밀 점검 및 펌프 성능 확인"
        },
        "No/None Pressure": {
            "Cause": "대형 누출, 압력계 고장, 펌프 정지",
            "Consequence": "공급 압력 상실 → 공정 정지",
            "Existing": "정기점검",
            "Recommended": "압력 감시 인터록 및 비상 차단 강화"
        },
        "More Temperature": {
            "Cause": "펌프 베어링 과열, 마찰 증가",
            "Consequence": "Seal 손상 → 누출 위험 증가",
            "Existing": "온도센서",
            "Recommended": "윤활유 주기 점검 및 베어링 상태 모니터링"
        },
        "Less Temperature": {
            "Cause": "과도한 냉각, 저온 유체 영향",
            "Consequence": "배관 취성 증가, 결빙 가능성",
            "Existing": "온도 모니터링",
            "Recommended": "저온 운전기준 설정 및 보온 점검"
        },
        "No/None Temperature": {
            "Cause": "온도센서 고장",
            "Consequence": "이상 온도 감지 실패 → 대응 지연",
            "Existing": "기본 계기 점검",
            "Recommended": "온도센서 이중화 및 교정 관리"
        }
    }
}

# ✅ hazop_db 키 한글화: 영어 → 한글로 변환
for node in hazop_db:
    for dev in hazop_db[node]:
        entry = hazop_db[node][dev]
        hazop_db[node][dev] = {
            "원인": entry.get("Cause", ""),
            "결과": entry.get("Consequence", ""),
            "현재 안전조치": entry.get("Existing", ""),
            "개선 조치": entry.get("Recommended", "")
        }

# ✅ FAISS DB 불러오기
law_index = faiss.read_index("law_faiss.index")
with open("law_chunks.pkl", "rb") as f:
    law_chunks = pickle.load(f)

guide_index = faiss.read_index("index.faiss")
with open("index.pkl", "rb") as f:
    guide_chunks = pickle.load(f)

handbook_index = faiss.read_index("handbook_index.faiss")
with open("handbook_chunks.pkl", "rb") as f:
    handbook_chunks = pickle.load(f)

# ✅ 검색 함수 (KOSHA 출처만 우선적으로 필터링)
def search_db(index, chunks, query, k=5):
    if client is None:
        return ["API 키가 설정되지 않아 DB 검색이 비활성화되었습니다."]

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_vector = np.array(response.data[0].embedding).astype("float32").reshape(1, -1)
        distances, indices = index.search(query_vector, k * 2)

        results = []
        for i in indices[0]:
            if i < len(chunks):
                chunk = chunks[i]

                if isinstance(chunk, str):
                    content = chunk
                    source = "알 수 없음"
                elif isinstance(chunk, dict):
                    content = chunk.get("content", "")
                    source = chunk.get("source", "")
                else:
                    continue

                if "KOSHA" in source.upper():
                    entry = f"{content} (출처: {source})"
                    results.append(entry)

        return results[:2]

    except Exception as e:
        return [f"DB 검색 중 오류 발생: {e}"]
    
# ✅ 사이드바
st.sidebar.header("분석 설정")
process_name = st.sidebar.text_input("대상 공정", value="에틸렌 저장탱크 공정")
analysis_method = st.sidebar.text_input("분석 기법", value="HAZOP Lite")
selected_node = st.sidebar.selectbox("단일 편차 분석 Node 선택", list(hazop_db.keys()), key="sidebar_node_select")
# ✅ 세션 초기화
if "data" not in st.session_state:
    st.session_state["data"] = []

# ✅ 페이지 제목 표시
st.markdown("""
<style>
.header {
    position: sticky;
    top: 0;
    background-color: #0B3D91;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.header h1 {
    color: white;
    font-size: 32px;
    margin: 0;
}

.header p {
    color: #E6E6E6;
    margin: 5px 0 0 0;
}
</style>

<div class="header">
<h1>AI-Based HAZOP Safety Analysis Tool</h1>
<p>Process Hazard Analysis with AI-based Safeguard Recommendation</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

section.main > div {
    padding-top: 1rem;
}

.card {
    background-color: #f8f9fc;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e1e4eb;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.streamlit-expanderHeader {
    font-size: 18px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

if client is None:
    st.warning("현재 OpenAI API 키가 설정되지 않아 AI 추천 기능은 데모 모드로 표시됩니다.")
else:
    st.info("OpenAI API 키가 설정되었습니다. 실제 유효성은 AI 실행 시 검증됩니다.")

st.caption("저장탱크 공정을 대상으로 단일·복합 편차를 분석하고, 위험도 평가 및 AI 기반 개선권고사항 도출 구조를 구현한 HAZOP 프로그램")

# ✅ ------------------- 메인 2-Column UI -------------------
st.markdown("## 1) AI 단일 편차 HAZOP 분석")

# ✅ 2-Column UI 시작
col1, col2 = st.columns([1, 1], gap="large")

# ✅ [왼쪽] – Cause/Consequence + 현재조치 + 현재 위험도 평가
with col1:
    st.subheader("현재 위험도 평가")

    # ✅ 편차 선택
    selected_deviation = st.selectbox(
        "편차 선택",
        list(hazop_db[selected_node].keys()),
        key="deviation_select_left"
    )

    # ✅ 현재 정보 표시
    st.write(f"원인: {hazop_db[selected_node][selected_deviation]['원인']}")
    st.write(f"결과: {hazop_db[selected_node][selected_deviation]['결과']}")
    st.write(f"현재 안전조치: {hazop_db[selected_node][selected_deviation]['현재 안전조치']}")

    # ✅ 발생빈도 / 발생강도
    freq = st.selectbox("발생빈도 [1-5]", [1, 2, 3, 4, 5], key="freq_single")
    sev = st.selectbox("발생강도 [1-4]", [1, 2, 3, 4], key="sev_single")

    # ✅ 위험도 계산
    risk_score = freq * sev

    # ✅ 위험 등급 판정
    if risk_score <= 3:
        risk_level = "매우 낮음 (허용 가능)"
        color = "green"
    elif risk_score <= 6:
        risk_level = "낮음 (허용 가능)"
        color = "blue"
    elif risk_score == 8:
        risk_level = "보통 (허용 불가능)"
        color = "orange"
    elif 9 <= risk_score <= 12:
        risk_level = "약간 높음 (허용 불가능)"
        color = "darkorange"
    elif risk_score == 15:
        risk_level = "높음 (허용 불가능)"
        color = "red"
    else:
        risk_level = "매우 높음 (허용 불가능)"
        color = "darkred"

    # ✅ 현재 위험도 표시
    st.markdown(f"현재 위험도: {risk_score} (빈도 {freq} × 강도 {sev})")
    st.markdown(
        f"<h3 style='color:{color}; font-weight:700;'>현재 위험도: {risk_score}점 → {risk_level}</h3>",
        unsafe_allow_html=True
    )

    # ✅ 빈 줄
    st.markdown(" ")

    # ✅ 빈도/강도 기준 + 위험도 결정 기준
    freq_col, matrix_col = st.columns([3, 5])

    with freq_col:
        st.markdown("#### 빈도·강도 설정 기준")
        st.markdown("""
- **빈도 (1~5)**  
1 = 극히 드뭄  
2 = 드뭄  
3 = 보통  
4 = 자주 발생  
5 = 매우 자주 발생

- **강도 (1~4)**  
1 = 경미  
2 = 보통  
3 = 심각  
4 = 치명적
""")

    with matrix_col:
        st.markdown("#### 위험도 결정 기준")
        st.markdown("""
| 점수 범위 | 위험도 등급 | 허용 여부 | 조치 권고사항 |
|-----------|-------------|-----------|----------------|
| 16~20 | 매우 높음 | 허용 불가능 | 즉시 개선 / 작업 중단 |
| 15 | 높음 | 허용 불가능 | 신속한 개선 조치 |
| 9~12 | 약간 높음 | 허용 불가능 | 가능한 빨리 개선 |
| 8 | 보통 | 허용 불가능 | 계획적인 개선 필요 |
| 4~6 | 낮음 | 허용 가능 | 필요시 개선 |
| 1~3 | 매우 낮음 | 허용 가능 | 개선 불요 또는 필요시 개선 |
""")
        
# ✅ [오른쪽] – AI 개선 Safeguard & 개선 후 위험도

# ✅ 사고사례가 존재하는 deviation 목록 및 사고사례 내용
accident_cases = {
    "More Pressure": """[관련 사고사례]
탱크로리에서 액체를 하역하는 작업 도중 내부 압력이 비정상적으로 상승했음에도 불구하고, 설치된 안전밸브가 작동하지 않아 탱크가 파열되고 대규모 폭발이 발생하였습니다. 사고 조사 결과, 안전밸브의 미작동은 주기적인 점검과 유지보수가 제대로 이루어지지 않은 것이 원인이었습니다. (출처: KOSHA 중소규모사업장_화재폭발사고_예방_핸드북)"""
}

# ✅ AI 개선 Safeguard 생성 함수
def generate_ai_safeguard(deviation, guide_results, law_results, accident_results_str=None):
    if client is None:
        return """
### AI 기능 안내
현재 API 키가 설정되지 않아 AI 개선권고사항 생성 기능은 비활성화되어 있습니다.

대신 본 앱에서는 다음 기능을 확인할 수 있습니다.
- 단일 편차 HAZOP 분석
- 위험도 평가
- 복합 편차 조합 검토
- 사고사례 및 참고 DB 연계 구조
"""

    prompt = f"""
당신은 산업안전 컨설턴트입니다.

아래 Deviation에 대해 다음 항목을 각각 작성하세요:

1. KOSHA 가이드 기준의 기본 개선 권고사항 2가지
2. 산업안전보건법 등 법령에 근거한 필수 안전조치 및 관련 조문 2가지
3. 관련 사고사례

Deviation: {deviation}

참고 Guide:
{guide_results}

참고 Law:
{law_results}

참고 사고사례:
{accident_results_str if accident_results_str else "없음"}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 산업안전 전문가이자 위험성평가 컨설턴트야."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"AI 개선권고사항 생성 중 오류가 발생했습니다: {e}"
        
# ✅ Streamlit UI
if "gpt_output_single" not in st.session_state:
    st.session_state["gpt_output_single"] = ""

with col2:
    st.subheader("AI 활용 개선권고사항")

    manual_safeguard = st.text_area(
        "관리자 개선권고 입력",
        value="",
        height=120,
        placeholder="관리자가 최종 판단하여 개선권고사항을 직접 입력할 수 있습니다."
    )

    show_accident_case = False
    if selected_deviation in accident_cases:
        show_accident_case = st.checkbox("사고사례도 함께 보기", value=False)

    if st.button("AI 추천 개선권고사항"):
        with st.spinner("KOSHA & 법령 DB 검색 중..."):
            law_results = search_db(law_index, law_chunks, selected_deviation)
            guide_results = search_db(guide_index, guide_chunks, selected_deviation)

            kosha_results = [r for r in guide_results if "KOSHA" in r.upper()]
            nfpa_results = [r for r in guide_results if "NFPA" in r.upper()]
            guide_final = kosha_results if kosha_results else nfpa_results

            law_results_str = "\n".join(law_results)
            guide_results_str = "\n".join(guide_final)

            accident_results_str = accident_cases[selected_deviation] if show_accident_case else None

        with st.spinner("AI가 개선권고사항 생성 중..."):
            st.session_state["gpt_output_single"] = generate_ai_safeguard(
                selected_deviation,
                guide_results_str,
                law_results_str,
                accident_results_str
            )

    with st.expander("AI 추천 개선권고사항 분석 결과 (클릭하여 열기/닫기)", expanded=True):
        result_text = st.session_state.get("gpt_output_single", "")
        if result_text:
            st.markdown(result_text, unsafe_allow_html=True)
        else:
            st.info("아직 AI 분석 결과가 없습니다.")

    if manual_safeguard.strip():
        st.markdown("#### 관리자 입력 개선권고사항")
        st.write(manual_safeguard)

    st.markdown("### 개선 후 위험도 평가")

    freq_after = st.selectbox("개선 후 발생빈도 [1-5]", [1, 2, 3, 4, 5], key="freq_after_col2")
    sev_after = st.selectbox("개선 후 발생강도 [1-4]", [1, 2, 3, 4], key="sev_after_col2")

    risk_score_after = freq_after * sev_after
    st.markdown(f"개선 후 위험도: {risk_score_after} (빈도 {freq_after} × 강도 {sev_after})")

    if risk_score_after <= 3:
        risk_level_after = "매우 낮음 (허용 가능)"
        color_after = "green"
    elif risk_score_after <= 6:
        risk_level_after = "낮음 (허용 가능)"
        color_after = "blue"
    elif risk_score_after == 8:
        risk_level_after = "보통 (허용 불가능)"
        color_after = "orange"
    elif 9 <= risk_score_after <= 12:
        risk_level_after = "약간 높음 (허용 불가능)"
        color_after = "darkorange"
    elif risk_score_after == 15:
        risk_level_after = "높음 (허용 불가능)"
        color_after = "red"
    else:
        risk_level_after = "매우 높음 (허용 불가능)"
        color_after = "darkred"

    st.markdown(
        f"<h3 style='color:{color_after};'>개선 후 위험도: {risk_score_after}점 → {risk_level_after}</h3>",
        unsafe_allow_html=True
    )

# ✅ Node1 – 같은 변수 내 모순 + 진짜 불가능한 조합 포함
invalid_combinations_node1 = [
    # Flow
    ("More Flow", "Less Flow"),
    ("More Flow", "No/None Flow"),
    ("More Flow", "Reverse Flow"),
    ("Less Flow", "No/None Flow"),
    ("Less Flow", "Reverse Flow"),
    ("No/None Flow", "Reverse Flow"),

    # Pressure
    ("More Pressure", "Less Pressure"),
    ("More Pressure", "No/None Pressure"),
    ("Less Pressure", "No/None Pressure"),

    # Temperature
    ("More Temperature", "Less Temperature"),
    ("More Temperature", "No/None Temperature"),
    ("Less Temperature", "No/None Temperature"),

    # Level
    ("More Level", "Less Level"),
    ("More Level", "No/None Level"),
    ("Less Level", "No/None Level"),

    # 물리적으로 부자연스러운 조합
    ("No/None Flow", "More Level")
]

invalid_combinations_node2 = [
    # Flow
    ("More Flow", "Less Flow"),
    ("More Flow", "No/None Flow"),
    ("Less Flow", "No/None Flow"),

    # Pressure
    ("More Pressure", "Less Pressure"),
    ("More Pressure", "No/None Pressure"),
    ("Less Pressure", "No/None Pressure"),

    # Temperature
    ("More Temperature", "Less Temperature"),
    ("More Temperature", "No/None Temperature"),
    ("Less Temperature", "No/None Temperature"),

    # 공정상 부자연스러운 조합
    ("No/None Flow", "More Pressure")
]
# ✅ 복합 편차 유효성 검사 함수
def is_invalid_combination(devs, node):
    """
    주어진 노드(node)에서 선택된 편차 devs 중,
    동시에 존재할 수 없는 조합이 있는지 확인하는 함수
    """
    # 노드에 따라 해당 리스트 선택
    invalid_list = invalid_combinations_node1 if node == "Node1" else invalid_combinations_node2

    # 리스트에서 순회하며 devs 내 두 항목이 동시에 존재하는 경우 확인
    for pair in invalid_list:
        if pair[0] in devs and pair[1] in devs:
            return True
    return False

# ==========================================
# 6. 복합 Deviation 분석 (핸드북 우선 적용 버전)
# ==========================================
st.markdown("---")
st.markdown("## 2) AI 복합 편차 HAZOP 분석")

# ✅ 사이드바에서 노드 선택
node_ai = st.sidebar.selectbox("AI 복합 편차 분석 Node 선택", ["Node1", "Node2"], key="node_sidebar_ai")
deviation_list = list(hazop_db[node_ai].keys())

# ✅ 복합 Deviation 선택 (2~3개)
selected_devs = st.multiselect("AI 분석 대상 편차 선택 (2~3개)", deviation_list, max_selections=3)

# ✅ 실행 버튼
run_multi_ai = st.button("복합 편차 AI 분석 실행")

# ✅ 핸드북 사고사례 전용 검색 함수
def search_handbook_accidents(index, chunks, query, k=5):
    if client is None:
        return ["API 키가 설정되지 않아 사고사례 검색이 비활성화되었습니다."]

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_vector = np.array(response.data[0].embedding).astype("float32").reshape(1, -1)
        distances, indices = index.search(query_vector, k * 2)

        results = []
        for i in indices[0]:
            if i < len(chunks):
                content = chunks[i]
                results.append(f"{content} (KOSHA 중소규모사업장_화재폭발사고_예방_핸드북)")
        return results[:2]

    except Exception as e:
        return [f"사고사례 검색 중 오류 발생: {e}"]
    
# ✅ 실행 시 분석 시작
if run_multi_ai:
    if len(selected_devs) < 2:
        st.warning("⚠️ 최소 2개 이상의 편차를 선택하세요.")

    elif is_invalid_combination(selected_devs, node_ai):
        st.error("❌ 이는 물리적으로 불가능한 편차 조합입니다. 다시 선택해주세요.")

    else:
        with st.spinner("AI 기반 복합 편차 HAZOP 분석 중..."):
            query_text = ", ".join(selected_devs)

            # ✅ 1순위: 핸드북 사고사례 검색
            handbook_results = search_handbook_accidents(handbook_index, handbook_chunks, query_text)

            if handbook_results and not handbook_results[0].startswith("API 키가 설정되지 않아"):
                reference_data = "\n".join(handbook_results)
                source_used = "핸드북 사고사례"
            else:
                guide_results = search_db(guide_index, guide_chunks, query_text)
                law_results = search_db(law_index, law_chunks, query_text)
                reference_data = "가이드:\n" + "\n".join(guide_results) + "\n\n법령:\n" + "\n".join(law_results)
                source_used = "KOSHA Guide + 법령"

            prompt = f"""
너는 산업안전 HAZOP 전문가야.

Node: {node_ai}
선택된 Deviation: {", ".join(selected_devs)}

✅ 아래 형식으로 하나의 통합 분석 결과만 작성해줘 (한국어):

1. 원인
- {source_used} 자료 기반 사고사례 또는 기술적 설명 반영

2. 결과
- 실제 피해(인명·설비 등) 중심의 구체적 결과 작성

3. 개선권고조치
- 아래 참고자료 기반으로 기술
- 가능한 경우 KOSHA Guide 코드 표기 포함

✅ 참고자료:
{reference_data}
"""

            st.markdown("### AI 복합 편차 HAZOP 분석 결과")

            try:
                if client is None:
                    st.error("현재 API 키를 읽지 못해 복합 편차 AI 분석을 실행할 수 없습니다.")
                else:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "너는 산업안전 전문가이자 HAZOP 컨설턴트야."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"복합 편차 AI 분석 중 오류가 발생했습니다: {e}")

else:
    st.info("AI 복합 편차 HAZOP 분석 실행 버튼을 눌러주세요.")