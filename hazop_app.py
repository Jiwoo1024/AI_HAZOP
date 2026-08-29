import os
import streamlit as st
import pandas as pd
import faiss
import pickle
import re
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

# ✅ 검색 함수
# source_filter를 넘기면 그 문자열이 출처(source)에 포함된 청크만 사용합니다.
# 법령 DB(law_chunks)는 출처가 "산업안전보건법(법률)" 등으로 저장돼 있어서
# "KOSHA" 필터를 걸면 전부 걸러져 법령 검색 결과가 항상 빈 리스트가 되는 버그가
# 있었습니다 — 법령 검색에는 source_filter를 주지 않도록 호출부를 분리했습니다.
def search_db(index, chunks, query, k=5, source_filter=None):
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
                else:                   continue

                if source_filter and source_filter.upper() not in source.upper():
                    continue

                entry = f"{content} (출처: {source})"
                results.append(entry)

        return results[:8]

    except Exception as e:
        return [f"DB 검색 중 오류 발생: {e}"]


# ✅ 벡터 검색은 "유사한 단어"를 찾을 뿐, "이 편차에 실제로 적용되는가"는 판단하지 않는다.
# 그래서 search_db()가 찾아온 후보를 AI에게 다시 보여주고, 이 편차의 원인/결과에 실질적으로
# 관련 있는 것만 추리게 한다. 후보가 하나도 관련 없으면 억지로 채우지 않고 빈 리스트를 반환한다.
def rerank_relevant(query, raw_results, limit=3):
    if not raw_results or client is None:
        return raw_results[:limit]
    # search_db()가 에러/비활성 메시지를 반환한 경우 그대로 통과
    if raw_results[0].startswith("API 키") or raw_results[0].startswith("DB 검색 중 오류"):
        return raw_results

    numbered = "\n".join(f"[{i+1}] {r[:400]}" for i, r in enumerate(raw_results))
    prompt = f"""아래는 '{query}' 라는 공정 편차(Deviation)와 관련해 벡터 검색으로 찾아온 후보 자료입니다.
각 후보가 이 편차의 원인 또는 결과와 실질적으로 관련되어, 개선권고사항을 뒷받침하는 근거로
쓸 수 있는지 판단하십시오. 완전히 동일한 설비명이 아니어도 같은 기능(예: 안전밸브/릴리프밸브/
압력방출장치는 모두 과압 방지 설비로 동일 취급, Level Gauge와 액위계·레벨센서도 동일 취급)을
하는 설비·조항이면 관련 있다고 판단하십시오. 다만 전혀 다른 주제(무관한 화학물질, 무관한 공정
단계 등)까지 억지로 끼워맞추지는 마십시오.

관련 있는 후보의 번호만 쉼표로 구분해 답하십시오. 관련 있는 게 하나도 없으면 "없음"이라고만 답하십시오.
번호나 "없음" 외의 다른 설명은 절대 쓰지 마십시오.

{numbered}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 산업안전 자료의 관련성을 엄격하게 판단하는 검토자야."},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content.strip()
        picked_idx = [int(t) - 1 for t in re.findall(r"\d+", answer)]
        picked = [raw_results[i] for i in picked_idx if 0 <= i < len(raw_results)]
        return picked[:limit]
    except Exception:
        # 재랭킹 자체가 실패하면(API 오류 등) 기존 동작으로 폴백 — 검색은 됐으니 결과 없이 끝내지 않음
        return raw_results[:limit]


def relevance_note(raw_results, reranked_results, label):
    """법령/가이드 참고자료가 비어있는 이유를 구분해서 AI 프롬프트에 넣어줄 안내문."""
    if reranked_results:
        return ""
    if not raw_results or (raw_results[0].startswith("API 키") or raw_results[0].startswith("DB 검색 중 오류")):
        return f"[{label} 안내] DB에서 이 편차와 유사한 후보 자체를 찾지 못했습니다."
    return f"[{label} 안내] DB에서 후보 {len(raw_results)}건을 찾았으나, 검토 결과 이 편차의 원인/결과와 실질적으로 관련된 내용이 없어 모두 제외했습니다."

# ✅ 사이드바
st.sidebar.header("분석 설정")
process_name = st.sidebar.text_input("대상 공정", value="에틸렌 저장탱크 공정")
analysis_method = st.sidebar.text_input("분석 기법", value="HAZOP Lite")
selected_node = st.sidebar.selectbox("단일 편차 분석 Node 선택", list(hazop_db.keys()), key="sidebar_node_select")

# ✅ P&ID / Node 참고자료 — Node1·Node2가 실제로 어떤 설비를 가리키는지
# 도면으로 바로 확인할 수 있도록 사이드바에 항상 노출
with st.sidebar.expander("P&ID / Node 설명 보기", expanded=False):
    st.image(
        os.path.join(os.path.dirname(__file__), "images", "PID_ethylene_node1_2.png"),
        caption="ETHYLENE SUPPLY SYSTEM P&ID (NODE 1 & 2)",
        use_container_width=True,
    )
    st.markdown("""
**NODE 1 (ET-01)** — 에틸렌 저장탱크 본체 및 충전 계통
질소 블랭킷/PSV 라인, 레벨·온도·압력 계측(LT-101, TT-101, PT-101A 등), 탱크 하부 배출 라인을 포함. 이 앱의 **Node1**과 매칭됩니다.

**NODE 2 (ET-02)** — 탱크 상부 배기/이송 계통
증기라인 계측(PT-201A/B, FI-201), 비상차단밸브 ESDV-201, 후단 펌프 2대를 포함. 이 앱의 **Node2**와 매칭됩니다.
""")
# ✅ 세션 초기화
if "data" not in st.session_state:
    st.session_state["data"] = []

# ✅ 페이지 제목 표시 — 중립 다크 테마 (특정 회사 CI에 치우치지 않는 톤)
st.markdown("""
<style>
:root {
    --bg-page: #14161A;
    --bg-nav: #0B0C0F;
    --bg-card: #1E2126;
    --bg-card-alt: #262A31;
    --accent: #A5B4FC;
    --accent-dark: #8B98F0;
    --accent-soft: rgba(165,180,252,0.14);
    --on-accent: #1E2126;
    --text-primary: #F2F3F5;
    --text-secondary: #9AA0A8;
    --border: #33373E;
}

/* ── 전체 앱 배경 다크화 ───────────────────────── */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: var(--bg-page) !important;
}
[data-testid="stHeader"] { background-color: transparent !important; }

.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stMarkdown, .stCaption, div[data-testid="stCaptionContainer"] {
    color: var(--text-primary);
}
.stCaption, div[data-testid="stCaptionContainer"] p {
    color: var(--text-secondary) !important;
}

/* ── 상단 고정 헤더 (뉴스룸 블랙 내비 느낌) ─────── */
.header {
    position: sticky;
    top: 0;
    z-index: 999;
    background: var(--bg-nav);
    padding: 12px 22px;
    border-radius: 10px;
    margin-bottom: 12px;
    border-bottom: 3px solid var(--accent);
}

.header h1 {
    color: #ffffff;
    font-size: 21px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.01em;
}

.header p {
    color: var(--text-secondary);
    font-size: 13px;
    margin: 3px 0 0 0;
}
</style>

<div class="header">
<h1>AI-Based <span style="color:var(--accent);">HAZOP</span> Safety Analysis Tool</h1>
<p>Process Hazard Analysis with AI-based Safeguard Recommendation</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    max-width: 1200px;
}

section.main > div {
    padding-top: 0.4rem;
}

/* ── 카드 컨테이너 ─────────────────────────────── */
.card {
    background-color: var(--bg-card);
    padding: 14px 18px;
    border-radius: 12px;
    border: 1px solid var(--border);
    margin-bottom: 10px;
    color: var(--text-primary);
}

/* 원인/결과/현재 안전조치 미니 카드 */
.fact-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 14px;
    border-left: 3px solid var(--accent);
    background-color: var(--bg-card-alt);
    border-radius: 6px;
    margin-bottom: 8px;
}
.fact-row .fact-label {
    flex-shrink: 0;
    width: 84px;
    font-weight: 700;
    font-size: 13px;
    color: var(--accent);
}
.fact-row .fact-value {
    font-size: 14px;
    color: var(--text-primary);
    line-height: 1.5;
}
.fact-row.safeguard { border-left-color: var(--text-secondary); }
.fact-row.safeguard .fact-label { color: var(--text-secondary); }

/* ── 버튼 ─────────────────────────────────────── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid var(--accent);
    background-color: transparent;
    color: var(--accent);
    transition: all 0.15s ease;
}
.stButton > button[kind="primary"],
.stButton > button:not([kind]) {
    background-color: var(--accent);
    color: var(--on-accent);
    font-weight: 700;
    border-color: var(--accent);
}
.stButton > button:hover {
    background-color: var(--accent-dark);
    border-color: var(--accent-dark);
    color: var(--on-accent);
}

/* ── 선택박스 / 입력창 ────────────────────────── */
/* 최신 Streamlit(react-aria-components 기반)은 BaseWeb이 아니라
   role="group"/"combobox" 구조를 쓴다. 실제로 흰 배경이 그려지는
   지점은 data-rac 속성이 붙은 role="group" div다. */
div[data-testid="stSelectbox"] div[role="group"],
div[data-testid="stNumberInput"] div[role="group"],
.react-aria-ComboBox div[role="group"] {
    background-color: var(--bg-card-alt) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}
div[data-testid="stSelectbox"] input,
div[data-testid="stNumberInput"] input,
.react-aria-ComboBox input {
    background-color: transparent !important;
    color: var(--text-primary) !important;
}
div[data-testid="stSelectbox"] svg,
div[data-testid="stNumberInput"] svg,
.react-aria-ComboBox svg {
    fill: var(--text-secondary) !important;
    color: var(--text-secondary) !important;
}

.stTextInput input,
.stTextArea textarea {
    border-radius: 8px !important;
    border-color: var(--border) !important;
    background-color: var(--bg-card-alt) !important;
    color: var(--text-primary) !important;
}
/* text_input/text_area도 selectbox와 동일하게 흰 배경의 RootElement 래퍼가
   따로 있어서, input/textarea 자체만 색을 바꿔도 테두리가 흰색으로 남는다. */
div[data-testid="stTextInputRootElement"],
div[data-testid="stTextAreaRootElement"] {
    background-color: var(--bg-card-alt) !important;
    border-color: var(--border) !important;
}

/* 드롭다운 펼침 목록은 body 하위 별도 레이어(포탈)로 렌더링되므로 전역 선택자로 처리 */
div[role="listbox"], ul[role="listbox"],
div[data-baseweb="popover"], div[popover], [data-rac][role="presentation"] {
    background-color: var(--bg-card-alt) !important;
}
div[role="listbox"] [role="option"], ul[role="listbox"] li,
div[role="listbox"] [role="option"] *, ul[role="listbox"] li * {
    background-color: var(--bg-card-alt) !important;
    color: var(--text-primary) !important;
}
div[role="listbox"] [role="option"]:hover, ul[role="listbox"] li:hover {
    background-color: var(--bg-card) !important;
}

/* ── 알림 배너 (st.info / st.warning / st.success 등 공통) ──── */
div[data-testid="stAlert"] {
    background-color: var(--bg-card-alt) !important;
    border: 1px solid var(--border) !important;
}
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div {
    color: var(--text-primary) !important;
}

/* ── 확장영역(expander) ──────────────────────── */
.streamlit-expanderHeader {
    font-size: 17px;
    font-weight: 700;
    border-radius: 8px;
}
div[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    background-color: var(--bg-card);
}
/* Streamlit 기본 테마가 summary(확장영역 헤더 바)에 거의 흰색(rgb(250,250,250))
   배경을 하드코딩해서 넣어놔서, hover 전에는 이 흰 배경이 그대로 보였다.
   호버 여부와 상관없이 항상 다크 배경을 강제한다. */
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary:hover,
div[data-testid="stExpander"] summary:focus {
    background-color: var(--bg-card) !important;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span,
.streamlit-expanderHeader {
    color: var(--text-primary) !important;
}
div[data-testid="stExpander"] summary svg {
    fill: var(--text-primary) !important;
}

/* AI 결과 내부 markdown 헤더가 너무 커서 스크롤이 과해지는 것 방지 */
div[data-testid="stExpander"] h1 { font-size: 20px !important; margin: 0.6em 0 0.3em !important; color: var(--text-primary); }
div[data-testid="stExpander"] h2 { font-size: 17px !important; margin: 0.6em 0 0.3em !important; color: var(--text-primary); }
div[data-testid="stExpander"] h3 { font-size: 15px !important; margin: 0.5em 0 0.2em !important; color: var(--accent); }
div[data-testid="stExpander"] p, div[data-testid="stExpander"] li { font-size: 14px !important; line-height: 1.55 !important; color: var(--text-primary); }

/* ── 서브헤더 여백 ────────────────────────────── */
h1, h2, h3, h4 {
    letter-spacing: -0.01em;
    color: var(--text-primary);
}
hr {
    margin: 1.6rem 0;
    border: none;
    border-top: 1px solid var(--border);
}

/* ── 모던 섹션 타이틀 (뉴스룸 필터탭 느낌의 뱃지 + 큰 제목) ──── */
.section-title { margin: 0.6rem 0 0.9rem; }
.section-title .eyebrow {
    display: inline-block;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: var(--on-accent);
    background: var(--accent);
    padding: 4px 12px;
    border-radius: 6px;
    margin-bottom: 8px;
}
.section-title .title {
    margin: 6px 0 0;
    font-size: 22px;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}

/* ── 서브섹션 타이틀 (카드 안쪽 소제목) ─────────── */
.subsection-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 10px;
    padding-bottom: 7px;
    border-bottom: 2px solid var(--accent);
    display: block;
}
.minor-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-secondary);
    letter-spacing: 0.02em;
    margin: 0.3em 0 0.4em;
}

/* ── 커스텀 상태 배너 (API 키 안내) ──────────────── */
.status-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 4px;
    border: 1px solid var(--border);
}
.status-banner::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.status-banner.ok { background: var(--bg-card-alt); color: #4ADE80; }
.status-banner.ok::before { background: #4ADE80; }
.status-banner.warn { background: var(--bg-card-alt); color: #FBBF24; }
.status-banner.warn::before { background: #FBBF24; }

/* ── 컬럼을 카드처럼 (뉴스룸 카드 그리드 느낌) ──────────── */
div[data-testid="column"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 18px;
}
/* 중첩된 컬럼(예: 빈도/강도 나란히 배치, 빈도·강도 기준표)은 톤을 살짝 낮추고
   여백도 최소화해서 이중 테두리+과한 패딩으로 세로 공간을 잡아먹지 않게 함 */
div[data-testid="column"] div[data-testid="column"] {
    background: var(--bg-card-alt);
    border: 1px solid var(--border);
    padding: 10px 12px;
}

/* stVerticalBlock(각 컬럼/컨테이너 내부 요소 간 기본 간격)이 기본값이 커서
   위아래로 공간을 많이 잡아먹는 것을 줄임 */
div[data-testid="stVerticalBlock"] { gap: 0.6rem !important; }

/* 표(markdown table) 다크 대응 */
.stApp table { color: var(--text-primary); border-color: var(--border); }
.stApp table th { background-color: var(--bg-card-alt); }
.stApp table td, .stApp table th { border-color: var(--border) !important; }

/* ── 사이드바 (뉴스룸 블랙 내비 톤) ───────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-nav);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-primary); }
section[data-testid="stSidebar"] h2 {
    font-size: 13px;
    font-weight: 800;
    color: var(--accent) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


def render_section(eyebrow, title):
    """모던 홈페이지 스타일의 섹션 대제목 (STEP n 뱃지 + 큰 타이틀)."""
    st.markdown(
        f'<div class="section-title"><span class="eyebrow">{eyebrow}</span>'
        f'<div class="title">{title}</div></div>',
        unsafe_allow_html=True,
    )


def render_subsection(title):
    """카드 안쪽에 쓰는 소제목 (밑줄 강조)."""
    st.markdown(f'<div class="subsection-title">{title}</div>', unsafe_allow_html=True)


def render_minor_title(title):
    """참고표/보조 정보용 아주 작은 타이틀."""
    st.markdown(f'<div class="minor-title">{title}</div>', unsafe_allow_html=True)


if client is None:
    st.markdown(
        '<div class="status-banner warn">현재 OpenAI API 키가 설정되지 않아 AI 추천 기능은 데모 모드로 표시됩니다.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-banner ok">OpenAI API 키가 설정되었습니다. 실제 유효성은 AI 실행 시 검증됩니다.</div>',
        unsafe_allow_html=True,
    )

st.caption("저장탱크 공정을 대상으로 단일·복합 편차를 분석하고, 위험도 평가 및 AI 기반 개선권고사항 도출 구조를 구현한 HAZOP 프로그램")

# ✅ ------------------- 메인 2-Column UI -------------------
tab1, tab2 = st.tabs(["STEP 1 · 단일 편차 분석", "STEP 2 · 복합 편차 분석"])

with tab1:
    render_section("STEP 1", "AI 단일 편차 HAZOP 분석")

    # ✅ 2-Column UI 시작
    col1, col2 = st.columns([1, 1], gap="large")

    # ✅ [왼쪽] – Cause/Consequence + 현재조치 + 현재 위험도 평가
    with col1:
        render_subsection("현재 위험도 평가")

        # ✅ 편차 선택
        selected_deviation = st.selectbox(
            "편차 선택",
            list(hazop_db[selected_node].keys()),
            key="deviation_select_left"
        )

        # ✅ 현재 정보 표시
        st.markdown(f"""
    <div class="fact-row"><div class="fact-label">원인</div><div class="fact-value">{hazop_db[selected_node][selected_deviation]['원인']}</div></div>
    <div class="fact-row"><div class="fact-label">결과</div><div class="fact-value">{hazop_db[selected_node][selected_deviation]['결과']}</div></div>
    <div class="fact-row safeguard"><div class="fact-label">현재 안전조치</div><div class="fact-value">{hazop_db[selected_node][selected_deviation]['현재 안전조치']}</div></div>
    """, unsafe_allow_html=True)

        # ✅ 발생빈도 / 발생강도 (한 줄에 배치해 세로 공간 절약)
        freq_sev_col1, freq_sev_col2 = st.columns(2)
        with freq_sev_col1:
            freq = st.selectbox("발생빈도 [1-5]", [1, 2, 3, 4, 5], key="freq_single")
        with freq_sev_col2:
            sev = st.selectbox("발생강도 [1-4]", [1, 2, 3, 4], key="sev_single")

        # ✅ 위험도 계산
        risk_score = freq * sev

        # ✅ 위험 등급 판정
        if risk_score <= 3:
            risk_level = "매우 낮음 (허용 가능)"
            color = "#4ADE80"
        elif risk_score <= 6:
            risk_level = "낮음 (허용 가능)"
            color = "#60A5FA"
        elif risk_score == 8:
            risk_level = "보통 (허용 불가능)"
            color = "#FB923C"
        elif 9 <= risk_score <= 12:
            risk_level = "약간 높음 (허용 불가능)"
            color = "#F97316"
        elif risk_score == 15:
            risk_level = "높음 (허용 불가능)"
            color = "#F87171"
        else:
            risk_level = "매우 높음 (허용 불가능)"
            color = "#DC2626"

        # ✅ 현재 위험도 표시
        st.markdown(
            f"""<div class="card" style="border-left: 5px solid {color}; padding: 14px 18px; margin-top: 4px;">
    <div style="font-size:13px; color:#9C9CA8; font-weight:600;">현재 위험도 (빈도 {freq} × 강도 {sev})</div>
    <div style="font-size:22px; font-weight:800; color:{color}; margin-top:2px;">{risk_score}점 &nbsp;→&nbsp; {risk_level}</div>
    </div>""",
            unsafe_allow_html=True
        )

        # ✅ 빈도/강도 기준 + 위험도 결정 기준 (기본은 접어서 세로 공간 절약)
        with st.expander("빈도·강도 및 위험도 판정 기준 보기"):
            freq_col, matrix_col = st.columns([3, 5])

            with freq_col:
                render_minor_title("빈도·강도 설정 기준")
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
                render_minor_title("위험도 결정 기준")
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
    def generate_ai_safeguard(deviation, cause, consequence, existing, guide_results, law_results, accident_results_str=None):
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
당신은 산업안전 컨설턴트입니다. HAZOP 방법론에 따라 개선권고사항을 작성하되,
아래 원칙을 반드시 지키십시오.

[원칙 1] 개선권고는 성격이 다른 두 축으로 나누어 작성하십시오.
- (A) 원인 감소(예방): 아래 [원인]이 애초에 발생하지 않도록 막는 조치
- (B) 결과 완화(방호): [원인]이 발생하더라도 아래 [결과]까지 이어지지 않도록 막거나
      피해를 줄이는 조치 (이미 있는 [현재 안전조치]와 중복되지 않게 작성)
각 축마다 최소 1개, 최대 2개씩 작성하십시오.

[원칙 2] 법령은 편차 자체가 아니라, 당신이 위에서 작성한 "개별 개선권고사항"에
실질적으로 대응하는 경우에만 붙이십시오. 즉 "이 개선권고를 시행해야 하는 근거가
아래 [참고 Guide]/[참고 Law]에 실제로 적혀 있는가"를 개선권고마다 따로 판단하십시오.

**절대 규칙 (가장 중요): 법령명, 기준명(OSHA 등), 조문·조항 번호는 반드시 아래
[참고 Guide]와 [참고 Law]의 텍스트 안에 실제로 적혀 있는 것만 그대로 옮겨 쓰십시오.
그 안에 없는 법령명/기준명/조항번호는 절대 언급하지도, 추측하지도, 지어내지도 마십시오.
"그럴듯하게 들리는" 조항번호를 만들어내는 것도 금지입니다.**

이때 실제로 인용하는 자료의 출처 종류를 반드시 구분해서 표기하십시오.
- 한국 법령(출처에 "산업안전보건법", "산업안전보건기준에 관한 규칙" 등이 포함된 자료)에
  대응되는 조문이 있으면: "법적 근거: <법령명 조문번호> (국내 법적 의무사항)"
- 해외 기준(출처에 "OSHA" 등 해외 기관/규격명이 포함된 자료)에 대응되는 조항이 있으면:
  "참고 기준: <출처 조항번호> (국제 기술기준 참고 — 한국 내 법적 강제력 없음, 모범사례로만 활용)"
  이라고 명시하여 한국 법령과 절대 같은 급으로 "법적 근거"라고 쓰지 마십시오.
- MSDS(물질안전보건자료, 출처에 "MSDS"가 포함된 자료)에서 가져온 물성·유해성 수치
  (인화점, 폭발범위, 증기압 등)를 개선권고의 배경 설명에 쓸 경우: "참고자료: 에틸렌 MSDS"
  라고만 표기하고, 이것도 "법적 근거"나 "참고 기준"(국제기술기준)이 아니라 순수한
  물질 특성 데이터임을 구분하십시오.
- 아래 [참고 Guide]/[참고 Law]가 비어있거나("없음" 표시) 대응되는 조문/자료가 전혀 없으면:
  억지로 끼워맞추지 말고 "법적 근거: 없음(업계 모범사례 기준)"이라고만 명시하십시오.
- [참고 Guide]/[참고 Law]에 있는 내용이라도 이 편차의 원인/결과와 실질적으로 무관하면 인용하지 마십시오.

[원칙 3] 마지막에 관련 사고사례를 별도 항목으로 정리하십시오.

Deviation: {deviation}
[원인]: {cause}
[결과]: {consequence}
[현재 안전조치]: {existing}

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
            answer = response.choices[0].message.content

            # 안전장치: 프롬프트로 "DB에 없는 출처는 언급 금지"라고 지시해도 LLM이 가끔
            # 그럴듯한 법령/기준명을 지어내는 경우가 있다. 실제로 검색된 참고자료(guide_results,
            # law_results)에 전혀 등장하지 않는 출처명이 답변에 나오면, 화면에 경고를 붙여
            # 사용자가 바로 알아볼 수 있게 한다.
            combined_sources = f"{guide_results}\n{law_results}"
            KNOWN_SOURCE_KEYWORDS = ["NFPA", "ISO", "ANSI", "API RP", "EN 1"]
            hallucinated = [
                kw for kw in KNOWN_SOURCE_KEYWORDS
                if kw in answer and kw not in combined_sources
            ]
            if hallucinated:
                warning = (
                    f"\n\n---\n**주의**: 위 답변에 {', '.join(hallucinated)} 관련 언급이 있으나, "
                    "실제로 검색된 참고자료(DB)에는 해당 출처가 없습니다. AI가 잘못 생성했을 가능성이 "
                    "높으니 이 부분은 법적 근거로 신뢰하지 말고 반드시 원문을 직접 확인하세요."
                )
                answer += warning

            return answer

        except Exception as e:
            return f"AI 개선권고사항 생성 중 오류가 발생했습니다: {e}"
        
    # ✅ Streamlit UI
    if "gpt_output_single" not in st.session_state:
        st.session_state["gpt_output_single"] = ""

    with col2:
        render_subsection("AI 활용 개선권고사항")

        manual_safeguard = st.text_area(
            "관리자 개선권고 입력",
            value="",
            height=80,
            placeholder="관리자가 최종 판단하여 개선권고사항을 직접 입력할 수 있습니다."
        )

        show_accident_case = False
        if selected_deviation in accident_cases:
            show_accident_case = st.checkbox("사고사례도 함께 보기", value=False)

        if st.button("AI 추천 개선권고사항"):
            current_entry = hazop_db[selected_node][selected_deviation]
            # 검색 쿼리로 "More Flow" 같은 영문 라벨만 쓰면 한국어 법령 원문과 임베딩 유사도가
            # 잘 안 잡혀서(교차 언어 매칭이 약함), 실제 원인·결과 한국어 문장을 붙여서 검색한다.
            search_query = f"{selected_deviation}. 원인: {current_entry['원인']}. 결과: {current_entry['결과']}"

            # 설비명이 영문 약칭(Relief Valve 등)으로만 적혀 있으면 한국어 법령 원문의 용어
            # (안전밸브, 액위계 등)와 임베딩 유사도가 잘 안 잡힌다. 알려진 영/한 동의어를
            # 검색어에 덧붙여 교차 언어 매칭을 보강한다.
            EQUIPMENT_SYNONYMS = {
                "Relief Valve": "안전밸브 릴리프밸브 압력방출장치",
                "Level Gauge": "액위계 레벨센서 레벨게이지",
                "Check Valve": "체크밸브 역지밸브",
                "Alarm": "경보장치 알람",
            }
            extra_terms = " ".join(
                ko for en, ko in EQUIPMENT_SYNONYMS.items()
                if en in search_query
            )
            if extra_terms:
                search_query = f"{search_query} {extra_terms}"

            with st.spinner("KOSHA & 법령 DB 검색 중..."):
                # 법령 DB는 출처 이름으로 거르지 않음 (law_chunks의 source는 "산업안전보건법(법률)" 등
                # 법령명이지 "KOSHA"가 아니라서, 여기에 KOSHA 필터를 걸면 결과가 항상 비었음)
                law_results_raw = search_db(law_index, law_chunks, search_query)
                # 가이드 DB는 KOSHA 출처만 우선 사용
                guide_results_raw = search_db(guide_index, guide_chunks, search_query, source_filter="KOSHA")

                # 벡터 유사도로 찾아온 후보를 AI가 다시 보고, 실제로 이 편차에 적용되는 것만 추림
                law_results = rerank_relevant(search_query, law_results_raw)
                guide_results = rerank_relevant(search_query, guide_results_raw)

                law_note = relevance_note(law_results_raw, law_results, "법령")
                guide_note = relevance_note(guide_results_raw, guide_results, "KOSHA 가이드")

                law_results_str = "\n".join(law_results) if law_results else law_note
                guide_results_str = "\n".join(guide_results) if guide_results else guide_note

                accident_results_str = accident_cases[selected_deviation] if show_accident_case else None

            with st.spinner("AI가 개선권고사항 생성 중..."):
                st.session_state["gpt_output_single"] = generate_ai_safeguard(
                    selected_deviation,
                    current_entry["원인"],
                    current_entry["결과"],
                    current_entry["현재 안전조치"],
                    guide_results_str,
                    law_results_str,
                    accident_results_str
                )

        with st.expander("AI 추천 개선권고사항 분석 결과 (클릭하여 열기/닫기)", expanded=True):
            result_text = st.session_state.get("gpt_output_single", "")
            if result_text:
                # 결과가 길어서 페이지 전체를 계속 스크롤해야 하는 문제를 막기 위해,
                # 고정 높이의 내부 스크롤 영역 안에서만 결과를 보여준다.
                with st.container(height=460, border=False):
                    st.markdown(result_text, unsafe_allow_html=True)
            else:
                st.info("아직 AI 분석 결과가 없습니다.")

        if manual_safeguard.strip():
            render_minor_title("관리자 입력 개선권고사항")
            st.markdown(f'<div class="card">{manual_safeguard}</div>', unsafe_allow_html=True)

        render_subsection("개선 후 위험도 평가")

        freq_after_col1, freq_after_col2 = st.columns(2)
        with freq_after_col1:
            freq_after = st.selectbox("개선 후 발생빈도 [1-5]", [1, 2, 3, 4, 5], key="freq_after_col2")
        with freq_after_col2:
            sev_after = st.selectbox("개선 후 발생강도 [1-4]", [1, 2, 3, 4], key="sev_after_col2")

        risk_score_after = freq_after * sev_after

        if risk_score_after <= 3:
            risk_level_after = "매우 낮음 (허용 가능)"
            color_after = "#4ADE80"
        elif risk_score_after <= 6:
            risk_level_after = "낮음 (허용 가능)"
            color_after = "#60A5FA"
        elif risk_score_after == 8:
            risk_level_after = "보통 (허용 불가능)"
            color_after = "#FB923C"
        elif 9 <= risk_score_after <= 12:
            risk_level_after = "약간 높음 (허용 불가능)"
            color_after = "#F97316"
        elif risk_score_after == 15:
            risk_level_after = "높음 (허용 불가능)"
            color_after = "#F87171"
        else:
            risk_level_after = "매우 높음 (허용 불가능)"
            color_after = "#DC2626"

        st.markdown(
            f"""<div class="card" style="border-left: 5px solid {color_after}; padding: 14px 18px; margin-top: 4px;">
    <div style="font-size:13px; color:#9C9CA8; font-weight:600;">개선 후 위험도 (빈도 {freq_after} × 강도 {sev_after})</div>
    <div style="font-size:22px; font-weight:800; color:{color_after}; margin-top:2px;">{risk_score_after}점 &nbsp;→&nbsp; {risk_level_after}</div>
    </div>""",
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


with tab2:
    # ==========================================
    # 6. 복합 Deviation 분석 (핸드북 우선 적용 버전)
    # ==========================================
    render_section("STEP 2", "AI 복합 편차 HAZOP 분석")

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
            st.warning("최소 2개 이상의 편차를 선택하세요.")

        elif is_invalid_combination(selected_devs, node_ai):
            st.error("이는 물리적으로 불가능한 편차 조합입니다. 다시 선택해주세요.")

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

아래 형식으로 하나의 통합 분석 결과만 작성해줘 (한국어):

1. 원인
- {source_used} 자료 기반 사고사례 또는 기술적 설명 반영

2. 결과
- 실제 피해(인명·설비 등) 중심의 구체적 결과 작성

3. 개선권고조치
- 아래 참고자료 기반으로 기술
- 가능한 경우 KOSHA Guide 코드 표기 포함

참고자료:
{reference_data}
"""

                render_subsection("AI 복합 편차 HAZOP 분석 결과")

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
