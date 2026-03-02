from fastapi import FastAPI
from pydantic import BaseModel
import json
from pinecone import Pinecone
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel

app = FastAPI(title="EEHO AI API", version="0.2")

# ============================================================
# 설정
# ============================================================
PINECONE_API_KEY = "pcsk_5p2m3k_6vU1BHznUnFiiZT2xpXov2Y9Hpx7KqG1dXRBkmmZVLyS4kcQgg9aAceNcDMwcti"
BUCKET_NAME = "eeho-tax-knowledge-base-01"
GUIDE_PATH = "실무서가이드/yangdo_2025_guide.json"

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("eeho-tax-cases")

# Gemini
vertexai.init(project="project-9fb5ee59-ec65-4d2a-aa6", location="us-central1")
gemini = GenerativeModel("gemini-2.0-flash-001")

# ============================================================
# 요청 모델
# ============================================================
class QueryRequest(BaseModel):
    question: str

class Step2Request(BaseModel):
    question: str
    판례검색: list
    추출키워드: list
    참조섹션: list

# ============================================================
# 실무서 가이드 로드 (캐싱)
# ============================================================
_guide_cache = None

def load_guide():
    global _guide_cache
    if _guide_cache is not None:
        return _guide_cache
    gcs = storage.Client()
    bucket = gcs.bucket(BUCKET_NAME)
    blob = bucket.blob(GUIDE_PATH)
    data = blob.download_as_text()
    _guide_cache = json.loads(data)
    return _guide_cache

# ============================================================
# 키워드 매칭
# ============================================================
TAX_KEYWORDS = [
    "1세대1주택", "비과세", "동거봉양", "합가", "혼인", "상속주택",
    "일시적2주택", "장애인", "직계존속", "5년", "5년이내", "60세",
    "다주택자", "중과세", "재개발", "재건축", "농어촌주택",
    "부담부증여", "고가주택", "비사업용토지", "비사업용",
    "장기보유특별공제", "장특공", "양도시기", "취득시기",
    "세대", "주택수", "보유기간", "거주기간", "2주택", "3주택",
    "조합원입주권", "분양권", "겸용주택", "다가구주택",
    "상속", "증여", "대물변제", "교환", "문화재주택",
    "임대주택", "자경농지", "출국세", "비거주자",
    "실거래가", "기준시가", "필요경비", "양도차익",
    "예정신고", "확정신고", "경정청구", "가산세",
    "조정대상지역", "중과배제", "사업용의제"
]

def extract_keywords_from_case(fields: dict) -> set:
    text = json.dumps(fields, ensure_ascii=False)
    found = set()
    for kw in TAX_KEYWORDS:
        if kw in text:
            found.add(kw)
    return found

def match_guide_sections(keywords: list, guide: list) -> list:
    results = []
    for section in guide:
        section_kw = set(section.get("키워드", []))
        matched = section_kw.intersection(set(keywords))
        if matched:
            results.append({
                "섹션": section["섹션"],
                "출처": section["출처"],
                "챕터": section["챕터"],
                "페이지": section["페이지"],
                "파일명": section["파일명"],
                "매칭키워드": list(matched),
                "매칭수": len(matched)
            })
    results.sort(key=lambda x: x["매칭수"], reverse=True)
    return results[:3]

# ============================================================
# Phase 1: 판례 검색 + 실무서 매칭
# ============================================================
@app.get("/")
def health():
    return {"service": "EEHO AI API", "version": "0.2", "status": "running"}

@app.post("/analyze")
async def analyze(req: QueryRequest):
    question = req.question

    # STEP 1: Pinecone 유사 판례 검색
    search_results = index.search_records(
        namespace="tax_cases",
        query={"inputs": {"text": question}, "top_k": 3},
    )

    판례목록 = []
    전체키워드 = set()

    for hit in search_results["result"]["hits"]:
        case = {
            "사건번호": hit["fields"].get("사건번호", ""),
            "주제": hit["fields"].get("주제", ""),
            "결과": hit["fields"].get("결과", ""),
            "유사도": round(hit["_score"], 3),
            "판단근거": hit["fields"].get("판단근거", ""),
            "관련법령": hit["fields"].get("관련법령", ""),
            "과세관청주장": hit["fields"].get("과세관청주장", ""),
        }
        판례목록.append(case)
        keywords = extract_keywords_from_case(hit["fields"])
        전체키워드.update(keywords)

    키워드리스트 = list(전체키워드)

    # STEP 2: 실무서 매칭
    guide = load_guide()
    매칭결과 = match_guide_sections(키워드리스트, guide)

    참조섹션 = []
    for m in 매칭결과:
        참조섹션.append({
            "섹션": m["섹션"],
            "챕터": m["챕터"],
            "페이지": m["페이지"],
            "파일명": m["파일명"],
            "매칭키워드": m["매칭키워드"]
        })

    return {
        "질문": question,
        "step1_판례검색": 판례목록,
        "step2_추출키워드": 키워드리스트,
        "step3_실무서매칭": 매칭결과,
        "step4_참조섹션": 참조섹션,
    }

# ============================================================
# Phase 2: Gemini 추가 질문 생성
# ============================================================
@app.post("/generate-questions")
async def generate_questions(req: QueryRequest):
    """
    Phase 1 + Phase 2 통합:
    고객 질문 → 판례 검색 → 실무서 매칭 → Gemini 추가 질문 생성
    """
    # Phase 1 실행
    phase1 = await analyze(req)

    # Phase 2: Gemini에게 추가 질문 생성 요청
    prompt = f"""당신은 양도소득세 전문 세무사 AI입니다.

고객이 다음과 같은 질문을 했습니다:
"{req.question}"

관련 판례 분석 결과:
{json.dumps(phase1["step1_판례검색"], ensure_ascii=False, indent=2)}

추출된 핵심 키워드:
{phase1["step2_추출키워드"]}

참조할 실무서 섹션:
{json.dumps(phase1["step4_참조섹션"], ensure_ascii=False, indent=2)}

위 정보를 바탕으로, 고객의 양도소득세 비과세/감면 여부를 정확히 판단하기 위해
추가로 확인해야 할 사항 중 비과세/감면 판단에 가장 핵심적인 질문 3개만 선별하여 Yes/No 체크리스트로 생성해주세요. 반드시 3개만 생성하세요.

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요:

{{
  "체크리스트": [
    {{
      "질문": "양도일 기준 해당 주택의 보유기간이 2년 이상인가요?",
      "카테고리": "보유기간",
      "중요도": "필수",
      "설명": "1세대 1주택 비과세를 위해서는 2년 이상 보유해야 합니다."
    }}
  ],
  "추가정보_필요": [
    {{
      "항목": "양도가액",
      "설명": "예상 세액 계산을 위해 양도가액(매매대금)이 필요합니다.",
      "입력형식": "금액"
    }}
  ]
}}
"""

    response = gemini.generate_content(prompt)
    response_text = response.text.strip()

    # JSON 파싱
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    try:
        gemini_result = json.loads(response_text)
    except json.JSONDecodeError:
        gemini_result = {"raw_response": response_text, "parse_error": True}

    return {
        "질문": req.question,
        "판례검색": phase1["step1_판례검색"],
        "추출키워드": phase1["step2_추출키워드"],
        "참조섹션": phase1["step4_참조섹션"],
        "추가질문": gemini_result,
    }

# ============================================================
# 테스트 엔드포인트
# ============================================================
@app.get("/test")
async def test():
    req = QueryRequest(question="부모님을 모시려고 합가했는데 아파트를 팔면 비과세 되나요?")
    return await analyze(req)

@app.get("/test-questions")
async def test_questions():
    req = QueryRequest(question="부모님을 모시려고 합가했는데 아파트를 팔면 비과세 되나요?")
    return await generate_questions(req)

# ============================================================
# Phase 3: 사실관계 정리
# ============================================================
class ConfirmRequest(BaseModel):
    question: str
    체크리스트응답: list  # [{"질문": "...", "답변": "Yes"}, ...]
    추가정보: dict = {}  # {"양도가액": "500000000", "취득가액": "300000000"}
    판례검색: list = []
    추출키워드: list = []
    참조섹션: list = []

@app.post("/confirm")
async def confirm(req: ConfirmRequest):
    """
    Phase 3: 체크리스트 답변 + 추가정보 → 사실관계 정리
    사용자가 컨펌하면 Phase 4(리포트)로 진행
    """
    prompt = f"""당신은 양도소득세 전문 세무사 AI입니다.

고객의 초기 질문:
"{req.question}"

고객이 답변한 체크리스트:
{json.dumps(req.체크리스트응답, ensure_ascii=False, indent=2)}

고객이 제공한 추가 정보:
{json.dumps(req.추가정보, ensure_ascii=False, indent=2)}

관련 판례:
{json.dumps(req.판례검색[:1], ensure_ascii=False, indent=2)}

위 정보를 종합하여 고객의 사실관계를 세법 관점에서 정리해주세요.

반드시 아래 JSON 형식으로만 응답하세요:

{{
  "사실관계_요약": {{
    "양도자_현황": "1주택 보유자가 동거봉양 목적으로 직계존속과 합가하여 일시적 2주택이 된 상태에서 본인 주택을 양도",
    "주택_현황": "양도 주택: OOO, 보유기간: O년, 거주기간: O년",
    "합가_현황": "합가일: OOOO, 합가사유: 동거봉양, 양도일까지 경과기간: O년",
    "기타": "해당 사항 정리"
  }},
  "적용_검토_조문": [
    {{
      "조문": "소득세법 시행령 제155조 제4항",
      "내용": "동거봉양 합가 특례",
      "충족여부": "충족 / 미충족 / 확인필요",
      "판단근거": "합가일로부터 5년 이내 양도하였으므로 요건 충족"
    }}
  ],
  "비과세_가능성": "높음/보통/낮음",
  "비과세_가능성_설명": "동거봉양 합가 특례 요건을 충족하는 것으로 보이나, 세무사의 최종 검토가 필요합니다."
}}
"""

    response = gemini.generate_content(prompt)
    response_text = response.text.strip()

    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    try:
        gemini_result = json.loads(response_text)
    except json.JSONDecodeError:
        gemini_result = {"raw_response": response_text, "parse_error": True}

    return {
        "질문": req.question,
        "사실관계": gemini_result,
        "안내": "위 사실관계가 맞으면 확인 버튼을 눌러 최종 리포트를 생성합니다."
    }

# ============================================================
# Phase 3 테스트
# ============================================================
@app.get("/test-confirm")
async def test_confirm():
    req = ConfirmRequest(
        question="부모님을 모시려고 합가했는데 아파트를 팔면 비과세 되나요?",
        체크리스트응답=[
            {"질문": "합가로 인해 일시적으로 2주택이 된 경우인가요?", "답변": "Yes"},
            {"질문": "합가일로부터 5년 이내에 해당 주택을 양도하는 것인가요?", "답변": "Yes"},
            {"질문": "양도하는 주택이 유일한 주택이었을 때 부모님을 모시기 위해 합가한 것인가요?", "답변": "Yes"},
        ],
        추가정보={"양도가액": "800000000", "취득가액": "500000000"},
        판례검색=[{
            "사건번호": "조심-2015-전-4851",
            "주제": "1세대 1주택 비과세 대상 여부 (혼인+동거봉양 합가 3주택)",
            "결과": "기각",
            "판단근거": "혼인 전에 배우자와 직계존속이 이미 각자 1주택을 보유하여 혼인으로 3주택자가 된 점, 양도일이 합가일부터 5년 도과 후인 점에 비추어 1세대 1주택 비과세 특례 요건을 충족하지 못함."
        }],
        추출키워드=["동거봉양", "합가", "비과세", "5년", "직계존속"],
        참조섹션=[{"섹션": "제4절 1세대 2주택 비과세 특례", "페이지": "749-898"}]
    )
    return await confirm(req)

# ============================================================
# Phase 4: 최종 리포트 생성
# ============================================================
class ReportRequest(BaseModel):
    question: str
    사실관계: dict
    체크리스트응답: list
    추가정보: dict = {}
    판례검색: list = []
    추출키워드: list = []
    참조섹션: list = []

@app.post("/report")
async def report(req: ReportRequest):
    """
    Phase 4: 사실관계 컨펌 후 → 예상세액 + 판단근거 + 리스크 리포트 생성
    """
    prompt = f"""당신은 양도소득세 전문 세무사 AI입니다. 최종 분석 리포트를 작성해주세요.

고객 질문: "{req.question}"

확인된 사실관계:
{json.dumps(req.사실관계, ensure_ascii=False, indent=2)}

체크리스트 답변:
{json.dumps(req.체크리스트응답, ensure_ascii=False, indent=2)}

고객 제공 정보:
{json.dumps(req.추가정보, ensure_ascii=False, indent=2)}

관련 판례:
{json.dumps(req.판례검색, ensure_ascii=False, indent=2)}

관련 실무서 섹션:
{json.dumps(req.참조섹션, ensure_ascii=False, indent=2)}

위 정보를 종합하여 최종 분석 리포트를 작성하세요.

반드시 아래 JSON 형식으로만 응답하세요:

{{
  "예상세액": {{
    "비과세_적용시": "0원 (1세대 1주택 비과세 적용)",
    "비과세_미적용시": "약 OOO만원 (양도차익 기준 추정)"
  }},
  "판단근거": [
    {{
      "조문": "소득세법 시행령 제155조 제4항",
      "내용": "동거봉양 합가 특례",
      "판단": "요건 충족 여부 및 근거 설명"
    }}
  ],
  "관련예판": [
    {{
      "사건번호": "조심-2015-전-4851",
      "결과": "기각",
      "시사점": "본 사례와의 차이점 또는 유사점 설명"
    }}
  ],
  "리스크": [
    {{
      "유형": "미확인사항",
      "내용": "제출 데이터로 확인되지 않는 요건 설명",
      "대응방안": "확인 방법 안내"
    }},
    {{
      "유형": "사후관리",
      "내용": "비과세 적용 후 주의해야 할 사항",
      "대응방안": "관리 방법 안내"
    }}
  ],
  "종합의견": "전체적인 판단 요약"
}}
"""

    response = gemini.generate_content(prompt)
    response_text = response.text.strip()

    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    try:
        gemini_result = json.loads(response_text)
    except json.JSONDecodeError:
        gemini_result = {"raw_response": response_text, "parse_error": True}

    return {
        "리포트": gemini_result,
        "면책안내": "본 분석은 참고용이며, 해당 조문의 적용 가능 여부 및 정확한 세액은 세무사의 최종 검토가 필요합니다.",
        "상담문의": {
            "안내": "보다 정확한 상담을 원하시면 아래 버튼을 눌러주세요.",
            "카카오톡채널": "https://pf.kakao.com/YOUR_CHANNEL_ID"
        }
    }

# ============================================================
# Phase 4 테스트
# ============================================================
@app.get("/test-report")
async def test_report():
    req = ReportRequest(
        question="부모님을 모시려고 합가했는데 아파트를 팔면 비과세 되나요?",
        사실관계={
            "사실관계_요약": {
                "양도자_현황": "1주택 보유자가 동거봉양 목적으로 직계존속과 합가하여 일시적 2주택이 된 상태에서 본인 주택을 양도",
                "주택_현황": "양도 주택: 고객 소유 주택, 보유기간: 확인 필요, 거주기간: 확인 필요",
                "합가_현황": "합가사유: 동거봉양, 양도일까지 경과기간: 5년 이내 (고객 답변 기준)"
            },
            "적용_검토_조문": [{
                "조문": "소득세법 시행령 제155조 제4항",
                "내용": "동거봉양 합가 특례",
                "충족여부": "충족",
                "판단근거": "합가일로부터 5년 이내 양도, 동거봉양 목적 합가 확인"
            }],
            "비과세_가능성": "높음"
        },
        체크리스트응답=[
            {"질문": "합가로 인해 일시적으로 2주택이 된 경우인가요?", "답변": "Yes"},
            {"질문": "합가일로부터 5년 이내에 해당 주택을 양도하는 것인가요?", "답변": "Yes"},
            {"질문": "양도하는 주택이 유일한 주택이었을 때 부모님을 모시기 위해 합가한 것인가요?", "답변": "Yes"},
        ],
        추가정보={"양도가액": "800000000", "취득가액": "500000000"},
        판례검색=[{
            "사건번호": "조심-2015-전-4851",
            "주제": "1세대 1주택 비과세 대상 여부 (혼인+동거봉양 합가 3주택)",
            "결과": "기각",
            "판단근거": "혼인으로 3주택자가 된 점, 합가일부터 5년 도과 후 양도한 점에서 특례 미충족"
        }],
        추출키워드=["동거봉양", "합가", "비과세", "5년", "직계존속"],
        참조섹션=[{"섹션": "제4절 1세대 2주택 비과세 특례", "페이지": "749-898"}]
    )
    return await report(req)
