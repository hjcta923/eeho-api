import os
import re
import uuid
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel, Field
from pinecone import Pinecone
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel

app = FastAPI(title="EEHO AI API", version="1.0")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://eehotax.com", "https://www.eehotax.com", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 환경변수
# ============================================================
PINECONE_API_KEY    = os.environ.get("PINECONE_API_KEY", "")
BUCKET_NAME         = os.environ.get("GCS_BUCKET_NAME", "")
GUIDE_PATH          = os.environ.get("GUIDE_PATH", "")
GCP_PROJECT_ID      = os.environ.get("GCP_PROJECT_ID", "")
GCP_LOCATION        = os.environ.get("GCP_LOCATION", "")
GEMINI_MODEL        = os.environ.get("GEMINI_MODEL", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "")
KAKAO_CHANNEL_URL   = os.environ.get("KAKAO_CHANNEL_URL", "")

# ============================================================
# 싱글턴 초기화
# ============================================================
pc    = None
index = None

def get_index():
    global pc, index
    if index is None:
        pc    = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
    return index

gemini = None

def get_gemini():
    global gemini
    if gemini is None:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        gemini = GenerativeModel(GEMINI_MODEL)
    return gemini

_guide_cache = None

def load_guide():
    global _guide_cache
    if _guide_cache is not None:
        return _guide_cache
    gcs    = storage.Client()
    bucket = gcs.bucket(BUCKET_NAME)
    blob   = bucket.blob(GUIDE_PATH)
    data   = blob.download_as_text()
    _guide_cache = json.loads(data)
    return _guide_cache

# ============================================================
# [순서 1] 공통 요청 모델 — buildPayload() 1:1 대응
# ============================================================
class TaxCategory(BaseModel):
    type: str = "모름"

class AssetInfo(BaseModel):
    asset_type: str = "모름"
    address:    str = "모름"
    area_size:  str = "모름"

class PriceInfo(BaseModel):
    buy_price:  int = 0
    sell_price: int = 0

class DateInfo(BaseModel):
    buy_date:  str = "모름"
    sell_date: str = "모름"

class ConditionInfo(BaseModel):
    is_regulated_area: str = "모름"
    house_count:       str = "모름"
    residence_period:  str = "모름"
    # 주식 CGT 전용 (선택)
    stock_listed:    Optional[str] = None
    stock_major:     Optional[str] = None
    stock_corp_size: Optional[str] = None
    # 상속세 전용 (선택)
    has_spouse: Optional[str] = None

class StructuredData(BaseModel):
    asset_info:     AssetInfo     = AssetInfo()
    price_info:     PriceInfo     = PriceInfo()
    date_info:      DateInfo      = DateInfo()
    condition_info: ConditionInfo = ConditionInfo()

class CalculatedData(BaseModel):
    estimated_total_tax:         int = 0
    estimated_yangdo_tax:        Optional[int] = None
    estimated_local_tax:         Optional[int] = None
    estimated_surcharge:         Optional[int] = None
    estimated_acq_tax:           Optional[int] = None
    estimated_local_edu_tax:     Optional[int] = None
    estimated_rural_tax:         Optional[int] = None
    estimated_property_tax:      Optional[int] = None
    estimated_comprehensive_tax: Optional[int] = None
    estimated_prop_edu_tax:      Optional[int] = None
    estimated_jonbu_rural_tax:   Optional[int] = None
    estimated_city_tax:          Optional[int] = None
    estimated_gift_tax:          Optional[int] = None
    estimated_inherit_tax:       Optional[int] = None
    estimated_filing_deduction:  Optional[int] = None

class UnstructuredData(BaseModel):
    user_context: str = ""

class AnalyzeRequest(BaseModel):
    """프론트엔드 buildPayload()가 보내는 계층적 JSON과 1:1 대응"""
    session_id:        str              = ""
    tax_category:      TaxCategory      = TaxCategory()
    structured_data:   StructuredData   = StructuredData()
    calculated_data:   CalculatedData   = CalculatedData()
    unstructured_data: UnstructuredData = UnstructuredData()
    additional_data:   Optional[dict]   = None

# ============================================================
# 세목별 프롬프트 분기
# ============================================================
ROLE_BY_TAX_TYPE = {
    "양도소득세":          "당신은 양도소득세 전문 세무사 AI입니다.",
    "상속세":              "당신은 상속세 전문 세무사 AI입니다.",
    "증여세":              "당신은 증여세 전문 세무사 AI입니다.",
    "취득세":              "당신은 취득세 전문 세무사 AI입니다.",
    "재산세/종합부동산세":  "당신은 보유세(재산세/종합부동산세) 전문 세무사 AI입니다.",
}

def get_role_prompt(tax_type: str) -> str:
    return ROLE_BY_TAX_TYPE.get(tax_type, "당신은 세무 전문가 AI입니다.")

# ============================================================
# Gap Detection 모델
# ============================================================
class GapResult(BaseModel):
    req_name:      str
    data_field:    str
    status:        str
    user_value:    str = ""
    threshold:     str = ""
    legal_basis:   str = ""
    priority:      str = ""
    question_hint: str = ""

class GapAnalysisResult(BaseModel):
    algorithm_stage:     str   = "deterministic_gap_detection"
    provisions_checked:  list  = []
    total_requirements:  int   = 0
    satisfied_count:     int   = 0
    gap_count:           int   = 0
    ambiguous_count:     int   = 0
    completeness_ratio:  float = 0.0
    gap_items:           list  = []
    ambiguous_items:     list  = []
    satisfied_items:     list  = []

# ============================================================
# /confirm 전용 모델
# ============================================================
class ChecklistAnswer(BaseModel):
    variable: str
    answer:   str

class ConfirmRequest(BaseModel):
    session_id:         str                    = ""
    original_request:   Optional[dict]         = None
    checklist_answers:  list[ChecklistAnswer]   = []
    user_corrections:   Optional[dict]         = None

# ============================================================
# /feedback 전용 모델
# ============================================================
class FeedbackRequest(BaseModel):
    session_id:      str  = ""
    original_report: dict = {}
    feedback_text:   str  = ""
    rating:          int  = 3

class TriageResult(BaseModel):
    classification:         str
    confidence:             float = Field(ge=0.0, le=1.0)
    reason:                 str   = ""
    has_factual_correction: bool  = False
    has_missing_info:       bool  = False
    has_legal_dispute:      bool  = False

class ErrorNote(BaseModel):
    note_id:            str
    session_id:         str
    timestamp:          str
    original_question:  str
    keywords:           list = []
    ai_report_summary:  str  = ""
    ai_비과세_판정:     str  = ""
    feedback_text:      str
    triage:             dict = {}
    deltas:             list = []
    embed_text:         str  = ""

# ============================================================
# 세무 키워드 사전
# ============================================================
TAX_KEYWORDS = [
    "1세대1주택","비과세","동거봉양","합가","혼인","상속주택","일시적2주택","장애인","직계존속",
    "5년","5년이내","60세","다주택자","중과세","재개발","재건축","농어촌주택","부담부증여",
    "고가주택","비사업용토지","비사업용","장기보유특별공제","장특공","양도시기","취득시기",
    "세대","주택수","보유기간","거주기간","2주택","3주택","조합원입주권","분양권","겸용주택",
    "다가구주택","상속","증여","대물변제","교환","문화재주택","임대주택","자경농지","출국세",
    "비거주자","실거래가","기준시가","필요경비","양도차익","예정신고","확정신고","경정청구",
    "가산세","조정대상지역","중과배제","사업용의제"
]

# ============================================================
# 공통 유틸
# ============================================================
def extract_keywords(fields):
    text = json.dumps(fields, ensure_ascii=False)
    return {kw for kw in TAX_KEYWORDS if kw in text}

def match_guide_sections(keywords, guide):
    results = []
    for section in guide:
        matched = set(section.get("키워드", [])).intersection(set(keywords))
        if matched:
            results.append({
                "섹션": section["섹션"], "출처": section["출처"],
                "챕터": section["챕터"], "페이지": section["페이지"],
                "파일명": section["파일명"],
                "매칭키워드": list(matched), "매칭수": len(matched)
            })
    results.sort(key=lambda x: x["매칭수"], reverse=True)
    return results[:3]

def flatten_user_data(d):
    """중첩 JSON → 1차원 dict 평탄화"""
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat[k2] = str(v2) if v2 is not None else ""
        else:
            flat[k] = str(v) if v is not None else ""
    return flat

def is_missing(v):
    return not v or str(v).strip() in ["", "모름", "null", "None", "없음", "미입력", "0"]

def evaluate_requirement(req, user_value):
    """data_type별 판정 규칙"""
    if is_missing(user_value):
        return "gap"
    dt = req.get("data_type", "text")
    th = req.get("threshold", "")

    if dt == "boolean":
        if user_value.lower() in ["yes", "true", "예", "여", "y"]:
            return "satisfied"
        elif user_value.lower() in ["no", "false", "아니오", "부", "n"]:
            return "ambiguous"
        return "ambiguous"
    elif dt == "number" and th:
        try:
            val = float(re.sub(r'[^\d.]', '', user_value))
            nums = re.findall(r'[\d,.]+', th)
            if nums:
                threshold_val = float(nums[0].replace(",", ""))
                if "이하" in th:
                    return "satisfied" if val <= threshold_val else "ambiguous"
                elif "이상" in th:
                    return "satisfied" if val >= threshold_val else "ambiguous"
                elif "초과" in th:
                    return "satisfied" if val > threshold_val else "ambiguous"
                elif "미만" in th:
                    return "satisfied" if val < threshold_val else "ambiguous"
        except:
            pass
        return "ambiguous"
    elif dt == "duration" and th:
        un = re.findall(r'(\d+)', user_value)
        tn = re.findall(r'(\d+)', th)
        if un and tn:
            if "이상" in th:
                return "satisfied" if int(un[0]) >= int(tn[0]) else "ambiguous"
            elif "이내" in th or "이하" in th:
                return "satisfied" if int(un[0]) <= int(tn[0]) else "ambiguous"
            elif "초과" in th:
                return "satisfied" if int(un[0]) > int(tn[0]) else "ambiguous"
        return "ambiguous"
    elif dt in ("text", "date"):
        # 값 존재하면 ambiguous (자동 판정 불가)
        return "ambiguous" if user_value and not is_missing(user_value) else "gap"

    return "ambiguous" if user_value and not is_missing(user_value) else "gap"

def run_gap_detection(ai_requirements: list, user_data: dict, provisions_checked=None) -> GapAnalysisResult:
    """
    Stage 2: LLM을 호출하지 않는 순수 규칙 기반 비교 알고리즘.
    """
    flat = flatten_user_data(user_data)
    gap_items, ambiguous_items, satisfied_items = [], [], []

    for req in ai_requirements:
        df = req.get("data_field", "")
        uv = flat.get(df, "")
        st = evaluate_requirement(req, uv)
        r  = GapResult(
            req_name=req.get("req_name", ""),
            data_field=df,
            status=st,
            user_value=uv if not is_missing(uv) else "(미입력)",
            threshold=req.get("threshold", ""),
            legal_basis=req.get("legal_basis", ""),
            priority=req.get("priority", "important"),
            question_hint=req.get("question_hint", "")
        )
        if st == "gap":
            gap_items.append(r.model_dump())
        elif st == "ambiguous":
            ambiguous_items.append(r.model_dump())
        else:
            satisfied_items.append(r.model_dump())

    total = len(ai_requirements)
    return GapAnalysisResult(
        provisions_checked=provisions_checked or [],
        total_requirements=total,
        satisfied_count=len(satisfied_items),
        gap_count=len(gap_items),
        ambiguous_count=len(ambiguous_items),
        completeness_ratio=round(len(satisfied_items) / total, 2) if total > 0 else 0,
        gap_items=gap_items,
        ambiguous_items=ambiguous_items,
        satisfied_items=satisfied_items
    )

def lookup_prior_errors(query: str, top_k: int = 3) -> list:
    """Pinecone error_notes namespace 선행 검색"""
    try:
        idx     = get_index()
        results = idx.search_records(
            namespace="error_notes",
            query={"inputs": {"text": query}, "top_k": top_k}
        )
        return [
            {
                "note_id":  h.get("_id", ""),
                "유사도":   round(h.get("_score", 0), 3),
                "원래질문": h["fields"].get("original_question", ""),
                "교훈":     h["fields"].get("교훈", ""),
                "오류유형": h["fields"].get("오류유형", ""),
                "lesson_learned": h["fields"].get("교훈", ""),
            }
            for h in results.get("result", {}).get("hits", [])
            if h.get("_score", 0) > 0.5
        ]
    except:
        return []

def build_search_query(payload: AnalyzeRequest) -> str:
    parts = []
    sd = payload.structured_data
    ci = sd.condition_info
    if payload.tax_category.type not in ["모름", ""]:
        parts.append(payload.tax_category.type)
    if sd.asset_info.asset_type not in ["모름", ""]:
        parts.append(sd.asset_info.asset_type)
    if sd.asset_info.address not in ["모름", ""]:
        parts.append(sd.asset_info.address)
    if ci.house_count not in ["모름", ""]:
        parts.append(f"{ci.house_count} 보유")
    if ci.is_regulated_area == "여":
        parts.append("조정대상지역")
    if ci.residence_period not in ["모름", "없음", ""]:
        parts.append(f"거주기간 {ci.residence_period}")
    if sd.price_info.sell_price > 1_200_000_000:
        parts.append("고가주택 12억 초과")
    ctx = payload.unstructured_data.user_context.strip()
    if ctx:
        parts.append(ctx)
    return " ".join(parts) if parts else "양도소득세 비과세 감면"

def build_llm_context(payload: AnalyzeRequest) -> str:
    sd = payload.structured_data
    cd = payload.calculated_data
    ci = sd.condition_info
    lines = [
        f"[세목] {payload.tax_category.type}",
        f"[자산] {sd.asset_info.asset_type} | 주소: {sd.asset_info.address} | 면적: {sd.asset_info.area_size}",
        f"[가격] 취득가: {sd.price_info.buy_price:,}원 / 양도가: {sd.price_info.sell_price:,}원",
        f"[날짜] 취득일: {sd.date_info.buy_date} / 양도일: {sd.date_info.sell_date}",
        f"[조건] 조정지역: {ci.is_regulated_area} | 주택수: {ci.house_count} | 거주기간: {ci.residence_period}",
    ]
    if ci.has_spouse:
        lines.append(f"[배우자] {ci.has_spouse}")
    if cd.estimated_total_tax > 0:
        tax_parts = [f"총 {cd.estimated_total_tax:,}원"]
        if cd.estimated_yangdo_tax:
            tax_parts.append(f"양도세 {cd.estimated_yangdo_tax:,}원")
        if cd.estimated_local_tax:
            tax_parts.append(f"지방세 {cd.estimated_local_tax:,}원")
        if cd.estimated_surcharge and cd.estimated_surcharge > 0:
            tax_parts.append(f"중과가산 {cd.estimated_surcharge:,}원")
        if cd.estimated_acq_tax and cd.estimated_acq_tax > 0:
            tax_parts.append(f"취득세 {cd.estimated_acq_tax:,}원")
        if cd.estimated_gift_tax and cd.estimated_gift_tax > 0:
            tax_parts.append(f"증여세 {cd.estimated_gift_tax:,}원")
        if cd.estimated_inherit_tax and cd.estimated_inherit_tax > 0:
            tax_parts.append(f"상속세 {cd.estimated_inherit_tax:,}원")
        lines.append(f"[1차예상세액] {' | '.join(tax_parts)}")
    ctx = payload.unstructured_data.user_context.strip()
    if ctx:
        lines.append(f"[사용자상황] {ctx}")
    add = payload.additional_data
    if add:
        lines.append(f"[추가답변] {json.dumps(add, ensure_ascii=False)}")
    return "\n".join(lines)

def strip_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    return t

# ============================================================
# 헬스체크
# ============================================================
@app.get("/")
def health():
    return {"service": "EEHO AI API", "version": "1.0", "status": "running"}

# ============================================================
# [순서 2] POST /analyze — Phase 1 진입점
# ============================================================
@app.post("/analyze")
async def analyze(payload: AnalyzeRequest):
    # 1. 검색 쿼리 구성
    search_query = build_search_query(payload)

    # 2. Pinecone tax_cases namespace 검색 (top_k=3)
    idx = get_index()
    search_results = idx.search_records(
        namespace="tax_cases",
        query={"inputs": {"text": search_query}, "top_k": 3}
    )
    판례목록 = []
    전체키워드 = set()
    for hit in search_results["result"]["hits"]:
        판례목록.append({
            "사건번호": hit["fields"].get("사건번호", ""),
            "주제":     hit["fields"].get("주제", ""),
            "결과":     hit["fields"].get("결과", ""),
            "유사도":   round(hit["_score"], 3),
            "판단근거": hit["fields"].get("판단근거", ""),
            "관련법령": hit["fields"].get("관련법령", ""),
        })
        전체키워드.update(extract_keywords(hit["fields"]))

    # 3. Pinecone error_notes namespace 선행 검색 (top_k=3)
    prior_errors = lookup_prior_errors(search_query, top_k=3)
    오답노트_교훈 = [
        {"lesson_learned": e.get("lesson_learned", ""), "유사도": e.get("유사도", 0)}
        for e in prior_errors
        if e.get("lesson_learned")
    ]

    # 4. 키워드 추출 + 실무서 매칭
    키워드리스트 = list(전체키워드)
    매칭결과 = match_guide_sections(키워드리스트, load_guide())
    참조섹션 = [m.get("섹션", "") for m in 매칭결과]

    # 5. 응답 반환 — calculated_data 패스스루
    return {
        "session_id":    payload.session_id,
        "tax_type":      payload.tax_category.type,
        "판례검색":      판례목록,
        "추출키워드":    키워드리스트,
        "실무서매칭":    매칭결과,
        "참조섹션":      참조섹션,
        "오답노트_교훈": 오답노트_교훈,
        "calculated_data": payload.calculated_data.model_dump(),
    }

# ============================================================
# [순서 3] POST /generate-questions — 3-Stage Gap Analysis
# ============================================================
@app.post("/generate-questions")
async def generate_questions(payload: AnalyzeRequest):
    model        = get_gemini()
    search_query = build_search_query(payload)
    llm_context  = build_llm_context(payload)
    role_prompt  = get_role_prompt(payload.tax_category.type)

    # Pinecone 판례 검색
    idx            = get_index()
    search_results = idx.search_records(
        namespace="tax_cases",
        query={"inputs": {"text": search_query}, "top_k": 3}
    )
    판례목록 = []
    전체키워드 = set()
    for hit in search_results["result"]["hits"]:
        판례목록.append({
            "사건번호": hit["fields"].get("사건번호", ""),
            "주제":     hit["fields"].get("주제", ""),
            "결과":     hit["fields"].get("결과", ""),
            "유사도":   round(hit["_score"], 3),
            "판단근거": hit["fields"].get("판단근거", ""),
            "관련법령": hit["fields"].get("관련법령", ""),
        })
        전체키워드.update(extract_keywords(hit["fields"]))

    키워드리스트 = list(전체키워드)
    매칭결과     = match_guide_sections(키워드리스트, load_guide())

    # 오답노트 선행참조
    prior_errors  = lookup_prior_errors(search_query)
    error_context = ""
    if prior_errors:
        lessons = [e.get("교훈", "") for e in prior_errors if e.get("교훈")]
        if lessons:
            error_context = (
                "\n[주의-과거오류교훈 반드시 반영]\n"
                f"과거 유사 사례에서 다음 사항을 놓쳤으므로 반드시 확인:\n"
                f"{json.dumps(lessons, ensure_ascii=False)}\n"
            )

    # ── Stage 1: 동적 요건 추출 (LLM) ──
    s1_prompt = f"""{role_prompt}
고객 상황:
{llm_context}

관련 판례:
{json.dumps(판례목록, ensure_ascii=False, indent=2)}

추출 키워드: {키워드리스트}
{error_context}
적용 가능한 비과세/감면 규정과 필수 충족 요건을 구조화하세요.

★ 절대 규칙: question_hint는 반드시 "예/아니오"로 답할 수 있는 이진 질문이어야 합니다.
- 올바른 예: "혼인신고를 하셨나요?", "보유기간이 2년 이상인가요?", "실거주 기간이 2년 이상인가요?"
- 잘못된 예: "거주기간이 얼마나 되나요?", "취득일이 언제인가요?" (개방형 질문 절대 금지)
- 수치/기간 요건은 반드시 임계값과 함께 이진 질문으로 작성하세요.

JSON만 응답:
{{
  "적용_검토_규정": [{{"규정명": "...", "근거조문": "...", "적용가능성": "높음/보통/낮음"}}],
  "필수요건": [{{"req_name": "...", "data_field": "영문snake_case", "data_type": "boolean/number/duration/date/text", "threshold": "...", "legal_basis": "...", "priority": "critical/important/optional", "question_hint": "예/아니오로 답할 수 있는 이진 질문"}}]
}}
필수요건 5~8개. question_hint는 모두 이진 질문."""

    r1 = model.generate_content(s1_prompt)
    t1 = strip_json(r1.text)
    try:
        s1r = json.loads(t1)
    except:
        return {"status": "error", "message": "요건 추출 실패: " + t1[:200]}

    reqs  = s1r.get("필수요건", [])
    provs = s1r.get("적용_검토_규정", [])

    applicable_law = provs[0].get("규정명", "") if provs else ""
    law_summary    = provs[0].get("근거조문", "") if provs else ""

    # ── Stage 2: 결정론적 Gap Detection (LLM 미호출) ──
    gap = run_gap_detection(
        reqs,
        payload.structured_data.model_dump(),
        [p.get("규정명", "") for p in provs]
    )

    needs = gap.gap_items + gap.ambiguous_items
    needs.sort(key=lambda x: {"critical": 0, "important": 1, "optional": 2}.get(x.get("priority", ""), 2))
    top = needs[:5]

    # checklist 구성
    checklist = []
    for p in provs:
        fields_for_prov = [
            item.get("data_field", "")
            for item in needs
            if item.get("legal_basis", "") and p.get("근거조문", "") and
               item.get("legal_basis", "") in p.get("근거조문", "")
        ]
        if not fields_for_prov:
            fields_for_prov = [item.get("data_field", "") for item in top]
        checklist.append({
            "alias":  p.get("규정명", ""),
            "fields": fields_for_prov[:5]
        })

    missing_variables = [item.get("data_field", "") for item in top]

    # ── Stage 3: 표적 질문 생성 (LLM) ──
    s3_prompt = f"""{role_prompt}
고객 상황: {llm_context}
미확인 항목: {json.dumps(top, ensure_ascii=False, indent=2)}

★ 절대 규칙 (위반 금지):
1. 모든 질문은 반드시 "예/아니오"로만 답할 수 있어야 합니다.
2. 수치·날짜·기간을 직접 묻는 개방형 질문은 절대 생성하지 마세요.
   - 금지: "거주기간이 얼마나 되나요?", "취득일이 언제인가요?"
   - 허용: "실거주 기간이 2년 이상인가요?", "취득일이 양도일로부터 2년 이전인가요?"
3. threshold가 있는 요건은 그 임계값을 질문에 명시하세요.
4. 질문은 일반인이 이해하기 쉬운 평어체로 작성하세요.
5. 각 질문에 category, priority, legal_basis, description, input_type을 포함하세요.

JSON만 응답:
{{
  "questions": [
    {{
      "variable": "data_field명",
      "question": "예/아니오로 답할 수 있는 질문",
      "category": "카테고리명",
      "priority": "critical/important/optional",
      "legal_basis": "근거조문",
      "description": "이 질문이 중요한 이유 1~2문장",
      "input_type": "yes_no"
    }}
  ]
}}
정확히 {len(top)}개."""

    r3 = model.generate_content(s3_prompt)
    t3 = strip_json(r3.text)
    try:
        s3r       = json.loads(t3)
        questions = s3r.get("questions", [])
    except:
        questions = [
            {
                "variable":    item.get("data_field", ""),
                "question":    item.get("question_hint", ""),
                "category":    item.get("req_name", ""),
                "priority":    item.get("priority", "important"),
                "legal_basis": item.get("legal_basis", ""),
                "description": "",
                "input_type":  "yes_no"
            }
            for item in top
        ]

    return {
        "status":            "need_more_info",
        "message":           "정확한 분석을 위해 추가 정보가 필요합니다.",
        "applicable_law":    applicable_law,
        "law_summary":       law_summary,
        "gap_analysis": {
            "completeness_ratio": gap.completeness_ratio,
            "satisfied_count":    gap.satisfied_count,
            "gap_count":          gap.gap_count,
            "ambiguous_count":    gap.ambiguous_count,
        },
        "checklist":          checklist,
        "missing_variables":  missing_variables,
        "questions":          questions,
        "_debug": {
            "search_query":       search_query,
            "prior_errors_count": len(prior_errors),
            "stage1_provisions":  len(provs),
            "stage2_total_reqs":  gap.total_requirements,
        }
    }

# ============================================================
# [순서 4] POST /confirm — HITL 전처리
# ============================================================
@app.post("/confirm")
async def confirm(payload: ConfirmRequest):
    model = get_gemini()

    # original_request에서 AnalyzeRequest 복원
    orig = payload.original_request or {}
    tax_type = orig.get("tax_category", {}).get("type", "모름")
    role_prompt = get_role_prompt(tax_type)

    # structured_data 복원
    sd_raw = orig.get("structured_data", {})
    cd_raw = orig.get("calculated_data", {})

    # ── DataQualityMetrics 산출 ──
    # completeness_before: 체크리스트 응답률(60%) + 추가정보 입력률(40%)
    total_checklist = len(payload.checklist_answers)
    answered_checklist = sum(1 for a in payload.checklist_answers if a.answer.strip())
    checklist_ratio = answered_checklist / total_checklist if total_checklist > 0 else 0

    # 추가정보 = structured_data에서 "모름"이 아닌 필드 비율
    flat_sd = flatten_user_data(sd_raw)
    total_fields = len(flat_sd) if flat_sd else 1
    filled_fields = sum(1 for v in flat_sd.values() if v and v not in ["모름", "", "0", "None", "null"])
    info_ratio = filled_fields / total_fields

    completeness_before = round(checklist_ratio * 0.6 + info_ratio * 0.4, 2)

    # user_corrections 보너스 (건당 5%, 최대 20%)
    corrections = payload.user_corrections or {}
    correction_bonus = min(len(corrections) * 0.05, 0.20)
    completeness_after = round(min(completeness_before + correction_bonus, 1.0), 2)
    quality_delta = round(completeness_after - completeness_before, 2)

    # ── FieldDiff 기록 ──
    field_diffs = []
    for field, user_value in corrections.items():
        ai_value = flat_sd.get(field, "(없음)")
        if ai_value == user_value:
            mod_type = "unchanged"
        elif ai_value in ["모름", "", "(없음)", "0", "None", "null"]:
            mod_type = "supplemented"
        else:
            mod_type = "corrected"
        field_diffs.append({
            "field":             field,
            "ai_value":          ai_value,
            "user_value":        user_value,
            "modification_type": mod_type,
        })

    human_corrections_count = sum(1 for d in field_diffs if d["modification_type"] == "corrected")

    # LLM 컨텍스트 구성
    llm_lines = []
    if sd_raw:
        llm_lines.append(f"[자산정보] {json.dumps(sd_raw, ensure_ascii=False)}")
    if cd_raw:
        estimated_total = cd_raw.get("estimated_total_tax", 0)
        if estimated_total > 0:
            llm_lines.append(f"[1차예상세액(절세비교기준)] 총 {estimated_total:,}원")
    if payload.checklist_answers:
        answers_str = json.dumps(
            [{"variable": a.variable, "answer": a.answer} for a in payload.checklist_answers],
            ensure_ascii=False, indent=2
        )
        llm_lines.append(f"[체크리스트 답변]\n{answers_str}")
    if corrections:
        llm_lines.append(f"[사용자 수정사항] {json.dumps(corrections, ensure_ascii=False)}")
    ctx_text = orig.get("unstructured_data", {}).get("user_context", "")
    if ctx_text:
        llm_lines.append(f"[사용자상황] {ctx_text}")

    prompt = f"""{role_prompt}
고객 상황:
{chr(10).join(llm_lines)}

위 정보를 종합하여 사실관계를 정리하세요.
반드시 아래 JSON 형식으로만 응답하세요:
{{
  "사실관계_요약": {{
    "양도자_현황": "양도자의 주택 보유 및 거주 상황 요약",
    "양도대상_자산": "양도 대상 자산 정보",
    "적용_검토_규정": "적용 가능한 비과세/감면 규정",
    "핵심_쟁점": "판단이 필요한 핵심 쟁점"
  }},
  "요건_충족_판단": [
    {{"요건": "요건명", "충족여부": "충족/미충족/확인필요", "근거": "판단 근거"}}
  ]
}}"""

    r  = model.generate_content(prompt)
    t  = strip_json(r.text)
    try:
        gr = json.loads(t)
    except:
        gr = {"사실관계_요약": {"원문": t[:500]}, "요건_충족_판단": []}

    return {
        "session_id": payload.session_id,
        "사실관계":   gr,
        "data_quality": {
            "completeness_before":      completeness_before,
            "completeness_after":       completeness_after,
            "quality_delta":            quality_delta,
            "human_corrections_count":  human_corrections_count,
            "field_diffs":              field_diffs,
        },
        "안내": "위 사실관계가 맞으면 확인 버튼을 눌러 최종 리포트를 생성합니다.",
    }

# ============================================================
# [순서 5] POST /report — 최종 분석 리포트
# ============================================================
@app.post("/report")
async def report(payload: AnalyzeRequest):
    model       = get_gemini()
    llm_context = build_llm_context(payload)
    add         = payload.additional_data or {}
    cd          = payload.calculated_data
    role_prompt = get_role_prompt(payload.tax_category.type)

    # 세액 비교 기준
    base_tax = cd.estimated_total_tax if cd.estimated_total_tax > 0 else 0
    tax_baseline = ""
    if base_tax > 0:
        tax_parts = [f"현재 예상세액 {base_tax:,}원"]
        if cd.estimated_yangdo_tax:
            tax_parts.append(f"양도세 {cd.estimated_yangdo_tax:,}원")
        if cd.estimated_local_tax:
            tax_parts.append(f"지방세 {cd.estimated_local_tax:,}원")
        tax_baseline = (
            f"\n[절세비교기준] {' + '.join(tax_parts)}"
            f"\n→ 비과세/감면 적용 시 절감액을 반드시 구체적 수치로 제시.\n"
            f"→ 리포트에 '기존 예상 세액: {base_tax:,}원' 대비 "
            f"'절세 적용 후 예상 세액' 비교를 반드시 포함할 것.\n"
        )

    fact_ctx = ""
    if add.get("fact_summary"):
        fact_ctx = f"\n[확인된 사실관계]\n{add['fact_summary']}\n"

    checklist_ctx = ""
    if add.get("checklist_answers"):
        checklist_ctx = f"\n[체크리스트 최종 답변]\n{json.dumps(add['checklist_answers'], ensure_ascii=False, indent=2)}\n"

    # 오답노트 선행참조
    prior_errors  = lookup_prior_errors(build_search_query(payload))
    error_context = ""
    if prior_errors:
        lessons = [e.get("교훈", "") for e in prior_errors if e.get("교훈")]
        if lessons:
            error_context = f"\n[과거오류교훈 반드시 반영]\n{json.dumps(lessons, ensure_ascii=False)}\n"

    prompt = f"""{role_prompt} 최종 분석 리포트를 작성하세요.

세무사 3대 원칙:
1. 리스크 시각화: 미충족 요건의 구체적 위험을 수치와 함께 제시
2. 전략적 생략: 확실히 충족된 요건은 간략히, 쟁점에 집중
3. 인사이트 강화: 단순 법령 나열이 아닌 실무적 조언 제공

고객 상황:
{llm_context}
{fact_ctx}{checklist_ctx}{tax_baseline}{error_context}

프론트엔드 간이 계산 1차 예상 세액: {base_tax:,}원

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "result_type": "PASS 또는 FAIL 또는 REVIEW",
  "세액비교": {{
    "기존_예상세액": {base_tax},
    "절세_적용후_세액": 0,
    "절감액": 0,
    "절감율": "0%"
  }},
  "applicable_law": "적용 법령명",
  "law_summary": "법령 핵심 내용 1~2문장",
  "details": "판단 근거를 3~5문장으로 설명. 적용 조문과 판단 이유를 포함하되, 긍정적 측면을 먼저 설명.",
  "risk_warning": "미충족 또는 불확실한 요건을 2~3문장으로 설명.",
  "tax_saving": "특례 조문 적용 시 절세 효과 요약",
  "예상세액": {{
    "비과세_적용시": "비과세 적용 시 세액",
    "일반과세_적용시": "일반 과세 시 세액"
  }},
  "판단근거": ["근거1", "근거2"],
  "관련예판": ["예판1"],
  "리스크": ["리스크1"],
  "종합의견": "최종 종합 의견 2~3문장",
  "confidence_pct": 0,
  "tax_after_applied": 0
}}
result_type: PASS(요건 충족), FAIL(미충족 명확), REVIEW(추가검토 필요)
세액비교의 절세_적용후_세액, 절감액은 원 단위 정수. 절감율은 백분율 문자열.
confidence_pct와 tax_after_applied는 반드시 숫자(정수)로만 반환."""

    r  = model.generate_content(prompt)
    t  = strip_json(r.text)
    try:
        gr = json.loads(t)
    except:
        gr = {
            "result_type": "REVIEW", "details": t[:300], "risk_warning": "",
            "applicable_law": "", "law_summary": "", "tax_saving": "",
            "confidence_pct": 50, "tax_after_applied": 0,
            "세액비교": {"기존_예상세액": base_tax, "절세_적용후_세액": 0, "절감액": 0, "절감율": "0%"},
        }

    # 타입 보정
    try:
        conf_pct = int(str(gr.get("confidence_pct", 50)).replace("%", "").strip())
    except:
        conf_pct = 50
    conf_pct = max(0, min(100, conf_pct))

    try:
        tax_after = int(str(gr.get("tax_after_applied", 0)).replace(",", "").replace("원", "").strip())
    except:
        tax_after = 0

    # 세액비교 보정
    세액비교 = gr.get("세액비교", {})
    if not 세액비교:
        세액비교 = {
            "기존_예상세액": base_tax,
            "절세_적용후_세액": tax_after,
            "절감액": base_tax - tax_after if base_tax > 0 else 0,
            "절감율": f"{round((base_tax - tax_after) / base_tax * 100)}%" if base_tax > 0 else "0%"
        }

    return {
        "status":            "success",
        "result_type":       gr.get("result_type", "REVIEW"),
        "리포트": {
            "세액비교":      세액비교,
            "예상세액":      gr.get("예상세액", {}),
            "판단근거":      gr.get("판단근거", []),
            "관련예판":      gr.get("관련예판", []),
            "리스크":        gr.get("리스크", []),
            "종합의견":      gr.get("종합의견", ""),
        },
        "applicable_law":    gr.get("applicable_law", ""),
        "law_summary":       gr.get("law_summary", ""),
        "details":           gr.get("details", ""),
        "risk_warning":      gr.get("risk_warning", ""),
        "tax_saving":        gr.get("tax_saving", ""),
        "confidence_pct":    conf_pct,
        "tax_after_applied": tax_after,
        "base_tax":          base_tax,
        "면책안내":           "본 분석은 참고용이며, 정확한 세액은 세무사의 최종 검토가 필요합니다.",
        "상담문의": {
            "안내":         "더 정확한 상담을 원하시면 아래 버튼을 눌러주세요.",
            "카카오톡채널": KAKAO_CHANNEL_URL,
        }
    }

# ============================================================
# [순서 6] POST /feedback — 3-Stage Triage + 오답노트
# ============================================================
async def triage_feedback(feedback_text: str, original_report: dict, rating: int) -> TriageResult:
    model = get_gemini()
    report_summary = json.dumps(original_report, ensure_ascii=False)[:2000]
    p = f"""세무 AI 품질관리 분석가.
[AI리포트] {report_summary}
[피드백] "{feedback_text}"
[평점] {rating}/5

판정 기준:
- actionable: 사실관계 오류 지적, 누락 정보 보충, 법령 해석 이의
- emotional: 구체적 보정 정보 없이 감정만 표현
- ambiguous: 유용 정보 있을 수 있으나 자동 판단 어려움

JSON만:
{{"classification":"actionable/emotional/ambiguous","confidence":0.85,"reason":"판정 이유","has_factual_correction":true,"has_missing_info":false,"has_legal_dispute":false}}"""

    r = model.generate_content(p)
    t = strip_json(r.text)
    try:
        return TriageResult(**json.loads(t))
    except:
        return TriageResult(classification="ambiguous", confidence=0.0, reason="파싱실패")

async def extract_deltas(feedback_text: str, original_report: dict) -> tuple:
    model = get_gemini()
    report_summary = json.dumps(original_report, ensure_ascii=False)[:2000]
    p = f"""세무 AI 오류 분석가.
[AI리포트] {report_summary}
[피드백] "{feedback_text}"

AI 리포트의 어떤 항목이 틀렸는지 구조화하세요.

JSON만:
{{"deltas":[{{"error_field":"틀린필드경로","ai_judgment":"AI가 판단한 내용","user_correction":"사용자가 지적한 내용","lesson_learned":"향후 반영할 교훈","error_type":"factual_error/missing_info/legal_interpretation/other"}}],"embed_summary":"1~2문장 요약"}}"""

    r = model.generate_content(p)
    t = strip_json(r.text)
    try:
        res = json.loads(t)
        return res.get("deltas", []), res.get("embed_summary", "")
    except:
        return [], ""

def save_error_note_gcs(en: ErrorNote) -> bool:
    try:
        gcs    = storage.Client()
        bucket = gcs.bucket(BUCKET_NAME)
        blob   = bucket.blob(f"오답노트/{en.note_id}.json")
        blob.upload_from_string(
            json.dumps(en.model_dump(), ensure_ascii=False, indent=2),
            content_type="application/json"
        )
        return True
    except:
        return False

def save_error_note_pinecone(en: ErrorNote) -> bool:
    try:
        idx = get_index()
        idx.upsert_records(namespace="error_notes", records=[{
            "_id":               en.note_id,
            "text":              en.embed_text,
            "original_question": en.original_question,
            "keywords":          json.dumps(en.keywords, ensure_ascii=False),
            "ai_판정":           en.ai_비과세_판정,
            "오류유형":          json.dumps([d.get("error_type", "") for d in en.deltas], ensure_ascii=False),
            "교훈":              json.dumps([d.get("lesson_learned", "") for d in en.deltas], ensure_ascii=False),
            "timestamp":         en.timestamp,
        }])
        return True
    except:
        return False

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    sid = req.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    nid = f"err_{uuid.uuid4().hex[:12]}"
    ts  = datetime.utcnow().isoformat()

    # Stage 1: Feedback Triage
    triage = await triage_feedback(req.feedback_text, req.original_report, req.rating)

    # emotional + rating 3점 이하 → 저장 제외
    if triage.classification == "emotional":
        return {
            "triage_result":       "emotional",
            "confidence":          triage.confidence,
            "saved_to_error_notes": False,
            "message":             "더 정확한 분석을 위해 세무 전문가 상담을 추천드립니다." if req.rating <= 3
                                   else "소중한 의견 감사합니다.",
            "kakao_channel":       KAKAO_CHANNEL_URL if req.rating <= 3 else None,
        }

    # Stage 2: Delta Extraction (actionable / ambiguous)
    deltas, embed_summary = await extract_deltas(req.feedback_text, req.original_report)

    # Stage 3: Error Note Persistence
    # 원본 리포트에서 질문/키워드 추출
    original_question = req.original_report.get("종합의견", req.feedback_text[:100])
    keywords = list(extract_keywords(req.original_report))

    ai_b = ""
    if req.original_report:
        예상 = req.original_report.get("예상세액", {})
        if isinstance(예상, dict):
            ai_b = 예상.get("비과세_적용시", "")
        if not ai_b:
            ai_b = req.original_report.get("종합의견", "")

    en = ErrorNote(
        note_id=nid,
        session_id=sid,
        timestamp=ts,
        original_question=original_question,
        keywords=keywords,
        ai_report_summary=json.dumps(req.original_report, ensure_ascii=False)[:1000],
        ai_비과세_판정=ai_b,
        feedback_text=req.feedback_text,
        triage=triage.model_dump(),
        deltas=deltas,
        embed_text=embed_summary if embed_summary else f"{original_question} | {req.feedback_text[:200]}"
    )

    gs = save_error_note_gcs(en)
    ps = save_error_note_pinecone(en)

    return {
        "triage_result":        triage.classification,
        "confidence":           triage.confidence,
        "deltas_extracted":     len(deltas),
        "saved_to_error_notes": gs or ps,
        "message":              "피드백이 반영되었습니다. 향후 유사 사례에서 이 교훈이 자동으로 참조됩니다.",
        "_detail": {
            "gcs":           "성공" if gs else "실패",
            "pinecone":      "성공" if ps else "실패",
            "needs_review":  triage.classification == "ambiguous",
        }
    }

# ============================================================
# [순서 7] 테스트 엔드포인트
# ============================================================

# GET /test — Phase 1 (AnalyzeRequest 샘플, LLM 미호출)
@app.get("/test")
async def test_analyze():
    dummy = AnalyzeRequest(
        session_id="test_analyze_001",
        tax_category=TaxCategory(type="양도소득세"),
        structured_data=StructuredData(
            asset_info=AssetInfo(asset_type="아파트", address="서울시 강남구 역삼동 123-45", area_size="85㎡ 이하"),
            price_info=PriceInfo(buy_price=500000000, sell_price=1500000000),
            date_info=DateInfo(buy_date="2015-05-20", sell_date="2026-04-10"),
            condition_info=ConditionInfo(is_regulated_area="여", house_count="1주택", residence_period="2년+")
        ),
        calculated_data=CalculatedData(
            estimated_total_tax=200145000,
            estimated_yangdo_tax=181950000,
            estimated_local_tax=18195000
        ),
        unstructured_data=UnstructuredData(user_context="올해 결혼을 하면서 2주택이 되었어요.")
    )
    return await analyze(dummy)

# GET /test-gap — Gap Detection Engine 단독 테스트 (LLM 미호출)
@app.get("/test-gap")
async def test_gap():
    sample_requirements = [
        {"req_name": "보유기간 2년 이상", "data_field": "residence_period", "data_type": "duration", "threshold": "2년 이상", "legal_basis": "소득세법 제89조", "priority": "critical", "question_hint": "보유기간이 2년 이상인가요?"},
        {"req_name": "1세대 1주택", "data_field": "house_count", "data_type": "text", "threshold": "", "legal_basis": "소득세법 제89조", "priority": "critical", "question_hint": "1세대 1주택자인가요?"},
        {"req_name": "조정대상지역 여부", "data_field": "is_regulated_area", "data_type": "boolean", "threshold": "", "legal_basis": "소득세법 시행령 제154조", "priority": "important", "question_hint": "조정대상지역 소재 주택인가요?"},
        {"req_name": "고가주택 여부", "data_field": "sell_price", "data_type": "number", "threshold": "1200000000 이하", "legal_basis": "소득세법 시행령 제156조의2", "priority": "important", "question_hint": "양도가액이 12억원 이하인가요?"},
        {"req_name": "배우자 유무", "data_field": "has_spouse", "data_type": "boolean", "threshold": "", "legal_basis": "소득세법 시행령 제152조의3", "priority": "optional", "question_hint": "배우자가 있으신가요?"},
    ]
    sample_user_data = {
        "asset_info": {"asset_type": "아파트", "address": "서울시 강남구", "area_size": "85㎡ 이하"},
        "price_info": {"buy_price": 500000000, "sell_price": 1500000000},
        "date_info":  {"buy_date": "2015-05-20", "sell_date": "2026-04-10"},
        "condition_info": {"is_regulated_area": "여", "house_count": "1주택", "residence_period": "2년+", "has_spouse": "모름"},
    }
    result = run_gap_detection(sample_requirements, sample_user_data, ["소득세법 제89조 비과세"])
    return {
        "test":   "gap_detection_engine",
        "result": result.model_dump(),
    }

# GET /test-questions — 3-Stage Gap Analysis 전체 테스트 (LLM 호출)
@app.get("/test-questions")
async def test_questions():
    dummy = AnalyzeRequest(
        session_id="test_questions_001",
        tax_category=TaxCategory(type="양도소득세"),
        structured_data=StructuredData(
            asset_info=AssetInfo(asset_type="아파트", address="서울시 강남구 역삼동 123-45", area_size="85㎡ 이하"),
            price_info=PriceInfo(buy_price=500000000, sell_price=1500000000),
            date_info=DateInfo(buy_date="2015-05-20", sell_date="2026-04-10"),
            condition_info=ConditionInfo(is_regulated_area="여", house_count="1주택", residence_period="2년+")
        ),
        calculated_data=CalculatedData(
            estimated_total_tax=200145000,
            estimated_yangdo_tax=181950000,
            estimated_local_tax=18195000
        ),
        unstructured_data=UnstructuredData(user_context="올해 결혼을 하면서 2주택이 되었어요. 기존 아파트는 언제 팔아야 하나요?")
    )
    return await generate_questions(dummy)

# GET /test-confirm — HITL 전처리 + 사실관계 테스트 (LLM 호출)
@app.get("/test-confirm")
async def test_confirm():
    dummy = ConfirmRequest(
        session_id="test_confirm_001",
        original_request={
            "session_id": "test_confirm_001",
            "tax_category": {"type": "양도소득세"},
            "structured_data": {
                "asset_info": {"asset_type": "아파트", "address": "서울시 강남구 역삼동 123-45", "area_size": "85㎡ 이하"},
                "price_info": {"buy_price": 500000000, "sell_price": 1500000000},
                "date_info": {"buy_date": "2015-05-20", "sell_date": "2026-04-10"},
                "condition_info": {"is_regulated_area": "여", "house_count": "1주택", "residence_period": "2년+"}
            },
            "calculated_data": {"estimated_total_tax": 200145000, "estimated_yangdo_tax": 181950000, "estimated_local_tax": 18195000},
            "unstructured_data": {"user_context": "부모님을 모시려고 합가했는데 아파트를 팔면 비과세 되나요?"}
        },
        checklist_answers=[
            ChecklistAnswer(variable="merge_reason", answer="동거봉양"),
            ChecklistAnswer(variable="years_since_merge", answer="3년"),
            ChecklistAnswer(variable="sole_house_before_merge", answer="예"),
        ],
        user_corrections={
            "residence_period": "3년 6개월"
        }
    )
    return await confirm(dummy)

# GET /test-report — 최종 리포트 테스트 (LLM 호출)
@app.get("/test-report")
async def test_report():
    dummy = AnalyzeRequest(
        session_id="test_report_001",
        tax_category=TaxCategory(type="양도소득세"),
        structured_data=StructuredData(
            asset_info=AssetInfo(asset_type="아파트", address="서울시 강남구 역삼동 123-45"),
            price_info=PriceInfo(buy_price=500000000, sell_price=1500000000),
            condition_info=ConditionInfo(house_count="1주택", is_regulated_area="여", residence_period="2년+")
        ),
        calculated_data=CalculatedData(
            estimated_total_tax=200145000,
            estimated_yangdo_tax=181950000,
            estimated_local_tax=18195000
        ),
        unstructured_data=UnstructuredData(user_context="부모님 동거봉양 합가 후 아파트 양도"),
        additional_data={
            "fact_summary": "1주택 보유자가 직계존속 동거봉양 목적으로 합가하여 일시적 2주택 상태에서 본인 주택 양도",
            "checklist_answers": [
                {"variable": "merge_reason", "answer": "동거봉양"},
                {"variable": "years_since_merge", "answer": "3년"}
            ]
        }
    )
    return await report(dummy)

# GET /test-feedback-actionable — actionable 피드백 테스트 (LLM 호출)
@app.get("/test-feedback-actionable")
async def test_feedback_actionable():
    dummy = FeedbackRequest(
        session_id="test_fb_action_001",
        original_report={
            "result_type": "PASS",
            "세액비교": {"기존_예상세액": 200145000, "절세_적용후_세액": 0, "절감액": 200145000},
            "종합의견": "동거봉양 합가 특례에 의한 비과세 적용 가능",
            "예상세액": {"비과세_적용시": "0원"}
        },
        feedback_text="합가 전에 이미 분양권이 하나 더 있었어요. 그래서 합가 전에도 2주택이었습니다.",
        rating=2
    )
    return await feedback(dummy)

# GET /test-feedback-emotional — emotional 피드백 테스트 (LLM 호출)
@app.get("/test-feedback-emotional")
async def test_feedback_emotional():
    dummy = FeedbackRequest(
        session_id="test_fb_emot_001",
        original_report={
            "result_type": "FAIL",
            "종합의견": "비과세 요건 미충족"
        },
        feedback_text="이게 맞아? 너무 실망이네... 세금이 너무 많아요 ㅠㅠ",
        rating=1
    )
    return await feedback(dummy)

# GET /test-feedback-ambiguous — ambiguous 피드백 테스트 (LLM 호출)
@app.get("/test-feedback-ambiguous")
async def test_feedback_ambiguous():
    dummy = FeedbackRequest(
        session_id="test_fb_ambig_001",
        original_report={
            "result_type": "REVIEW",
            "종합의견": "추가 검토 필요"
        },
        feedback_text="제가 듣기로는 5년 안에 팔면 된다고 하던데 맞나요?",
        rating=3
    )
    return await feedback(dummy)

# GET /test-prior-errors — 오답 노트 선행 참조 테스트 (LLM 미호출)
@app.get("/test-prior-errors")
async def test_prior_errors():
    results = lookup_prior_errors("부모님 동거봉양 합가 양도세 비과세")
    return {
        "test":           "prior_error_lookup",
        "발견된_과거_오류": results,
        "count":           len(results),
    }

# GET /test-gemini — Gemini 환경변수 확인 (LLM 미호출)
@app.get("/test-gemini")
async def test_gemini():
    return {
        "project":  GCP_PROJECT_ID or "EMPTY",
        "location": GCP_LOCATION or "EMPTY",
        "model":    GEMINI_MODEL or "EMPTY",
        "pinecone_index": PINECONE_INDEX_NAME or "EMPTY",
        "gcs_bucket":     BUCKET_NAME or "EMPTY",
    }
