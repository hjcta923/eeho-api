import os
import re
import uuid
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel, Field
import json
from datetime import datetime
from typing import Optional
from pinecone import Pinecone
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel

app = FastAPI(title="EEHO AI API", version="0.5")

PINECONE_API_KEY   = os.environ.get("PINECONE_API_KEY", "")
BUCKET_NAME        = os.environ.get("GCS_BUCKET_NAME", "")
GUIDE_PATH         = os.environ.get("GUIDE_PATH", "")
GCP_PROJECT_ID     = os.environ.get("GCP_PROJECT_ID", "")
GCP_LOCATION       = os.environ.get("GCP_LOCATION", "")
GEMINI_MODEL       = os.environ.get("GEMINI_MODEL", "")
PINECONE_INDEX_NAME= os.environ.get("PINECONE_INDEX_NAME", "")
KAKAO_CHANNEL_URL  = os.environ.get("KAKAO_CHANNEL_URL", "")

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

TAX_KEYWORDS = [
    "1세대1주택","비과세","동거봉양","합가","혼인","상속주택","일시적2주택","장애인","직계존속",
    "5년","5년이내","60세","다주택자","중과세","재개발","재건축","농어촌주택","부담부증여",
    "고가주택","비사업용토지","비사업용","장기보유특별공제","장특공","양도시기","취득시기",
    "세대","주택수","보유기간","거주기간","2주택","3주택","조합원입주권","분양권","겸용주택",
    "다가구주택","상속","증여","대물변제","교환","문화재주택","임대주택","자경농지","출국세",
    "비거주자","실거래가","기준시가","필요경비","양도차익","예정신고","확정신고","경정청구",
    "가산세","조정대상지역","중과배제","사업용의제"
]

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

# ============================================================
# ★ 프론트엔드 Payload 수신 모델 (buildPayload 구조와 1:1 매칭)
# ============================================================
class TaxCategory(BaseModel):
    type: str = "모름"

class AssetInfo(BaseModel):
    asset_type: str = "모름"
    address:    str = "모름"
    area_size:  str = "모름"

class PriceInfo(BaseModel):
    buy_price:  float = 0
    sell_price: float = 0

class DateInfo(BaseModel):
    buy_date:  str = "모름"
    sell_date: str = "모름"

class ConditionInfo(BaseModel):
    is_regulated_area: str = "모름"
    house_count:       str = "모름"
    residence_period:  str = "모름"

class StructuredData(BaseModel):
    asset_info:     AssetInfo     = AssetInfo()
    price_info:     PriceInfo     = PriceInfo()
    date_info:      DateInfo      = DateInfo()
    condition_info: ConditionInfo = ConditionInfo()

class CalculatedData(BaseModel):
    estimated_total_tax:         float = 0
    estimated_yangdo_tax:        float = 0
    estimated_local_tax:         float = 0
    estimated_surcharge:         float = 0
    estimated_acq_tax:           float = 0
    estimated_local_edu_tax:     float = 0
    estimated_rural_tax:         float = 0
    estimated_property_tax:      float = 0
    estimated_comprehensive_tax: float = 0
    estimated_gift_tax:          float = 0
    estimated_filing_deduction:  float = 0

class UnstructuredData(BaseModel):
    user_context: str = ""

class EEHOPayload(BaseModel):
    session_id:        str              = ""
    tax_category:      TaxCategory      = TaxCategory()
    structured_data:   StructuredData   = StructuredData()
    calculated_data:   CalculatedData   = CalculatedData()
    unstructured_data: UnstructuredData = UnstructuredData()
    additional_data:   dict             = {}

# ============================================================
# Gap Detection
# ============================================================
class GapResult(BaseModel):
    req_name: str; data_field: str; status: str
    user_value: str = ""; threshold: str = ""
    legal_basis: str = ""; priority: str = ""; question_hint: str = ""

class GapAnalysisResult(BaseModel):
    algorithm_stage: str = "deterministic_gap_detection"
    provisions_checked: list = []; total_requirements: int = 0
    satisfied_count: int = 0; gap_count: int = 0; ambiguous_count: int = 0
    completeness_ratio: float = 0.0
    gap_items: list = []; ambiguous_items: list = []; satisfied_items: list = []

def flatten_user_data(d):
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items(): flat[k2] = str(v2) if v2 is not None else ""
        else: flat[k] = str(v) if v is not None else ""
    return flat

def is_missing(v):
    return not v or v.strip() in ["", "모름", "null", "None", "없음", "미입력"]

def evaluate_requirement(req, user_value):
    if is_missing(user_value): return "gap"
    dt, th = req.get("data_type", "text"), req.get("threshold", "")
    if dt == "boolean":
        return "satisfied" if user_value.lower() in ["yes","true","예","여","y"] else "ambiguous"
    elif dt == "number" and th:
        try:
            val = float(user_value.replace(",","").replace("원",""))
            nums = re.findall(r'[\d.]+', th)
            if nums:
                threshold_val = float(nums[0])
                if "이하" in th: return "satisfied" if val <= threshold_val else "ambiguous"
                elif "이상" in th: return "satisfied" if val >= threshold_val else "ambiguous"
        except: pass
        return "ambiguous"
    elif dt == "duration" and th:
        un = re.findall(r'(\d+)', user_value)
        tn = re.findall(r'(\d+)', th)
        if un and tn:
            if "이상" in th: return "satisfied" if int(un[0]) >= int(tn[0]) else "ambiguous"
            elif "이내" in th: return "satisfied" if int(un[0]) <= int(tn[0]) else "ambiguous"
        return "ambiguous"
    return "ambiguous" if user_value and not is_missing(user_value) else "gap"

def run_gap_detection(ai_requirements, user_data, provisions_checked=None):
    flat = flatten_user_data(user_data)
    gap_items, ambiguous_items, satisfied_items = [], [], []
    for req in ai_requirements:
        df = req.get("data_field","")
        uv = flat.get(df,"")
        st = evaluate_requirement(req, uv)
        r  = GapResult(
            req_name=req.get("req_name",""), data_field=df, status=st,
            user_value=uv if not is_missing(uv) else "(미입력)",
            threshold=req.get("threshold",""), legal_basis=req.get("legal_basis",""),
            priority=req.get("priority","important"), question_hint=req.get("question_hint","")
        )
        if st == "gap":         gap_items.append(r.model_dump())
        elif st == "ambiguous": ambiguous_items.append(r.model_dump())
        else:                   satisfied_items.append(r.model_dump())
    total = len(ai_requirements)
    return GapAnalysisResult(
        provisions_checked=provisions_checked or [], total_requirements=total,
        satisfied_count=len(satisfied_items), gap_count=len(gap_items),
        ambiguous_count=len(ambiguous_items),
        completeness_ratio=round(len(satisfied_items)/total,2) if total>0 else 0,
        gap_items=gap_items, ambiguous_items=ambiguous_items, satisfied_items=satisfied_items
    )

# ============================================================
# 공통 유틸
# ============================================================
def lookup_prior_errors(question, top_k=2):
    try:
        idx     = get_index()
        results = idx.search_records(namespace="error_notes", query={"inputs":{"text":question},"top_k":top_k})
        return [
            {"note_id": h.get("_id",""), "유사도": round(h.get("_score",0),3),
             "원래질문": h["fields"].get("original_question",""),
             "교훈": h["fields"].get("교훈",""), "오류유형": h["fields"].get("오류유형","")}
            for h in results.get("result",{}).get("hits",[]) if h.get("_score",0) > 0.5
        ]
    except: return []

def build_search_query(payload: EEHOPayload) -> str:
    parts = []
    sd, ci = payload.structured_data, payload.structured_data.condition_info
    if payload.tax_category.type not in ["모름",""]:   parts.append(payload.tax_category.type)
    if sd.asset_info.asset_type not in ["모름",""]:    parts.append(sd.asset_info.asset_type)
    if sd.asset_info.address not in ["모름",""]:       parts.append(sd.asset_info.address)
    if ci.house_count not in ["모름",""]:              parts.append(f"{ci.house_count} 보유")
    if ci.is_regulated_area == "여":                   parts.append("조정대상지역")
    if ci.residence_period not in ["모름","없음",""]:  parts.append(f"거주기간 {ci.residence_period}")
    if sd.price_info.sell_price > 1_200_000_000:       parts.append("고가주택 12억 초과")
    ctx = payload.unstructured_data.user_context.strip()
    if ctx: parts.append(ctx)
    return " ".join(parts) if parts else "양도소득세 비과세 감면"

def build_llm_context(payload: EEHOPayload) -> str:
    sd, cd, ci = payload.structured_data, payload.calculated_data, payload.structured_data.condition_info
    lines = [
        f"[세목] {payload.tax_category.type}",
        f"[자산] {sd.asset_info.asset_type} | 주소: {sd.asset_info.address} | 면적: {sd.asset_info.area_size}",
        f"[가격] 취득가: {int(sd.price_info.buy_price):,}원 / 양도가: {int(sd.price_info.sell_price):,}원",
        f"[날짜] 취득일: {sd.date_info.buy_date} / 양도일: {sd.date_info.sell_date}",
        f"[조건] 조정지역: {ci.is_regulated_area} | 주택수: {ci.house_count} | 거주기간: {ci.residence_period}",
    ]
    if cd.estimated_total_tax > 0:
        lines.append(
            f"[1차예상세액] 총 {int(cd.estimated_total_tax):,}원"
            f" (양도세 {int(cd.estimated_yangdo_tax):,}원"
            f" + 지방세 {int(cd.estimated_local_tax):,}원"
            + (f" + 중과가산 {int(cd.estimated_surcharge):,}원" if cd.estimated_surcharge > 0 else "")
            + ")"
        )
    ctx = payload.unstructured_data.user_context.strip()
    if ctx: lines.append(f"[사용자상황] {ctx}")
    add = payload.additional_data
    if add: lines.append(f"[추가답변] {json.dumps(add, ensure_ascii=False)}")
    return "\n".join(lines)

def strip_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"): t = t[4:]
        t = t.strip()
    return t

# ============================================================
# 헬스체크
# ============================================================
@app.get("/")
def health():
    return {"service": "EEHO AI API", "version": "0.5", "status": "running"}

# ============================================================
# ★ POST /generate-questions
# 프론트 기대 응답:
# { status:"checklist", applicable_law, law_summary, questions:[{id, category, question}] }
# ============================================================
@app.post("/generate-questions")
async def generate_questions(payload: EEHOPayload):
    model        = get_gemini()
    search_query = build_search_query(payload)
    llm_context  = build_llm_context(payload)

    # Pinecone 판례 검색
    idx            = get_index()
    search_results = idx.search_records(
        namespace="tax_cases",
        query={"inputs": {"text": search_query}, "top_k": 3}
    )
    판례목록, 전체키워드 = [], set()
    for hit in search_results["result"]["hits"]:
        판례목록.append({
            "사건번호": hit["fields"].get("사건번호",""),
            "주제":     hit["fields"].get("주제",""),
            "결과":     hit["fields"].get("결과",""),
            "유사도":   round(hit["_score"],3),
            "판단근거": hit["fields"].get("판단근거",""),
            "관련법령": hit["fields"].get("관련법령",""),
        })
        전체키워드.update(extract_keywords(hit["fields"]))

    키워드리스트 = list(전체키워드)
    매칭결과     = match_guide_sections(키워드리스트, load_guide())

    # 오답노트 선행참조
    prior_errors  = lookup_prior_errors(search_query)
    error_context = ""
    if prior_errors:
        lessons = [e.get("교훈","") for e in prior_errors if e.get("교훈")]
        if lessons:
            error_context = f"\n[주의-과거오류교훈 반드시 반영]\n{json.dumps(lessons,ensure_ascii=False)}\n"

    # Stage 1: LLM 동적 요건 추출
    s1_prompt = f"""당신은 양도소득세 전문 세무사 AI입니다.
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
  "적용_검토_규정": [{{"규정명": "소득세법 시행령 제155조 ...", "근거조문": "...", "적용가능성": "높음"}}],
  "필수요건": [{{"req_name": "...", "data_field": "영문snake_case", "data_type": "boolean/number/duration/date/text", "threshold": "...", "legal_basis": "...", "priority": "critical/important/optional", "question_hint": "예/아니오로 답할 수 있는 이진 질문"}}]
}}
필수요건 5~8개. question_hint는 모두 이진 질문."""

    r1  = model.generate_content(s1_prompt)
    t1  = strip_json(r1.text)
    try: s1r = json.loads(t1)
    except: return {"status": "error", "message": "요건 추출 실패: " + t1[:200]}

    reqs  = s1r.get("필수요건", [])
    provs = s1r.get("적용_검토_규정", [])

    # 대표 적용 법령
    applicable_law = provs[0].get("규정명","") if provs else ""
    law_summary    = provs[0].get("근거조문","") if provs else ""

    # Stage 2: 결정론적 Gap Detection
    gap = run_gap_detection(reqs, payload.structured_data.model_dump(), [p.get("규정명","") for p in provs])

    needs = gap.gap_items + gap.ambiguous_items
    needs.sort(key=lambda x: {"critical":0,"important":1,"optional":2}.get(x.get("priority",""),2))
    top = needs[:5]

    # Stage 3: LLM 표적 질문 생성 (반드시 YES/NO 이진 질문)
    s3_prompt = f"""양도소득세 전문 세무사 AI.
고객 상황: {llm_context}
미확인 항목: {json.dumps(top, ensure_ascii=False, indent=2)}

★ 절대 규칙 (위반 금지):
1. 모든 질문은 반드시 "예/아니오"로만 답할 수 있어야 합니다.
2. 수치·날짜·기간을 직접 묻는 개방형 질문은 절대 생성하지 마세요.
   - 금지: "거주기간이 얼마나 되나요?", "취득일이 언제인가요?"
   - 허용: "실거주 기간이 2년 이상인가요?", "취득일이 양도일로부터 2년 이전인가요?"
3. threshold가 있는 요건은 그 임계값을 질문에 명시하세요.
   - 예: threshold=2년 → "보유기간이 2년 이상인가요?"
   - 예: threshold=1,200,000,000 → "양도가액이 12억원을 초과하나요?"
4. 질문은 일반인이 이해하기 쉬운 평어체로 작성하세요.

JSON만 응답:
{{
  "questions": [
    {{"id": "q1", "category": "카테고리명", "question": "예/아니오로 답할 수 있는 질문"}}
  ]
}}
정확히 {len(top)}개. id는 q1, q2, q3... 순서."""

    r3  = model.generate_content(s3_prompt)
    t3  = strip_json(r3.text)
    try:
        s3r       = json.loads(t3)
        questions = s3r.get("questions", [])
    except:
        questions = [
            {"id": f"q{i+1}", "category": item.get("req_name",""), "question": item.get("question_hint","")}
            for i, item in enumerate(top)
        ]

    # ★ 프론트 기대 형식으로 반환
    return {
        "status":         "checklist",
        "applicable_law": applicable_law,
        "law_summary":    law_summary,
        "questions":      questions,
        "_debug": {
            "search_query":       search_query,
            "completeness_ratio": gap.completeness_ratio,
            "gap_count":          gap.gap_count,
            "prior_errors_count": len(prior_errors),
        }
    }

# ============================================================
# ★ POST /confirm
# 프론트 기대 응답:
# { status:"confirm", fact_summary, applicable_law, law_summary,
#   requirement_check:[{요건, 충족여부, 근거}],
#   tax_impact:{before, after_pass, saving}, confidence }
# ============================================================
@app.post("/confirm")
async def confirm(payload: EEHOPayload):
    model       = get_gemini()
    llm_context = build_llm_context(payload)
    add         = payload.additional_data
    cd          = payload.calculated_data

    tax_ctx = ""
    if cd.estimated_total_tax > 0:
        tax_ctx = f"\n[1차예상세액(절세비교기준)] 총 {int(cd.estimated_total_tax):,}원\n"

    second_ctx = ""
    if add.get("is_second_round"):
        second_ctx = f"\n[보완 추가 상황] {add.get('supplement_text','')}\n"

    checklist_ctx = ""
    if add.get("checklist_answers"):
        checklist_ctx = f"\n[체크리스트 답변]\n{json.dumps(add['checklist_answers'], ensure_ascii=False, indent=2)}\n"

    prompt = f"""당신은 양도소득세 전문 세무사 AI입니다.
고객 상황:
{llm_context}
{tax_ctx}{second_ctx}{checklist_ctx}
위 정보를 종합하여 사실관계를 정리하고 요건 충족 여부를 판단하세요.
반드시 아래 JSON 형식으로만 응답하세요:
{{
  "fact_summary": "고객의 전체 상황을 2~3문장으로 요약",
  "applicable_law": "적용 검토 법령명 (예: 소득세법 시행령 제155조 제4항 동거봉양 합가 특례)",
  "law_summary": "해당 법령의 핵심 내용 1~2문장",
  "requirement_check": [
    {{"요건": "요건명", "충족여부": "충족/미충족/확인필요", "근거": "판단 근거 1문장"}}
  ],
  "tax_impact": {{
    "before": "현재 예상 세액 (예: 약 1억 8,720만원)",
    "after_pass": "비과세 적용 시 (예: 0원)",
    "saving": "절세 효과 (예: 약 1억 8,720만원 절감)"
  }},
  "confidence": "높음/보통/낮음"
}}"""

    r  = model.generate_content(prompt)
    t  = strip_json(r.text)
    try: gr = json.loads(t)
    except: gr = {}

    # ★ 프론트 기대 형식으로 반환
    return {
        "status":            "confirm",
        "fact_summary":      gr.get("fact_summary", ""),
        "applicable_law":    gr.get("applicable_law", ""),
        "law_summary":       gr.get("law_summary", ""),
        "requirement_check": gr.get("requirement_check", []),
        "tax_impact":        gr.get("tax_impact", {}),
        "confidence":        gr.get("confidence", "보통"),
    }

# ============================================================
# ★ POST /report
# 프론트 기대 응답:
# { status:"success", result_type:"PASS"|"FAIL"|"REVIEW",
#   details, risk_warning, applicable_law, law_summary, tax_saving }
# ============================================================
@app.post("/report")
async def report(payload: EEHOPayload):
    model       = get_gemini()
    llm_context = build_llm_context(payload)
    add         = payload.additional_data
    cd          = payload.calculated_data

    tax_baseline = ""
    if cd.estimated_total_tax > 0:
        tax_baseline = (
            f"\n[절세비교기준] 현재 예상세액 {int(cd.estimated_total_tax):,}원"
            f" (양도세 {int(cd.estimated_yangdo_tax):,}원"
            f" + 지방세 {int(cd.estimated_local_tax):,}원)"
            f"\n→ 비과세/감면 적용 시 절감액을 반드시 구체적 수치로 제시.\n"
        )

    fact_ctx = f"\n[확인된 사실관계]\n{add['fact_summary']}\n" if add.get("fact_summary") else ""

    checklist_ctx = ""
    if add.get("checklist_answers"):
        checklist_ctx = f"\n[체크리스트 최종 답변]\n{json.dumps(add['checklist_answers'], ensure_ascii=False, indent=2)}\n"

    prior_errors  = lookup_prior_errors(build_search_query(payload))
    error_context = ""
    if prior_errors:
        lessons = [e.get("교훈","") for e in prior_errors if e.get("교훈")]
        if lessons:
            error_context = f"\n[과거오류교훈 반드시 반영]\n{json.dumps(lessons,ensure_ascii=False)}\n"

    base_tax = int(cd.estimated_total_tax) if cd.estimated_total_tax > 0 else 0

    prompt = f"""당신은 양도소득세 전문 세무사 AI입니다. 최종 분석 리포트를 작성하세요.

고객 상황:
{llm_context}
{fact_ctx}{checklist_ctx}{tax_baseline}{error_context}
반드시 아래 JSON 형식으로만 응답하세요:
{{
  "result_type": "PASS 또는 FAIL 또는 REVIEW",
  "applicable_law": "적용 법령명",
  "law_summary": "법령 핵심 내용 1~2문장",
  "details": "판단 근거를 3~5문장으로 설명. 적용 조문과 판단 이유를 포함하되, 긍정적 측면을 먼저 설명.",
  "risk_warning": "미충족 또는 불확실한 요건을 2~3문장으로 설명. 충족하지 못한 구체적 요건명을 명시하고 해결 방법을 안내.",
  "tax_saving": "특례 조문 적용 시 절세 효과 요약 (예: 최대 약 1억 8,720만원 절감 가능)",
  "confidence_pct": 해당 조문이 실제 적용될 가능성을 0~100 사이 정수. 요건 모두 충족=85~95, 일부 미확인=40~70, 충족 어려움=10~35,
  "tax_after_applied": 해당 조문이 적용되었을 때의 예상 세액을 원 단위 정수로만 반환. 비과세면 0, 부분감면이면 감면 후 세액 숫자
}}
result_type: PASS(요건 충족), FAIL(미충족 명확), REVIEW(추가검토 필요)
confidence_pct와 tax_after_applied는 반드시 숫자(정수)로만 반환."""

    r  = model.generate_content(prompt)
    t  = strip_json(r.text)
    try: gr = json.loads(t)
    except: gr = {"result_type":"REVIEW","details":t[:300],"risk_warning":"","applicable_law":"","law_summary":"","tax_saving":"","confidence_pct":50,"tax_after_applied":0}

    # 타입 보정
    try:    conf_pct = int(str(gr.get("confidence_pct", 50)).replace("%","").strip())
    except: conf_pct = 50
    conf_pct = max(0, min(100, conf_pct))

    try:    tax_after = int(str(gr.get("tax_after_applied", 0)).replace(",","").replace("원","").strip())
    except: tax_after = 0

    # ★ 프론트 기대 형식으로 반환
    return {
        "status":            "success",
        "result_type":       gr.get("result_type", "REVIEW"),
        "applicable_law":    gr.get("applicable_law", ""),
        "law_summary":       gr.get("law_summary", ""),
        "details":           gr.get("details", ""),
        "risk_warning":      gr.get("risk_warning", ""),
        "tax_saving":        gr.get("tax_saving", ""),
        "confidence_pct":    conf_pct,
        "tax_after_applied": tax_after,
        "base_tax":          base_tax,
        "면책안내":           "본 분석은 참고용이며, 정확한 세액은 세무사의 최종 검토가 필요합니다.",
        "상담문의":           {"안내": "더 정확한 상담을 원하시면 아래 버튼을 눌러주세요.", "카카오톡채널": KAKAO_CHANNEL_URL}
    }

# ============================================================
# /feedback — 오답노트 저장
# ============================================================
class FeedbackRequest(BaseModel):
    session_id:    str  = ""
    question:      str
    feedback_text: str
    ai_report:     dict = {}
    ai_사실관계:   dict = {}
    추출키워드:    list = []

class TriageResult(BaseModel):
    classification: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    has_factual_correction: bool = False
    has_missing_info:       bool = False
    has_legal_dispute:      bool = False

class ErrorNote(BaseModel):
    note_id: str; session_id: str; timestamp: str
    original_question: str; keywords: list = []
    ai_report_summary: str = ""; ai_비과세_판정: str = ""
    feedback_text: str; triage: dict = {}; deltas: list = []; embed_text: str = ""

async def triage_feedback(feedback_text, question, ai_report):
    model = get_gemini()
    p = f"""세무 AI 품질관리 분석가.
[질문] "{question}"
[AI리포트] {json.dumps(ai_report,ensure_ascii=False)[:2000]}
[피드백] "{feedback_text}"
판정: actionable(사실오류/누락보충/법령이의), emotional(감정만), ambiguous(불명확)
JSON만: {{"classification":"...","confidence":0.85,"reason":"...","has_factual_correction":true,"has_missing_info":false,"has_legal_dispute":false}}"""
    r = model.generate_content(p)
    t = strip_json(r.text)
    try: return TriageResult(**json.loads(t))
    except: return TriageResult(classification="ambiguous", confidence=0.0, reason="파싱실패")

async def extract_deltas(feedback_text, question, ai_report, ai_사실관계):
    model = get_gemini()
    p = f"""세무 AI 오류 분석가.
[질문] "{question}"
[AI사실관계] {json.dumps(ai_사실관계,ensure_ascii=False)[:2000]}
[AI리포트] {json.dumps(ai_report,ensure_ascii=False)[:2000]}
[피드백] "{feedback_text}"
JSON만:
{{"deltas":[{{"error_field":"...","ai_judgment":"...","user_correction":"...","lesson_learned":"...","error_type":"factual_error/missing_info/legal_interpretation/other"}}],"embed_summary":"1~2문장 요약"}}"""
    r = model.generate_content(p)
    t = strip_json(r.text)
    try:
        res = json.loads(t)
        return res.get("deltas",[]), res.get("embed_summary","")
    except: return [], ""

def save_error_note_gcs(en):
    try:
        gcs    = storage.Client()
        bucket = gcs.bucket(BUCKET_NAME)
        blob   = bucket.blob(f"오답노트/{en.note_id}.json")
        blob.upload_from_string(json.dumps(en.model_dump(), ensure_ascii=False, indent=2), content_type="application/json")
        return True
    except: return False

def save_error_note_pinecone(en):
    try:
        idx = get_index()
        idx.upsert_records(namespace="error_notes", records=[{
            "_id": en.note_id, "text": en.embed_text,
            "original_question": en.original_question,
            "keywords":  json.dumps(en.keywords, ensure_ascii=False),
            "ai_판정":   en.ai_비과세_판정,
            "오류유형":  json.dumps([d.get("error_type","") for d in en.deltas], ensure_ascii=False),
            "교훈":      json.dumps([d.get("lesson_learned","") for d in en.deltas], ensure_ascii=False),
            "timestamp": en.timestamp
        }])
        return True
    except: return False

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    sid    = req.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    nid    = f"err_{uuid.uuid4().hex[:12]}"
    ts     = datetime.utcnow().isoformat()
    triage = await triage_feedback(req.feedback_text, req.question, req.ai_report)
    result = {"session_id":sid,"note_id":nid,"timestamp":ts,"stage1_triage":triage.model_dump()}

    if triage.classification == "emotional":
        result.update({"처리결과":"emotional_excluded","안내":"소중한 의견 감사합니다.","카카오톡채널":KAKAO_CHANNEL_URL,"stage2_delta":None,"stage3_저장":None})
        return result

    deltas, embed = await extract_deltas(req.feedback_text, req.question, req.ai_report, req.ai_사실관계)
    result["stage2_delta"] = {"deltas":deltas,"embed_summary":embed,"delta_count":len(deltas)}

    ai_b = ""
    if req.ai_report:
        e = req.ai_report.get("예상세액",{})
        if isinstance(e,dict): ai_b = e.get("비과세_적용시","")
        if not ai_b: ai_b = req.ai_report.get("종합의견","")

    en = ErrorNote(
        note_id=nid, session_id=sid, timestamp=ts,
        original_question=req.question, keywords=req.추출키워드,
        ai_report_summary=json.dumps(req.ai_report,ensure_ascii=False)[:1000] if req.ai_report else "",
        ai_비과세_판정=ai_b, feedback_text=req.feedback_text,
        triage=triage.model_dump(), deltas=deltas,
        embed_text=embed if embed else f"{req.question} | {req.feedback_text[:200]}"
    )
    gs = save_error_note_gcs(en)
    ps = save_error_note_pinecone(en)
    result["stage3_저장"] = {"gcs":"성공" if gs else "실패","pinecone":"성공" if ps else "실패","needs_review":triage.classification=="ambiguous"}
    result["처리결과"]    = "actionable_saved" if triage.classification=="actionable" else "ambiguous_saved"
    result["안내"]        = "소중한 의견이 반영되었습니다."
    return result

# ============================================================
# 테스트 엔드포인트
# ============================================================
@app.get("/test-generate-questions")
async def test_generate_questions():
    dummy = EEHOPayload(
        session_id="test_001",
        tax_category=TaxCategory(type="양도소득세"),
        structured_data=StructuredData(
            asset_info=AssetInfo(asset_type="아파트", address="서울시 강남구 역삼동 123-45", area_size="85㎡ 이하"),
            price_info=PriceInfo(buy_price=500000000, sell_price=1500000000),
            date_info=DateInfo(buy_date="2015-05-20", sell_date="2026-04-10"),
            condition_info=ConditionInfo(is_regulated_area="여", house_count="1주택", residence_period="2년+")
        ),
        calculated_data=CalculatedData(estimated_total_tax=187200000, estimated_yangdo_tax=144000000, estimated_local_tax=14400000),
        unstructured_data=UnstructuredData(user_context="올해 결혼을 하면서 2주택이 되었어요. 기존 아파트는 언제 팔아야 하나요?")
    )
    return await generate_questions(dummy)

@app.get("/test-confirm")
async def test_confirm():
    dummy = EEHOPayload(
        session_id="test_002",
        tax_category=TaxCategory(type="양도소득세"),
        structured_data=StructuredData(
            asset_info=AssetInfo(asset_type="아파트", address="서울시 강남구 역삼동 123-45"),
            price_info=PriceInfo(buy_price=500000000, sell_price=1500000000),
            condition_info=ConditionInfo(house_count="1주택", is_regulated_area="여", residence_period="2년+")
        ),
        calculated_data=CalculatedData(estimated_total_tax=187200000, estimated_yangdo_tax=144000000, estimated_local_tax=14400000),
        unstructured_data=UnstructuredData(user_context="부모님을 모시려고 합가했는데 아파트를 팔면 비과세 되나요?"),
        additional_data={
            "checklist_answers": [
                {"id":"q1","question":"합가로 2주택이 된 경우인가요?","answer":"예"},
                {"id":"q2","question":"합가일로부터 10년 이내 양도인가요?","answer":"예"},
                {"id":"q3","question":"양도 주택이 합가 전 유일한 주택이었나요?","answer":"예"}
            ],
            "applicable_law": "소득세법 시행령 제155조 제4항"
        }
    )
    return await confirm(dummy)

@app.get("/test-report")
async def test_report():
    dummy = EEHOPayload(
        session_id="test_003",
        tax_category=TaxCategory(type="양도소득세"),
        structured_data=StructuredData(
            asset_info=AssetInfo(asset_type="아파트", address="서울시 강남구 역삼동 123-45"),
            price_info=PriceInfo(buy_price=500000000, sell_price=1500000000),
            condition_info=ConditionInfo(house_count="1주택", is_regulated_area="여", residence_period="2년+")
        ),
        calculated_data=CalculatedData(estimated_total_tax=187200000, estimated_yangdo_tax=144000000, estimated_local_tax=14400000),
        unstructured_data=UnstructuredData(user_context="부모님 동거봉양 합가 후 아파트 양도"),
        additional_data={
            "fact_summary": "1주택 보유자가 직계존속 동거봉양 목적으로 합가하여 일시적 2주택 상태에서 본인 주택 양도",
            "checklist_answers": [{"id":"q1","question":"합가로 2주택?","answer":"예"}]
        }
    )
    return await report(dummy)

@app.get("/test-prior-errors")
async def test_prior_errors():
    return {"발견된_과거_오류": lookup_prior_errors("부모님 동거봉양 합가 양도세 비과세")}

@app.get("/test-gemini")
async def test_gemini():
    return {
        "project":  os.environ.get("GCP_PROJECT_ID","EMPTY"),
        "location": os.environ.get("GCP_LOCATION","EMPTY"),
        "model":    os.environ.get("GEMINI_MODEL","EMPTY")
    }
