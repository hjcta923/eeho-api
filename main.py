import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
from pinecone import Pinecone
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel

app = FastAPI(title="EEHO AI API", version="0.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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

pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
gemini = GenerativeModel(GEMINI_MODEL)

# ============================================================
# Pydantic 모델
# ============================================================
class AssetInfo(BaseModel):
    asset_type: Optional[str] = "모름"
    address:    Optional[str] = "모름"
    area_size:  Optional[str] = "모름"

class PriceInfo(BaseModel):
    buy_price:  Optional[float] = 0
    sell_price: Optional[float] = 0

class DateInfo(BaseModel):
    buy_date:  Optional[str] = "모름"
    sell_date: Optional[str] = "모름"

class ConditionInfo(BaseModel):
    is_regulated_area: Optional[str] = "모름"
    house_count:       Optional[str] = "모름"
    residence_period:  Optional[str] = "모름"

class StructuredData(BaseModel):
    asset_info:     Optional[AssetInfo]     = None
    price_info:     Optional[PriceInfo]     = None
    date_info:      Optional[DateInfo]      = None
    condition_info: Optional[ConditionInfo] = None

class AnalyzeRequest(BaseModel):
    session_id:        Optional[str]  = ""
    tax_category:      Optional[dict] = {}
    structured_data:   Optional[StructuredData] = None
    calculated_data:   Optional[dict] = {}
    unstructured_data: Optional[dict] = {}
    additional_data:   Optional[dict] = {}

    def get_question(self) -> str:
        if self.unstructured_data:
            return self.unstructured_data.get("user_context", "")
        return ""

    def get_structured_summary(self) -> str:
        lines = []
        if self.tax_category:
            lines.append(f"세목: {self.tax_category.get('type','')}")
        if self.structured_data:
            sd = self.structured_data
            if sd.asset_info:
                lines.append(f"자산: {sd.asset_info.asset_type} / 주소: {sd.asset_info.address} / 면적: {sd.asset_info.area_size}")
            if sd.price_info:
                lines.append(f"취득가: {sd.price_info.buy_price:,.0f}원 / 양도가: {sd.price_info.sell_price:,.0f}원")
            if sd.date_info:
                lines.append(f"취득일: {sd.date_info.buy_date} / 양도일: {sd.date_info.sell_date}")
            if sd.condition_info:
                lines.append(f"조정대상지역: {sd.condition_info.is_regulated_area} / 주택수: {sd.condition_info.house_count} / 거주기간: {sd.condition_info.residence_period}")
        if self.calculated_data:
            total = self.calculated_data.get("estimated_total_tax", 0)
            if total:
                lines.append(f"1차 예상세액: {total:,.0f}원")
        if self.additional_data:
            checklist = self.additional_data.get("checklist_answers", [])
            if checklist:
                lines.append("체크리스트 답변: " + json.dumps(checklist, ensure_ascii=False))
            supplement = self.additional_data.get("supplement_text", "")
            if supplement:
                lines.append(f"보완 내용: {supplement}")
        return "\n".join(lines)

# ============================================================
# 실무서 가이드 캐싱
# ============================================================
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

def extract_keywords(fields: dict) -> set:
    text = json.dumps(fields, ensure_ascii=False)
    return {kw for kw in TAX_KEYWORDS if kw in text}

def match_guide_sections(keywords: list, guide: list) -> list:
    results = []
    for section in guide:
        matched = set(section.get("키워드", [])).intersection(set(keywords))
        if matched:
            results.append({
                "섹션": section["섹션"], "챕터": section["챕터"],
                "페이지": section["페이지"], "파일명": section["파일명"],
                "매칭수": len(matched)
            })
    results.sort(key=lambda x: x["매칭수"], reverse=True)
    return results[:3]

def parse_gemini_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text  = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def retrieve_context(question: str) -> dict:
    try:
        res = index.search_records(
            namespace="tax_cases",
            query={"inputs": {"text": question}, "top_k": 3},
        )
        판례목록   = []
        전체키워드 = set()
        for hit in res["result"]["hits"]:
            판례목록.append({
                "사건번호": hit["fields"].get("사건번호", ""),
                "주제":     hit["fields"].get("주제", ""),
                "결과":     hit["fields"].get("결과", ""),
                "유사도":   round(hit["_score"], 3),
                "판단근거": hit["fields"].get("판단근거", ""),
                "관련법령": hit["fields"].get("관련법령", ""),
                "사실관계": hit["fields"].get("사실관계", ""),
            })
            전체키워드.update(extract_keywords(hit["fields"]))
    except Exception as e:
        print(f"[EEHO] Pinecone 오류: {e}")
        판례목록, 전체키워드 = [], set()

    키워드리스트 = list(전체키워드)
    try:
        guide    = load_guide()
        매칭결과 = match_guide_sections(키워드리스트, guide)
        참조섹션 = [{"섹션": m["섹션"], "챕터": m["챕터"], "페이지": m["페이지"]} for m in 매칭결과]
    except Exception as e:
        print(f"[EEHO] 실무서 오류: {e}")
        참조섹션 = []

    return {"판례목록": 판례목록, "키워드리스트": 키워드리스트, "참조섹션": 참조섹션}

# ============================================================
# 헬스체크
# ============================================================
@app.get("/")
def health():
    return {"service": "EEHO AI API", "version": "0.4", "status": "running"}

# ============================================================
# [Step 1] /generate-questions
# 정형/비정형 데이터 → 적용 가능 조문 탐색 → 체크리스트 3문항
# ============================================================
@app.post("/generate-questions")
async def generate_questions(req: AnalyzeRequest):
    question           = req.get_question()
    structured_summary = req.get_structured_summary()

    if not question:
        return {
            "status": "checklist", "applicable_law": "정보 부족",
            "law_summary": "상황 설명이 비어 있습니다.", "questions": []
        }

    ctx = retrieve_context(question)

    prompt = f"""당신은 양도소득세 전문 세무사 AI입니다.

[고객 정형 데이터]
{structured_summary}

[고객 자유 기술]
"{question}"

[관련 판례]
{json.dumps(ctx["판례목록"], ensure_ascii=False, indent=2)}

[추출 키워드]
{ctx["키워드리스트"]}

[참조 실무서]
{json.dumps(ctx["참조섹션"], ensure_ascii=False, indent=2)}

위 정보를 분석하여:
1. 고객 상황에 가장 적합한 비과세·감면 특례 조문 1개를 선택하세요
2. 해당 조문 적용 여부 판단에 필수적인 핵심 질문 3개를 생성하세요
3. 각 질문은 "예 / 아니오 / 모름" 으로 답할 수 있어야 합니다

반드시 아래 JSON 형식으로만 응답하세요:

{{
  "status": "checklist",
  "applicable_law": "소득세법 시행령 제155조 제1항 일시적 2주택 특례",
  "law_summary": "이사 등의 사유로 종전 주택 취득일로부터 1년 이상 지난 후 신규 주택을 취득하고, 신규 취득일로부터 3년 이내 종전 주택을 양도하면 1세대 1주택 비과세 적용",
  "questions": [
    {{
      "id": "q1",
      "question": "종전 주택(기존 아파트)을 취득한 날로부터 1년 이상 지난 후에 새로운 주택을 취득하셨나요?",
      "category": "취득 시기 요건"
    }},
    {{
      "id": "q2",
      "question": "새로운 주택을 취득한 날로부터 3년 이내에 종전 주택을 양도할 예정인가요?",
      "category": "양도 기한 요건"
    }},
    {{
      "id": "q3",
      "question": "종전 주택(양도 예정 주택)의 보유기간이 2년 이상인가요?",
      "category": "보유기간 요건"
    }}
  ]
}}
"""

    response = gemini.generate_content(prompt)
    result   = parse_gemini_json(response.text)

    if not result or "questions" not in result:
        result = {
            "status": "checklist",
            "applicable_law": "1세대 1주택 비과세 특례",
            "law_summary": "해당 상황에 적용 가능한 특례를 검토 중입니다.",
            "questions": [
                {"id": "q1", "question": "현재 양도 예정 주택이 1세대 1주택 상태에서 취득한 주택인가요?", "category": "주택 수 요건"},
                {"id": "q2", "question": "양도 예정 주택의 보유기간이 2년 이상인가요?", "category": "보유기간 요건"},
                {"id": "q3", "question": "해당 주택에서 2년 이상 실거주하셨나요?", "category": "거주기간 요건"}
            ]
        }

    result["status"] = "checklist"
    return result


# ============================================================
# [Step 2] /confirm
# 체크리스트 답변 → 사실관계 정리본 생성
# ============================================================
@app.post("/confirm")
async def confirm(req: AnalyzeRequest):
    question           = req.get_question()
    structured_summary = req.get_structured_summary()
    additional         = req.additional_data or {}
    checklist_answers  = additional.get("checklist_answers", [])
    applicable_law     = additional.get("applicable_law", "")
    law_summary        = additional.get("law_summary", "")
    calc               = req.calculated_data or {}
    estimated_total    = calc.get("estimated_total_tax", 0)

    ctx = retrieve_context(question)

    prompt = f"""당신은 양도소득세 전문 세무사 AI입니다.

[고객 정형 데이터]
{structured_summary}

[고객 자유 기술]
"{question}"

[검토 중인 조문]
{applicable_law}: {law_summary}

[체크리스트 답변]
{json.dumps(checklist_answers, ensure_ascii=False, indent=2)}

[관련 판례]
{json.dumps(ctx["판례목록"][:2], ensure_ascii=False, indent=2)}

[비과세 적용 전 1차 예상세액]
{estimated_total:,.0f}원

세무사 3대 원칙을 적용하여 사실관계를 정리하세요:
1. 리스크와 기회비용: 비과세 적용 시 절세액을 수치로 명시
2. 전략적 생략: 판단 결과 중심 서술
3. 전문가적 인사이트: 판례가 현재 상황에 갖는 의미 해석

반드시 아래 JSON 형식으로만 응답하세요:

{{
  "status": "confirm",
  "fact_summary": "3-5문장의 사실관계 정리. 비과세 적용 시 예상 절세액 수치 포함 필수.",
  "applied_law": "{applicable_law}",
  "requirement_check": [
    {{"요건": "요건명", "충족여부": "충족 또는 미충족 또는 확인필요", "근거": "판단 근거 1줄"}}
  ],
  "tax_impact": {{
    "before": "{estimated_total:,.0f}원",
    "after_pass": "0원 (비과세)",
    "saving": "{estimated_total:,.0f}원"
  }},
  "confidence": "높음 또는 보통 또는 낮음"
}}
"""

    response = gemini.generate_content(prompt)
    result   = parse_gemini_json(response.text)

    if not result or "fact_summary" not in result:
        result = {
            "status":       "confirm",
            "fact_summary": f"{applicable_law} 적용 가능성이 있습니다. 제출하신 체크리스트 답변을 바탕으로 최종 분석을 진행합니다.",
            "applied_law":  applicable_law,
            "requirement_check": [],
            "tax_impact": {"before": f"{estimated_total:,.0f}원", "after_pass": "0원 (비과세)", "saving": f"{estimated_total:,.0f}원"},
            "confidence": "검토 중"
        }

    result["status"] = "confirm"
    return result


# ============================================================
# [Step 3] /report
# 최종 확정 → 절세 리포트 생성
# ============================================================
@app.post("/report")
async def report(req: AnalyzeRequest):
    question           = req.get_question()
    structured_summary = req.get_structured_summary()
    additional         = req.additional_data or {}
    supplement_text    = additional.get("supplement_text", "")
    fact_summary       = additional.get("fact_summary", "")
    applicable_law     = additional.get("applicable_law", "")
    calc               = req.calculated_data or {}
    estimated_total    = calc.get("estimated_total_tax", 0)

    full_context = question + " " + supplement_text
    ctx = retrieve_context(full_context)

    prompt = f"""당신은 양도소득세 전문 세무사 AI입니다. 최종 절세 분석 리포트를 작성하세요.

[고객 정형 데이터]
{structured_summary}

[고객 자유 기술]
"{question}"

{"[고객 보완 내용]" + chr(10) + supplement_text + chr(10) if supplement_text else ""}
[사실관계 정리본]
{fact_summary}

[검토 조문]
{applicable_law}

[관련 판례]
{json.dumps(ctx["판례목록"], ensure_ascii=False, indent=2)}

[비과세 적용 전 1차 예상세액]
{estimated_total:,.0f}원

세무사 3대 원칙 철저 적용:
1. 리스크·기회비용 시각화: "비과세 적용 시 약 OOO만원 절세" 수치 필수
2. 전략적 생략: 계산 과정 생략, 판단 결과와 근거만 제시
3. 전문가적 인사이트: 판례 기각 사유가 현 상황에 해당하는지 명시적 해석

반드시 아래 JSON 형식으로만 응답하세요:

{{
  "status": "success",
  "result_type": "PASS",
  "applicable_law": "{applicable_law}",
  "details": "【판단】 ...\\n\\n【절세 효과】 비과세 적용 시 약 OOO만원 절세 예상\\n\\n【근거】 ...\\n\\n【판례 시사점】 ...",
  "risk_warning": "주의해야 할 리스크 또는 사후관리 사항"
}}

result_type: PASS(비과세/감면 가능) / FAIL(요건 미충족) / REVIEW(전문가 추가 검토 필요)
"""

    response = gemini.generate_content(prompt)
    result   = parse_gemini_json(response.text)

    if not result:
        result = {
            "status": "success", "result_type": "REVIEW",
            "applicable_law": applicable_law,
            "details": response.text,
            "risk_warning": "AI 응답 파싱 오류. 세무 전문가 상담을 권장드립니다."
        }

    result["status"] = "success"
    return result


# ============================================================
# 테스트
# ============================================================
@app.get("/test")
def test():
    return {"status": "ok", "version": "0.4"}

@app.get("/test-questions")
async def test_questions():
    req = AnalyzeRequest(
        session_id="test_001",
        tax_category={"type": "양도소득세"},
        structured_data=StructuredData(
            asset_info=AssetInfo(asset_type="아파트", address="서울시 서초구", area_size="85㎡ 이하"),
            price_info=PriceInfo(buy_price=500000000, sell_price=1500000000),
            date_info=DateInfo(buy_date="2015-05-20", sell_date="2026-04-10"),
            condition_info=ConditionInfo(is_regulated_area="여", house_count="2주택", residence_period="2년+")
        ),
        calculated_data={"estimated_total_tax": 187200000},
        unstructured_data={"user_context": "작년에 이사하려고 분당 아파트를 하나 더 샀어요. 기존 서초구 아파트는 언제 팔아야 하나요?"}
    )
    return await generate_questions(req)
