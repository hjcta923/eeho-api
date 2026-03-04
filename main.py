import os
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

app = FastAPI(title="EEHO AI API", version="0.3")

# ============================================================
# 설정 (환경변수에서 로드)
# ============================================================
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")
GUIDE_PATH = os.environ.get("GUIDE_PATH", "")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "")
KAKAO_CHANNEL_URL = os.environ.get("KAKAO_CHANNEL_URL", "")

# Pinecone (지연 로딩)
pc = None
index = None


def get_index():
    global pc, index
    if index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
    return index


# Gemini (지연 로딩)
gemini = None


def get_gemini():
    global gemini
    if gemini is None:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        gemini = GenerativeModel(GEMINI_MODEL)
    return gemini


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
# Phase 1: 판례 검색 + 실무서 매칭
# ============================================================

@app.get("/")
def health():
    return {"service": "EEHO AI API", "version": "0.3", "status": "running"}


@app.post("/analyze")
async def analyze(req: QueryRequest):
    question = req.question
    idx = get_index()

    search_results = idx.search_records(
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
    phase1 = await analyze(req)
    model = get_gemini()

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

    response = model.generate_content(prompt)
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
        "판례검색": phase1["step1_판례검색"],
        "추출키워드": phase1["step2_추출키워드"],
        "참조섹션": phase1["step4_참조섹션"],
        "추가질문": gemini_result,
    }


# ============================================================
# [보강 ④] Human-in-the-Loop 데이터 전처리 스키마
# ─────────────────────────────────────────────────────────────
# [특허 기술 명세 대응]
#
# 본 모듈은 사용자 컨펌(확인) UI를 단순 '동의 버튼'이 아닌,
# '학습 데이터 품질 제어 메커니즘(Human-in-the-Loop Data
# Preprocessing)'으로 정의합니다.
#
# 기술적 의의:
# 1. AI가 정리한 사실관계를 사용자가 검토·수정하는 과정을
#    '데이터 전처리 파이프라인의 일부'로 편입
# 2. 수정 전/후 diff(차분)를 구조화하여 기록함으로써,
#    사용자 개입이 데이터 품질에 미치는 영향을 정량화
# 3. 이 diff 데이터는 향후 오답 노트 DB(RLHF DB)의
#    학습 피드백으로 활용 가능
#
# 이로써 '사용자 컨펌'은 특허 청구 범위에서 독립적인
# 기술적 수단(데이터 전처리 단계)으로 기능합니다.
# ============================================================


class DataQualityMetrics(BaseModel):
    """
    [특허 명세서 대응] 데이터 품질 정량화 스키마
    ─ 사용자 컨펌 전/후의 데이터 완성도를 수치화
    """
    completeness_before: float = Field(
        ..., ge=0.0, le=1.0,
        description="컨펌 전 데이터 완성도 (0.0~1.0). 체크리스트 응답률 + 추가정보 입력률 기반"
    )
    completeness_after: float = Field(
        ..., ge=0.0, le=1.0,
        description="컨펌 후 데이터 완성도 (0.0~1.0). 사용자 수정 반영 후 재산정"
    )
    quality_delta: float = Field(
        ...,
        description="품질 개선 폭 (after - before). 양수면 품질 향상, 음수면 정보 손실"
    )
    human_corrections_count: int = Field(
        ..., ge=0,
        description="사용자가 수정한 항목 수. 0이면 AI 판단을 그대로 수용"
    )
    preprocessing_type: str = Field(
        default="human_in_the_loop_validation",
        description="전처리 유형 식별자 (특허 청구항 연계)"
    )


class FieldDiff(BaseModel):
    """
    [특허 명세서 대응] 개별 필드의 수정 전/후 차분 기록
    ─ AI 생성 값 vs 사용자 확인 값의 차이를 구조화
    """
    field_name: str = Field(..., description="수정된 필드명 (예: '보유기간', '합가사유')")
    ai_generated_value: str = Field(..., description="AI가 추론/정리한 값 (수정 전)")
    user_confirmed_value: str = Field(..., description="사용자가 확인/수정한 값 (수정 후)")
    modification_type: str = Field(
        ...,
        description="수정 유형: 'corrected'(오류수정), 'supplemented'(누락보충), 'unchanged'(변경없음)"
    )
    impact_on_analysis: str = Field(
        default="",
        description="이 수정이 최종 분석에 미치는 영향 (예: '비과세 요건 충족 여부 변경')"
    )


# ============================================================
# Phase 3: 사실관계 정리 + 데이터 전처리 (보강)
# ============================================================

class ConfirmRequest(BaseModel):
    question: str
    체크리스트응답: list  # [{"질문": "...", "답변": "Yes"}, ...]
    추가정보: dict = {}  # {"양도가액": "500000000", "취득가액": "300000000"}
    판례검색: list = []
    추출키워드: list = []
    참조섹션: list = []
    # [보강 ④] 사용자 수정 데이터 (컨펌 UI에서 사용자가 수정한 내역)
    사용자수정사항: list = []  # [{"필드": "보유기간", "AI값": "2년", "수정값": "3년"}, ...]


def compute_data_quality(
    checklist: list,
    extra_info: dict,
    user_corrections: list,
) -> tuple[DataQualityMetrics, list[FieldDiff]]:
    """
    [특허 명세서 대응] 데이터 품질 산정 알고리즘
    ────────────────────────────────────────────
    - 입력 완성도 = (체크리스트 응답률 × 0.6) + (추가정보 입력률 × 0.4)
    - 수정 반영 후 완성도 = 사용자 수정 건수에 비례하여 보정
    - diff 목록을 구조화하여 반환
    """
    # (1) 체크리스트 응답 완성도
    total_checks = len(checklist) if checklist else 1
    answered_checks = sum(
        1 for item in checklist
        if item.get("답변") and item["답변"] not in ["", "모름"]
    )
    check_completeness = answered_checks / total_checks

    # (2) 추가정보 입력 완성도
    if extra_info:
        total_fields = len(extra_info)
        filled_fields = sum(
            1 for v in extra_info.values()
            if v and str(v).strip() not in ["", "모름"]
        )
        info_completeness = filled_fields / total_fields if total_fields > 0 else 0
    else:
        info_completeness = 0.0

    # (3) 컨펌 전 완성도 (가중 평균)
    completeness_before = round(check_completeness * 0.6 + info_completeness * 0.4, 3)

    # (4) 사용자 수정사항 → diff 목록 생성
    diffs = []
    corrections_count = 0

    for correction in user_corrections:
        ai_val = str(correction.get("AI값", ""))
        user_val = str(correction.get("수정값", ""))

        if ai_val != user_val and user_val.strip():
            mod_type = "corrected" if ai_val.strip() else "supplemented"
            corrections_count += 1
        else:
            mod_type = "unchanged"

        diffs.append(FieldDiff(
            field_name=correction.get("필드", ""),
            ai_generated_value=ai_val,
            user_confirmed_value=user_val,
            modification_type=mod_type,
            impact_on_analysis=correction.get("영향", ""),
        ))

    # (5) 컨펌 후 완성도 (수정 반영 보정)
    correction_bonus = min(corrections_count * 0.05, 0.2)
    completeness_after = round(min(completeness_before + correction_bonus, 1.0), 3)

    quality_delta = round(completeness_after - completeness_before, 3)

    metrics = DataQualityMetrics(
        completeness_before=completeness_before,
        completeness_after=completeness_after,
        quality_delta=quality_delta,
        human_corrections_count=corrections_count,
        preprocessing_type="human_in_the_loop_validation",
    )

    return metrics, diffs


@app.post("/confirm")
async def confirm(req: ConfirmRequest):
    """
    ══════════════════════════════════════════════════════════
    Phase 3: Human-in-the-Loop 데이터 전처리 + 사실관계 정리
    ══════════════════════════════════════════════════════════

    [특허 기술 명세 대응]
    본 엔드포인트는 다음 2가지 기능을 수행합니다:

    (A) 데이터 전처리 (Data Preprocessing)
        - 사용자 체크리스트 응답 + 추가정보를 종합하여
          데이터 품질 지표(DataQualityMetrics)를 산정
        - 사용자가 AI 생성 사실관계를 수정한 경우,
          수정 전/후 diff(FieldDiff)를 구조화하여 기록
        - 이 diff 데이터는 오답 노트 DB(RLHF DB)의
          학습 피드백으로 축적 가능

    (B) 사실관계 정리 (Fact Summarization)
        - Gemini를 통해 세법 관점의 사실관계 요약을 생성
        - 사용자가 이를 컨펌하면 Phase 4(리포트)로 진행

    반환값의 'preprocessing' 필드가 (A)의 결과이며,
    '사실관계' 필드가 (B)의 결과입니다.
    ══════════════════════════════════════════════════════════
    """

    # ── (A) 데이터 전처리: 품질 지표 산정 + diff 생성 ──
    quality_metrics, field_diffs = compute_data_quality(
        checklist=req.체크리스트응답,
        extra_info=req.추가정보,
        user_corrections=req.사용자수정사항,
    )

    preprocessing_result = {
        "pipeline_stage": "human_in_the_loop_preprocessing",
        "timestamp": datetime.utcnow().isoformat(),
        "data_quality": quality_metrics.model_dump(),
        "field_diffs": [d.model_dump() for d in field_diffs],
        "summary": {
            "총_입력_필드": len(req.체크리스트응답) + len(req.추가정보),
            "사용자_수정_건수": quality_metrics.human_corrections_count,
            "품질_개선_폭": quality_metrics.quality_delta,
            "판정": (
                "AI 판단 수용 (수정 없음)"
                if quality_metrics.human_corrections_count == 0
                else f"사용자 보정 {quality_metrics.human_corrections_count}건 반영"
            ),
        },
    }

    # ── (B) 사실관계 정리: Gemini 호출 ──
    # 사용자 수정사항이 있으면, 수정된 값을 우선 반영하도록 프롬프트에 포함
    correction_context = ""
    if req.사용자수정사항:
        corrections_text = json.dumps(req.사용자수정사항, ensure_ascii=False, indent=2)
        correction_context = f"""
[중요] 아래는 사용자가 AI의 초기 분석을 검토한 후 수정한 사항입니다.
수정된 값을 원래 값보다 우선하여 사실관계에 반영하세요:
{corrections_text}
"""

    model = get_gemini()

    prompt = f"""당신은 양도소득세 전문 세무사 AI입니다.

고객의 초기 질문:
"{req.question}"

고객이 답변한 체크리스트:
{json.dumps(req.체크리스트응답, ensure_ascii=False, indent=2)}

고객이 제공한 추가 정보:
{json.dumps(req.추가정보, ensure_ascii=False, indent=2)}
{correction_context}
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

    response = model.generate_content(prompt)
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
        # [보강 ④] 데이터 전처리 결과 (특허 청구항 연계)
        "preprocessing": preprocessing_result,
        # 기존 사실관계 정리 결과
        "사실관계": gemini_result,
        "안내": "위 사실관계가 맞으면 확인 버튼을 눌러 최종 리포트를 생성합니다.",
    }


# ============================================================
# Phase 3 테스트 (수정사항 포함 버전)
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
            "판단근거": "혼인으로 3주택자가 된 점, 합가일부터 5년 도과 후 양도한 점에서 특례 미충족"
        }],
        추출키워드=["동거봉양", "합가", "비과세", "5년", "직계존속"],
        참조섹션=[{"섹션": "제4절 1세대 2주택 비과세 특례", "페이지": "749-898"}],
        # [보강 ④] 사용자가 AI 초기 분석을 수정한 내역
        사용자수정사항=[
            {
                "필드": "보유기간",
                "AI값": "2년",
                "수정값": "3년 6개월",
                "영향": "장기보유특별공제율 변경 가능"
            },
            {
                "필드": "합가사유",
                "AI값": "동거봉양",
                "수정값": "동거봉양",  # 변경 없음
                "영향": ""
            },
            {
                "필드": "거주기간",
                "AI값": "",
                "수정값": "2년 이상",
                "영향": "조정대상지역 2년 거주 요건 충족 여부 판단에 영향"
            },
        ],
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
    # [보강 ④] 전처리 결과를 리포트에도 포함
    preprocessing: dict = {}


@app.post("/report")
async def report(req: ReportRequest):
    """
    Phase 4: 사실관계 컨펌 후 → 예상세액 + 판단근거 + 리스크 리포트 생성

    [보강 ④] preprocessing 데이터가 포함된 경우,
    리포트에 '데이터 품질 기반 신뢰도'를 추가로 표기합니다.
    """
    model = get_gemini()

    # 전처리 품질 정보가 있으면 프롬프트에 반영
    quality_context = ""
    if req.preprocessing:
        dq = req.preprocessing.get("data_quality", {})
        corrections = dq.get("human_corrections_count", 0)
        quality_context = f"""
[참고] 데이터 품질 정보:
- 사용자 검증 완료 (Human-in-the-Loop)
- 사용자 수정 건수: {corrections}건
- 데이터 완성도: {dq.get('completeness_after', 'N/A')}
이 정보를 바탕으로 분석 신뢰도를 판단하세요.
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
{quality_context}
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
  "분석_신뢰도": "높음/보통/낮음",
  "종합의견": "전체적인 판단 요약"
}}
"""

    response = model.generate_content(prompt)
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
        # [보강 ④] 전처리 메타데이터를 리포트에 첨부
        "preprocessing_metadata": req.preprocessing if req.preprocessing else None,
        "면책안내": "본 분석은 참고용이며, 해당 조문의 적용 가능 여부 및 정확한 세액은 세무사의 최종 검토가 필요합니다.",
        "상담문의": {
            "안내": "보다 정확한 상담을 원하시면 아래 버튼을 눌러주세요.",
            "카카오톡채널": KAKAO_CHANNEL_URL
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
                "주택_현황": "양도 주택: 고객 소유 주택, 보유기간: 3년 6개월, 거주기간: 2년 이상",
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
        참조섹션=[{"섹션": "제4절 1세대 2주택 비과세 특례", "페이지": "749-898"}],
        # [보강 ④] 전처리 결과를 리포트 요청에 포함
        preprocessing={
            "data_quality": {
                "completeness_before": 0.76,
                "completeness_after": 0.86,
                "quality_delta": 0.1,
                "human_corrections_count": 2,
                "preprocessing_type": "human_in_the_loop_validation"
            }
        }
    )
    return await report(req)


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

@app.get("/test-gemini")
async def test_gemini():
    try:
        return {
            "project": GCP_PROJECT_ID,
            "location": GCP_LOCATION,
            "model": GEMINI_MODEL,
            "check": "vars loaded"
        }
    except Exception as e:
        return {"error": str(e)}
