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

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")
GUIDE_PATH = os.environ.get("GUIDE_PATH", "")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "")
KAKAO_CHANNEL_URL = os.environ.get("KAKAO_CHANNEL_URL", "")

pc = None
index = None

def get_index():
    global pc, index
    if index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
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
    gcs = storage.Client()
    bucket = gcs.bucket(BUCKET_NAME)
    blob = bucket.blob(GUIDE_PATH)
    data = blob.download_as_text()
    _guide_cache = json.loads(data)
    return _guide_cache

TAX_KEYWORDS = ["1세대1주택","비과세","동거봉양","합가","혼인","상속주택","일시적2주택","장애인","직계존속","5년","5년이내","60세","다주택자","중과세","재개발","재건축","농어촌주택","부담부증여","고가주택","비사업용토지","비사업용","장기보유특별공제","장특공","양도시기","취득시기","세대","주택수","보유기간","거주기간","2주택","3주택","조합원입주권","분양권","겸용주택","다가구주택","상속","증여","대물변제","교환","문화재주택","임대주택","자경농지","출국세","비거주자","실거래가","기준시가","필요경비","양도차익","예정신고","확정신고","경정청구","가산세","조정대상지역","중과배제","사업용의제"]

def extract_keywords_from_case(fields):
    text = json.dumps(fields, ensure_ascii=False)
    return {kw for kw in TAX_KEYWORDS if kw in text}

def match_guide_sections(keywords, guide):
    results = []
    for section in guide:
        matched = set(section.get("키워드", [])).intersection(set(keywords))
        if matched:
            results.append({"섹션": section["섹션"], "출처": section["출처"], "챕터": section["챕터"], "페이지": section["페이지"], "파일명": section["파일명"], "매칭키워드": list(matched), "매칭수": len(matched)})
    results.sort(key=lambda x: x["매칭수"], reverse=True)
    return results[:3]

class QueryRequest(BaseModel):
    question: str

def lookup_prior_errors(question, top_k=2):
    try:
        idx = get_index()
        results = idx.search_records(namespace="error_notes", query={"inputs": {"text": question}, "top_k": top_k})
        return [{"note_id": hit.get("_id",""), "유사도": round(hit.get("_score",0),3), "원래질문": hit["fields"].get("original_question",""), "교훈": hit["fields"].get("교훈",""), "오류유형": hit["fields"].get("오류유형","")} for hit in results.get("result",{}).get("hits",[]) if hit.get("_score",0) > 0.5]
    except Exception:
        return []

@app.get("/")
def health():
    return {"service": "EEHO AI API", "version": "0.5", "status": "running"}

@app.post("/analyze")
async def analyze(req: QueryRequest):
    idx = get_index()
    search_results = idx.search_records(namespace="tax_cases", query={"inputs": {"text": req.question}, "top_k": 3})
    판례목록, 전체키워드 = [], set()
    for hit in search_results["result"]["hits"]:
        판례목록.append({"사건번호": hit["fields"].get("사건번호",""), "주제": hit["fields"].get("주제",""), "결과": hit["fields"].get("결과",""), "유사도": round(hit["_score"],3), "판단근거": hit["fields"].get("판단근거",""), "관련법령": hit["fields"].get("관련법령",""), "과세관청주장": hit["fields"].get("과세관청주장","")})
        전체키워드.update(extract_keywords_from_case(hit["fields"]))
    키워드리스트 = list(전체키워드)
    매칭결과 = match_guide_sections(키워드리스트, load_guide())
    참조섹션 = [{"섹션": m["섹션"], "챕터": m["챕터"], "페이지": m["페이지"], "파일명": m["파일명"], "매칭키워드": m["매칭키워드"]} for m in 매칭결과]
    prior_errors = lookup_prior_errors(req.question)
    return {"질문": req.question, "step1_판례검색": 판례목록, "step2_추출키워드": 키워드리스트, "step3_실무서매칭": 매칭결과, "step4_참조섹션": 참조섹션, "step0_오답노트_선행참조": prior_errors}

class GapResult(BaseModel):
    req_name: str; data_field: str; status: str; user_value: str = ""; threshold: str = ""; legal_basis: str = ""; priority: str = ""; question_hint: str = ""

class GapAnalysisResult(BaseModel):
    algorithm_stage: str = "deterministic_gap_detection"; provisions_checked: list = []; total_requirements: int = 0; satisfied_count: int = 0; gap_count: int = 0; ambiguous_count: int = 0; completeness_ratio: float = 0.0; gap_items: list = []; ambiguous_items: list = []; satisfied_items: list = []

def flatten_user_data(d):
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items(): flat[k2] = str(v2) if v2 is not None else ""
        else: flat[k] = str(v) if v is not None else ""
    return flat

def is_missing(v):
    return not v or v.strip() in ["","모름","null","None","없음","미입력"]

def evaluate_requirement(req, user_value):
    if is_missing(user_value): return "gap"
    dt, th = req.get("data_type","text"), req.get("threshold","")
    if dt == "boolean":
        if user_value.lower() in ["yes","true","예","여","y"]: return "satisfied"
        return "ambiguous"
    elif dt == "number" and th:
        try:
            val = float(user_value.replace(",","").replace("원",""))
            if "이하" in th: return "satisfied" if val <= float(th.replace("이하","").strip().replace(",","")) else "ambiguous"
            elif "이상" in th: return "satisfied" if val >= float(th.replace("이상","").strip().replace(",","")) else "ambiguous"
        except: pass
        return "ambiguous"
    elif dt == "duration" and th:
        un, tn = re.findall(r'(\d+)', user_value), re.findall(r'(\d+)', th)
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
        r = GapResult(req_name=req.get("req_name",""), data_field=df, status=st, user_value=uv if not is_missing(uv) else "(미입력)", threshold=req.get("threshold",""), legal_basis=req.get("legal_basis",""), priority=req.get("priority","important"), question_hint=req.get("question_hint",""))
        if st == "gap": gap_items.append(r.model_dump())
        elif st == "ambiguous": ambiguous_items.append(r.model_dump())
        else: satisfied_items.append(r.model_dump())
    total = len(ai_requirements)
    return GapAnalysisResult(provisions_checked=provisions_checked or [], total_requirements=total, satisfied_count=len(satisfied_items), gap_count=len(gap_items), ambiguous_count=len(ambiguous_items), completeness_ratio=round(len(satisfied_items)/total,2) if total>0 else 0, gap_items=gap_items, ambiguous_items=ambiguous_items, satisfied_items=satisfied_items)

@app.post("/generate-questions")
async def generate_questions(req: QueryRequest):
    phase1 = await analyze(req)
    model = get_gemini()
    prior_errors = phase1.get("step0_오답노트_선행참조", [])
    error_context = ""
    if prior_errors:
        lessons = [e.get("교훈","") for e in prior_errors if e.get("교훈")]
        if lessons: error_context = f"\n[주의] 과거 유사 사례 오류 교훈:\n{json.dumps(lessons, ensure_ascii=False)}\n"

    s1p = f'''당신은 양도소득세 전문 세무사입니다.
고객 질문: "{req.question}"
관련 판례: {json.dumps(phase1["step1_판례검색"], ensure_ascii=False, indent=2)}
추출 키워드: {phase1["step2_추출키워드"]}
{error_context}
적용 가능 비과세/감면 규정과 필수 충족 요건을 구조화하세요.
JSON만 응답:
{{"적용_검토_규정": [{{"규정명": "...", "근거조문": "...", "적용가능성": "높음"}}], "필수요건": [{{"req_name": "...", "data_field": "영문snake", "data_type": "boolean/number/duration/date/text", "threshold": "...", "legal_basis": "...", "priority": "critical/important/optional", "question_hint": "..."}}]}}
필수요건 5~8개.'''

    r1 = model.generate_content(s1p)
    t1 = r1.text.strip()
    if t1.startswith("```"):
        t1 = t1.split("```")[1]
        if t1.startswith("json"): t1 = t1[4:]
        t1 = t1.strip()
    try: s1r = json.loads(t1)
    except: return {"질문": req.question, "stage": "stage1_error", "error": t1[:500]}

    reqs = s1r.get("필수요건", [])
    provs = s1r.get("적용_검토_규정", [])
    gap = run_gap_detection(reqs, {}, [p.get("규정명","") for p in provs])

    needs = gap.gap_items + gap.ambiguous_items
    needs.sort(key=lambda x: {"critical":0,"important":1,"optional":2}.get(x.get("priority",""),2))
    top = needs[:5]

    s3p = f'''양도소득세 전문 세무사 AI. 고객 질문: "{req.question}"
미확인 항목: {json.dumps(top, ensure_ascii=False, indent=2)}
각 항목에 대해 고객이 이해하기 쉬운 질문 생성.
JSON만 응답:
{{"체크리스트": [{{"질문": "...", "카테고리": "...", "중요도": "필수", "근거조문": "...", "설명": "...", "입력형식": "Yes/No/금액/날짜/기간/선택"}}]}}
{len(top)}개 질문 생성.'''

    r3 = model.generate_content(s3p)
    t3 = r3.text.strip()
    if t3.startswith("```"):
        t3 = t3.split("```")[1]
        if t3.startswith("json"): t3 = t3[4:]
        t3 = t3.strip()
    try: s3r = json.loads(t3)
    except: s3r = {"raw": t3, "parse_error": True}

    return {"질문": req.question, "stage1_적용규정": provs, "stage1_필수요건_수": len(reqs), "stage2_gap_analysis": gap.model_dump(), "stage3_추가질문": s3r, "오답노트_선행참조": prior_errors, "판례검색": phase1["step1_판례검색"], "추출키워드": phase1["step2_추출키워드"], "참조섹션": phase1["step4_참조섹션"]}

class DataQualityMetrics(BaseModel):
    completeness_before: float = Field(..., ge=0.0, le=1.0); completeness_after: float = Field(..., ge=0.0, le=1.0); quality_delta: float; human_corrections_count: int = Field(..., ge=0); preprocessing_type: str = "human_in_the_loop_validation"

class FieldDiff(BaseModel):
    field_name: str; ai_generated_value: str; user_confirmed_value: str; modification_type: str; impact_on_analysis: str = ""

class ConfirmRequest(BaseModel):
    question: str; 체크리스트응답: list; 추가정보: dict = {}; 판례검색: list = []; 추출키워드: list = []; 참조섹션: list = []; 사용자수정사항: list = []

def compute_data_quality(cl, ei, uc):
    tc = len(cl) if cl else 1
    ac = sum(1 for i in cl if i.get("답변") and i["답변"] not in ["","모름"])
    cc = ac/tc
    if ei:
        tf = len(ei); ff = sum(1 for v in ei.values() if v and str(v).strip() not in ["","모름"]); ic = ff/tf if tf>0 else 0
    else: ic = 0.0
    cb = round(cc*0.6+ic*0.4, 3)
    diffs, cnt = [], 0
    for c in uc:
        av, uv = str(c.get("AI값","")), str(c.get("수정값",""))
        if av != uv and uv.strip(): mt = "corrected" if av.strip() else "supplemented"; cnt += 1
        else: mt = "unchanged"
        diffs.append(FieldDiff(field_name=c.get("필드",""), ai_generated_value=av, user_confirmed_value=uv, modification_type=mt, impact_on_analysis=c.get("영향","")))
    ca = round(min(cb + min(cnt*0.05,0.2), 1.0), 3)
    return DataQualityMetrics(completeness_before=cb, completeness_after=ca, quality_delta=round(ca-cb,3), human_corrections_count=cnt), diffs

@app.post("/confirm")
async def confirm(req: ConfirmRequest):
    qm, fd = compute_data_quality(req.체크리스트응답, req.추가정보, req.사용자수정사항)
    pp = {"pipeline_stage": "human_in_the_loop_preprocessing", "timestamp": datetime.utcnow().isoformat(), "data_quality": qm.model_dump(), "field_diffs": [d.model_dump() for d in fd], "summary": {"총_입력_필드": len(req.체크리스트응답)+len(req.추가정보), "사용자_수정_건수": qm.human_corrections_count, "품질_개선_폭": qm.quality_delta, "판정": "AI 판단 수용" if qm.human_corrections_count==0 else f"사용자 보정 {qm.human_corrections_count}건 반영"}}
    cc = ""
    if req.사용자수정사항: cc = f"\n[중요] 사용자 수정:\n{json.dumps(req.사용자수정사항, ensure_ascii=False, indent=2)}\n"
    model = get_gemini()
    p = f'''양도소득세 전문 세무사 AI.
고객 질문: "{req.question}"
체크리스트: {json.dumps(req.체크리스트응답, ensure_ascii=False, indent=2)}
추가정보: {json.dumps(req.추가정보, ensure_ascii=False, indent=2)}
{cc}판례: {json.dumps(req.판례검색[:1], ensure_ascii=False, indent=2)}
사실관계를 세법 관점에서 정리. JSON만 응답:
{{"사실관계_요약": {{"양도자_현황": "...", "주택_현황": "...", "합가_현황": "...", "기타": "..."}}, "적용_검토_조문": [{{"조문": "...", "내용": "...", "충족여부": "충족/미충족/확인필요", "판단근거": "..."}}], "비과세_가능성": "높음/보통/낮음", "비과세_가능성_설명": "..."}}'''
    r = model.generate_content(p)
    t = r.text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"): t = t[4:]
        t = t.strip()
    try: gr = json.loads(t)
    except: gr = {"raw": t, "parse_error": True}
    return {"질문": req.question, "preprocessing": pp, "사실관계": gr, "안내": "위 사실관계가 맞으면 확인 버튼을 눌러 최종 리포트를 생성합니다."}

class ReportRequest(BaseModel):
    question: str; 사실관계: dict; 체크리스트응답: list; 추가정보: dict = {}; 판례검색: list = []; 추출키워드: list = []; 참조섹션: list = []; preprocessing: dict = {}

@app.post("/report")
async def report(req: ReportRequest):
    model = get_gemini()
    qc = ""
    if req.preprocessing:
        dq = req.preprocessing.get("data_quality",{})
        qc = f"\n[데이터 품질] 사용자 검증 완료, 수정 {dq.get('human_corrections_count',0)}건\n"
    p = f'''양도소득세 전문 세무사 AI. 최종 리포트 작성.
질문: "{req.question}"
사실관계: {json.dumps(req.사실관계, ensure_ascii=False, indent=2)}
체크리스트: {json.dumps(req.체크리스트응답, ensure_ascii=False, indent=2)}
추가정보: {json.dumps(req.추가정보, ensure_ascii=False, indent=2)}
판례: {json.dumps(req.판례검색, ensure_ascii=False, indent=2)}
실무서: {json.dumps(req.참조섹션, ensure_ascii=False, indent=2)}
{qc}JSON만 응답:
{{"예상세액": {{"비과세_적용시": "0원", "비과세_미적용시": "약 OOO만원"}}, "판단근거": [{{"조문": "...", "내용": "...", "판단": "..."}}], "관련예판": [{{"사건번호": "...", "결과": "...", "시사점": "..."}}], "리스크": [{{"유형": "...", "내용": "...", "대응방안": "..."}}], "분석_신뢰도": "높음/보통/낮음", "종합의견": "..."}}'''
    r = model.generate_content(p)
    t = r.text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"): t = t[4:]
        t = t.strip()
    try: gr = json.loads(t)
    except: gr = {"raw": t, "parse_error": True}
    return {"리포트": gr, "preprocessing_metadata": req.preprocessing or None, "면책안내": "본 분석은 참고용이며, 정확한 세액은 세무사의 최종 검토가 필요합니다.", "상담문의": {"안내": "보다 정확한 상담을 원하시면 아래 버튼을 눌러주세요.", "카카오톡채널": KAKAO_CHANNEL_URL}}

class FeedbackRequest(BaseModel):
    session_id: str = ""; question: str; feedback_text: str; ai_report: dict = {}; ai_사실관계: dict = {}; 추출키워드: list = []

class TriageResult(BaseModel):
    classification: str; confidence: float = Field(ge=0.0, le=1.0); reason: str = ""; has_factual_correction: bool = False; has_missing_info: bool = False; has_legal_dispute: bool = False

class ErrorNote(BaseModel):
    note_id: str; session_id: str; timestamp: str; original_question: str; keywords: list = []; ai_report_summary: str = ""; ai_비과세_판정: str = ""; feedback_text: str; triage: dict = {}; deltas: list = []; embed_text: str = ""

async def triage_feedback(feedback_text, question, ai_report):
    model = get_gemini()
    p = f'''세무 AI 품질관리 분석가.
사용자가 "보완하기" 후 피드백:
[질문] "{question}"
[AI 리포트] {json.dumps(ai_report, ensure_ascii=False)[:2000]}
[피드백] "{feedback_text}"
판정: actionable(사실오류/누락보충/법령이의), emotional(감정만), ambiguous(불명확)
JSON만: {{"classification": "...", "confidence": 0.85, "reason": "...", "has_factual_correction": true, "has_missing_info": false, "has_legal_dispute": false}}'''
    r = model.generate_content(p)
    t = r.text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"): t = t[4:]
        t = t.strip()
    try: return TriageResult(**json.loads(t))
    except: return TriageResult(classification="ambiguous", confidence=0.0, reason="파싱 실패")

async def extract_deltas(feedback_text, question, ai_report, ai_사실관계):
    model = get_gemini()
    p = f'''세무 AI 오류 분석가.
[질문] "{question}"
[AI 사실관계] {json.dumps(ai_사실관계, ensure_ascii=False)[:2000]}
[AI 리포트] {json.dumps(ai_report, ensure_ascii=False)[:2000]}
[피드백] "{feedback_text}"
AI 오류를 구조화. JSON만:
{{"deltas": [{{"error_field": "...", "ai_judgment": "...", "user_correction": "...", "lesson_learned": "...", "error_type": "factual_error/missing_info/legal_interpretation/other"}}], "embed_summary": "벡터검색용 1~2문장 요약"}}'''
    r = model.generate_content(p)
    t = r.text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"): t = t[4:]
        t = t.strip()
    try:
        res = json.loads(t)
        return res.get("deltas",[]), res.get("embed_summary","")
    except: return [], ""

def save_error_note_to_gcs(en):
    try:
        gcs = storage.Client()
        bucket = gcs.bucket(BUCKET_NAME)
        blob = bucket.blob(f"오답노트/{en.note_id}.json")
        blob.upload_from_string(json.dumps(en.model_dump(), ensure_ascii=False, indent=2), content_type="application/json")
        return True
    except: return False

def save_error_note_to_pinecone(en):
    try:
        idx = get_index()
        idx.upsert_records(namespace="error_notes", records=[{"_id": en.note_id, "text": en.embed_text, "original_question": en.original_question, "keywords": json.dumps(en.keywords, ensure_ascii=False), "ai_판정": en.ai_비과세_판정, "오류유형": json.dumps([d.get("error_type","") for d in en.deltas], ensure_ascii=False), "교훈": json.dumps([d.get("lesson_learned","") for d in en.deltas], ensure_ascii=False), "timestamp": en.timestamp}])
        return True
    except: return False

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    sid = req.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    nid = f"err_{uuid.uuid4().hex[:12]}"
    ts = datetime.utcnow().isoformat()
    triage = await triage_feedback(req.feedback_text, req.question, req.ai_report)
    result = {"session_id": sid, "note_id": nid, "timestamp": ts, "stage1_triage": triage.model_dump()}

    if triage.classification == "emotional":
        result.update({"처리결과": "emotional_feedback_excluded", "안내": "소중한 의견 감사합니다. 보다 정확한 상담이 필요하시면 세무사와 직접 상담을 권해드립니다.", "카카오톡채널": KAKAO_CHANNEL_URL, "stage2_delta": None, "stage3_저장": None})
        return result

    deltas, embed = await extract_deltas(req.feedback_text, req.question, req.ai_report, req.ai_사실관계)
    result["stage2_delta"] = {"deltas": deltas, "embed_summary": embed, "delta_count": len(deltas)}

    ai_b = ""
    if req.ai_report:
        e = req.ai_report.get("예상세액",{})
        if isinstance(e,dict): ai_b = e.get("비과세_적용시","")
        if not ai_b: ai_b = req.ai_report.get("종합의견","")

    en = ErrorNote(note_id=nid, session_id=sid, timestamp=ts, original_question=req.question, keywords=req.추출키워드, ai_report_summary=json.dumps(req.ai_report, ensure_ascii=False)[:1000] if req.ai_report else "", ai_비과세_판정=ai_b, feedback_text=req.feedback_text, triage=triage.model_dump(), deltas=deltas, embed_text=embed if embed else f"{req.question} | {req.feedback_text[:200]}")
    gs = save_error_note_to_gcs(en)
    ps = save_error_note_to_pinecone(en)
    result["stage3_저장"] = {"gcs": "성공" if gs else "실패", "gcs_path": f"gs://{BUCKET_NAME}/오답노트/{nid}.json" if gs else None, "pinecone": "성공" if ps else "실패", "pinecone_namespace": "error_notes", "needs_review": triage.classification == "ambiguous"}
    result["처리결과"] = "actionable_feedback_saved" if triage.classification == "actionable" else "ambiguous_feedback_saved_for_review"
    result["안내"] = "소중한 의견이 반영되었습니다. 향후 유사 사례 분석 시 더 나은 결과를 제공하겠습니다."
    return result

@app.get("/test")
async def test():
    return await analyze(QueryRequest(question="부모님을 모시려고 합가했는데 아파트를 팔면 비과세 되나요?"))

@app.get("/test-questions")
async def test_questions():
    return await generate_questions(QueryRequest(question="부모님을 모시려고 합가했는데 아파트를 팔면 비과세 되나요?"))

@app.get("/test-gap")
async def test_gap():
    mr = [{"req_name":"1세대 판정","data_field":"house_count","data_type":"text","threshold":"1주택","legal_basis":"소득세법 시행령 제152조","priority":"critical","question_hint":"보유 주택 수"},{"req_name":"보유기간 2년 이상","data_field":"residence_period","data_type":"duration","threshold":"2년 이상","legal_basis":"소득세법 제89조","priority":"critical","question_hint":"보유기간"},{"req_name":"조정대상지역","data_field":"is_regulated_area","data_type":"boolean","threshold":"","legal_basis":"소득세법 시행령 제154조","priority":"important","question_hint":"조정대상지역"},{"req_name":"고가주택(12억초과)","data_field":"sell_price","data_type":"number","threshold":"1200000000 이하","legal_basis":"소득세법 시행령 제156조의2","priority":"important","question_hint":"양도가액"},{"req_name":"합가사유","data_field":"merge_reason","data_type":"text","threshold":"","legal_basis":"소득세법 시행령 제155조 제4항","priority":"critical","question_hint":"합가사유"},{"req_name":"합가후양도기한","data_field":"years_since_merge","data_type":"duration","threshold":"10년 이내","legal_basis":"소득세법 시행령 제155조 제4항","priority":"critical","question_hint":"양도기한"}]
    md = {"condition_info":{"house_count":"1주택","is_regulated_area":"여","residence_period":"2년+"},"price_info":{"sell_price":"1500000000"}}
    return {"테스트_설명":"LLM없이 Gap Detection만 실행","입력_데이터":md,"판정_결과":run_gap_detection(mr,md,["동거봉양 합가 특례"]).model_dump()}

@app.get("/test-confirm")
async def test_confirm():
    return await confirm(ConfirmRequest(question="부모님을 모시려고 합가했는데 아파트를 팔면 비과세 되나요?",체크리스트응답=[{"질문":"합가로 인해 일시적으로 2주택이 된 경우인가요?","답변":"Yes"},{"질문":"합가일로부터 5년 이내에 양도하는 것인가요?","답변":"Yes"},{"질문":"양도하는 주택이 유일한 주택이었을 때 합가한 것인가요?","답변":"Yes"}],추가정보={"양도가액":"800000000","취득가액":"500000000"},판례검색=[{"사건번호":"조심-2015-전-4851","주제":"1세대 1주택 비과세","결과":"기각"}],추출키워드=["동거봉양","합가","비과세"],참조섹션=[{"섹션":"제4절 1세대 2주택 비과세 특례","페이지":"749-898"}],사용자수정사항=[{"필드":"보유기간","AI값":"2년","수정값":"3년 6개월","영향":"장기보유특별공제율 변경"},{"필드":"합가사유","AI값":"동거봉양","수정값":"동거봉양"},{"필드":"거주기간","AI값":"","수정값":"2년 이상","영향":"거주 요건 충족 여부"}]))

@app.get("/test-feedback-actionable")
async def test_feedback_actionable():
    return await feedback(FeedbackRequest(session_id="test_001",question="부모님을 모시려고 합가했는데 아파트를 팔면 비과세 되나요?",feedback_text="분석 결과에서 합가 전 1주택으로 판단하셨는데, 실제로는 남편 명의 분양권이 하나 더 있었습니다. 분양권도 주택 수에 포함되는 걸로 알고 있는데, 그러면 합가 전 2주택이라서 동거봉양 특례가 안 되는 거 아닌가요?",ai_report={"예상세액":{"비과세_적용시":"0원","비과세_미적용시":"약 1억 2천만원"},"종합의견":"비과세 가능성 높음"},ai_사실관계={"사실관계_요약":{"양도자_현황":"1주택 보유자가 동거봉양 목적으로 합가하여 일시적 2주택"},"비과세_가능성":"높음"},추출키워드=["동거봉양","합가","비과세","분양권","주택수"]))

@app.get("/test-feedback-emotional")
async def test_feedback_emotional():
    return await feedback(FeedbackRequest(session_id="test_002",question="아파트 팔면 세금이 얼마나 나오나요?",feedback_text="세금이 너무 많이 나와요ㅠㅠ 이게 맞나요? 너무 억울해요",ai_report={"예상세액":{"비과세_미적용시":"약 8천만원"},"종합의견":"양도차익에 대한 과세 대상"},추출키워드=["양도소득세"]))

@app.get("/test-feedback-ambiguous")
async def test_feedback_ambiguous():
    return await feedback(FeedbackRequest(session_id="test_003",question="부모님을 모시려고 합가했는데 아파트를 팔면 비과세 되나요?",feedback_text="상속받은 집도 있는데요",ai_report={"예상세액":{"비과세_적용시":"0원"},"종합의견":"비과세 가능성 높음"},추출키워드=["동거봉양","합가","비과세","상속"]))

@app.get("/test-prior-errors")
async def test_prior_errors():
    return {"테스트_질문":"부모님 동거봉양으로 합가했는데 양도세 비과세 가능한가요?","발견된_과거_오류":lookup_prior_errors("부모님 동거봉양으로 합가했는데 양도세 비과세 가능한가요?")}

@app.get("/test-gemini")
async def test_gemini():
    return {"project":os.environ.get("GCP_PROJECT_ID","EMPTY"),"location":os.environ.get("GCP_LOCATION","EMPTY"),"model":os.environ.get("GEMINI_MODEL","EMPTY")}
