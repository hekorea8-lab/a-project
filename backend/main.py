"""Streamlit 화면이 호출하는 회계·세무 리스크 PoC API다."""

import os

from fastapi import FastAPI, HTTPException

from backend.ai_review import AiReviewError, MODEL_NAME, prepare_attachments, review_with_openai
from backend.database import compare_with_previous_month, database_status, initialize_monthly_storage, load_active_history, save_monthly_analysis
from backend.evidence import EvidenceSearchError, search_local_evidence
from backend.ledger import REQUIRED_LEDGER_COLUMNS, validate_csv_headers
from backend.risk_engine import parse_ledger_csv, parse_related_parties, score_records
from backend.schemas import AiReviewRequest, AutoAiReviewRequest, CsvHeaderValidationRequest, ExpectedTransactionRequest, RiskScoreRequest


app = FastAPI(title="AI 회계·세무 리스크 PoC API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, object]:
    """화면에서 API와 PostgreSQL 설정 상태를 확인한다."""
    return {"api": "ok", "database": database_status()}


@app.get("/ledger/schema")
def ledger_schema() -> dict[str, object]:
    """SAP 원장 업로드 전 필요한 최소 컬럼을 반환한다."""
    return {"required_columns": REQUIRED_LEDGER_COLUMNS}


@app.post("/ledger/validate-csv")
def validate_csv(payload: CsvHeaderValidationRequest) -> dict[str, object]:
    """화면에서 전달한 CSV 헤더를 순수 원장 검증 로직으로 확인한다."""
    return validate_csv_headers(payload.csv_text)


@app.get("/ai-review/status")
def ai_review_status() -> dict[str, str]:
    """화면이 지정 모델과 키 설정 여부만 확인하도록 한다."""
    return {"model": MODEL_NAME, "status": "configured" if os.environ.get("OPENAI_API_KEY") else "not_configured"}


@app.post("/ai-review")
def ai_review(payload: AiReviewRequest) -> dict[str, object]:
    """승인된 근거 문서를 바탕으로 OpenAI 잠정 검토를 생성한다."""
    try:
        return review_with_openai(payload.transaction, [item.model_dump() for item in payload.evidence_documents])
    except AiReviewError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/ai-review/evidence-preview")
def ai_review_evidence_preview(payload: AutoAiReviewRequest) -> dict[str, object]:
    """AI 호출 전 자동 검색된 근거 문서와 검색어를 확인한다."""
    try:
        return search_local_evidence(payload.transaction, payload.issue_keywords, payload.evidence_limit)
    except EvidenceSearchError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/ai-review/with-auto-evidence")
def ai_review_with_auto_evidence(payload: AutoAiReviewRequest) -> dict[str, object]:
    """로컬 지식기반 검색 결과만 사용해 OpenAI 잠정 검토를 생성한다."""
    try:
        evidence_result = search_local_evidence(payload.transaction, payload.issue_keywords, payload.evidence_limit)
        review_result = review_with_openai(payload.transaction, evidence_result["evidence_documents"])
        return {**evidence_result, **review_result}
    except (EvidenceSearchError, AiReviewError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/risk-score/preview")
def risk_score_preview(payload: RiskScoreRequest) -> dict[str, object]:
    """실제 원장을 저장하지 않고 선택 월의 검토 후보와 점수를 계산한다."""
    try:
        records = parse_ledger_csv(payload.ledger_csv_text)
        related_parties = parse_related_parties(payload.related_party_csv_text)
        findings = score_records(records, related_parties, payload.analysis_year_month)
        risk_summary = {
            level: {
                "count": sum(1 for item in findings if item["risk_level"] == level),
                "amount": sum(item["amount"] for item in findings if item["risk_level"] == level),
            }
            for level in ("High", "Medium", "Low")
        }
        return {
            "analysis_year_month": payload.analysis_year_month,
            "ledger_record_count": len(records),
            "finding_count": len(findings),
            "risk_summary": risk_summary,
            "findings": findings,
        }
    except (ValueError, ArithmeticError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/risk-score/analyze-and-save")
def risk_score_analyze_and_save(payload: RiskScoreRequest) -> dict[str, object]:
    """선택 월 원장을 누적 저장하고 과거 활성 원장과 비교해 위험후보를 산정한다."""
    try:
        uploaded_records = parse_ledger_csv(payload.ledger_csv_text)
        year, month = map(int, payload.analysis_year_month.split("-"))
        current_records = [record for record in uploaded_records if (record["posting_date"].year, record["posting_date"].month) == (year, month)]
        if not current_records:
            raise ValueError("분석 대상 월에 해당하는 원장 행이 없습니다.")
        company_code = current_records[0]["company_code"] or "미지정"
        initialize_monthly_storage()
        history_records = load_active_history(company_code, payload.analysis_year_month)
        related_parties = parse_related_parties(payload.related_party_csv_text)
        findings = score_records([*history_records, *current_records], related_parties, payload.analysis_year_month)
        saved_run = save_monthly_analysis(current_records, findings, payload.analysis_year_month)
        comparison = compare_with_previous_month(company_code, payload.analysis_year_month, findings)
        risk_summary = {
            level: {"count": sum(1 for item in findings if item["risk_level"] == level), "amount": sum(item["amount"] for item in findings if item["risk_level"] == level)}
            for level in ("High", "Medium", "Low")
        }
        return {"analysis_year_month": payload.analysis_year_month, "ledger_record_count": len(current_records), "finding_count": len(findings), "risk_summary": risk_summary, "findings": findings, "storage": saved_run, "comparison": comparison}
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (ValueError, ArithmeticError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def expected_transaction_facts(payload: ExpectedTransactionRequest) -> dict[str, object]:
    """입력한 예정 거래를 기존 근거 검색·AI 검토에 맞는 사실 묶음으로 바꾼다."""
    return {
        "거래구분": "예상 거래",
        "법인명": payload.company_name,
        "예정거래일": payload.expected_date,
        "계정과목코드": payload.account_code,
        "계정과목명": payload.account_name,
        "거래처명": payload.counterparty_name,
        "특수관계자여부": "예" if payload.related_party else "아니오",
        "차대변구분자": payload.debit_credit,
        "검토 대상 거래금액": payload.amount,
        "거래설명": payload.description,
        "추가 사실관계": payload.facts,
    }


def expected_risk_assessment(payload: ExpectedTransactionRequest) -> dict[str, object]:
    """이력 없는 예상 거래에는 자동 판단만 제안하고 점수는 임의로 만들지 않는다."""
    return {
        "status": "산정 보류",
        "risk_score": None,
        "message": "과거 거래 이력 또는 담당자 승인 점수 규칙이 없어 Risk Score를 계산하지 않았습니다. AI가 별도의 검토 관점과 확인사항을 제안합니다.",
    }


@app.post("/expected-transaction/evidence-preview")
def expected_transaction_evidence_preview(payload: ExpectedTransactionRequest) -> dict[str, object]:
    """예상 거래의 사실관계로 로컬 지식기반 근거 후보를 검색한다."""
    try:
        transaction = expected_transaction_facts(payload)
        evidence_result = search_local_evidence(transaction, payload.issue_keywords, payload.evidence_limit)
        return {"transaction": transaction, "risk_assessment": expected_risk_assessment(payload), **evidence_result}
    except EvidenceSearchError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/expected-transaction/diagnose")
def expected_transaction_diagnose(payload: ExpectedTransactionRequest) -> dict[str, object]:
    """자동 검색한 근거만 사용해 예상 거래의 AI 잠정 검토를 생성한다."""
    try:
        transaction = expected_transaction_facts(payload)
        evidence_result = search_local_evidence(transaction, payload.issue_keywords, payload.evidence_limit)
        attachments = prepare_attachments([item.model_dump() for item in payload.attachments])
        review_result = review_with_openai(transaction, evidence_result["evidence_documents"], attachments)
        return {
            "transaction": transaction,
            "risk_assessment": expected_risk_assessment(payload),
            **evidence_result,
            **review_result,
        }
    except (EvidenceSearchError, AiReviewError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
