"""FastAPI 요청과 응답의 데이터 형식이다."""

from pydantic import BaseModel, Field


class CsvHeaderValidationRequest(BaseModel):
    """CSV 원문을 헤더 검증 용도로만 전달한다."""

    csv_text: str


class EvidenceDocument(BaseModel):
    """AI 검토에 제공할 승인된 근거 문서의 최소 정보다."""

    document_id: str
    title: str
    source: str
    source_url: str | None = None
    effective_date_or_version: str | None = None
    excerpt: str


class AiReviewRequest(BaseModel):
    """거래 사실과 승인된 근거 문서만으로 구성하는 AI 검토 요청이다."""

    transaction: dict[str, str | int | float | None]
    evidence_documents: list[EvidenceDocument]


class AutoAiReviewRequest(BaseModel):
    """거래 사실과 쟁점어로 로컬 지식기반을 자동 검색하는 AI 검토 요청이다."""

    transaction: dict[str, str | int | float | None]
    issue_keywords: list[str] = Field(default_factory=list)
    evidence_limit: int = 10


class RiskScoreRequest(BaseModel):
    """SAP 원장·특수관계자 CSV를 분석 월 기준으로 선별하는 요청이다."""

    ledger_csv_text: str
    related_party_csv_text: str
    analysis_year_month: str


class ExpectedTransactionAttachment(BaseModel):
    """예상 거래 사전진단에만 쓰는 사용자 첨부 자료다."""

    filename: str
    content_type: str
    content_base64: str


class ExpectedTransactionRequest(BaseModel):
    """원장 이력이 없는 예상 거래를 근거 기반으로 사전 검토하는 요청이다."""

    company_name: str = ""
    expected_date: str
    account_code: str = ""
    account_name: str
    counterparty_name: str = ""
    related_party: bool = False
    debit_credit: str
    amount: float = Field(gt=0)
    description: str
    facts: str = ""
    issue_keywords: list[str] = Field(default_factory=list)
    attachments: list[ExpectedTransactionAttachment] = Field(default_factory=list, max_length=5)
    evidence_limit: int = Field(default=10, ge=1, le=20)
