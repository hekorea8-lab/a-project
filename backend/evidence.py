"""거래 정보로 승인된 로컬 지식기반을 검색해 AI 근거 묶음을 만드는 로직이다."""

import sqlite3
from pathlib import Path
from typing import Any

from app import DEFAULT_DB_PATH, search_hybrid_documents


TRANSACTION_SEARCH_FIELDS = (
    "계정과목명",
    "전표적요",
    "전표적요상세",
    "고객명",
    "구매처명",
)


# 회계기준과 세무근거를 같은 질문에서 섞어 순위를 잃지 않도록 구분한다.
ACCOUNTING_TERMS = ("회계", "k-ifrs", "ifrs", "일반기업회계", "수익인식", "리스", "손상", "충당", "재고", "유형자산")
TAX_TERMS = ("세법", "세무", "법인세", "부가가치세", "지방세", "조세특례", "공제", "가산세", "판례", "유권", "예규", "시행령", "시행규칙", "조문")
ACCOUNTING_DOCUMENT_TYPES = {"accounting_standard", "internal_accounting_guideline"}
TAX_DOCUMENT_TYPES = {"law", "tax_interpretation", "interpretation", "precedent", "internal_tax_guideline"}


class EvidenceSearchError(RuntimeError):
    """로컬 지식기반 검색을 완료할 수 없을 때 발생한다."""


def classify_evidence_track(queries: list[str]) -> str:
    """질문의 명시 표현을 기준으로 회계·세무·복합 검색 트랙을 구분한다."""
    text = " ".join(queries).lower()
    accounting = any(term in text for term in ACCOUNTING_TERMS)
    tax = any(term in text for term in TAX_TERMS)
    if accounting and tax:
        return "복합"
    if accounting:
        return "회계"
    if tax:
        return "세무"
    return "복합"


def evidence_track(document_type: str | None) -> str:
    """문서 유형을 화면·AI에 전달할 회계 또는 세무 분류로 바꾼다."""
    if document_type in ACCOUNTING_DOCUMENT_TYPES:
        return "회계"
    if document_type in TAX_DOCUMENT_TYPES:
        return "세무"
    return "공통"


def build_search_queries(transaction: dict[str, Any], issue_keywords: list[str]) -> list[str]:
    """사용자·Risk Engine이 준 쟁점어와 거래 설명에서 검색 후보를 만든다."""
    candidates = [str(keyword).strip() for keyword in issue_keywords]
    candidates.extend(str(transaction.get(field) or "").strip() for field in TRANSACTION_SEARCH_FIELDS)
    queries: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in queries:
            queries.append(candidate[:150])
    return queries


def search_local_evidence(
    transaction: dict[str, Any], issue_keywords: list[str], limit: int = 10, db_path: Path = DEFAULT_DB_PATH
) -> dict[str, Any]:
    """지식기반에서 검색된 문서만 AI에 전달할 수 있는 근거 형식으로 변환한다."""
    queries = build_search_queries(transaction, issue_keywords)
    if not queries:
        return {"queries": [], "evidence_track": "복합", "evidence_documents": []}
    requested_track = classify_evidence_track(queries)
    try:
        connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True, timeout=2)
        connection.row_factory = sqlite3.Row
        try:
            documents: list[dict[str, Any]] = []
            document_ids: set[str] = set()
            for query in queries:
                for document in search_hybrid_documents(connection, query, limit=limit):
                    evidence_id = str(document.get("chunk_id") or document["document_id"])
                    if evidence_id in document_ids:
                        continue
                    document_ids.add(evidence_id)
                    documents.append(
                        {
                            "document_id": evidence_id,
                            "title": document["title"],
                            "source": document["source"],
                            "source_url": document["source_url"],
                            "effective_date_or_version": document["effective_date"] or document["version"],
                            "article": document.get("article"),
                            "hierarchy_path": document.get("hierarchy_path"),
                            "excerpt": document["excerpt"],
                            "metadata": {
                                "parent_document_id": document["document_id"],
                                "document_type": document.get("document_type"),
                                "evidence_track": evidence_track(document.get("document_type")),
                                **dict(document.get("metadata") or {}),
                            },
                            "relevance": document.get("relevance") or document.get("similarity"),
                            "relation_info": document.get("relation_info"),
                        }
                    )
                    if len(documents) >= limit * 2:
                        break
                if len(documents) >= limit * 2:
                    break
            preferred = [item for item in documents if item["metadata"]["evidence_track"] == requested_track]
            if requested_track == "복합":
                preferred = [item for item in documents if item["metadata"]["evidence_track"] in {"회계", "세무"}]
            remaining = [item for item in documents if item not in preferred]
            # 복합 질의는 회계와 세무 근거를 한쪽에 치우치지 않게 번갈아 제시한다.
            if requested_track == "복합":
                accounting = [item for item in preferred if item["metadata"]["evidence_track"] == "회계"]
                tax = [item for item in preferred if item["metadata"]["evidence_track"] == "세무"]
                preferred = [item for pair in zip(accounting, tax) for item in pair] + accounting[len(tax):] + tax[len(accounting):]
            documents = (preferred + remaining)[:limit]
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise EvidenceSearchError("지식기반이 갱신 중이거나 검색할 수 없습니다.") from error
    return {"queries": queries, "evidence_track": requested_track, "evidence_documents": documents}
