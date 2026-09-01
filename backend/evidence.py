"""거래 정보로 승인된 로컬 지식기반을 검색해 AI 근거 묶음을 만드는 로직이다."""

import sqlite3
from pathlib import Path
from typing import Any

from app import DEFAULT_DB_PATH, search_documents


TRANSACTION_SEARCH_FIELDS = (
    "계정과목명",
    "전표적요",
    "전표적요상세",
    "고객명",
    "구매처명",
)


class EvidenceSearchError(RuntimeError):
    """로컬 지식기반 검색을 완료할 수 없을 때 발생한다."""


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
        return {"queries": [], "evidence_documents": []}
    try:
        connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True, timeout=2)
        connection.row_factory = sqlite3.Row
        try:
            documents: list[dict[str, Any]] = []
            document_ids: set[str] = set()
            for query in queries:
                for document in search_documents(connection, query, limit=limit):
                    if document["document_id"] in document_ids:
                        continue
                    document_ids.add(document["document_id"])
                    documents.append(
                        {
                            "document_id": document["document_id"],
                            "title": document["title"],
                            "source": document["source"],
                            "source_url": document["source_url"],
                            "effective_date_or_version": document["effective_date"] or document["version"],
                            "excerpt": document["excerpt"],
                        }
                    )
                    if len(documents) >= limit:
                        return {"queries": queries, "evidence_documents": documents}
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise EvidenceSearchError("지식기반이 갱신 중이거나 검색할 수 없습니다.") from error
    return {"queries": queries, "evidence_documents": documents}
