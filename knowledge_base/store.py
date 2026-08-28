import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        initialize(connection)
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            document_type TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            source_url TEXT,
            effective_date TEXT,
            collected_at TEXT NOT NULL,
            version TEXT,
            local_path TEXT,
            standard_family TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_documents_source_type
            ON documents(source, document_type);
        """
    )
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(documents)")}
    if "standard_family" not in columns:
        connection.execute("ALTER TABLE documents ADD COLUMN standard_family TEXT")


def upsert_document(connection: sqlite3.Connection, document: dict[str, str | None]) -> None:
    required = {"document_id", "source", "document_type", "title", "content"}
    missing = required.difference(document)
    if missing:
        raise ValueError(f"Missing document fields: {sorted(missing)}")
    connection.execute(
        """
        INSERT INTO documents (
            document_id, source, document_type, title, content, source_url,
            effective_date, collected_at, version, local_path, standard_family
        ) VALUES (
            :document_id, :source, :document_type, :title, :content, :source_url,
            :effective_date, :collected_at, :version, :local_path, :standard_family
        )
        ON CONFLICT(document_id) DO UPDATE SET
            source=excluded.source,
            document_type=excluded.document_type,
            title=excluded.title,
            content=excluded.content,
            source_url=excluded.source_url,
            effective_date=excluded.effective_date,
            collected_at=excluded.collected_at,
            version=excluded.version,
            local_path=excluded.local_path,
            standard_family=excluded.standard_family
        """,
        {"collected_at": utc_now(), "standard_family": None, **document},
    )


def search_documents(connection: sqlite3.Connection, query: str, limit: int = 5) -> list[dict[str, str | None]]:
    terms = [term for term in re.split(r"\s+", query.strip()) if term]
    if not terms:
        return []
    where = " AND ".join("(title LIKE ? OR content LIKE ?)" for _ in terms)
    parameters: list[str | int] = []
    for term in terms:
        parameters.extend((f"%{term}%", f"%{term}%"))
    parameters.append(limit)
    rows = connection.execute(
        f"""
        SELECT document_id, source, document_type, title, source_url,
               effective_date, collected_at, version, standard_family,
               substr(content, 1, 600) AS excerpt
        FROM documents
        WHERE {where}
        ORDER BY CASE document_type WHEN 'law' THEN 0 WHEN 'precedent' THEN 1 ELSE 2 END,
                 title
        LIMIT ?
        """,
        parameters,
    ).fetchall()
    return [dict(row) for row in rows]


def get_document(connection: sqlite3.Connection, document_id: str) -> dict[str, str | None] | None:
    row = connection.execute(
        """
        SELECT document_id, source, document_type, title, content, source_url,
               effective_date, collected_at, version, local_path, standard_family
        FROM documents WHERE document_id = ?
        """,
        (document_id,),
    ).fetchone()
    return dict(row) if row else None
