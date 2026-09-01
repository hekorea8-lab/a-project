"""AI 회계·세무 리스크 PoC의 외부 기준 데이터 지식 기반 도구.

사용자 실행형 법령·판례 갱신, 회계기준 PDF 색인, 기준 검색과 읽기 전용 MCP를 제공한다.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "knowledge.db"
DEFAULT_IFRS_DIR = PROJECT_ROOT / "ifrs"
LAW_SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"
LAW_SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"

# PoC에서 우선 수집하는 세법과 하위 규정이다.
TAX_LAW_NAMES = (
    "법인세법",
    "법인세법 시행령",
    "법인세법 시행규칙",
    "부가가치세법",
    "부가가치세법 시행령",
    "부가가치세법 시행규칙",
    "조세특례제한법",
    "조세특례제한법 시행령",
    "조세특례제한법 시행규칙",
)

# 공식 판례 검색에 사용하는 세법별 검색어다.
PRECEDENT_QUERIES = ("법인세법", "부가가치세법", "조세특례제한법")


class LawApiError(RuntimeError):
    """국가법령정보 Open API 호출 또는 응답 처리 오류다."""


def utc_now() -> str:
    """수집 시각을 비교 가능한 UTC ISO 형식으로 반환한다."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """SQLite 연결을 열고 스키마 생성과 커밋을 함께 처리한다."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        initialize_database(connection)
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database(connection: sqlite3.Connection) -> None:
    """문서 원문과 근거 메타데이터를 보관할 테이블을 만든다."""
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
    """같은 공식 문서는 최신 수집 내용과 버전으로 갱신한다."""
    required = {"document_id", "source", "document_type", "title", "content"}
    missing = required.difference(document)
    if missing:
        raise ValueError(f"필수 문서 항목이 없습니다: {sorted(missing)}")
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


def search_documents(
    connection: sqlite3.Connection, query: str, limit: int = 5
) -> list[dict[str, str | None]]:
    """모든 검색어를 포함하는 기준 문서를 찾아 메타데이터와 발췌문을 반환한다."""
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


def get_document(
    connection: sqlite3.Connection, document_id: str
) -> dict[str, str | None] | None:
    """문서 ID로 원문과 출처·버전 정보를 조회한다."""
    row = connection.execute(
        """
        SELECT document_id, source, document_type, title, content, source_url,
               effective_date, collected_at, version, local_path, standard_family
        FROM documents WHERE document_id = ?
        """,
        (document_id,),
    ).fetchone()
    return dict(row) if row else None


def normalize_text(text: str) -> str:
    """PDF·XML 추출 결과의 과도한 빈 줄을 정리한다."""
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def ifrs_version_from_name(filename: str) -> str:
    """파일명에 표시된 수정목록 버전을 사용하고 없으면 current로 기록한다."""
    match = re.search(r"수정목록[_ ]?(\d{2}-\d)", filename)
    return match.group(1) if match else "current"


def classify_standard_family(path: Path, first_page_text: str) -> str | None:
    """PoC 범위인 K-IFRS와 일반기업회계기준만 구분한다."""
    filename = path.name
    if filename.startswith("시행중_K-IFRS_"):
        return "K-IFRS"
    if (
        "일반기업회계기준" in filename
        or filename.startswith("재무회계개념체계")
        or re.match(r"제\d+장_", filename)
        or "일반기업회계기준" in first_page_text
    ):
        return "일반기업회계기준"
    return None


def index_ifrs_directory(connection: sqlite3.Connection, directory: Path) -> dict[str, int]:
    """로컬 회계기준 PDF를 텍스트화해 검색 데이터베이스에 등록한다."""
    if not directory.is_dir():
        raise FileNotFoundError(f"회계기준 폴더를 찾을 수 없습니다: {directory}")
    indexed = {"K-IFRS": 0, "일반기업회계기준": 0, "excluded": 0}
    for path in sorted(directory.glob("*.pdf")):
        reader = PdfReader(path)
        first_page_text = reader.pages[0].extract_text() or ""
        standard_family = classify_standard_family(path, first_page_text)
        if standard_family is None:
            indexed["excluded"] += 1
            continue
        content = normalize_text("\n".join(page.extract_text() or "" for page in reader.pages))
        if not content:
            raise ValueError(f"PDF에서 텍스트를 추출할 수 없습니다: {path.name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        upsert_document(
            connection,
            {
                "document_id": f"ifrs:{digest}",
                "source": f"한국회계기준원 제공 {standard_family} PDF",
                "document_type": "accounting_standard",
                "title": path.stem,
                "content": content,
                "source_url": None,
                "effective_date": None,
                "version": ifrs_version_from_name(path.name),
                "local_path": str(path.resolve()),
                "standard_family": standard_family,
            },
        )
        indexed[standard_family] += 1
    return indexed


@dataclass(frozen=True)
class LawApiClient:
    """키를 외부에 출력하지 않고 국가법령정보 Open API를 호출한다."""

    oc: str
    timeout_seconds: int = 30

    @classmethod
    def from_environment(cls) -> "LawApiClient":
        oc = os.environ.get("LAW_API_OC")
        if not oc:
            raise LawApiError("LAW_API_OC가 비어 있습니다. .env에 직접 입력한 후 다시 실행하세요.")
        return cls(oc=oc)

    def request(self, endpoint: str, **parameters: str | int) -> ET.Element:
        """키가 포함된 요청 주소는 저장·출력하지 않고 XML 응답만 처리한다."""
        query = urllib.parse.urlencode({"OC": self.oc, **parameters})
        request = urllib.request.Request(
            f"{endpoint}?{query}", headers={"User-Agent": "tax-risk-poc/0.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            # 인증값과 요청 주소는 민감 정보이므로 HTTP 상태만 안내한다.
            raise LawApiError(f"국가법령정보 Open API가 HTTP {error.code} 응답을 반환했습니다.") from error
        except OSError as error:
            raise LawApiError("국가법령정보 Open API 요청에 실패했습니다.") from error
        try:
            root = ET.fromstring(body)
        except ET.ParseError as error:
            raise LawApiError("국가법령정보 Open API가 XML이 아닌 응답을 반환했습니다.") from error
        if root.tag.lower() == "error" or root.find(".//error") is not None:
            # API 본문의 상세 오류는 키 또는 요청 정보를 포함할 수 있어 출력하지 않는다.
            raise LawApiError("국가법령정보 Open API가 오류 응답을 반환했습니다.")
        return root

    def law_by_name(self, name: str) -> ET.Element:
        return self.request(LAW_SERVICE_URL, target="law", type="XML", LM=name)

    def precedent_page(self, query: str, page: int) -> ET.Element:
        return self.request(
            LAW_SEARCH_URL,
            target="prec",
            type="XML",
            search=2,
            query=query,
            display=100,
            page=page,
        )

    def precedent_by_id(self, precedent_id: str) -> ET.Element:
        return self.request(LAW_SERVICE_URL, target="prec", type="XML", ID=precedent_id)


def text_of(element: ET.Element, *names: str) -> str | None:
    """후보 XML 태그 중 실제 값이 있는 첫 값을 반환한다."""
    for name in names:
        found = element.find(f".//{name}")
        if found is not None and found.text:
            return " ".join(found.itertext()).strip()
    return None


def flatten_xml(element: ET.Element) -> str:
    """공식 XML의 모든 텍스트를 검색 가능한 원문으로 평탄화한다."""
    return "\n".join(part.strip() for part in element.itertext() if part and part.strip())


def collect_tax_laws(connection: sqlite3.Connection, client: LawApiClient) -> int:
    """법인세·부가가치세·조특법과 시행령·시행규칙을 함께 갱신한다."""
    count = 0
    for name in TAX_LAW_NAMES:
        root = client.law_by_name(name)
        title = text_of(root, "법령명한글", "법령명") or name
        law_id = text_of(root, "법령ID", "법령일련번호") or title
        effective_date = text_of(root, "시행일자")
        proclamation_number = text_of(root, "공포번호")
        upsert_document(
            connection,
            {
                "document_id": f"law:{law_id}",
                "source": "국가법령정보센터",
                "document_type": "law",
                "title": title,
                "content": flatten_xml(root),
                "source_url": f"https://www.law.go.kr/법령/{urllib.parse.quote(title)}",
                "effective_date": effective_date,
                "version": proclamation_number,
                "local_path": None,
            },
        )
        count += 1
    return count


def precedent_ids(root: ET.Element) -> list[str]:
    """판례 검색 목록에서 중복 없는 판례 식별자를 추출한다."""
    ids: list[str] = []
    for item in root.findall(".//prec"):
        value = item.attrib.get("id") or text_of(item, "판례일련번호", "판례정보일련번호")
        if value:
            ids.append(value)
    return list(dict.fromkeys(ids))


def total_count(root: ET.Element) -> int:
    """공식 검색 응답의 전체 건수를 안전하게 정수로 읽는다."""
    raw = text_of(root, "totalCnt", "총건수")
    return int(raw) if raw and raw.isdigit() else 0


def collect_precedents(
    connection: sqlite3.Connection, client: LawApiClient, pause_seconds: float = 0.1
) -> int:
    """세법별 공식 검색 결과의 판례 원문을 중복 없이 갱신한다."""
    seen: set[str] = set()
    count = 0
    for query in PRECEDENT_QUERIES:
        first_page = client.precedent_page(query, 1)
        pages = max(1, (total_count(first_page) + 99) // 100)
        page_roots = [first_page]
        for page in range(2, pages + 1):
            time.sleep(pause_seconds)
            page_roots.append(client.precedent_page(query, page))
        for page_root in page_roots:
            for precedent_id in precedent_ids(page_root):
                if precedent_id in seen:
                    continue
                seen.add(precedent_id)
                time.sleep(pause_seconds)
                root = client.precedent_by_id(precedent_id)
                title = text_of(root, "사건명") or f"판례 {precedent_id}"
                sentence_date = text_of(root, "선고일자")
                upsert_document(
                    connection,
                    {
                        "document_id": f"precedent:{precedent_id}",
                        "source": "국가법령정보센터",
                        "document_type": "precedent",
                        "title": title,
                        "content": flatten_xml(root),
                        "source_url": (
                            "https://www.law.go.kr/LSW/precInfoP.do?precSeq="
                            f"{urllib.parse.quote(precedent_id)}"
                        ),
                        "effective_date": sentence_date,
                        "version": sentence_date,
                        "local_path": None,
                    },
                )
                count += 1
    return count


def write_json(value: object) -> None:
    """Windows 콘솔 인코딩과 무관하게 UTF-8 JSON만 출력한다."""
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    sys.stdout.buffer.write((payload + "\n").encode("utf-8", errors="backslashreplace"))


def database_path(value: str | None) -> Path:
    """명령행 DB 경로가 없으면 프로젝트의 로컬 DB를 사용한다."""
    return Path(value) if value else DEFAULT_DB_PATH


def refresh_law(args: argparse.Namespace) -> None:
    """사용자가 직접 실행한 경우에만 전체 법령·판례 갱신을 수행한다."""
    client = LawApiClient.from_environment()
    with connect(database_path(args.db)) as connection:
        law_count = collect_tax_laws(connection, client)
        precedent_count = collect_precedents(connection, client)
    write_json({"laws": law_count, "precedents": precedent_count})


def index_ifrs(args: argparse.Namespace) -> None:
    """사용자 폴더의 K-IFRS·일반기업회계기준 PDF를 다시 색인한다."""
    directory = Path(args.ifrs_dir) if args.ifrs_dir else DEFAULT_IFRS_DIR
    with connect(database_path(args.db)) as connection:
        indexed = index_ifrs_directory(connection, directory)
    write_json({"indexed_standards": indexed})


def search(args: argparse.Namespace) -> None:
    """명령행에서 기준 문서를 검색한다."""
    with connect(database_path(args.db)) as connection:
        results = search_documents(connection, args.query, args.limit)
    write_json(results)


MCP_TOOLS = [
    {
        "name": "search_knowledge",
        "description": "색인된 한국 세법·판례·K-IFRS·일반기업회계기준 PDF를 검색합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색할 한글 키워드"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_document",
        "description": "문서 ID로 색인된 원문과 출처 메타데이터를 조회합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {"document_id": {"type": "string"}},
            "required": ["document_id"],
        },
    },
]


def mcp_response(message_id: object, result: object = None, error: object = None) -> dict:
    """MCP JSON-RPC 형식의 성공 또는 오류 응답을 만든다."""
    payload = {"jsonrpc": "2.0", "id": message_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    return payload


def mcp_text_result(value: object) -> dict:
    """MCP 도구 반환값을 텍스트 콘텐츠로 감싼다."""
    return {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=True)}]}


def handle_mcp_request(method: str, params: dict, db_path: Path) -> dict:
    """검색과 문서 조회만 허용하는 읽기 전용 MCP 요청을 처리한다."""
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "tax-risk-knowledge", "version": "0.1.0"},
        }
    if method == "tools/list":
        return {"tools": MCP_TOOLS}
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {})
        with connect(db_path) as connection:
            if name == "search_knowledge":
                query = arguments.get("query", "")
                limit = min(max(int(arguments.get("limit", 5)), 1), 20)
                return mcp_text_result(search_documents(connection, query, limit))
            if name == "get_document":
                document = get_document(connection, arguments.get("document_id", ""))
                return mcp_text_result(document or {"error": "문서를 찾을 수 없습니다."})
        raise ValueError(f"허용되지 않은 MCP 도구입니다: {name}")
    raise ValueError(f"지원하지 않는 MCP 메서드입니다: {method}")


def run_mcp(args: argparse.Namespace) -> None:
    """표준 입출력으로 읽기 전용 MCP 서버를 실행한다."""
    db_path = database_path(args.db)
    for line in sys.stdin:
        message: dict = {}
        try:
            message = json.loads(line)
            if "id" not in message:
                continue
            result = handle_mcp_request(message["method"], message.get("params", {}), db_path)
            print(json.dumps(mcp_response(message["id"], result=result), ensure_ascii=True), flush=True)
        except Exception as error:
            print(
                json.dumps(
                    mcp_response(message.get("id"), error={"code": -32000, "message": str(error)}),
                    ensure_ascii=True,
                ),
                flush=True,
            )


def build_parser() -> argparse.ArgumentParser:
    """사용자 실행형 색인·갱신·검색 명령을 정의한다."""
    parser = argparse.ArgumentParser(description="AI 회계·세무 리스크 PoC 지식 기반")
    parser.add_argument("--db", help="SQLite 데이터베이스 경로")
    subparsers = parser.add_subparsers(required=True)

    refresh = subparsers.add_parser("refresh-law", help="법령·시행령·시행규칙·판례를 수동 갱신")
    refresh.set_defaults(handler=refresh_law)

    index = subparsers.add_parser("index-ifrs", help="K-IFRS·일반기업회계기준 PDF 색인")
    index.add_argument("--ifrs-dir", help="회계기준 PDF 폴더")
    index.set_defaults(handler=index_ifrs)

    search_parser = subparsers.add_parser("search", help="색인 문서 검색")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.set_defaults(handler=search)

    mcp_parser = subparsers.add_parser("mcp", help="읽기 전용 MCP 서버 실행")
    mcp_parser.set_defaults(handler=run_mcp)
    return parser


def main() -> None:
    """명령행 인자를 처리하고 선택한 기능을 실행한다."""
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except (LawApiError, FileNotFoundError, ValueError) as error:
        # 오류 메시지에는 API 키나 요청 주소를 포함하지 않는다.
        write_json({"error": str(error)})
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
