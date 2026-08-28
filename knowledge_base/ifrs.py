import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from .store import upsert_document


def normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(path)
    return normalize_text("\n".join(page.extract_text() or "" for page in reader.pages))


def ifrs_version_from_name(filename: str) -> str:
    match = re.search(r"수정목록[_ ]?(\d{2}-\d)", filename)
    return match.group(1) if match else "current"


def classify_standard_family(path: Path, first_page_text: str) -> str | None:
    """Classify only the accounting-standard families approved for the PoC."""
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


def index_ifrs_directory(connection, directory: Path) -> dict[str, int]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Accounting-standard directory not found: {directory}")
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
            raise ValueError(f"Unable to extract text from PDF: {path.name}")
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
