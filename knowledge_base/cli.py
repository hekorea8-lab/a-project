import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_DB_PATH, DEFAULT_IFRS_DIR
from .ifrs import index_ifrs_directory
from .law_api import LawApiClient, collect_precedents, collect_tax_laws
from .store import connect, search_documents


def database_path(value: str | None) -> Path:
    return Path(value) if value else DEFAULT_DB_PATH


def write_json(value) -> None:
    """Write UTF-8 safely even when the Windows console uses CP949."""
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    sys.stdout.buffer.write((payload + "\n").encode("utf-8", errors="backslashreplace"))


def refresh_law(args: argparse.Namespace) -> None:
    client = LawApiClient.from_environment()
    with connect(database_path(args.db)) as connection:
        law_count = collect_tax_laws(connection, client)
        precedent_count = collect_precedents(connection, client)
    write_json({"laws": law_count, "precedents": precedent_count})


def index_ifrs(args: argparse.Namespace) -> None:
    directory = Path(args.ifrs_dir) if args.ifrs_dir else DEFAULT_IFRS_DIR
    with connect(database_path(args.db)) as connection:
        indexed = index_ifrs_directory(connection, directory)
    write_json({"indexed_standards": indexed})


def search(args: argparse.Namespace) -> None:
    with connect(database_path(args.db)) as connection:
        results = search_documents(connection, args.query, args.limit)
    write_json(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="External knowledge-base PoC")
    parser.add_argument("--db", help="SQLite database path")
    subparsers = parser.add_subparsers(required=True)

    refresh = subparsers.add_parser("refresh-law", help="Manually refresh laws and precedents")
    refresh.set_defaults(handler=refresh_law)

    index = subparsers.add_parser("index-ifrs", help="Index current K-IFRS and general-accounting-standard PDFs")
    index.add_argument("--ifrs-dir", help="Accounting-standard PDF directory")
    index.set_defaults(handler=index_ifrs)

    search_parser = subparsers.add_parser("search", help="Search indexed documents")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.set_defaults(handler=search)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
