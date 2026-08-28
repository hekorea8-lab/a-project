"""Minimal read-only MCP server over stdio for the local knowledge database."""

import json
import os
import sys
from pathlib import Path

from .config import DEFAULT_DB_PATH
from .store import connect, get_document, search_documents


TOOLS = [
    {
        "name": "search_knowledge",
        "description": "Search indexed Korean tax laws, precedents, K-IFRS, and general-accounting-standard PDFs. Results include source and version metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Korean search keywords"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_document",
        "description": "Retrieve one indexed source document by its document ID.",
        "inputSchema": {
            "type": "object",
            "properties": {"document_id": {"type": "string"}},
            "required": ["document_id"],
        },
    },
]

DATABASE_PATH = Path(os.environ.get("KNOWLEDGE_DB_PATH", DEFAULT_DB_PATH))


def response(message_id, result=None, error=None) -> dict:
    payload = {"jsonrpc": "2.0", "id": message_id}
    if error:
        payload["error"] = error
    else:
        payload["result"] = result
    return payload


def text_result(value) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=True)}]}


def handle(method: str, params: dict) -> dict:
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "tax-risk-knowledge", "version": "0.1.0"},
        }
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {})
        with connect(DATABASE_PATH) as connection:
            if name == "search_knowledge":
                query = arguments.get("query", "")
                limit = min(max(int(arguments.get("limit", 5)), 1), 20)
                return text_result(search_documents(connection, query, limit))
            if name == "get_document":
                document = get_document(connection, arguments.get("document_id", ""))
                return text_result(document or {"error": "Document not found"})
        raise ValueError(f"Unknown tool: {name}")
    raise ValueError(f"Unsupported method: {method}")


def main() -> None:
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if "id" not in message:
                continue
            result = handle(message["method"], message.get("params", {}))
            print(json.dumps(response(message["id"], result=result), ensure_ascii=True), flush=True)
        except Exception as error:  # Keep protocol errors structured and avoid exposing secrets.
            message_id = message.get("id") if "message" in locals() else None
            print(
                json.dumps(
                    response(message_id, error={"code": -32000, "message": str(error)}),
                    ensure_ascii=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
