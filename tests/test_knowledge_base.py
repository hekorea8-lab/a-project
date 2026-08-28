import tempfile
import unittest
from pathlib import Path

from knowledge_base.ifrs import classify_standard_family
from knowledge_base.store import connect, get_document, search_documents, upsert_document


class KnowledgeBaseStoreTests(unittest.TestCase):
    def test_upsert_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "knowledge.db"
            with connect(db_path) as connection:
                upsert_document(
                    connection,
                    {
                        "document_id": "law:52",
                        "source": "국가법령정보센터",
                        "document_type": "law",
                        "title": "법인세법 제52조",
                        "content": "특수관계인과의 거래로 조세 부담을 부당하게 감소시킨 경우",
                        "source_url": "https://example.test/law/52",
                        "effective_date": "20260701",
                        "version": "21217",
                        "local_path": None,
                    },
                )
                results = search_documents(connection, "특수관계인 거래")
                self.assertEqual([result["document_id"] for result in results], ["law:52"])
                self.assertEqual(get_document(connection, "law:52")["version"], "21217")

    def test_connect_creates_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "nested" / "knowledge.db"
            with connect(db_path) as connection:
                row = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
                ).fetchone()
            self.assertEqual(row[0], "documents")

    def test_classifies_approved_standard_families(self) -> None:
        self.assertEqual(
            classify_standard_family(Path("시행중_K-IFRS_제1024호_특수관계자공시.pdf"), ""),
            "K-IFRS",
        )
        self.assertEqual(
            classify_standard_family(Path("제10장_유형자산.pdf"), ""),
            "일반기업회계기준",
        )
        self.assertIsNone(classify_standard_family(Path("보험업회계처리준칙.pdf"), ""))
