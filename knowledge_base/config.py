from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "knowledge.db"
DEFAULT_IFRS_DIR = PROJECT_ROOT / "ifrs"
LAW_SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"
LAW_SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"

# The PRD names these three tax laws and their subordinate regulations.
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

# Used only to discover related cases through the official API.
PRECEDENT_QUERIES = ("법인세법", "부가가치세법", "조세특례제한법")
