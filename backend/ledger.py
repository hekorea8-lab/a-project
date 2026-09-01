"""SAP 원장 데이터의 화면·웹 프레임워크와 독립된 검증 로직이다."""

import csv
from io import StringIO


# 사용자가 제공한 SAP 원장 샘플을 기준으로 PoC에서 반드시 확인할 컬럼이다.
REQUIRED_LEDGER_COLUMNS = (
    "회사코드",
    "전표번호",
    "전기일자",
    "전표행번",
    "차대변구분자",
    "전표금액(기준통화)",
    "계정과목코드",
    "계정과목명",
)

# 특수관계자 목록과의 매칭에 사용할 수 있는 상대방 식별 컬럼이다.
COUNTERPARTY_COLUMNS = ("고객", "고객명", "구매처", "구매처명")


def normalize_column_name(name: str) -> str:
    """SAP 추출 파일의 공백·줄바꿈·NBSP 차이를 제거해 컬럼명을 비교한다."""
    return " ".join(name.replace("\u00a0", " ").replace("\ufeff", "").split())


def validate_ledger_headers(headers: list[str]) -> dict[str, list[str] | bool]:
    """원장 헤더에 필수 컬럼과 거래처 식별 컬럼이 있는지 확인한다."""
    normalized_headers = [normalize_column_name(header) for header in headers]
    available = set(normalized_headers)
    missing_required = [column for column in REQUIRED_LEDGER_COLUMNS if column not in available]
    available_counterparty = [column for column in COUNTERPARTY_COLUMNS if column in available]
    return {
        "valid": not missing_required,
        "normalized_headers": normalized_headers,
        "missing_required_columns": missing_required,
        "available_counterparty_columns": available_counterparty,
    }


def validate_csv_headers(csv_text: str) -> dict[str, list[str] | bool]:
    """CSV 첫 행만 읽어 원장 데이터의 컬럼 구성을 검증한다."""
    reader = csv.reader(StringIO(csv_text))
    headers = next(reader, [])
    if not headers:
        return {
            "valid": False,
            "normalized_headers": [],
            "missing_required_columns": list(REQUIRED_LEDGER_COLUMNS),
            "available_counterparty_columns": [],
        }
    return validate_ledger_headers(headers)
