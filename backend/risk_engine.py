"""SAP 원장과 특수관계자 목록에서 PoC Risk Score를 계산하는 순수 업무 로직이다."""

import csv
import statistics
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from io import StringIO
from typing import Any

from backend.ledger import normalize_column_name


HIGH_AMOUNT = Decimal("300000000")


def _date(value: str):
    """SAP 전기일자의 대표 형식을 날짜로 변환한다."""
    value = value.strip().replace(".", "-").replace("/", "-")
    for pattern in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            continue
    raise ValueError("전기일자 형식을 읽을 수 없습니다.")


def _amount(value: str) -> Decimal:
    """기준통화 금액의 콤마·공백을 정리해 절대 금액으로 변환한다."""
    value = value.replace(",", "").replace("\u00a0", "").strip()
    if value.startswith("(") and value.endswith(")"):
        value = f"-{value[1:-1]}"
    return abs(Decimal(value))


def _row(row: dict[str, str]) -> dict[str, str]:
    """SAP 추출물의 NBSP·공백 차이를 제거한 컬럼명으로 바꾼다."""
    return {normalize_column_name(key): (value or "").strip() for key, value in row.items()}


def parse_ledger_csv(csv_text: str) -> list[dict[str, Any]]:
    """SAP 원장 CSV를 리스크 산정용 거래 목록으로 정제한다."""
    records = []
    for source in csv.DictReader(StringIO(csv_text)):
        item = _row(source)
        customer_code, vendor_code = item.get("고객", ""), item.get("구매처", "")
        customer_name, vendor_name = item.get("고객명", ""), item.get("구매처명", "")
        records.append({
            "company_code": item.get("회사코드", ""), "voucher_number": item.get("전표번호", ""),
            "posting_date": _date(item.get("전기일자", "")), "line_number": item.get("전표행번", ""),
            "debit_credit": item.get("차대변구분자", ""), "amount": _amount(item.get("전표금액(기준통화)", "")),
            "account_code": item.get("계정과목코드", ""), "account_name": item.get("계정과목명", ""),
            "counterparty_code": customer_code or vendor_code or "미지정",
            "counterparty_name": customer_name or vendor_name,
            "description": item.get("전표적요상세", "") or item.get("전표적요", ""),
        })
    return records


def parse_related_parties(csv_text: str) -> set[tuple[str, str]]:
    """거래처코드·거래처명으로 된 특수관계자 CSV 목록을 읽는다."""
    parties = set()
    for source in csv.DictReader(StringIO(csv_text)):
        item = _row(source)
        if "거래처코드" not in item or "거래처명" not in item:
            raise ValueError("특수관계자 목록에는 거래처코드와 거래처명 컬럼이 필요합니다.")
        parties.add((item["거래처코드"], item["거래처명"]))
    return parties


def _group(record: dict[str, Any]) -> tuple[str, str, str, str]:
    """반복성·변동성 비교를 위한 거래군 식별자다."""
    return record["company_code"], record["account_code"], record["counterparty_code"], record["debit_credit"]


def _recurring(record: dict[str, Any], records: list[dict[str, Any]]) -> bool:
    """최근 12개월 3개월 이상 또는 3년 계절성 기준으로 반복성을 판단한다."""
    current = record["posting_date"]
    history = [item for item in records if _group(item) == _group(record) and item["posting_date"] < current]
    recent_months = {(item["posting_date"].year, item["posting_date"].month) for item in history if 0 < (current.year-item["posting_date"].year)*12+current.month-item["posting_date"].month <= 12}
    seasonal = [item for item in history if item["posting_date"].month == current.month or (item["posting_date"].month-1)//3 == (current.month-1)//3]
    return len(recent_months) >= 3 or len(seasonal) >= 2


def _volatility(record: dict[str, Any], records: list[dict[str, Any]]) -> str | None:
    """과거 월별 금액의 IQR 범위를 벗어난 거래를 주의·심각으로 구분한다."""
    current = record["posting_date"]
    months: dict[tuple[int, int], Decimal] = defaultdict(Decimal)
    for item in records:
        if _group(item) == _group(record) and item["posting_date"] < current:
            months[(item["posting_date"].year, item["posting_date"].month)] += item["amount"]
    values = sorted(months.values())
    if len(values) < 4:
        return None
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    iqr = q3 - q1
    if iqr == 0:
        return "심각" if record["amount"] != q3 else None
    if record["amount"] < q1-3*iqr or record["amount"] > q3+3*iqr:
        return "심각"
    if record["amount"] < q1-Decimal("1.5")*iqr or record["amount"] > q3+Decimal("1.5")*iqr:
        return "주의"
    return None


def score_records(records: list[dict[str, Any]], related_parties: set[tuple[str, str]], analysis_year_month: str) -> list[dict[str, Any]]:
    """합의된 PoC 규칙을 합산해 선택 월의 검토 후보와 Risk Score를 반환한다."""
    year, month = map(int, analysis_year_month.split("-"))
    findings = []
    for record in records:
        if (record["posting_date"].year, record["posting_date"].month) != (year, month):
            continue
        reasons = []
        volatility = _volatility(record, records)
        if volatility: reasons.append({"rule": f"변동성 {volatility}", "score": 35 if volatility == "심각" else 20})
        related = (record["counterparty_code"], record["counterparty_name"]) in related_parties
        recurring = _recurring(record, records)
        history = [item for item in records if _group(item) == _group(record) and item["posting_date"] < record["posting_date"]]
        if related and not recurring and record["amount"] >= HIGH_AMOUNT: reasons.append({"rule": "특수관계자 고액 특이거래", "score": 50})
        if related and not recurring and not history: reasons.append({"rule": "특수관계자 신규·무이력 거래", "score": 35})
        score = min(100, sum(reason["score"] for reason in reasons))
        if score:
            findings.append({**record, "related_party": related, "recurring": recurring, "risk_score": score, "risk_level": "High" if score >= 70 else "Medium" if score >= 40 else "Low", "reasons": reasons})
    return findings
