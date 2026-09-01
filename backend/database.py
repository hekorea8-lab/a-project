"""PostgreSQL 연결 설정과 최소 데이터베이스 상태 확인 기능이다."""

import json
import os
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text


load_dotenv()


def postgres_url_from_environment() -> URL | None:
    """환경변수가 모두 입력된 경우에만 PostgreSQL 접속 주소를 만든다."""
    host = os.environ.get("POSTGRES_HOST")
    database = os.environ.get("POSTGRES_DATABASE")
    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")
    port = os.environ.get("POSTGRES_PORT", "5432")
    if not all((host, database, user, password)):
        return None
    return URL.create("postgresql+psycopg", username=user, password=password, host=host, port=int(port), database=database)


def database_status() -> dict[str, str]:
    """접속 정보가 없거나 연결 실패 시에도 비밀값 없이 상태만 반환한다."""
    database_url = postgres_url_from_environment()
    if database_url is None:
        return {"status": "not_configured"}
    try:
        engine = create_engine(database_url, pool_pre_ping=True)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "connected"}
    except Exception:
        return {"status": "unavailable"}


def _engine():
    """설정된 PostgreSQL 엔진을 만들고, 접속 정보가 없으면 명확히 중단한다."""
    database_url = postgres_url_from_environment()
    if database_url is None:
        raise RuntimeError("PostgreSQL 접속 정보가 설정되지 않았습니다.")
    return create_engine(database_url, pool_pre_ping=True)


def initialize_monthly_storage() -> None:
    """월별 원장·분석 실행·위험후보 누적에 필요한 테이블과 인덱스를 만든다."""
    statements = (
        """CREATE TABLE IF NOT EXISTS analysis_runs (
            run_id UUID PRIMARY KEY, company_code TEXT NOT NULL, analysis_year_month CHAR(7) NOT NULL,
            version_number INTEGER NOT NULL, is_active BOOLEAN NOT NULL DEFAULT TRUE,
            ledger_record_count INTEGER NOT NULL, finding_count INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE IF NOT EXISTS ledger_entries (
            run_id UUID NOT NULL REFERENCES analysis_runs(run_id), company_code TEXT NOT NULL,
            voucher_number TEXT NOT NULL, line_number TEXT NOT NULL, posting_date DATE NOT NULL,
            debit_credit TEXT NOT NULL, amount NUMERIC(20, 2) NOT NULL, account_code TEXT NOT NULL,
            account_name TEXT NOT NULL, counterparty_code TEXT NOT NULL, counterparty_name TEXT NOT NULL,
            description TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS risk_findings (
            run_id UUID NOT NULL REFERENCES analysis_runs(run_id), finding_key TEXT NOT NULL,
            voucher_number TEXT NOT NULL, line_number TEXT NOT NULL, account_code TEXT NOT NULL,
            account_name TEXT NOT NULL, counterparty_code TEXT NOT NULL, counterparty_name TEXT NOT NULL,
            debit_credit TEXT NOT NULL, amount NUMERIC(20, 2) NOT NULL, risk_score INTEGER NOT NULL,
            risk_level TEXT NOT NULL, reasons JSONB NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS ix_analysis_runs_active_month ON analysis_runs (company_code, analysis_year_month DESC) WHERE is_active",
        "CREATE INDEX IF NOT EXISTS ix_ledger_entries_run_group ON ledger_entries (run_id, account_code, counterparty_code, debit_credit)",
        "CREATE INDEX IF NOT EXISTS ix_risk_findings_run_key ON risk_findings (run_id, finding_key)",
    )
    with _engine().begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def load_active_history(company_code: str, analysis_year_month: str) -> list[dict]:
    """선택 월보다 이전의 활성 원장만 불러와 전월·전년 비교 기준으로 사용한다."""
    statement = text("""SELECT entry.company_code, entry.voucher_number, entry.posting_date, entry.line_number,
        entry.debit_credit, entry.amount, entry.account_code, entry.account_name, entry.counterparty_code,
        entry.counterparty_name, entry.description FROM ledger_entries entry
        JOIN analysis_runs run ON run.run_id = entry.run_id
        WHERE run.company_code = :company_code AND run.analysis_year_month < :analysis_year_month AND run.is_active = TRUE""")
    with _engine().connect() as connection:
        return [dict(row) for row in connection.execute(statement, {"company_code": company_code, "analysis_year_month": analysis_year_month}).mappings()]


def finding_key(record: dict) -> str:
    """월이 달라도 같은 거래군 위험후보인지 비교하는 안정적인 식별자다."""
    return "|".join(str(record.get(name, "")) for name in ("company_code", "account_code", "counterparty_code", "debit_credit"))


def save_monthly_analysis(records: list[dict], findings: list[dict], analysis_year_month: str) -> dict:
    """대량 원장과 위험후보를 새 월 버전으로 저장하고 이전 최신본은 보관한다."""
    if not records:
        raise ValueError("저장할 원장 데이터가 없습니다.")
    company_code = records[0]["company_code"] or "미지정"
    if any((record["company_code"] or "미지정") != company_code for record in records):
        raise ValueError("월별 누적 저장은 한 번에 하나의 회사코드만 지원합니다.")
    run_id = uuid4()
    with _engine().begin() as connection:
        version_number = connection.execute(text("SELECT COALESCE(MAX(version_number), 0) + 1 FROM analysis_runs WHERE company_code = :company_code AND analysis_year_month = :analysis_year_month"), {"company_code": company_code, "analysis_year_month": analysis_year_month}).scalar_one()
        connection.execute(text("UPDATE analysis_runs SET is_active = FALSE WHERE company_code = :company_code AND analysis_year_month = :analysis_year_month AND is_active = TRUE"), {"company_code": company_code, "analysis_year_month": analysis_year_month})
        connection.execute(text("""INSERT INTO analysis_runs (run_id, company_code, analysis_year_month, version_number, ledger_record_count, finding_count)
            VALUES (:run_id, :company_code, :analysis_year_month, :version_number, :ledger_record_count, :finding_count)"""), {"run_id": run_id, "company_code": company_code, "analysis_year_month": analysis_year_month, "version_number": version_number, "ledger_record_count": len(records), "finding_count": len(findings)})
        ledger_statement = text("""INSERT INTO ledger_entries (run_id, company_code, voucher_number, line_number, posting_date, debit_credit, amount, account_code, account_name, counterparty_code, counterparty_name, description)
            VALUES (:run_id, :company_code, :voucher_number, :line_number, :posting_date, :debit_credit, :amount, :account_code, :account_name, :counterparty_code, :counterparty_name, :description)""")
        for offset in range(0, len(records), 5_000):
            connection.execute(ledger_statement, [{**record, "run_id": run_id} for record in records[offset:offset + 5_000]])
        finding_statement = text("""INSERT INTO risk_findings (run_id, finding_key, voucher_number, line_number, account_code, account_name, counterparty_code, counterparty_name, debit_credit, amount, risk_score, risk_level, reasons)
            VALUES (:run_id, :finding_key, :voucher_number, :line_number, :account_code, :account_name, :counterparty_code, :counterparty_name, :debit_credit, :amount, :risk_score, :risk_level, CAST(:reasons AS JSONB))""")
        rows = [{"run_id": run_id, "finding_key": finding_key(item), **item, "reasons": json.dumps(item["reasons"], ensure_ascii=False)} for item in findings]
        if rows:
            connection.execute(finding_statement, rows)
    return {"run_id": str(run_id), "company_code": company_code, "version_number": version_number}


def compare_with_previous_month(company_code: str, analysis_year_month: str, findings: list[dict]) -> dict:
    """직전 활성 분석의 후보와 비교해 신규·지속·해소·등급 변동을 계산한다."""
    query = text("""SELECT run_id, analysis_year_month FROM analysis_runs WHERE company_code = :company_code
        AND analysis_year_month < :analysis_year_month AND is_active = TRUE ORDER BY analysis_year_month DESC LIMIT 1""")
    with _engine().connect() as connection:
        previous = connection.execute(query, {"company_code": company_code, "analysis_year_month": analysis_year_month}).mappings().first()
        if previous is None:
            return {"comparison_month": None, "message": "비교할 이전 월 누적 데이터가 없습니다.", "new": 0, "continued": 0, "resolved": 0, "escalated": 0, "reduced": 0}
        prior_rows = connection.execute(text("SELECT finding_key, risk_level FROM risk_findings WHERE run_id = :run_id"), {"run_id": previous["run_id"]}).mappings()
        prior = {row["finding_key"]: row["risk_level"] for row in prior_rows}
    current = {finding_key(item): item["risk_level"] for item in findings}
    level_order = {"Low": 1, "Medium": 2, "High": 3}
    continued_keys = set(current) & set(prior)
    return {"comparison_month": previous["analysis_year_month"], "message": "직전 활성 분석월과 비교했습니다.", "new": len(set(current) - set(prior)), "continued": len(continued_keys), "resolved": len(set(prior) - set(current)), "escalated": sum(level_order[current[key]] > level_order[prior[key]] for key in continued_keys), "reduced": sum(level_order[current[key]] < level_order[prior[key]] for key in continued_keys)}
