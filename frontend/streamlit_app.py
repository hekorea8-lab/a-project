"""Stitch 디자인을 반영한 회계·세무 리스크 PoC Streamlit 화면이다."""

import base64
import html
import json
import os
import re
import urllib.error
import urllib.request

import streamlit as st
from dotenv import load_dotenv


load_dotenv()
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def call_api(path: str, method: str = "GET", payload: dict | None = None, timeout_seconds: int = 10) -> dict:
    """화면은 HTTP API만 호출하고 업무 로직을 직접 실행하지 않는다."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None
    request = urllib.request.Request(
        f"{API_BASE_URL}{path}", data=body, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except TimeoutError:
        return {"error": "AI 검토 응답이 지연되고 있습니다. 잠시 후 다시 시도하세요."}
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8")).get("detail", "")
        except json.JSONDecodeError:
            detail = ""
        return {"error": str(detail) or "요청 내용을 처리하지 못했습니다."}
    except (urllib.error.URLError, json.JSONDecodeError):
        return {"error": "백엔드 API에 연결할 수 없습니다."}


def apply_style() -> None:
    """Precision Analytical 디자인 체계를 적용한다."""
    st.markdown(
        """
        <style>
        .stApp {background:#f7f9fb;color:#191c1e;font-family:Inter,"Malgun Gothic",sans-serif;}
        [data-testid="stSidebar"] {background:#f2f4f6;border-right:1px solid #e2e8f0;}
        [data-testid="stSidebar"] * {color:#404752;}
        [data-testid="stSidebar"] .stRadio label {padding:10px 12px;border-radius:4px;}
        [data-testid="stSidebar"] .stRadio label:hover {background:#e6e8ea;}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#404752;}
        .brand {background:#005faa;color:#fff!important;font-size:20px;font-weight:700;margin:-1rem -1rem 0;padding:20px 24px 2px;}
        .brand-sub {background:#005faa;color:#d3e3ff!important;font-size:11px;letter-spacing:1.5px;margin:0 -1rem;padding:0 24px 18px;}
        .kicker {color:#005faa;font-size:12px;font-weight:700;letter-spacing:1.5px;font-family:"JetBrains Mono",monospace;}
        .title {font-size:32px;font-weight:700;margin:3px 0 8px;letter-spacing:-0.02em;}
        .description {color:#404752;font-size:15px;max-width:850px;}
        .card {background:#fff;border:1px solid #e2e8f0;border-radius:4px;padding:18px;box-shadow:none;}
        .label {color:#404752;font-size:12px;font-weight:700;letter-spacing:1px;font-family:"JetBrains Mono",monospace;}
        .value {font-size:24px;font-weight:700;margin-top:4px;}
        .high {color:#d11010;}.medium {color:#d97706;}.low {color:#107c10;}
        .empty {background:#fff;border:1px dashed #c0c7d4;border-radius:4px;padding:36px 20px;text-align:center;color:#404752;}
        .stButton > button {background:#005faa;color:#fff;border:1px solid #005faa;border-radius:4px;}
        .stButton > button:hover {background:#004883;border-color:#004883;}
        .dashboard-shell {background:#f7f9fb;}
        .dashboard-panel {background:#fff;border:1px solid #e2e8f0;border-radius:4px;padding:20px;}
        .metric-card {background:#fff;border:1px solid #e2e8f0;border-radius:4px;padding:18px 18px 12px;min-height:130px;}
        .metric-bar {height:4px;margin:18px -18px -12px;background:#005faa;}
        .metric-bar.high {background:#d11010;}.metric-bar.medium {background:#d97706;}.metric-bar.low {background:#107c10;}.metric-bar.none {background:#94a3b8;}
        .ai-conclusion {background:#fff;border:1px solid #e2e8f0;border-left:6px solid #d97706;border-radius:4px;padding:20px;margin:8px 0 18px;}
        .ai-conclusion.positive {border-left-color:#107c10;}.ai-conclusion.warning {border-left-color:#d97706;}.ai-conclusion.danger {border-left-color:#d11010;}
        .ai-conclusion-label {color:#404752;font-size:12px;font-weight:700;letter-spacing:1px;font-family:"JetBrains Mono",monospace;}
        .ai-conclusion-status {font-size:26px;font-weight:700;margin:6px 0;}.ai-conclusion-text {font-size:16px;line-height:1.65;}
        .report-draft {background:#fff;border:1px solid #cbd5e1;border-radius:6px;padding:24px;white-space:pre-wrap;line-height:1.85;color:#1f2937;}
        .ai-loading-card {background:#eef6ff;border:1px solid #a8cdf2;border-radius:8px;margin:18px 0;padding:28px;display:flex;align-items:center;gap:24px;}
        .ai-loading-spinner {width:52px;height:52px;border:7px solid #c5dff7;border-top-color:#005faa;border-radius:50%;animation:ai-loading-rotate 0.9s linear infinite;flex:0 0 auto;}
        .ai-loading-title {font-size:20px;font-weight:700;color:#003f72;margin-bottom:6px;}.ai-loading-detail {font-size:14px;color:#35546f;line-height:1.7;}
        @keyframes ai-loading-rotate {to {transform:rotate(360deg);}}
        .filter-title {font-size:18px;font-weight:600;margin:0 0 16px;}
        [data-testid="stToolbar"] {visibility:hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(kicker: str, title: str, description: str) -> None:
    """공통 제목 영역을 렌더링한다."""
    st.markdown(f'<div class="kicker">{kicker}</div><div class="title">{title}</div><div class="description">{description}</div>', unsafe_allow_html=True)
    st.divider()


def card(label: str, value: str, color: str = "") -> None:
    """실제 상태만 보여주는 대시보드 요약 카드를 만든다."""
    st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value {color}">{value}</div></div>', unsafe_allow_html=True)


def dashboard_metric(label: str, value: str, color: str = "none") -> None:
    """빈 상태를 포함해 대시보드 지표를 새 디자인의 하단 리스크 바로 표현한다."""
    value_color = {"high": "high", "medium": "medium", "low": "low"}.get(color, "")
    st.markdown(
        f'<div class="metric-card"><div class="label">{label}</div><div class="value {value_color}">{value}</div><div class="metric-bar {color}"></div></div>',
        unsafe_allow_html=True,
    )


def format_amount(value: object) -> str:
    """금액을 천 단위 쉼표로 표시하고 값이 없으면 하이픈으로 보여준다."""
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "-"


def transaction_for_ai_review(finding: dict) -> tuple[dict, list[str]]:
    """위험후보의 객체형 탐지 사유를 AI 요청에 안전한 텍스트로 바꾼다."""
    transaction = finding.copy()
    reasons = transaction.pop("reasons", [])
    keywords = [str(reason.get("rule", "")) for reason in reasons if reason.get("rule")]
    transaction["탐지사유"] = ", ".join(
        f"{reason['rule']} ({reason['score']}점)" for reason in reasons if reason.get("rule") and reason.get("score") is not None
    ) or "탐지 사유 정보 없음"
    transaction["전표적요"] = transaction.pop("description", "")
    return transaction, keywords


def show_ai_loading(placeholder) -> None:
    """AI 응답을 기다리는 동안 진행 중임을 명확히 보여주는 큰 로딩 카드를 표시한다."""
    placeholder.markdown(
        '<div class="ai-loading-card"><div class="ai-loading-spinner"></div><div>'
        '<div class="ai-loading-title">AI가 거래를 검토하고 있습니다</div>'
        '<div class="ai-loading-detail">거래 사실과 위험후보를 확인한 뒤 관련 기준·판례를 검색하고 있습니다.<br>AI 잠정 의견 작성과 근거 검증까지 잠시 기다려 주세요.</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def transaction_with_follow_up_answers(transaction: dict, history: list[dict], previous_review: dict) -> dict:
    """담당자 답변과 직전 결론을 별도 사실관계로 묶어 재검토 입력을 만든다."""
    enriched = transaction.copy()
    enriched["담당자 추가 확인 답변"] = "\n\n".join(
        f"질문: {item['question']}\n담당자 답변: {item['answer']}" for item in history
    )
    conclusion = previous_review.get("provisional_conclusion", {})
    enriched["직전 AI 잠정결론"] = json.dumps(
        {"status": conclusion.get("status", ""), "statement": conclusion.get("statement", "")},
        ensure_ascii=False,
    )
    return enriched


def expected_payload_with_follow_up_answers(payload: dict, history: list[dict], previous_review: dict) -> dict:
    """예상 거래의 원본 입력은 유지하고 담당자 답변만 추가 사실관계로 합친다."""
    enriched = payload.copy()
    conclusion = previous_review.get("provisional_conclusion", {})
    history_text = "\n\n".join(f"질문: {item['question']}\n담당자 답변: {item['answer']}" for item in history)
    enriched["facts"] = "\n\n".join(part for part in (
        payload.get("facts", ""),
        f"[직전 AI 잠정결론]\n{conclusion.get('status', '')}: {conclusion.get('statement', '')}",
        f"[담당자 추가 확인 답변]\n{history_text}",
    ) if part)
    return enriched


def render_ai_review(review: dict) -> None:
    """사용자에게 프롬프트나 원시 JSON 대신 표준 검토 Report를 표시한다."""
    conclusion = review.get("provisional_conclusion", {})
    status = conclusion.get("status", "판단 결과 없음")
    status_style = {
        "적정 가능성": "positive",
        "비적정 가능성": "danger",
    }.get(status, "warning")
    conclusion_text = conclusion.get("statement", "AI 잠정 결론이 반환되지 않았습니다.")
    conclusion_evidence = ", ".join(conclusion.get("evidence_ids", [])) or "연결된 근거 없음"
    st.markdown("### AI 잠정 의견")
    st.markdown(
        f'<div class="ai-conclusion {status_style}"><div class="ai-conclusion-label">잠정 결론</div>'
        f'<div class="ai-conclusion-status">{html.escape(str(status))}</div><div class="ai-conclusion-text">{html.escape(str(conclusion_text))}</div>'
        f'<div class="ai-conclusion-label" style="margin-top:14px;">근거 문서: {html.escape(str(conclusion_evidence))}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("아래 내용은 OpenAI가 승인된 근거 문서와 입력 사실을 바탕으로 작성한 잠정 검토입니다. 담당자의 최종 판단을 대체하지 않습니다.")
    confidence_level = conclusion.get("confidence_level", "미표시")
    st.caption(f"판단 신뢰 수준: {confidence_level} — 낮음은 결론을 내지 못했다는 뜻이 아니라, 추가 사실·증빙에 따라 결론이 바뀔 가능성이 크다는 뜻입니다.")
    report_draft = review.get("report_draft", "")
    if report_draft:
        st.markdown("### 보고서 문안")
        st.markdown(f'<div class="report-draft">{html.escape(str(report_draft))}</div>', unsafe_allow_html=True)
    expert_opinion = review.get("expert_opinion", "")
    if expert_opinion:
        st.markdown("### 전문가 검토 의견")
        st.info(expert_opinion, icon="🧾")
    refinement = review.get("refinement", {})
    if refinement.get("statement"):
        st.info(f"재검토 반영: {refinement['statement']}")
    attached_findings = review.get("attached_document_findings", [])
    review_focus = review.get("suggested_review_focus", [])
    if attached_findings or review_focus:
        attachment_column, focus_column = st.columns(2)
        with attachment_column:
            st.markdown("#### 첨부 자료 확인 내용")
            if attached_findings:
                for item in attached_findings:
                    st.markdown(f"- **{item.get('filename', '첨부 자료')}**: {item.get('statement', '')}")
            else:
                st.caption("첨부 자료가 없거나 AI가 별도 확인 내용을 제시하지 않았습니다.")
        with focus_column:
            st.markdown("#### AI 자동 제안 검토 관점")
            if review_focus:
                for item in review_focus:
                    st.markdown(f"- {item}")
            else:
                st.caption("AI가 별도 검토 관점을 제시하지 않았습니다.")
    sections = (
        ("확인된 사실", review.get("confirmed_facts", [])),
        ("적용 기준 및 근거", review.get("applicable_standards", [])),
        ("근거 기반 추론", review.get("reasoning", [])),
        ("반대 논리", review.get("counterarguments", [])),
    )
    for title, items in sections:
        with st.expander(title, expanded=title in ("확인된 사실", "적용 기준 및 근거")):
            if not items:
                st.caption("제시된 내용이 없습니다.")
                continue
            for item in items:
                statement = item.get("statement", "")
                evidence_ids = ", ".join(item.get("evidence_ids", [])) or "연결된 근거 없음"
                issue_type = item.get("issue_type", "")
                if issue_type:
                    st.markdown(f"**{issue_type}**")
                st.write(statement)
                st.caption(f"근거 문서 ID: {evidence_ids}")
    required_evidence = review.get("required_evidence", [])
    reviewer_actions = review.get("reviewer_actions", [])
    uncertainty = review.get("uncertainty", [])
    columns = st.columns(3)
    for column, title, items in zip(
        columns,
        ("확인 필요 증빙", "담당자 확인사항", "불확실성·제한사항"),
        (required_evidence, reviewer_actions, uncertainty),
    ):
        with column:
            st.markdown(f"#### {title}")
            if items:
                for item in items:
                    st.markdown(f"- {item}")
            else:
                st.caption("제시된 내용이 없습니다.")


def render_follow_up_dialogue(transaction: dict, keywords: list[str], review: dict) -> None:
    """사용자가 필요할 때만 추가 사실을 반영해 잠정 결론을 재검토한다."""
    questions = review.get("follow_up_questions", [])
    if not st.session_state.get("review_follow_up_open"):
        if st.button("추가 사실 반영 재검토", key="open_review_follow_up"):
            st.session_state["review_follow_up_open"] = True
        else:
            return
    if not questions:
        questions = [{"question_id": "USER", "question": "추가로 반영할 사실관계 또는 증빙 내용은 무엇입니까?", "why_needed": "사용자가 필요하다고 판단한 추가 정보를 반영합니다.", "conclusion_impact": "현재 잠정 결론의 신뢰 수준 또는 방향이 달라질 수 있습니다.", "priority": "보통"}]
        return
    st.divider()
    st.subheader("결론을 좁히기 위한 추가 확인")
    st.caption("아래 질문에 확인된 사실 또는 보유 증빙 기준으로 답변하면 AI가 기존 근거와 함께 재검토합니다. 답변은 현재 브라우저 세션에만 보관됩니다.")
    for question in questions:
        with st.container(border=True):
            st.markdown(f"**{question.get('priority', '보통')} · {question.get('question_id', '확인 질문')}**")
            st.write(question.get("question", ""))
            st.caption(f"필요한 이유: {question.get('why_needed', '제시된 이유 없음')}")
            st.caption(f"결론 영향: {question.get('conclusion_impact', '제시된 영향 없음')}")
    history = st.session_state.setdefault("review_follow_up_history", [])
    if history:
        st.markdown("#### 답변 이력")
        for item in history:
            with st.chat_message("assistant"):
                st.write(item["question"])
            with st.chat_message("user"):
                st.write(item["answer"])
    selected_question = st.selectbox(
        "답변할 추가 확인 질문",
        questions,
        format_func=lambda item: f"{item.get('question_id', '질문')} · {item.get('question', '')}",
        key="review_follow_up_question",
    )
    answer = st.chat_input(
        "선택한 질문에 대한 확인된 사실 또는 증빙 내용을 입력하세요",
        key="review_follow_up_answer",
        submit_mode="disable",
    )
    if answer:
        history.append({"question_id": selected_question.get("question_id", ""), "question": selected_question.get("question", ""), "answer": answer})
        enriched_transaction = transaction_with_follow_up_answers(transaction, history, review)
        loading_area = st.empty()
        show_ai_loading(loading_area)
        try:
            with st.status("담당자 답변을 반영해 AI 재검토 진행 중", expanded=True, width="stretch") as status:
                st.write("추가 사실관계와 직전 잠정 결론을 비교하고 있습니다.")
                st.write("기존 근거와 답변을 함께 검토하고 있습니다.")
                result = call_api("/ai-review/with-auto-evidence", "POST", {"transaction": enriched_transaction, "issue_keywords": keywords, "evidence_limit": 10}, timeout_seconds=120)
                st.session_state["ai_review_result"] = result
                if "error" in result:
                    status.update(label="AI 재검토를 완료하지 못했습니다", state="error", expanded=True)
                else:
                    status.update(label="AI 재검토가 완료되었습니다", state="complete", expanded=False)
        finally:
            loading_area.empty()
        st.rerun()


def render_expected_follow_up_dialogue(payload: dict, review: dict) -> None:
    """사용자가 필요할 때만 예상 거래의 추가 사실을 반영해 재검토한다."""
    questions = review.get("follow_up_questions", [])
    if not st.session_state.get("expected_follow_up_open"):
        if st.button("추가 사실 반영 재검토", key="open_expected_follow_up"):
            st.session_state["expected_follow_up_open"] = True
        else:
            return
    if not questions:
        questions = [{"question_id": "USER", "question": "추가로 반영할 사실관계 또는 증빙 내용은 무엇입니까?", "why_needed": "사용자가 필요하다고 판단한 추가 정보를 반영합니다.", "conclusion_impact": "현재 잠정 결론의 신뢰 수준 또는 방향이 달라질 수 있습니다.", "priority": "보통"}]
        return
    st.divider()
    st.subheader("결론을 좁히기 위한 추가 확인")
    st.caption("예상 거래의 확인된 사실 또는 보유 증빙 내용을 답변하면 Risk Score는 산정 보류 상태로 유지한 채 AI 잠정 의견만 재검토합니다.")
    for question in questions:
        with st.container(border=True):
            st.markdown(f"**{question.get('priority', '보통')} · {question.get('question_id', '확인 질문')}**")
            st.write(question.get("question", ""))
            st.caption(f"필요한 이유: {question.get('why_needed', '제시된 이유 없음')}")
            st.caption(f"결론 영향: {question.get('conclusion_impact', '제시된 영향 없음')}")
    history = st.session_state.setdefault("expected_follow_up_history", [])
    if history:
        st.markdown("#### 답변 이력")
        for item in history:
            with st.chat_message("assistant"):
                st.write(item["question"])
            with st.chat_message("user"):
                st.write(item["answer"])
    selected_question = st.selectbox(
        "답변할 추가 확인 질문",
        questions,
        format_func=lambda item: f"{item.get('question_id', '질문')} · {item.get('question', '')}",
        key="expected_follow_up_question",
    )
    answer = st.chat_input(
        "선택한 질문에 대한 확인된 사실 또는 증빙 내용을 입력하세요",
        key="expected_follow_up_answer",
        submit_mode="disable",
    )
    if answer:
        history.append({"question_id": selected_question.get("question_id", ""), "question": selected_question.get("question", ""), "answer": answer})
        enriched_payload = expected_payload_with_follow_up_answers(payload, history, review)
        loading_area = st.empty()
        show_ai_loading(loading_area)
        try:
            with st.status("담당자 답변을 반영해 예상 거래 재검토 진행 중", expanded=True, width="stretch") as status:
                st.write("추가 사실관계와 직전 잠정 결론을 비교하고 있습니다.")
                st.write("승인된 근거와 답변을 함께 검토하고 있습니다.")
                result = call_api("/expected-transaction/diagnose", "POST", enriched_payload, timeout_seconds=120)
                st.session_state["expected_transaction_result"] = result
                if "error" in result:
                    status.update(label="예상 거래 재검토를 완료하지 못했습니다", state="error", expanded=True)
                else:
                    status.update(label="예상 거래 재검토가 완료되었습니다", state="complete", expanded=False)
        finally:
            loading_area.empty()
        st.rerun()


def dashboard() -> None:
    """SAP 분석 전의 대시보드를 새 시안의 3열 분석 화면으로 표시한다."""
    result = st.session_state.get("risk_analysis_result")
    header("대시보드", "월간 거래 분석 현황", "거래 분석을 실행하면 현재 세션의 결과가 즉시 반영됩니다. 월별 영구 누적은 PostgreSQL 연결 후 제공됩니다.")
    content, filters = st.columns([4, 1], gap="large")
    with content:
        summary = result.get("risk_summary", {}) if result else {}
        high = summary.get("High", {"count": 0, "amount": 0})
        medium = summary.get("Medium", {"count": 0, "amount": 0})
        low = summary.get("Low", {"count": 0, "amount": 0})
        metrics = st.columns(4)
        metric_values = (
            ("분석 원장 행", f"{result['ledger_record_count']:,}건" if result else "미등록", "none"),
            ("고위험 후보", f"{high['count']:,}건" if result else "미분석", "high"),
            ("중위험 후보", f"{medium['count']:,}건" if result else "미분석", "medium"),
            ("저위험 후보", f"{low['count']:,}건" if result else "미분석", "low"),
        )
        for column, (label, value, color) in zip(metrics, metric_values):
            with column:
                dashboard_metric(label, value, color)
        st.markdown("<br>", unsafe_allow_html=True)
        if result:
            if result["finding_count"]:
                st.success(f"{result['analysis_year_month']} 분석 완료: 검토 후보 {result['finding_count']:,}건이 선별되었습니다.")
            else:
                st.info(f"{result['analysis_year_month']} 분석 완료: 위험후보는 없지만 원장 데이터 분석은 완료되었습니다.")
        chart, severity = st.columns([3, 2], gap="large")
        with chart:
            st.markdown('<div class="dashboard-panel"><div class="label">계정과목별 위험점수</div><br><div class="empty">월별 누적 저장이 시작되면 계정과목별 거래량과 위험점수 추이를 표시합니다.</div></div>', unsafe_allow_html=True)
        with severity:
            distribution = "분석 결과가 없습니다." if not result else f"고위험 {high['count']}건 ({format_amount(high['amount'])}) · 중위험 {medium['count']}건 ({format_amount(medium['amount'])}) · 저위험 {low['count']}건 ({format_amount(low['amount'])})"
            st.markdown(f'<div class="dashboard-panel"><div class="label">위험등급 분포</div><br><div class="empty">{distribution}</div></div>', unsafe_allow_html=True)
        if result and "comparison" in result:
            comparison = result["comparison"]
            if comparison["comparison_month"]:
                st.info(f"{comparison['comparison_month']} 대비: 신규 {comparison['new']:,}건 · 지속 {comparison['continued']:,}건 · 해소 {comparison['resolved']:,}건 · 등급상승 {comparison['escalated']:,}건 · 등급하락 {comparison['reduced']:,}건")
            else:
                st.caption(comparison["message"])
        st.markdown("<br>", unsafe_allow_html=True)
        if result and result["finding_count"]:
            top_findings = sorted(result["findings"], key=lambda item: item["risk_score"], reverse=True)[:5]
            st.markdown("#### 우선 검토 거래")
            st.dataframe(
                [{"전표번호": item["voucher_number"], "계정과목": item["account_name"], "거래처": item["counterparty_name"], "거래금액": format_amount(item["amount"]), "위험점수": item["risk_score"], "등급": item["risk_level"]} for item in top_findings],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.markdown('<div class="dashboard-panel"><div class="label">우선 검토 거래</div><br><div class="empty">현재 즉시 검토 대상 거래가 없습니다. 거래 분석에서 원장을 분석하세요.</div></div>', unsafe_allow_html=True)
    with filters:
        with st.container(border=True):
            st.markdown('<div class="filter-title">분석 조건</div>', unsafe_allow_html=True)
            st.selectbox("법인", ["포스코퓨처엠"], disabled=True, key="dashboard_entity")
            st.markdown("<br>", unsafe_allow_html=True)
            st.selectbox("분석 월", [result["analysis_year_month"] if result else "데이터 등록 후 선택"], disabled=True, key="dashboard_period")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="label">위험등급</div>', unsafe_allow_html=True)
            st.checkbox("고위험", disabled=True, key="dashboard_high")
            st.checkbox("중위험", disabled=True, key="dashboard_medium")
            st.checkbox("저위험", disabled=True, key="dashboard_low")
            st.button("조건 적용", disabled=True, key="dashboard_apply")


def risk_analysis() -> None:
    """원장·특수관계자 CSV를 FastAPI Risk Score API로 전달해 결과를 표시한다."""
    header("거래 분석", "거래 위험 분석", "원장 파일을 분석해 검토 후보와 탐지 사유를 제공합니다.")
    filters, content = st.columns([1, 2])
    with filters:
        st.markdown("#### 분석 조건")
        st.selectbox("위험등급", ["전체", "고위험", "중위험", "저위험"], disabled=True)
        st.text_input("회사코드", placeholder="원장 업로드 후 선택", disabled=True)
        st.text_input("계정과목", placeholder="원장 업로드 후 선택", disabled=True)
        st.caption("분석 데이터가 등록되면 조건을 활성화합니다.")
    with content:
        st.markdown("#### 위험점수 분석 실행")
        ledger_file = st.file_uploader("SAP 원장 UTF-8 CSV", type=["csv"], key="ledger")
        related_file = st.file_uploader("특수관계자 목록 UTF-8 CSV", type=["csv"], key="related")
        analysis_month = st.text_input("분석 대상 월", placeholder="예: 2026-08")
        if ledger_file is None or related_file is None or not analysis_month:
            st.info("원장·특수관계자 목록·분석 대상 월을 입력하면 위험점수를 분석합니다.")
            return
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", analysis_month):
            st.warning("분석 대상 월은 2026-08처럼 YYYY-MM 형식으로 입력하세요.")
            return
        try:
            payload = {"ledger_csv_text": ledger_file.getvalue().decode("utf-8-sig"), "related_party_csv_text": related_file.getvalue().decode("utf-8-sig"), "analysis_year_month": analysis_month}
        except UnicodeDecodeError:
            st.error("현재는 UTF-8 CSV만 지원합니다. Excel/XLSX 처리는 다음 단계에서 추가합니다.")
            return
        if st.button("위험점수 분석 실행"):
            database = call_api("/health").get("database", {})
            endpoint = "/risk-score/analyze-and-save" if database.get("status") == "connected" else "/risk-score/preview"
            with st.status("원장을 검증하고 위험후보를 선별하고 있습니다.", expanded=True) as progress:
                progress.write("원장 형식과 분석 대상 월을 확인하고 있습니다.")
                result = call_api(endpoint, "POST", payload, timeout_seconds=180)
                progress.write("위험점수 산정이 완료되었습니다.")
                progress.update(label="분석 완료", state="complete", expanded=False)
            if "error" in result:
                st.error(result["error"])
                st.session_state.pop("risk_analysis_result", None)
            else:
                st.session_state["risk_analysis_result"] = result
                st.session_state["risk_findings"] = result["findings"]
                if result["finding_count"]:
                    st.success(f"분석 완료: 검토 후보 {result['finding_count']:,}건을 선별했습니다. 대시보드에도 결과가 반영되었습니다.")
                else:
                    st.info("분석 완료: 위험후보는 없지만 원장 데이터 분석은 완료되었습니다. 대시보드에서 요약을 확인할 수 있습니다.")
                if endpoint == "/risk-score/preview":
                    st.warning("PostgreSQL이 연결되지 않아 이번 분석은 누적 저장되지 않았습니다. 연결 후에는 다음 달 분석에서 자동 비교됩니다.")
        analysis_result = st.session_state.get("risk_analysis_result")
        findings = analysis_result.get("findings", []) if analysis_result else []
        if findings:
            st.dataframe([{"전표번호": item["voucher_number"], "계정과목": item["account_name"], "거래처": item["counterparty_name"], "거래금액": format_amount(item["amount"]), "위험점수": item["risk_score"], "등급": item["risk_level"]} for item in findings], use_container_width=True, hide_index=True)
            selected = st.selectbox("AI 검토 대상 거래", range(len(findings)), format_func=lambda index: findings[index]["voucher_number"])
            st.session_state["selected_finding"] = findings[selected]


def expected_transaction() -> None:
    """과거 원장 없이 사용자가 입력한 예정 거래를 사전진단한다."""
    header("예상 거래", "예상 거래 사전진단", "예상 거래의 사실관계와 금액을 입력하면 근거 검색 및 AI 잠정 검토를 제공합니다.")
    st.info("과거 거래 이력 또는 확정 점수 규칙이 없으면 Risk Score는 계산하지 않습니다. AI 의견은 담당자의 최종 판단을 대체하지 않습니다.")
    with st.form("expected_transaction_form"):
        left, right = st.columns(2)
        with left:
            company_name = st.text_input("법인명", placeholder="예: A법인")
            expected_date = st.text_input("예정 거래일 *", placeholder="예: 2026-09-30")
            account_code = st.text_input("계정과목코드", placeholder="예: 610000")
            account_name = st.text_input("계정과목명 *", placeholder="예: 지급수수료")
            counterparty_name = st.text_input("거래처명", placeholder="예: 거래처 A")
        with right:
            related_party = st.checkbox("특수관계자 거래")
            debit_credit = st.selectbox("차대변 구분 *", ["차변", "대변"])
            amount = st.number_input("검토 대상 거래금액 *", min_value=0.0, step=1_000_000.0, format="%.0f")
            description = st.text_area("거래 설명 *", placeholder="거래 목적, 계약 조건, 처리 예정 내용을 입력하세요.")
            facts = st.text_area("추가 사실관계", placeholder="계약 조건, 가격 산정 방식, 거래 배경, 확보 증빙 등을 입력하세요.")
        issue_text = st.text_input("쟁점 키워드", placeholder="예: 특수관계자, 시가, 손금, 세금계산서 (쉼표로 구분)")
        attachments = st.file_uploader("합의서·계약서 등 첨부 자료", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True, help="최대 5개, 파일당 10MB까지 첨부할 수 있습니다. 첨부 자료는 이번 AI 검토에만 사용됩니다.")
        st.caption("첨부 자료는 AI 검토를 위해 OpenAI API에 전달됩니다. 민감정보·불필요한 개인정보는 첨부 전에 제거하세요.")
        search_requested = st.form_submit_button("자동 근거 검색")
        diagnose_requested = st.form_submit_button("gpt-5.6-terra 사전진단 실행")
    if search_requested or diagnose_requested:
        if not expected_date or not account_name or not description or amount <= 0:
            st.error("예정 거래일, 계정과목명, 거래 설명, 검토 대상 거래금액은 필수입니다.")
            return
        if len(attachments) > 5:
            st.error("첨부 자료는 최대 5개까지 등록할 수 있습니다.")
            return
        encoded_attachments = []
        for attachment in attachments:
            if attachment.size > 10 * 1024 * 1024:
                st.error(f"'{attachment.name}'은 10MB 이하만 첨부할 수 있습니다.")
                return
            content_type = attachment.type or ""
            if content_type == "image/jpg":
                content_type = "image/jpeg"
            encoded_attachments.append({"filename": attachment.name, "content_type": content_type, "content_base64": base64.b64encode(attachment.getvalue()).decode("ascii")})
        payload = {
            "company_name": company_name, "expected_date": expected_date, "account_code": account_code,
            "account_name": account_name, "counterparty_name": counterparty_name,
            "related_party": related_party, "debit_credit": debit_credit, "amount": amount,
            "description": description, "facts": facts,
            "issue_keywords": [item.strip() for item in issue_text.split(",") if item.strip()],
            "attachments": encoded_attachments, "evidence_limit": 10,
        }
        path = "/expected-transaction/evidence-preview" if search_requested else "/expected-transaction/diagnose"
        if diagnose_requested:
            with st.status("AI 사전진단 요청 중…", expanded=True, width="stretch") as status:
                st.write("입력한 거래 사실을 정리하고 있습니다.")
                st.write("승인된 근거 문서를 검색하고 AI 잠정 검토를 요청하고 있습니다.")
                result = call_api(path, "POST", payload, timeout_seconds=120)
                status.update(label="AI 사전진단을 완료하지 못했습니다" if "error" in result else "AI 사전진단이 완료되었습니다", state="error" if "error" in result else "complete", expanded="error" in result)
        else:
            with st.spinner("승인된 근거 문서를 검색하고 있습니다…"):
                result = call_api(path, "POST", payload)
        if "error" in result:
            st.error(result["error"])
            return
        st.session_state["expected_transaction_result"] = result
        if diagnose_requested:
            st.session_state["expected_transaction_payload"] = payload
            st.session_state["expected_follow_up_history"] = []
            st.session_state["expected_follow_up_open"] = False
    result = st.session_state.get("expected_transaction_result")
    if not result:
        return
    if "error" in result:
        st.error(result["error"])
        return
    risk_assessment = result["risk_assessment"]
    st.markdown("#### 리스크 산정 상태")
    st.info(f"{risk_assessment['status']}: {risk_assessment['message']}")
    st.markdown(f"#### 자동 검색 근거 ({len(result['evidence_documents'])}건)")
    st.dataframe(result["evidence_documents"], use_container_width=True)
    if "review" in result:
        render_ai_review(result["review"])
        if result["invalid_evidence_ids"]:
            st.warning("허용되지 않은 근거 문서 ID가 감지되었습니다: " + ", ".join(result["invalid_evidence_ids"]))
        expected_payload = st.session_state.get("expected_transaction_payload")
        if expected_payload:
            render_expected_follow_up_dialogue(expected_payload, result["review"])


def standard_data() -> None:
    """색인된 기준 데이터와 인프라 설정 상태를 보여준다."""
    header("기준 데이터", "기준 데이터 관리", "AI가 인용할 회계기준·법령·판례의 출처와 갱신 상태를 관리합니다.")
    health = call_api("/health")
    database = health.get("database", {}).get("status", "API 연결 필요")
    columns = st.columns(3)
    with columns[0]: card("K-IFRS 색인", "53건")
    with columns[1]: card("일반기업회계기준 색인", "36건")
    with columns[2]: card("POSTGRESQL 상태", database)
    st.table([
        {"유형": "회계기준", "원천": "ifrs 폴더", "상태": "색인 완료"},
        {"유형": "법령·판례", "원천": "국가법령정보 API", "상태": "갱신 완료"},
        {"유형": "사내지침", "원천": "담당자 업로드", "상태": "준비 필요"},
    ])


def review_report() -> None:
    """실제 거래가 선택된 뒤 채워질 Report의 확정 목차를 표시한다."""
    header("AI 검토 보고서", "AI 거래별 검토 보고서", "AI 의견은 사실·근거 기반 추론·미확인 사항을 구분하고 담당자가 최종 확정합니다.")
    finding = st.session_state.get("selected_finding")
    if finding:
        review_context_key = f"{finding.get('voucher_number', '')}|{finding.get('line_number', '')}"
        if st.session_state.get("review_context_key") != review_context_key:
            st.session_state["review_context_key"] = review_context_key
            st.session_state.pop("evidence_preview", None)
            st.session_state.pop("ai_review_result", None)
            st.session_state["review_follow_up_history"] = []
            st.session_state["review_follow_up_open"] = False
        st.success(f"선택 거래: {finding['voucher_number']} / 위험점수 {finding['risk_score']}")
        transaction, keywords = transaction_for_ai_review(finding)
        if st.button("자동 근거 검색"):
            st.session_state["evidence_preview"] = call_api("/ai-review/evidence-preview", "POST", {"transaction": transaction, "issue_keywords": keywords, "evidence_limit": 10})
        evidence = st.session_state.get("evidence_preview")
        if evidence and "error" not in evidence:
            st.write(f"검색 근거: {len(evidence['evidence_documents'])}건")
            st.dataframe(evidence["evidence_documents"], use_container_width=True)
        elif evidence:
            st.error(evidence["error"])
        if st.button("gpt-5.6-terra 검토 실행"):
            loading_area = st.empty()
            show_ai_loading(loading_area)
            try:
                with st.status("AI 거래 검토 진행 중", expanded=True, width="stretch") as status:
                    st.write("1/3 거래 사실과 위험후보를 확인하고 있습니다.")
                    st.write("2/3 관련 기준·판례 근거를 검색하고 있습니다.")
                    st.write("3/3 AI 잠정 의견과 근거 연결을 검증하고 있습니다.")
                    result = call_api("/ai-review/with-auto-evidence", "POST", {"transaction": transaction, "issue_keywords": keywords, "evidence_limit": 10}, timeout_seconds=120)
                    st.session_state["ai_review_result"] = result
                    if "error" in result:
                        status.update(label="AI 거래 검토를 완료하지 못했습니다", state="error", expanded=True)
                    else:
                        status.update(label="AI 거래 검토가 완료되었습니다", state="complete", expanded=False)
            finally:
                loading_area.empty()
        result = st.session_state.get("ai_review_result")
        if result and "error" not in result:
            render_ai_review(result["review"])
            if result["invalid_evidence_ids"]:
                st.warning("허용되지 않은 근거 문서 ID가 감지되었습니다: " + ", ".join(result["invalid_evidence_ids"]))
            render_follow_up_dialogue(transaction, keywords, result["review"])
        elif result:
            st.error(result["error"])
    else:
        st.markdown('<div class="empty">Risk Analysis에서 검토 대상 거래를 선택하면 근거 검색과 AI 검토를 실행할 수 있습니다.</div>', unsafe_allow_html=True)
    for section in (
        "1. 거래 기본정보 및 Risk 요약", "2. SAP 데이터 기반 사실관계", "3. 회계 쟁점 및 근거",
        "4. 세무 쟁점 및 근거", "5. AI 종합 의견·반대 논리·확인 필요 증빙", "6. 담당자 최종 검토",
    ):
        st.checkbox(section, value=False, disabled=True)


def main() -> None:
    """공통 사이드바와 선택된 업무 화면을 렌더링한다."""
    st.set_page_config(page_title="AI 회계·세무 리스크 PoC", layout="wide")
    apply_style()
    with st.sidebar:
        st.markdown('<p class="brand">회계·세무 리스크 분석</p><p class="brand-sub">포스코퓨처엠 AI</p>', unsafe_allow_html=True)
        st.divider()
        page = st.radio("메뉴", ["대시보드", "거래 분석", "예상 거래 사전진단", "기준 데이터 관리", "AI 검토 보고서"], label_visibility="collapsed")
        st.divider()
        st.caption("● AI 분석 준비 상태")
    st.text_input("통합 검색", placeholder="거래 또는 법령을 검색합니다.", disabled=True)
    {"대시보드": dashboard, "거래 분석": risk_analysis, "예상 거래 사전진단": expected_transaction, "기준 데이터 관리": standard_data, "AI 검토 보고서": review_report}[page]()


if __name__ == "__main__":
    main()
