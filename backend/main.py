"""FastAPI 웹 화면이 호출하는 회계·세무 리스크 PoC API다."""

import os
import re
import sqlite3
import json
import hashlib
import secrets
from pathlib import Path
from collections import Counter
from datetime import date, datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph

from backend.ai_review import AiReviewError, EXPERT_CHAT_TIMEOUT_SECONDS, MODEL_NAME, answer_natural_language_question, prepare_attachments, review_with_openai
from backend.database import compare_with_previous_month, database_status, initialize_monthly_storage, load_active_history, load_read_only_chat_context, save_monthly_analysis
from backend.evidence import EvidenceSearchError, evidence_track, search_local_evidence
from app import DEFAULT_DB_PATH, PROJECT_ROOT, neo4j_settings, read_refresh_status
from backend.ledger import REQUIRED_LEDGER_COLUMNS, validate_csv_headers
from backend.risk_engine import parse_ledger_csv, parse_related_parties, score_records
from backend.schemas import AiReviewRequest, AutoAiReviewRequest, CsvHeaderValidationRequest, ExpectedTransactionRequest, NaturalLanguageQueryRequest, RiskScoreRequest


app = FastAPI(title="AI 회계·세무 리스크 PoC API", version="0.1.0")
ANALYTICS_DB_PATH = DEFAULT_DB_PATH.parent / "chat_analytics.db"
ANALYTICS_STOPWORDS = {"알려줘", "알려주세요", "얼마", "계산", "어떻게", "경우", "대한", "관련", "이것", "그것", "있나요", "입니다"}
ADMIN_CREDENTIALS = HTTPBasic(auto_error=False)


def initialize_chat_analytics() -> None:
    """관리자 검토용 질문·답변 이력을 첨부 원문 없이 저장한다."""
    ANALYTICS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(ANALYTICS_DB_PATH) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS chat_events (
                event_id TEXT PRIMARY KEY,
                question_text TEXT NOT NULL,
                question_hash TEXT NOT NULL,
                answer_summary TEXT NOT NULL,
                answer_text TEXT NOT NULL DEFAULT '',
                answer_mode TEXT NOT NULL,
                calculation_used INTEGER NOT NULL,
                evidence_articles_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(chat_events)")}
        if "answer_text" not in columns:
            # 기존 운영 이력은 삭제하지 않고, 기존 핵심 답변을 상세 답변의 초기값으로 보존한다.
            connection.execute("ALTER TABLE chat_events ADD COLUMN answer_text TEXT NOT NULL DEFAULT ''")
            connection.execute("UPDATE chat_events SET answer_text = answer_summary WHERE answer_text = ''")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_chat_events_created_at ON chat_events(created_at)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_chat_events_question_hash ON chat_events(question_hash)")


def record_chat_event(question: str, answer: dict[str, object], evidence_documents: list[dict[str, object]]) -> None:
    """첨부 원문·사용자 식별자 없이 질문과 최종 답변만 관리자 기록으로 남긴다."""
    try:
        initialize_chat_analytics()
        normalized = re.sub(r"\s+", " ", question).strip()
        articles = list(dict.fromkeys(
            f"{item.get('title')} {item.get('article')}".strip()
            for item in evidence_documents if item.get("article")
        ))[:8]
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        event_id = hashlib.sha256(f"{created_at}:{normalized}".encode("utf-8")).hexdigest()
        answer_text = "\n\n".join(
            text for text in (str(answer.get("key_answer") or "").strip(), str(answer.get("answer") or "").strip()) if text
        )[:8_000]
        with sqlite3.connect(ANALYTICS_DB_PATH) as connection:
            connection.execute(
                "INSERT INTO chat_events (event_id, question_text, question_hash, answer_summary, answer_text, answer_mode, calculation_used, evidence_articles_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event_id, normalized[:1000], hashlib.sha256(normalized.encode("utf-8")).hexdigest(), str(answer.get("key_answer") or answer.get("answer") or "")[:1000], answer_text, str(answer.get("generation_mode") or ("calculation" if answer.get("calculation") else "ai")), int(bool(answer.get("calculation"))), json.dumps(articles, ensure_ascii=False), created_at),
            )
    except sqlite3.Error:
        # 통계 저장 오류가 지식 챗봇의 답변 자체를 막지 않게 한다.
        return


class TaxCalculationRequest(BaseModel):
    """조문 근거가 확인된 범위에서만 세액·가산세를 산정하는 입력값이다."""

    calculation_type: str = Field(pattern="^(national_strategy_credit|unreported_penalty|late_payment_penalty)$")
    amount: float | None = Field(default=None, gt=0)
    enterprise_type: str | None = Field(default=None, pattern="^(small|graduating|other)?$")
    semiconductor: bool = False
    tax_year: int | None = Field(default=None, ge=2021, le=2100)
    violation_type: str | None = Field(default=None, pattern="^(ordinary|fraudulent)?$")
    statutory_due_date: str | None = None
    actual_payment_date: str | None = None
    daily_rate_percent: float | None = Field(default=None, gt=0, le=1)


WEB_APP_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>회계·세무 리스크 분석</title>
  <style>
    :root { --blue:#0868b8; --ink:#15253a; --muted:#687587; --line:#dbe3eb; --bg:#f4f7fa; --accent:#fff3b5; }
    * { box-sizing:border-box; } body { margin:0; color:var(--ink); background:var(--bg); font-family:Arial,"Noto Sans KR",sans-serif; }
    .layout { min-height:100vh; display:grid; grid-template-columns:280px 1fr; }
    aside { padding:34px 22px; background:#fff; border-right:1px solid var(--line); }
    .brand { padding:24px; background:var(--blue); color:#fff; font-size:20px; font-weight:800; line-height:1.6; }
    nav { margin-top:36px; border-top:1px solid var(--line); padding-top:24px; } nav div { padding:14px 12px; color:#46566c; } nav .active { color:var(--blue); font-weight:800; background:#eef7ff; border-radius:8px; }
    main { max-width:1040px; width:100%; margin:0 auto; padding:54px 42px 150px; }
    .eyebrow { color:var(--blue); font-weight:800; letter-spacing:.12em; font-size:13px; } h1 { margin:12px 0 10px; font-size:38px; } .subtitle { color:var(--muted); margin:0 0 28px; }
    .status { display:inline-flex; gap:8px; align-items:center; padding:9px 12px; background:#fff; border:1px solid var(--line); border-radius:18px; color:var(--muted); font-size:13px; }
    .chat { margin-top:26px; display:grid; gap:18px; } .message { background:#fff; border:1px solid var(--line); border-radius:14px; padding:20px; line-height:1.75; white-space:pre-wrap; }
    .question { border-left:5px solid #ff5454; font-weight:700; } .key { margin:14px 0; padding:16px; background:#e5f1ff; border-radius:10px; font-weight:700; white-space:pre-wrap; }
    mark { background:var(--accent); color:inherit; padding:0 2px; border-radius:3px; font-weight:700; } .meta { margin-top:14px; color:var(--muted); font-size:13px; }
    details { margin-top:14px; } summary { cursor:pointer; color:var(--blue); font-weight:700; } .sources { margin:9px 0 0; padding-left:20px; } .sources li { margin:6px 0; }
    .followups { display:grid; gap:8px; margin-top:14px; } button.follow { text-align:left; border:1px solid #b9d8f5; color:#075da8; background:#f5faff; border-radius:8px; padding:11px 13px; cursor:pointer; font-size:14px; }
    form { position:fixed; left:280px; right:0; bottom:0; padding:18px max(42px,calc((100vw - 1040px)/2)); background:rgba(244,247,250,.96); border-top:1px solid var(--line); display:flex; gap:10px; }
    input { min-width:0; flex:1; padding:16px; border:1px solid var(--line); border-radius:10px; font-size:16px; } button.send { border:0; border-radius:10px; padding:0 22px; background:var(--blue); color:#fff; font-weight:700; cursor:pointer; }
    .error { color:#ae2732; background:#fff0f1; border-color:#f4bcc2; } .loading { color:var(--muted); }
    @media (max-width:760px) { .layout { display:block; } aside { display:none; } main { padding:30px 18px 120px; } h1 { font-size:30px; } form { left:0; padding:14px 18px; } }
  </style>
</head>
<body>
  <div class="layout"><aside><div class="brand">회계·세무 리스크 분석<br>포스코퓨처엠 AI</div><nav><div>대시보드</div><div>거래 분석</div><div>예상 거래 사전진단</div><div>기준 데이터 관리</div><div class="active">지식 챗봇</div><div>AI 검토 보고서</div></nav></aside>
  <main><div class="eyebrow">자연어 질의</div><h1>회계·세무 지식 챗봇</h1><p class="subtitle">승인된 법령·판례·유권해석·회계기준을 근거로 답변합니다.</p><div id="status" class="status">지식기반 상태 확인 중</div><section id="chat" class="chat"></section></main></div>
  <form id="question-form"><input id="question" autocomplete="off" placeholder="예: 통합투자세액공제 국가전략기술 공제율을 알려줘" aria-label="질문"><button class="send" type="submit">질문</button></form>
  <script>
    const chat = document.getElementById('chat'); const input = document.getElementById('question'); const history = [];
    function element(tag, className, text) { const node=document.createElement(tag); if(className) node.className=className; if(text !== undefined) node.textContent=text; return node; }
    function highlight(target, text, terms) { const safeTerms=(terms||[]).filter(term=>term && text.includes(term)).sort((a,b)=>b.length-a.length); if(!safeTerms.length){target.textContent=text; return;} const pattern=new RegExp('('+safeTerms.map(term=>term.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\$&')).join('|')+')','g'); let cursor=0; text.split(pattern).forEach(part=>{if(!part)return; const marked=safeTerms.includes(part); const node=marked?element('mark','',part):document.createTextNode(part); target.append(node); cursor+=part.length;}); }
    function addQuestion(question) { chat.append(element('article','message question',question)); }
    function sourceLabel(source) { const meta=source.metadata||{}; const bits=[source.title]; if(source.article) bits.push(source.article); if(meta.paragraph_number) bits.push('문단 '+meta.paragraph_number+(meta.page_start?' (p.'+meta.page_start+')':'')); return bits.join(' · '); }
    function addAnswer(payload) { const answer=payload.answer; const card=element('article','message'); if(answer.key_answer){card.append(element('div','key','핵심 답변\n'+answer.key_answer));} const body=element('div',''); highlight(body,answer.answer||'답변을 생성하지 못했습니다.',answer.highlight_terms); card.append(body);
      const sourceById=new Map((payload.evidence_documents||[]).map(source=>[source.document_id,source])); const selected=(answer.evidence_ids||[]).map(id=>sourceById.get(id)).filter(Boolean);
      if(selected.length){const details=element('details',''); const summary=element('summary','', '답변에 사용한 근거'); details.append(summary); const list=element('ul','sources'); selected.forEach(source=>{const li=element('li',''); const link=element('a','',sourceLabel(source)); if(source.source_url){link.href=source.source_url; link.target='_blank'; link.rel='noopener';} li.append(link); list.append(li);}); details.append(list); card.append(details);}
      const followups=answer.follow_up_questions||[]; if(followups.length){const box=element('div','followups'); followups.forEach(question=>{const button=element('button','follow',question); button.type='button'; button.onclick=()=>ask(question); box.append(button);}); card.append(box);} chat.append(card); history.push({question:payload.question,key_answer:answer.key_answer||answer.answer||''}); window.scrollTo({top:document.body.scrollHeight,behavior:'smooth'}); }
    async function ask(question) { if(!question.trim()) return; input.value=''; addQuestion(question); const loading=element('article','message loading','근거 문서를 검색하고 답변을 준비하고 있습니다.'); chat.append(loading); try { const response=await fetch('/knowledge-chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,conversation:history.slice(-3)})}); const payload=await response.json(); loading.remove(); if(!response.ok) throw new Error(payload.detail||'답변을 불러오지 못했습니다.'); payload.question=question; addAnswer(payload); } catch(error) { loading.classList.add('error'); loading.classList.remove('loading'); loading.textContent=error.message; } }
    document.getElementById('question-form').addEventListener('submit',event=>{event.preventDefault(); ask(input.value);});
    fetch('/knowledge-refresh/status').then(response=>response.json()).then(data=>{document.getElementById('status').textContent=data.state==='running'?'지식기반 갱신 중 · '+data.stage+' '+data.completed+'/'+data.total:data.state==='stopped'?'지식기반 갱신 중단 · 보존된 수집 '+data.completed+'/'+data.total:'지식기반 '+(data.state==='completed'?'갱신 완료':'준비 상태');}).catch(()=>{document.getElementById('status').textContent='지식기반 상태를 확인할 수 없습니다.';});
  </script>
</body></html>"""


INTEGRATED_WEB_APP_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>회계·세무 리스크 분석</title><style>
:root{--blue:#0668b9;--ink:#17263a;--muted:#66758a;--line:#dce4ed;--bg:#f5f8fb;--pale:#edf6ff;--warn:#fff5d8;--red:#bd3c48}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,"Noto Sans KR",sans-serif}.layout{min-height:100vh;display:grid;grid-template-columns:270px 1fr}aside{background:#fff;border-right:1px solid var(--line);padding:30px 18px}.brand{background:var(--blue);color:#fff;padding:22px;font-size:19px;line-height:1.6;font-weight:800}.nav{margin-top:28px;border-top:1px solid var(--line);padding-top:16px}.nav button{width:100%;border:0;background:transparent;color:#455870;text-align:left;padding:13px;border-radius:8px;font-size:15px;cursor:pointer}.nav button.active{color:var(--blue);font-weight:800;background:var(--pale)}main{max-width:1160px;width:100%;margin:0 auto;padding:42px 42px 90px}.view{display:none}.view.active{display:block}.eyebrow{font-size:13px;font-weight:800;letter-spacing:.1em;color:var(--blue)}h1{margin:9px 0 8px;font-size:34px}h2{font-size:23px;margin:0 0 8px}.subtitle{color:var(--muted);margin:0 0 26px;line-height:1.7}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.card,.panel{background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px}.metric .label,.small{color:var(--muted);font-size:13px}.metric .value{font-size:28px;font-weight:800;margin-top:8px}.panel{margin-top:16px}.panel h3{margin:0 0 13px;font-size:16px}.notice{padding:13px 15px;border-radius:9px;background:var(--pale);color:#075da8;line-height:1.6}.notice.warn{background:#fff4dd;color:#805d04}.muted{color:var(--muted)}label{display:block;font-size:13px;font-weight:700;margin:12px 0 6px}input,textarea,select{width:100%;padding:11px;border:1px solid var(--line);border-radius:8px;font:inherit;background:#fff}textarea{min-height:90px;resize:vertical}.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}.actions{display:flex;gap:9px;flex-wrap:wrap;margin-top:16px}button.primary,button.secondary{border:0;border-radius:8px;padding:11px 15px;font:inherit;font-weight:700;cursor:pointer}button.primary{background:var(--blue);color:#fff}button.secondary{color:#075da8;background:var(--pale)}button:disabled{opacity:.6;cursor:wait}.result{margin-top:18px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:10px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:700}.pill{display:inline-block;padding:3px 8px;border-radius:13px;font-size:12px;font-weight:800}.High{color:#a51f2b;background:#ffe8ea}.Medium{color:#885300;background:#fff1cc}.Low{color:#08703c;background:#e5f7ed}.chat{display:grid;gap:14px}.message{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px;line-height:1.75;white-space:pre-wrap}.question{border-left:5px solid #ff535a;font-weight:700}.key{padding:15px;background:#ddecff;color:#064f91;font-weight:800;border-radius:9px;margin-bottom:13px}.message mark{background:#fff0a9;font-weight:700;padding:0 2px}.sources{padding-left:20px}.sources li{margin:6px 0}.followups{display:grid;gap:7px;margin-top:13px}.followups button{text-align:left;color:#075da8;background:#f5faff;border:1px solid #b8d7f4;padding:10px;border-radius:8px;cursor:pointer}.chat-input{display:flex;gap:8px;margin-top:15px}.chat-input input{flex:1}.error{color:#a52734;background:#fff0f1}.empty{padding:22px;color:var(--muted);text-align:center}.report-section{padding:14px 0;border-bottom:1px solid var(--line);white-space:pre-wrap;line-height:1.75}.report-section:last-child{border:0}.status{display:inline-block;padding:8px 11px;border-radius:16px;background:#fff;border:1px solid var(--line);color:var(--muted);font-size:13px}.file-note{font-size:12px;color:var(--muted);margin-top:5px}@media(max-width:850px){.layout{display:block}aside{padding:14px}.brand{display:inline-block}.nav{display:flex;overflow:auto;margin-top:12px;padding-top:8px}.nav button{white-space:nowrap;width:auto}main{padding:30px 18px}.grid,.two{grid-template-columns:1fr}h1{font-size:29px}}
</style></head><body><div class="layout"><aside><div class="brand">회계·세무 리스크 분석<br>포스코퓨처엠 AI</div><div class="nav"><button data-view="dashboard" class="active">대시보드</button><button data-view="analysis">거래 분석</button><button data-view="expected">예상 거래 사전진단</button><button data-view="reference">기준 데이터 관리</button><button data-view="chat">지식 챗봇</button><button data-view="report">AI 검토 보고서</button></div></aside><main>
<section id="dashboard" class="view active"><div class="eyebrow">AI 회계·세무 리스크 PoC</div><h1>대시보드</h1><p class="subtitle">월간 위험거래 선별부터 근거 확인과 담당자 검토까지 한 흐름으로 관리합니다.</p><div id="dashboard-status" class="status">상태를 확인하고 있습니다.</div><div class="grid" style="margin-top:18px"><div class="card metric"><div class="label">High Risk</div><div id="high-count" class="value">-</div><div id="high-amount" class="small">거래 분석 후 표시</div></div><div class="card metric"><div class="label">Medium Risk</div><div id="medium-count" class="value">-</div><div id="medium-amount" class="small">거래 분석 후 표시</div></div><div class="card metric"><div class="label">Low Risk</div><div id="low-count" class="value">-</div><div id="low-amount" class="small">거래 분석 후 표시</div></div></div><div class="panel"><h3>오늘의 작업 흐름</h3><div class="notice">① 원장·특수관계자 Master를 업로드해 검토 후보를 선별하고 ② 후보를 선택해 AI 검토 보고서를 생성한 뒤 ③ 필요한 세법·판례·회계기준은 지식 챗봇에서 근거 조문과 함께 확인합니다.</div></div></section>
<section id="analysis" class="view"><div class="eyebrow">SAP 원장 분석</div><h1>거래 분석</h1><p class="subtitle">선택 월의 CSV 원장과 특수관계자 Master를 바탕으로 PRD의 명시 규칙만 적용해 Risk Score를 계산합니다.</p><div class="panel"><div class="two"><div><label>분석 대상 월</label><input id="analysis-month" type="month"></div><div><label>분석 방식</label><select id="analysis-mode"><option value="preview">미리보기 — 서버에 저장하지 않음</option><option value="save">월별 분석 이력으로 저장</option></select></div></div><div class="two"><div><label>SAP 원장 CSV</label><input id="ledger-file" type="file" accept=".csv,text/csv"><div class="file-note">필수: 전기일자, 전표금액(기준통화), 계정과목, 거래처 식별정보 등</div></div><div><label>특수관계자 Master CSV</label><input id="related-file" type="file" accept=".csv,text/csv"><div class="file-note">필수: 거래처코드, 거래처명</div></div></div><div class="actions"><button id="run-analysis" class="primary">Risk Check 실행</button><button class="secondary" data-go="reference">필수 컬럼 확인</button></div></div><div id="analysis-result" class="result"></div></section>
<section id="expected" class="view"><div class="eyebrow">사전 검토</div><h1>예상 거래 사전진단</h1><p class="subtitle">예정 거래 사실과 첨부 자료를 바탕으로 관련 기준을 먼저 찾고, 필요한 경우 AI 잠정 검토를 생성합니다.</p><div class="panel"><div class="two"><div><label>예정 거래일</label><input id="expected-date" type="date"></div><div><label>검토 대상 거래금액</label><input id="expected-amount" type="number" min="1" placeholder="예: 300000000"></div></div><div class="two"><div><label>법인명</label><input id="expected-company" value="포스코퓨처엠"></div><div><label>계정과목명</label><input id="expected-account" placeholder="예: 유형자산"></div></div><div class="two"><div><label>거래처명</label><input id="expected-counterparty"></div><div><label>차변/대변 또는 매출/매입</label><input id="expected-debit" placeholder="예: 차변"></div></div><label>거래 설명</label><textarea id="expected-description" placeholder="거래 목적, 자산·용역, 계약 조건을 적어주세요."></textarea><label>쟁점 키워드 (쉼표로 구분)</label><input id="expected-keywords" placeholder="예: 특수관계자, 시가, 국가전략기술"><label><input id="expected-related" type="checkbox" style="width:auto"> 특수관계자 거래</label><label>참고 첨부자료 (최대 5개)</label><input id="expected-files" type="file" multiple accept=".pdf,.txt,.png,.jpg,.jpeg"><div class="actions"><button id="expected-evidence" class="secondary">근거 먼저 찾기</button><button id="expected-diagnose" class="primary">AI 사전진단 실행</button></div></div><div id="expected-result" class="result"></div></section>
<section id="reference" class="view"><div class="eyebrow">승인된 외부 기준</div><h1>기준 데이터 관리</h1><p class="subtitle">법령·시행령·시행규칙·예규·유권해석·판례 및 K-IFRS·일반기업회계기준의 검색 준비 상태를 확인합니다.</p><div id="reference-refresh" class="notice">상태를 불러오는 중입니다.</div><div class="panel"><h3>검색 대상 요약</h3><div id="reference-summary" class="empty">문서 현황을 불러오는 중입니다.</div></div><div class="panel"><h3>운영 원칙</h3><p class="muted">공식 원천에서 정제·승인된 문서만 검색합니다. 외부 최신자료 조회는 사용자가 별도로 요청할 때만 공식 출처로 제한합니다. 법령 갱신 중에는 기존 지식기반의 쓰기 작업을 하지 않습니다.</p></div></section>
<section id="chat" class="view"><div class="eyebrow">자연어 질의</div><h1>회계·세무 지식 챗봇</h1><p class="subtitle">승인된 법령·판례·유권해석·회계기준 및 허용된 내부 조회 결과를 근거로 답변합니다.</p><div id="chat-status" class="status">지식기반 상태 확인 중</div><div id="chat-messages" class="chat" style="margin-top:18px"></div><div class="chat-input"><input id="chat-question" placeholder="예: 이 캡처에 적힌 거래의 세무 쟁점을 알려줘"><button id="chat-send" class="primary">질문</button></div><label>현업 자료 첨부 (선택)</label><input id="chat-files" type="file" multiple accept=".pdf,.png,.jpg,.jpeg,.txt,.eml,application/pdf,image/png,image/jpeg,text/plain,message/rfc822"><div id="chat-attachment-status" class="file-note">메일 저장본(EML)·텍스트·PDF·화면 캡처를 최대 5개, 파일당 10MB까지 첨부할 수 있습니다. 캡처 도구에서 이미지를 복사한 뒤 질문 입력창에 Ctrl+V로 붙여넣을 수도 있습니다.</div><div class="panel"><h3>세액·가산세 계산</h3><p class="muted">입력값과 현재 지식기반의 공식 조문에 확인된 요율만 사용합니다.</p><div class="two"><div><label>계산 유형</label><select id="calc-type"><option value="national_strategy_credit">국가전략기술 통합투자세액공제</option><option value="unreported_penalty">무신고가산세</option><option value="late_payment_penalty">납부지연가산세</option></select></div><div><label>투자금액·미납세액 (원)</label><input id="calc-amount" type="number" min="1" placeholder="예: 10000000000"></div></div><div class="two"><div><label>기업유형 (투자공제)</label><select id="calc-enterprise"><option value="small">중소기업</option><option value="graduating">중소기업 졸업 유예기업</option><option value="other">그 밖의 기업</option></select></div><div><label>투자 과세연도 (투자공제)</label><input id="calc-year" type="number" min="2021" max="2100" value="2026"></div></div><label><input id="calc-semiconductor" type="checkbox" style="width:auto"> 반도체 분야 국가전략기술 시설</label><div class="two"><div><label>법정납부기한 (납부지연)</label><input id="calc-due-date" type="date"></div><div><label>실제 납부일 (납부지연)</label><input id="calc-paid-date" type="date"></div></div><div class="two"><div><label>일일요율 % (납부지연)</label><input id="calc-daily-rate" type="number" step="0.000001" min="0.000001" max="1" placeholder="해당 연도 법정 요율 입력"></div><div><label>무신고 구분</label><select id="calc-violation"><option value="ordinary">일반 무신고</option><option value="fraudulent">부정행위 무신고</option></select></div></div><div class="actions"><button id="calc-run" class="primary">근거 기반 계산</button></div><div id="calc-result" class="result"></div></div></section>
<section id="report" class="view"><div class="eyebrow">근거 기반 잠정 검토</div><h1>AI 검토 보고서</h1><p class="subtitle">거래 사실 → 기준 원문 → 적용 논리 → 반대 논리 → AI 결론 순서로 확인합니다. 최종 판단과 조치는 담당자가 수행합니다.</p><div id="report-context" class="notice warn">거래 분석 결과에서 검토 후보를 선택하면 이 화면에서 AI 보고서를 생성할 수 있습니다.</div><div class="actions"><button id="generate-report" class="primary" disabled>선택 거래 AI 검토 생성</button><button class="secondary" data-go="analysis">거래 분석으로 이동</button></div><div id="report-result" class="result"></div></section>
</main></div><script>
const state={history:[],risk:null,selected:null,chatAttachments:[]};const $=id=>document.getElementById(id);const money=n=>new Intl.NumberFormat('ko-KR',{maximumFractionDigits:0}).format(Number(n||0))+'원';const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
function go(view){document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x.id===view));document.querySelectorAll('.nav button').forEach(x=>x.classList.toggle('active',x.dataset.view===view));window.scrollTo({top:0,behavior:'smooth'})}document.querySelectorAll('[data-view]').forEach(x=>x.onclick=()=>go(x.dataset.view));document.querySelectorAll('[data-go]').forEach(x=>x.onclick=()=>go(x.dataset.go));
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});const data=await r.json();if(!r.ok)throw new Error(data.detail||'요청을 처리하지 못했습니다.');return data}function setBusy(button,busy,label){button.disabled=busy;if(busy)button.dataset.label=button.textContent;button.textContent=busy?label:(button.dataset.label||button.textContent)}
function renderSummary(data){['High','Medium','Low'].forEach(level=>{const key=level.toLowerCase();$(key+'-count').textContent=data.risk_summary[level].count+'건';$(key+'-amount').textContent=money(data.risk_summary[level].amount)});$('dashboard-status').textContent=data.analysis_year_month+' 분석 · 원장 '+data.ledger_record_count+'건 · 검토 후보 '+data.finding_count+'건'}
async function fileText(id){const file=$(id).files[0];if(!file)throw new Error('CSV 파일을 선택해주세요.');return await file.text()}
$('run-analysis').onclick=async()=>{const button=$('run-analysis');try{const month=$('analysis-month').value;if(!month)throw new Error('분석 대상 월을 선택해주세요.');setBusy(button,true,'분석 중…');const body={analysis_year_month:month,ledger_csv_text:await fileText('ledger-file'),related_party_csv_text:await fileText('related-file')};const path=$('analysis-mode').value==='save'?'/risk-score/analyze-and-save':'/risk-score/preview';const data=await api(path,body);state.risk=data;renderSummary(data);renderFindings(data)}catch(e){$('analysis-result').innerHTML='<div class="message error">'+esc(e.message)+'</div>'}finally{setBusy(button,false)}};
function renderFindings(data){if(!data.findings.length){$('analysis-result').innerHTML='<div class="message">선택한 월에는 PRD의 현재 Risk Score 규칙에 해당하는 후보가 없습니다.</div>';return}let rows=data.findings.map((x,i)=>'<tr><td><span class="pill '+x.risk_level+'">'+x.risk_level+'</span></td><td><b>'+x.risk_score+'점</b></td><td>'+esc(x.counterparty_name||x.counterparty_code)+'</td><td>'+esc(x.account_name)+'</td><td>'+money(x.amount)+'</td><td>'+esc(x.reasons.map(r=>r.rule).join(', '))+'</td><td><button class="secondary choose" data-index="'+i+'">검토</button></td></tr>').join('');$('analysis-result').innerHTML='<div class="panel"><h3>검토 후보 '+data.finding_count+'건</h3><div class="table-wrap"><table><thead><tr><th>등급</th><th>점수</th><th>거래처</th><th>계정과목</th><th>거래금액</th><th>탐지 사유</th><th></th></tr></thead><tbody>'+rows+'</tbody></table></div></div>';document.querySelectorAll('.choose').forEach(b=>b.onclick=()=>selectFinding(Number(b.dataset.index)))}
function selectFinding(index){state.selected=state.risk.findings[index];$('report-context').className='notice';$('report-context').textContent='선택 거래: '+(state.selected.counterparty_name||state.selected.counterparty_code)+' · '+state.selected.account_name+' · '+money(state.selected.amount)+' · '+state.selected.risk_score+'점';$('generate-report').disabled=false;go('report')}
async function attachmentsFromInput(inputId,extraFiles=[]){const files=[...$(inputId).files,...extraFiles].slice(0,5);if(files.some(file=>file.size>10*1024*1024))throw new Error('첨부 파일은 각각 10MB 이하만 지원합니다.');return await Promise.all(files.map(file=>new Promise((ok,fail)=>{const r=new FileReader();r.onload=()=>ok({filename:file.name,content_type:file.type||({'.eml':'message/rfc822','.txt':'text/plain'}[(file.name.match(/[.][^.]+$/)||[''])[0].toLowerCase()]||'application/octet-stream'),content_base64:String(r.result).split(',')[1]});r.onerror=fail;r.readAsDataURL(file)})))}
async function expectedPayload(){const date=$('expected-date').value,amount=Number($('expected-amount').value),account=$('expected-account').value.trim(),debit=$('expected-debit').value.trim(),description=$('expected-description').value.trim();if(!date||!amount||!account||!debit||!description)throw new Error('예정일·금액·계정과목·차대변 구분·거래 설명을 입력해주세요.');return {company_name:$('expected-company').value,expected_date:date,account_name:account,counterparty_name:$('expected-counterparty').value,related_party:$('expected-related').checked,debit_credit:debit,amount,description,issue_keywords:$('expected-keywords').value.split(',').map(x=>x.trim()).filter(Boolean),attachments:await attachmentsFromInput('expected-files')}}
function evidenceHtml(items){if(!items?.length)return '<div class="muted">검색된 근거가 없습니다.</div>';return '<ul class="sources">'+items.map(x=>{const meta=x.metadata||{},where=[x.article,meta.paragraph_number?'문단 '+meta.paragraph_number:'',meta.page_start?'p.'+meta.page_start:''].filter(Boolean).join(' · ');const title=esc(x.title+(where?' · '+where:''));const track=meta.evidence_track?'<span class="pill '+(meta.evidence_track==='세무'?'Medium':'Low')+'">'+esc(meta.evidence_track)+'</span> ':'';return '<li>'+track+(x.source_url?'<a target="_blank" rel="noopener" href="'+esc(x.source_url)+'">'+title+'</a>':title)+'</li>'}).join('')+'</ul>'}
async function runExpected(diagnose){const id=diagnose?'expected-diagnose':'expected-evidence',button=$(id);try{setBusy(button,true,diagnose?'AI 검토 중…':'근거 검색 중…');const result=await api(diagnose?'/expected-transaction/diagnose':'/expected-transaction/evidence-preview',await expectedPayload());let html='<div class="panel"><h3>Risk Score: '+esc(result.risk_assessment.status)+'</h3><p class="muted">'+esc(result.risk_assessment.message)+'</p><h3>검색된 근거</h3>'+evidenceHtml(result.evidence_documents);if(result.review)html+='<h3>AI 잠정 검토</h3><div class="report-section">'+esc(result.review)+'</div>';if(result.answer)html+='<h3>AI 잠정 검토</h3><div class="report-section">'+esc(result.answer)+'</div>';html+='</div>';$('expected-result').innerHTML=html}catch(e){$('expected-result').innerHTML='<div class="message error">'+esc(e.message)+'</div>'}finally{setBusy(button,false)}}$('expected-evidence').onclick=()=>runExpected(false);$('expected-diagnose').onclick=()=>runExpected(true);
function addChatQuestion(q){$('chat-messages').insertAdjacentHTML('beforeend','<article class="message question">'+esc(q)+'</article>')}function marked(text,terms){let value=esc(text);(terms||[]).filter(x=>x&&x.length>1).sort((a,b)=>b.length-a.length).forEach(x=>{const safe=esc(x).replace(/[.*+?^${}()|[\\]\\\\]/g,'\\$&');value=value.replace(new RegExp('('+safe+')','g'),'<mark>$1</mark>')});return value}function addChatAnswer(payload){const a=payload.answer,docs=new Map((payload.evidence_documents||[]).map(x=>[x.document_id,x])),sources=(a.evidence_ids||[]).map(x=>docs.get(x)).filter(Boolean);let html='<article class="message">'+(a.key_answer?'<div class="key">핵심 답변<br>'+esc(a.key_answer)+'</div>':'')+'<div>'+marked(a.answer||'답변을 생성하지 못했습니다.',a.highlight_terms)+'</div>';if(sources.length)html+='<details><summary>답변에 사용한 근거</summary>'+evidenceHtml(sources)+'</details>';if(a.follow_up_questions?.length)html+='<div class="followups">'+a.follow_up_questions.map(q=>'<button data-q="'+esc(q)+'">'+esc(q)+'</button>').join('')+'</div>';html+='</article>';$('chat-messages').insertAdjacentHTML('beforeend',html);document.querySelectorAll('.followups button').forEach(x=>x.onclick=()=>ask(x.dataset.q));state.history.push({question:payload.question,key_answer:a.key_answer||a.answer||''})}
function renderChatAttachments(){const status=$('chat-attachment-status');if(!state.chatAttachments.length){status.textContent='메일 저장본(EML)·텍스트·PDF·화면 캡처를 최대 5개, 파일당 10MB까지 첨부할 수 있습니다. 캡처 도구에서 이미지를 복사한 뒤 질문 입력창에 Ctrl+V로 붙여넣을 수도 있습니다.';return}status.innerHTML='붙여넣은 캡처 '+state.chatAttachments.length+'개: '+state.chatAttachments.map(file=>esc(file.name)).join(', ')+' <button id="clear-chat-captures" class="secondary" type="button">캡처 지우기</button>';$('clear-chat-captures').onclick=()=>{state.chatAttachments=[];renderChatAttachments()}}
function capturePaste(event){const images=[...event.clipboardData.items].filter(item=>item.type.startsWith('image/')).map(item=>item.getAsFile()).filter(Boolean);if(!images.length)return;event.preventDefault();const remaining=5-state.chatAttachments.length-[...$('chat-files').files].length;if(remaining<=0){$('chat-attachment-status').textContent='첨부는 최대 5개까지 가능합니다.';return}const stamp=new Date().toISOString().replace(/[:.]/g,'-');state.chatAttachments.push(...images.slice(0,remaining).map((file,index)=>new File([file],'clipboard-capture-'+stamp+'-'+(index+1)+'.png',{type:file.type||'image/png'})));renderChatAttachments()}
async function ask(question){const q=(question||$('chat-question').value).trim();if(!q)return;$('chat-question').value='';addChatQuestion(q);const loader=document.createElement('article');loader.className='message muted';loader.textContent='근거 문서를 검색하고 답변을 준비하고 있습니다.';$('chat-messages').append(loader);try{const payload=await api('/knowledge-chat',{question:q,conversation:state.history.slice(-3),attachments:await attachmentsFromInput('chat-files',state.chatAttachments)});$('chat-files').value='';state.chatAttachments=[];renderChatAttachments();loader.remove();payload.question=q;addChatAnswer(payload)}catch(e){loader.className='message error';loader.textContent=e.message}}$('chat-send').onclick=()=>ask();$('chat-question').addEventListener('paste',capturePaste);
async function runTaxCalculation(){const button=$('calc-run');try{setBusy(button,true,'계산 중…');const type=$('calc-type').value,amount=Number($('calc-amount').value)||null;const result=await api('/tax-calculations',{calculation_type:type,amount:amount,enterprise_type:$('calc-enterprise').value,semiconductor:$('calc-semiconductor').checked,tax_year:Number($('calc-year').value)||null,violation_type:$('calc-violation').value,statutory_due_date:$('calc-due-date').value||null,actual_payment_date:$('calc-paid-date').value||null,daily_rate_percent:Number($('calc-daily-rate').value)||null});if(result.status==='input_required'){$('calc-result').innerHTML='<div class="notice warn">'+esc(result.message)+'<br>필요 입력: '+result.required_fields.map(esc).join(', ')+'</div>'+evidenceHtml(result.evidence_documents);return}let html='<div class="notice"><b>'+esc(result.result_label)+'</b><br><span style="font-size:22px;font-weight:800">'+money(result.result_amount)+'</span><br>'+esc(result.formula)+'</div><div class="small" style="margin-top:10px">'+result.assumptions.map(esc).join('<br>')+'</div><h3 style="margin-top:16px">계산 근거</h3>'+evidenceHtml(result.evidence_documents);$('calc-result').innerHTML=html}catch(e){$('calc-result').innerHTML='<div class="message error">'+esc(e.message)+'</div>'}finally{setBusy(button,false)}}$('calc-run').onclick=runTaxCalculation;
$('generate-report').onclick=async()=>{const button=$('generate-report');if(!state.selected)return;try{setBusy(button,true,'AI 검토 중…');const x=state.selected,transaction={'전표번호':x.voucher_number,'전기일자':String(x.posting_date),'계정과목명':x.account_name,'거래처명':x.counterparty_name,'특수관계자여부':x.related_party?'예':'아니오','검토 대상 거래금액':Number(x.amount),'전표적요':x.description||'', 'Risk Score':x.risk_score};const result=await api('/ai-review/with-auto-evidence',{transaction,issue_keywords:x.reasons.map(r=>r.rule),evidence_limit:10});const text=result.review||result.answer||result.ai_review||'AI 검토 결과를 받지 못했습니다.';$('report-result').innerHTML='<div class="panel"><h3>사용 근거</h3>'+evidenceHtml(result.evidence_documents)+'<h3>AI 잠정 검토</h3><div class="report-section">'+esc(text)+'</div></div>'}catch(e){$('report-result').innerHTML='<div class="message error">'+esc(e.message)+'</div>'}finally{setBusy(button,false)}};
async function loadStatus(){try{const [refresh,summary,health]=await Promise.all([api('/knowledge-refresh/status'),api('/knowledge-base/summary'),api('/health')]);const label=refresh.state==='running'?'지식기반 갱신 중 · '+refresh.stage+' '+refresh.completed+'/'+refresh.total:refresh.state==='stopped'?'지식기반 갱신 중단 · 보존된 수집 '+refresh.completed+'/'+refresh.total:refresh.state==='completed'?'지식기반 갱신 완료':'지식기반 준비 상태';$('chat-status').textContent=label;$('reference-refresh').textContent=label;$('dashboard-status').textContent=health.database?.configured?'데이터 저장소 연결 준비됨 · '+label:'PoC 미리보기 모드 · '+label;const tracks=summary.tracks||{};let html='<div class="grid">'+['회계','세무','공통'].map(k=>'<div class="card metric"><div class="label">'+k+' 지식기반</div><div class="value">'+(tracks[k]??0)+'건</div><div class="small">'+(k==='회계'?'K-IFRS·일반기업회계기준·사내지침':k==='세무'?'법령·유권해석·판례·사내지침':'공통 문서')+'</div></div>').join('')+'</div><p class="small">검색 청크 '+(summary.chunk_count??'-')+'개 · '+esc(summary.status||'')+'</p>';$('reference-summary').innerHTML=html}catch(e){$('reference-refresh').className='notice warn';$('reference-refresh').textContent='상태를 확인하지 못했습니다: '+e.message;$('reference-summary').textContent='갱신 중이거나 데이터베이스를 사용할 수 없습니다.'}}loadStatus();$('analysis-month').value=new Date().toISOString().slice(0,7);$('expected-date').value=new Date().toISOString().slice(0,10);
</script></body></html>"""


ADMIN_WEB_HTML = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>관리자 분석</title><style>body{margin:0;padding:40px;max-width:1180px;background:#f5f8fb;color:#17263a;font-family:Arial,'Noto Sans KR',sans-serif}h1{margin:0 0 8px}.sub{color:#66758a}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:22px}.card{background:#fff;border:1px solid #dce4ed;border-radius:12px;padding:18px}.metric{font-size:28px;font-weight:800;color:#0668b9}li{margin:9px 0}.count{float:right;color:#66758a}@media(max-width:700px){body{padding:20px}.grid{grid-template-columns:1fr}}</style></head><body><h1>관리자 분석</h1><p class="sub">개인 식별정보와 첨부 원문은 저장하지 않고, 지식 챗봇의 운영 통계만 집계합니다.</p><div id="content" class="grid">불러오는 중입니다.</div><script>const esc=v=>String(v??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#039;'}[c]));const list=(title,items,key)=>'<section class="card"><h3>'+title+'</h3><ul>'+(items.length?items.map(x=>'<li>'+esc(x[key])+'<span class="count">'+x.count+'회</span></li>').join(''):'<li>아직 기록이 없습니다.</li>')+'</ul></section>';fetch('/admin/chat-analytics').then(r=>r.json()).then(d=>{document.getElementById('content').innerHTML='<section class="card"><h3>전체 질문</h3><div class="metric">'+d.event_count+'건</div><p>계산형 질문 '+d.calculation_count+'건</p></section>'+list('자주 묻는 질문',d.frequent_questions,'question')+list('반복 키워드',d.frequent_keywords,'keyword')+list('자주 사용된 근거 조문',d.frequent_articles,'article')}).catch(()=>document.getElementById('content').textContent='통계를 불러오지 못했습니다.');</script></body></html>"""


def admin_web_html() -> str:
    """관리자만 보는 누적 질문·답변과 운영 통계 화면을 만든다."""
    return """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>관리자 분석</title><style>body{margin:0;padding:40px;max-width:1180px;background:#f5f8fb;color:#17263a;font-family:Arial,'Noto Sans KR',sans-serif}h1{margin:0 0 8px}.sub{color:#66758a;line-height:1.6}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:22px}.card{background:#fff;border:1px solid #dce4ed;border-radius:12px;padding:18px}.metric{font-size:28px;font-weight:800;color:#0668b9}li{margin:9px 0}.count{float:right;color:#66758a}.history{grid-column:1/-1}.event{border-top:1px solid #e5ebf1;padding:14px 0}.event:first-of-type{border-top:0}.event-question{font-weight:800;margin-bottom:7px}.event-meta{font-size:12px;color:#66758a;margin-bottom:8px}.event-answer{white-space:pre-wrap;line-height:1.65}.event details{margin-top:9px}.event summary{cursor:pointer;color:#0668b9;font-weight:700}.event-source{font-size:13px;color:#46566c;margin:8px 0 0;padding-left:18px}@media(max-width:700px){body{padding:20px}.grid{grid-template-columns:1fr}}</style></head><body><h1>관리자 분석</h1><p class="sub">질문·답변과 사용 근거를 운영 품질 개선용으로 기록합니다. 첨부 원문과 사용자 식별정보는 저장하지 않습니다.</p><div id="content" class="grid">불러오는 중입니다.</div><script>const esc=v=>String(v??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#039;'}[c]));const list=(title,items,key)=>'<section class="card"><h3>'+title+'</h3><ul>'+(items.length?items.map(x=>'<li>'+esc(x[key])+'<span class="count">'+x.count+'회</span></li>').join(''):'<li>아직 기록이 없습니다.</li>')+'</ul></section>';const history=items=>'<section class="card history"><h3>최근 질문·답변</h3>'+(items.length?items.map(x=>'<article class="event"><div class="event-question">'+esc(x.question)+'</div><div class="event-meta">'+esc(x.created_at)+' · '+esc(x.answer_mode)+(x.calculation_used?' · 계산형':'')+'</div><details><summary>답변 보기</summary><div class="event-answer">'+esc(x.answer_text||x.answer_summary||'저장된 답변이 없습니다.')+'</div>'+(x.evidence_articles?.length?'<ul class="event-source">'+x.evidence_articles.map(esc).map(value=>'<li>'+value+'</li>').join('')+'</ul>':'')+'</details></article>').join(''):'<p>아직 기록이 없습니다.</p>')+'</section>';fetch('/admin/chat-analytics').then(r=>{if(!r.ok)throw new Error('관리자 통계를 불러오지 못했습니다.');return r.json()}).then(d=>{document.getElementById('content').innerHTML='<section class="card"><h3>전체 질문</h3><div class="metric">'+d.event_count+'건</div><p>계산형 질문 '+d.calculation_count+'건</p></section>'+list('자주 묻는 질문',d.frequent_questions,'question')+list('반복 키워드',d.frequent_keywords,'keyword')+list('자주 사용된 근거 조문',d.frequent_articles,'article')+history(d.latest_events||[])}).catch(error=>document.getElementById('content').textContent=error.message);</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def web_app() -> HTMLResponse:
    """별도 FastAPI 기반으로 PoC의 주요 업무 흐름을 직접 제공하는 웹 화면이다."""
    # 계산은 자연어 답변 안에서만 제공하고, 예상 거래 사전진단은 챗봇의 접힌 보조정보로 통합한다.
    html = re.sub(r'<div class="panel"><h3>세액·가산세 계산</h3>.*?<div id="calc-result" class="result"></div></div>', '', INTEGRATED_WEB_APP_HTML)
    # 제거한 계산 화면의 버튼 초기화 코드가 남으면 null.onclick 예외로 이후 챗봇 이벤트까지 등록되지 않는다.
    html = re.sub(r'async function runTaxCalculation\(\).*?\$\(\'calc-run\'\)\.onclick=runTaxCalculation;', '', html)
    html = html.replace('<button data-view="expected">예상 거래 사전진단</button>', '')
    html = re.sub(r'<section id="expected" class="view">.*?</section>', '', html)
    # 예상 거래 화면의 보조 함수는 줄바꿈을 포함하므로 DOTALL로 함께 제거한다.
    html = re.sub(r'async function expectedPayload\(\).*?\$\(\'expected-diagnose\'\)\.onclick=\(\)=>runExpected\(true\);', '', html, flags=re.S)
    html = html.replace("loadStatus();$('analysis-month').value=new Date().toISOString().slice(0,7);$('expected-date').value=new Date().toISOString().slice(0,10);", "loadStatus();if($('analysis-month'))$('analysis-month').value=new Date().toISOString().slice(0,7);")
    # 클립보드 첨부 목록을 갱신하는 사이 버튼이 없어질 수 있으므로, 존재할 때만 클릭 이벤트를 연결한다.
    html = html.replace("$('clear-chat-captures').onclick=()=>{state.chatAttachments=[];renderChatAttachments()}", "const clearCaptures=$('clear-chat-captures');if(clearCaptures)clearCaptures.onclick=()=>{state.chatAttachments=[];renderChatAttachments()}")
    # 챗봇 전송은 다른 대시보드 스크립트와 분리한다. 부가 화면의 오류가 있어도 질문·후속 질문은 계속 동작한다.
    chat_script = """(()=>{const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#039;'}[c]));const chat=$('chat-messages'),input=$('chat-question'),send=$('chat-send');if(!chat||!input||!send)return;const add=(kind,content)=>{const node=document.createElement('article');node.className='message '+kind;node.innerHTML=content;chat.append(node);return node};const attachments=async()=>{const files=[...($('chat-files')?.files||[])].slice(0,5);return Promise.all(files.map(file=>new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve({filename:file.name,content_type:file.type||'application/octet-stream',content_base64:String(reader.result).split(',')[1]});reader.onerror=reject;reader.readAsDataURL(file)})))};const expertBlocks=text=>{const parts=String(text||'').replace(/\\r/g,'').split(/\\[(핵심 판단|적용 기준|담당자 조치)\\]/);if(parts.length<3)return '<div class="expert-card expert-card-wide">'+esc(text||'검토 내용을 생성하지 못했습니다.').replace(/\\n/g,'<br>')+'</div>';let blocks='';for(let i=1;i<parts.length;i+=2){const title=parts[i],content=(parts[i+1]||'').trim();blocks+='<section class="expert-card"><h4>'+esc(title)+'</h4><p>'+esc(content).replace(/\\n/g,'<br>')+'</p></section>'}return '<div class="expert-grid">'+blocks+'</div>'};const render=(payload)=>{const answer=payload.answer||{},docs=new Map((payload.evidence_documents||[]).map(item=>[item.document_id,item])),used=(answer.evidence_ids||[]).map(id=>docs.get(id)).filter(Boolean),isExpert=answer.generation_mode==='expert_review';let body=answer.key_answer?'<div class="key">핵심 안내<br>'+esc(answer.key_answer)+'</div>':'';body+=isExpert?expertBlocks(answer.answer):'<div>'+esc(answer.answer||'답변을 생성하지 못했습니다.').replace(/\\n/g,'<br>')+'</div>';if(used.length)body+='<details class="evidence-fold"><summary>근거 조문·기준서 '+used.length+'건</summary><ul class="sources">'+used.map(item=>'<li>'+esc(item.title+(item.article?' · '+item.article:''))+'</li>').join('')+'</ul></details>';if(answer.follow_up_questions?.length)body+='<div class="followups">'+answer.follow_up_questions.map(question=>'<button type="button" data-followup="'+esc(question)+'">'+esc(question)+'</button>').join('')+'</div>';const hint=payload.transaction_hint;if(hint)body+='<details><summary>거래 검토 보조정보</summary><div class="small"><b>간단 거래설명 </b>'+esc(hint.summary)+'</div><div class="small"><b>쟁점 키워드 </b>'+esc((hint.issue_keywords||[]).join(' · '))+'</div><div class="small"><b>관련 회계계정 </b>'+esc((hint.related_accounts||[]).join(' · '))+'</div></details>';add('answer',body)};const submit=async(question)=>{const value=(question||input.value).trim();if(!value)return;input.value='';add('question',esc(value));const loading=add('muted','근거 문서를 검색하고 답변을 준비하고 있습니다.');try{const response=await fetch('/knowledge-chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:value,conversation:[],attachments:await attachments()})});const payload=await response.json();if(!response.ok)throw new Error(payload.detail||'답변을 불러오지 못했습니다.');loading.remove();render(payload)}catch(error){loading.className='message error';loading.textContent=error.message||'답변을 불러오지 못했습니다.'}};send.onclick=event=>{event.preventDefault();submit()};input.onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();submit()}};chat.addEventListener('click',event=>{const button=event.target.closest('[data-followup]');if(button)submit(button.dataset.followup)});})();"""
    # 일반 입력은 독립 질의로 시작하고, 후속 질문 또는 사용자의 명시 선택 때만 직전 검토 문맥을 전달한다.
    chat_script = chat_script.replace(
        "const chat=$('chat-messages'),input=$('chat-question'),send=$('chat-send');if(!chat||!input||!send)return;",
        "const chat=$('chat-messages'),input=$('chat-question'),send=$('chat-send'),conversation=[];if(!chat||!input||!send)return;const controls=document.createElement('div');controls.className='chat-session-controls';controls.innerHTML='<span id=\"chat-session-mode\" class=\"chat-session-mode\">새 질의</span><label class=\"chat-context-toggle\"><input id=\"continue-context\" type=\"checkbox\"> 이전 검토 이어가기</label><button id=\"new-chat\" type=\"button\" class=\"secondary\">새 채팅</button>';const inputArea=input.closest('.chat-input');if(inputArea)inputArea.parentNode.insertBefore(controls,inputArea);const continueContext=$('continue-context'),sessionMode=$('chat-session-mode'),newChat=$('new-chat');const updateMode=()=>{sessionMode.textContent=continueContext.checked?'이전 검토 이어가기':'새 질의'};continueContext.onchange=updateMode;newChat.onclick=()=>{conversation.length=0;chat.innerHTML='';input.value='';if($('chat-files'))$('chat-files').value='';continueContext.checked=false;updateMode()};updateMode();",
    )
    chat_script = chat_script.replace(
        "const submit=async(question)=>{const value=(question||input.value).trim();if(!value)return;",
        "const submit=async(question,followup=false)=>{const value=(question||input.value).trim();if(!value)return;if(!followup&&!continueContext.checked)conversation.length=0;",
    )
    chat_script = chat_script.replace(
        "body:JSON.stringify({question:value,conversation:[],attachments:await attachments()})",
        "body:JSON.stringify({question:value,conversation:(followup||continueContext.checked)?conversation.slice(-3):[],attachments:await attachments()})",
    )
    chat_script = chat_script.replace(
        "loading.remove();render(payload)",
        "render(payload);conversation.push({question:value,key_answer:(payload.answer||{}).key_answer||(payload.answer||{}).answer||''});loading.remove()",
    )
    chat_script = chat_script.replace(
        "send.onclick=event=>{event.preventDefault();submit()}",
        "send.onclick=event=>{event.preventDefault();submit('',false)}",
    )
    chat_script = chat_script.replace(
        "event.preventDefault();submit()",
        "event.preventDefault();submit('',false)",
    )
    chat_script = chat_script.replace(
        "if(button)submit(button.dataset.followup)",
        "if(button)submit(button.dataset.followup,true)",
    )
    # 법령 근거는 문서 전체가 아니라 검색된 조문으로 바로 이동하고, 그 밖의 근거는 원래 상세 URL을 유지한다.
    chat_script = chat_script.replace(
        "const render=(payload)=>{",
        "const evidenceHref=item=>{const source=String(item.source_url||''),meta=item.metadata||{};if(meta.document_type==='accounting_standard'){const parent=String(meta.parent_document_id||'');const page=Number(meta.page_start||0);if(parent)return '/knowledge-source/'+encodeURIComponent(parent)+(page?'#page='+page:'')}const article=String(item.article||'').match(/제\\d+(?:의\\d+)?조/);const isLawDocument=source.includes('/법령/')||source.includes('/%EB%B2%95%EB%A0%B9/');if(isLawDocument&&article)return 'https://www.law.go.kr/법령/'+encodeURIComponent(String(item.title||''))+'/'+encodeURIComponent(article[0]);return source};const render=(payload)=>{",
    )
    chat_script = chat_script.replace(
        "used.map(item=>'<li>'+esc(item.title+(item.article?' · '+item.article:''))+'</li>').join('')",
        "used.map(item=>{const label=item.title+(item.article?' · '+item.article:'');const href=evidenceHref(item);return '<li>'+(href?'<a target=\"_blank\" rel=\"noopener\" href=\"'+esc(href)+'\">'+esc(label)+'</a>':esc(label))+'</li>'}).join('')",
    )
    # 단순 답변은 프롬프트가 반환한 '핵심 설명'과 '관련 근거'를 별도 카드로 보여 주되, 형식을 알 수 없는 답변은 기존 본문으로 안전하게 표시한다.
    chat_script = chat_script.replace(
        "const render=(payload)=>{",
        "const simpleAnswerBlocks=text=>{const match=String(text||'').replace(/\\r/g,'').match(/^\\s*핵심 설명:\\s*([\\s\\S]*?)(?:\\n\\s*관련 근거:\\s*([\\s\\S]*))?\\s*$/);return match?{explanation:match[1].trim(),rationale:(match[2]||'').trim()}:null};const evidenceLinks=items=>items.map(item=>{const label=item.title+(item.article?' · '+item.article:'');const href=evidenceHref(item);const version=item.effective_date_or_version?'시행·버전 '+item.effective_date_or_version:'';const content='<span class=\"evidence-link-title\">'+esc(label)+'</span>'+(version?'<small>'+esc(version)+'</small>':'')+'<b>원문 보기 ↗</b>';return href?'<a class=\"evidence-link\" target=\"_blank\" rel=\"noopener\" href=\"'+esc(href)+'\">'+content+'</a>':'<span class=\"evidence-link disabled\">'+content+'</span>'}).join('');const render=(payload)=>{",
    )
    chat_script = chat_script.replace(
        "body+=isExpert?expertBlocks(answer.answer):'<div>'+esc(answer.answer||'답변을 생성하지 못했습니다.').replace(/\\n/g,'<br>')+'</div>';if(used.length)body+='<details class=\"evidence-fold\"><summary>근거 조문·기준서 '+used.length+'건</summary><ul class=\"sources\">'+used.map(item=>{const label=item.title+(item.article?' · '+item.article:'');const href=evidenceHref(item);return '<li>'+(href?'<a target=\"_blank\" rel=\"noopener\" href=\"'+esc(href)+'\">'+esc(label)+'</a>':esc(label))+'</li>'}).join('')+'</ul></details>';",
        "const simple=simpleAnswerBlocks(answer.answer);body+=isExpert?expertBlocks(answer.answer):simple?'<section class=\"answer-explanation\"><div class=\"answer-section-label\">실무 해설</div><p>'+esc(simple.explanation).replace(/\\n/g,'<br>')+'</p></section>'+(simple.rationale?'<section class=\"answer-rationale\"><div class=\"answer-section-label\">관련 근거 요약</div><p>'+esc(simple.rationale).replace(/\\n/g,'<br>')+'</p></section>':''):'<div class=\"answer-body\">'+esc(answer.answer||'답변을 생성하지 못했습니다.').replace(/\\n/g,'<br>')+'</div>';if(used.length)body+=simple?'<section class=\"answer-sources\"><div class=\"answer-section-label\">확인 근거</div><div class=\"evidence-links\">'+evidenceLinks(used)+'</div></section>':'<details class=\"evidence-fold\"><summary>근거 조문·기준서 '+used.length+'건</summary><div class=\"evidence-links\">'+evidenceLinks(used)+'</div></details>';",
    )
    # 질문 난이도나 회계·세무 구분과 무관하게 동일한 검토 메모 틀을 쓰고, 필요한 항목만 세로로 표시한다.
    chat_script = re.sub(
        r"const expertBlocks=text=>\{.*?\};const render=",
        "const reviewBlocks=text=>{const parts=String(text||'').replace(/\\r/g,'').split(/\\[(사실관계·쟁점|적용 기준|검토 의견|추가 확인)\\]/);if(parts.length<3)return '<section class=\"review-card\"><div class=\"review-card-title\">검토 내용</div><p>'+esc(text||'검토 내용을 생성하지 못했습니다.').replace(/\\n/g,'<br>')+'</p></section>';let blocks='';for(let i=1;i<parts.length;i+=2){const title=parts[i],content=(parts[i+1]||'').trim();if(content)blocks+='<section class=\"review-section\"><h4>'+esc(title)+'</h4><p>'+esc(content).replace(/\\n/g,'<br>')+'</p></section>'}return '<section class=\"review-card\"><div class=\"review-card-title\">검토 내용</div>'+blocks+'</section>'};const render=",
        chat_script,
        count=1,
        flags=re.S,
    )
    chat_script = re.sub(
        r"const expertBlocks=text=>\{.*?\};(?=const evidenceHref=)",
        lambda _match: "const reviewBlocks=text=>{const parts=String(text||'').replace(/\\r/g,'').split(/\\[(사실관계·쟁점|적용 기준|검토 의견|추가 확인)\\]/);if(parts.length<3)return '<section class=\"review-card\"><div class=\"review-card-title\">검토 내용</div><p>'+esc(text||'검토 내용을 생성하지 못했습니다.').replace(/\\n/g,'<br>')+'</p></section>';let blocks='';for(let i=1;i<parts.length;i+=2){const title=parts[i],content=(parts[i+1]||'').trim();if(content)blocks+='<section class=\"review-section\"><h4>'+esc(title)+'</h4><p>'+esc(content).replace(/\\n/g,'<br>')+'</p></section>'}return '<section class=\"review-card\"><div class=\"review-card-title\">검토 내용</div>'+blocks+'</section>'};",
        chat_script,
        count=1,
        flags=re.S,
    )
    chat_script = re.sub(
        r"const simple=simpleAnswerBlocks\(answer\.answer\);body\+=.*?;if\(answer\.follow_up_questions",
        "body+=reviewBlocks(answer.answer);if(used.length)body+='<section class=\"answer-sources\"><div class=\"answer-section-label\">확인 근거</div><div class=\"evidence-links\">'+evidenceLinks(used)+'</div></section>';if(answer.follow_up_questions",
        chat_script,
        count=1,
        flags=re.S,
    )
    # 모델이 선택한 핵심어만 이스케이프 후 표시해, 답변 본문에서 조문·금액·기한을 빠르게 식별하게 한다.
    chat_script = re.sub(
        r"const reviewBlocks=text=>\{.*?\};const evidenceHref=",
        lambda _match: "const reviewBlocks=(text,terms)=>{const parts=String(text||'').replace(/\\r/g,'').split(/\\[(사실관계·쟁점|적용 기준|검토 의견|추가 확인)\\]/);if(parts.length<3)return '<section class=\"review-card\"><div class=\"review-card-title\">검토 내용</div><p>'+highlighted(text||'검토 내용을 생성하지 못했습니다.',terms).replace(/\\n/g,'<br>')+'</p></section>';let blocks='';for(let i=1;i<parts.length;i+=2){const title=parts[i],content=(parts[i+1]||'').trim();if(content)blocks+='<section class=\"review-section\"><h4>'+esc(title)+'</h4><p>'+highlighted(content,terms).replace(/\\n/g,'<br>')+'</p></section>'}return '<section class=\"review-card\"><div class=\"review-card-title\">검토 내용</div>'+blocks+'</section>'};const evidenceHref=",
        chat_script,
        count=1,
        flags=re.S,
    )
    chat_script = chat_script.replace(
        "const evidenceHref=item=>",
        "const highlighted=(text,terms)=>{let value=esc(text||'');[...new Set((terms||[]).filter(term=>String(term).length>1))].sort((left,right)=>String(right).length-String(left).length).forEach(term=>{const safe=esc(term);value=value.split(safe).join('<mark class=\"answer-mark\"><strong>'+safe+'</strong></mark>')});return value};const evidenceHref=item=>",
    )
    chat_script = chat_script.replace(
        "let body=answer.key_answer?'<div class=\"key\">핵심 안내<br>'+esc(answer.key_answer)+'</div>':'';",
        "let body=answer.key_answer?'<div class=\"key\">핵심 안내<br>'+highlighted(answer.key_answer,answer.highlight_terms)+'</div>':'';",
    )
    chat_script = chat_script.replace(
        "body+=reviewBlocks(answer.answer);",
        "body+=reviewBlocks(answer.answer,answer.highlight_terms);",
    )
    # 모든 사용자의 실행 요청에는 공통 진행 표시를 적용한다. 챗봇은 카드형 진행 표시도 함께 유지한다.
    chat_script = chat_script.replace(
        "})();",
        "const globalLoader=document.createElement('div');globalLoader.className='global-request-loader';globalLoader.setAttribute('aria-live','polite');globalLoader.innerHTML='<span class=\"global-orbit\"></span><span>요청을 처리하고 있습니다</span>';document.body.append(globalLoader);const originalFetch=window.fetch.bind(window);let activeRequests=0;window.fetch=async(...args)=>{const options=args[1]||{},showLoader=String(options.method||'GET').toUpperCase()!=='GET';if(showLoader){activeRequests+=1;globalLoader.classList.add('visible')}try{return await originalFetch(...args)}finally{if(showLoader&&--activeRequests===0)globalLoader.classList.remove('visible')}};})();",
    )
    # 서식 렌더링에 실패해도 이미 받은 핵심 답변을 숨기지 않고, 텍스트 답변으로 안전하게 표시한다.
    chat_script = chat_script.replace(
        "render(payload);conversation.push({question:value,key_answer:(payload.answer||{}).key_answer||(payload.answer||{}).answer||''});loading.remove()",
        "try{render(payload);conversation.push({question:value,key_answer:(payload.answer||{}).key_answer||(payload.answer||{}).answer||''});clearTimeout(progressTimer);clearTimeout(reviewTimer);loading.remove()}catch(renderError){clearTimeout(progressTimer);clearTimeout(reviewTimer);const answer=payload.answer||{};loading.className='message answer';loading.textContent=[answer.key_answer,answer.answer].filter(Boolean).join('\\n\\n')||'답변을 표시하지 못했습니다.'}",
    )
    chat_script = chat_script.replace(
        "const loading=add('muted','근거 문서를 검색하고 답변을 준비하고 있습니다.');",
        "const loading=add('muted loading','<div class=\"loading-panel\" role=\"status\"><span class=\"chat-orbit\"></span><div class=\"loading-copy\"><strong>AI 검토 준비 중</strong><span class=\"chat-progress\">질문 분석 및 근거 검색 중…</span><span class=\"loading-track\"><i></i></span></div></div>');const progressTimer=setTimeout(()=>{const node=loading.querySelector('.chat-progress');if(node)node.textContent='근거 연결 및 Evidence Pack 준비 중…'},1200);const reviewTimer=setTimeout(()=>{const node=loading.querySelector('.chat-progress');if(node)node.textContent='회계·세무 전문가 검토 중…'},3500);",
    )
    chat_script = chat_script.replace(
        "catch(error){loading.className='message error';",
        "catch(error){clearTimeout(progressTimer);clearTimeout(reviewTimer);loading.className='message error';",
    )
    # 화면 조합 과정에서 로딩 효과가 빠지면 조용히 배포하지 않고 즉시 오류로 드러낸다.
    loading_contract = ("loading-panel", "chat-orbit", "chat-progress", "loading-track", "progressTimer", "reviewTimer", "global-request-loader", "originalFetch")
    if any(marker not in chat_script for marker in loading_contract):
        raise RuntimeError("CHAT_LOADING_CONTRACT_OK 위반: 챗봇 로딩 효과 구성이 누락되었습니다.")
    html = html.replace("</style>", ".chat-spinner{display:inline-block;width:14px;height:14px;margin-right:9px;border:2px solid #bdd7ef;border-top-color:#0668b9;border-radius:50%;vertical-align:-2px;animation:chat-spin .8s linear infinite}@keyframes chat-spin{to{transform:rotate(360deg)}}.loading{display:flex;align-items:center}.chat-session-controls{display:flex;align-items:center;gap:10px;margin:14px 0 8px;font-size:13px}.chat-session-mode{padding:5px 10px;background:#eaf3fb;color:#0768b4;border-radius:14px;font-weight:800}.chat-context-toggle{display:flex;align-items:center;gap:5px;color:#52687c}.chat-context-toggle input{width:auto}.chat-session-controls .secondary{margin-left:auto;padding:6px 11px;font-size:13px}.expert-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:16px 0}.expert-card{background:#f5f8fb;border:1px solid #dbe6ef;border-top:3px solid #1675bc;border-radius:3px;padding:14px 16px;min-height:118px}.expert-card h4{color:#0b5f9f;font-size:14px;margin:0 0 9px;font-weight:800}.expert-card p{margin:0;color:#253746;font-size:14px;line-height:1.7}.expert-card-wide{margin:16px 0}.evidence-fold{border-top:1px solid #d7e0e8;margin-top:18px;padding-top:11px}.evidence-fold summary{font-weight:700;color:#38536b}@media(max-width:760px){.chat-session-controls{flex-wrap:wrap}.chat-session-controls .secondary{margin-left:0}.expert-grid{grid-template-columns:1fr}.expert-card{min-height:auto}}</style>")
    html = html.replace("</style>", ".answer-explanation,.answer-rationale,.answer-sources{margin:16px 0;border-radius:12px}.answer-explanation{padding:18px 20px;background:#fff;border:1px solid #dce7f0;border-left:5px solid #0874bd;box-shadow:0 5px 16px rgba(20,79,122,.05)}.answer-rationale{padding:16px 20px;background:#f5faff;border:1px solid #cfe2f2}.answer-sources{padding:16px 18px;background:linear-gradient(135deg,#f8fbfd,#eff7fc);border:1px solid #d7e7f1}.answer-section-label{margin-bottom:9px;color:#0868b8;font-size:12px;font-weight:800;letter-spacing:.08em}.answer-explanation p,.answer-rationale p{margin:0;color:#253746;line-height:1.8}.answer-body{line-height:1.8}.evidence-links{display:grid;gap:9px}.evidence-link{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 16px;align-items:center;padding:12px 14px;background:#fff;border:1px solid #cfe0ec;border-radius:9px;color:#1e415d;text-decoration:none;transition:transform .16s ease,border-color .16s ease,box-shadow .16s ease}.evidence-link:hover{border-color:#1482c7;box-shadow:0 5px 14px rgba(8,104,184,.12);transform:translateY(-1px)}.evidence-link-title{min-width:0;color:#075e9f;font-weight:800}.evidence-link small{grid-column:1;color:#6d8091;font-size:11px}.evidence-link b{grid-column:2;grid-row:1 / span 2;color:#1675bc;font-size:12px;white-space:nowrap}.evidence-link.disabled{opacity:.66}.evidence-fold .evidence-links{margin-top:12px}@media(max-width:760px){.evidence-link{grid-template-columns:1fr}.evidence-link b{grid-column:1;grid-row:auto}}</style>")
    html = html.replace("</style>", ".review-card{margin:16px 0;padding:0 20px 5px;background:#fff;border:1px solid #dbe6ef;border-radius:12px;box-shadow:0 5px 16px rgba(20,79,122,.04)}.review-card-title{padding:15px 0 11px;color:#0868b8;font-size:13px;font-weight:800;letter-spacing:.08em;border-bottom:1px solid #dce7f0}.review-card>p{margin:14px 0 16px;line-height:1.8}.review-section{padding:14px 0;border-bottom:1px solid #e5edf3}.review-section:last-child{border-bottom:0}.review-section h4{margin:0 0 7px;color:#254a68;font-size:14px;font-weight:800}.review-section p{margin:0;color:#253746;line-height:1.8}@media(max-width:760px){.review-card{padding:0 15px 4px}}</style>")
    html = html.replace("</style>", ".answer-mark{padding:1px 3px;background:linear-gradient(120deg,#fff5ad,#ffe987);border-radius:3px;box-decoration-break:clone;-webkit-box-decoration-break:clone;color:#17344c;font-weight:800}</style>")
    html = html.replace("</style>", ".loading{padding:0!important;background:transparent!important;border:0!important}.loading-panel{display:flex;align-items:center;gap:14px;width:100%;padding:16px 18px;background:linear-gradient(110deg,#f7fbff,#e9f4fd);border:1px solid #c9e0f2;border-radius:9px;box-shadow:0 8px 22px rgba(24,98,158,.09);animation:loading-enter .28s ease-out}.chat-orbit{position:relative;display:block;flex:0 0 34px;width:34px;height:34px;border:3px solid #b8d9ef;border-top-color:#0874bd;border-right-color:#0874bd;border-radius:50%;animation:chat-spin .75s linear infinite}.chat-orbit:after{content:'';position:absolute;inset:7px;border:2px solid transparent;border-bottom-color:#ff5a5f;border-radius:50%;animation:chat-spin 1.05s linear infinite reverse}.loading-copy{display:flex;flex:1;flex-direction:column;gap:4px;min-width:0}.loading-copy strong{font-size:13px;color:#075e9f;letter-spacing:.01em}.chat-progress{font-size:14px;color:#334d63}.loading-track{display:block;overflow:hidden;width:100%;height:4px;background:#d5e7f4;border-radius:6px;margin-top:5px}.loading-track i{display:block;width:42%;height:100%;border-radius:6px;background:linear-gradient(90deg,#0874bd,#63b7e7,#0874bd);animation:loading-sweep 1.35s ease-in-out infinite}@keyframes loading-enter{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}@keyframes loading-sweep{from{transform:translateX(-110%)}to{transform:translateX(270%)}}</style>")
    html = html.replace("</style>", ".global-request-loader{position:fixed;z-index:9999;top:18px;right:22px;display:flex;align-items:center;gap:9px;padding:10px 14px;background:#073e69;color:#fff;border:1px solid #4da6dc;border-radius:24px;box-shadow:0 8px 22px rgba(4,48,82,.22);font-size:13px;font-weight:800;opacity:0;transform:translateY(-12px);pointer-events:none;transition:opacity .18s,transform .18s}.global-request-loader.visible{opacity:1;transform:translateY(0)}.global-orbit{width:15px;height:15px;border:2px solid rgba(255,255,255,.35);border-top-color:#fff;border-radius:50%;animation:chat-spin .65s linear infinite}@media(max-width:760px){.global-request-loader{top:10px;right:10px}}</style>")
    html = html.replace("</script></body>", "</script><script>" + chat_script + "</script></body>")
    # 인라인 이벤트가 포함된 단일 화면은 이전 HTML이 남으면 버튼 수정도 반영되지 않으므로 캐시하지 않는다.
    return HTMLResponse(html, headers={"Cache-Control": "no-store, max-age=0"})


def require_admin(credentials: HTTPBasicCredentials | None = Depends(ADMIN_CREDENTIALS)) -> None:
    """환경변수의 관리자 계정으로만 운영 기록에 접근하게 한다."""
    expected_username = os.environ.get("ADMIN_DASHBOARD_USERNAME", "admin")
    expected_password = os.environ.get("ADMIN_DASHBOARD_PASSWORD", "")
    if not expected_password:
        raise HTTPException(status_code=503, detail="관리자 비밀번호가 설정되지 않았습니다.")
    is_valid = bool(credentials) and secrets.compare_digest(credentials.username, expected_username) and secrets.compare_digest(credentials.password, expected_password)
    if not is_valid:
        raise HTTPException(status_code=401, detail="관리자 인증이 필요합니다.", headers={"WWW-Authenticate": "Basic"})


@app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
def admin_web(_: None = Depends(require_admin)) -> HTMLResponse:
    """운영 담당자용 익명 집계 화면을 제공한다."""
    return HTMLResponse(admin_web_html(), headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/health")
def health() -> dict[str, object]:
    """화면에서 API와 PostgreSQL 설정 상태를 확인한다."""
    return {"api": "ok", "database": database_status()}


@app.get("/knowledge-refresh/status")
def knowledge_refresh_status() -> dict[str, object]:
    """DB 잠금과 무관하게 법령·판례 수집 진행 상태를 반환한다."""
    return read_refresh_status()


@app.get("/knowledge-base/summary")
def knowledge_base_summary() -> dict[str, object]:
    """기준 데이터 화면에 읽기 전용 문서·검색 청크 현황만 제공한다."""
    if not DEFAULT_DB_PATH.is_file():
        return {"status": "not_initialized", "document_types": {}, "chunk_count": None}
    try:
        with sqlite3.connect(f"file:{DEFAULT_DB_PATH}?mode=ro", uri=True, timeout=1) as connection:
            document_types = {
                str(document_type): int(count)
                for document_type, count in connection.execute(
                    "SELECT document_type, COUNT(*) FROM documents GROUP BY document_type ORDER BY document_type"
                )
            }
            chunk_count = int(connection.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0])
        tracks: dict[str, int] = {"회계": 0, "세무": 0, "공통": 0}
        for document_type, count in document_types.items():
            tracks[evidence_track(document_type)] += count
        return {"status": "ready", "document_types": document_types, "tracks": tracks, "chunk_count": chunk_count}
    except sqlite3.Error as error:
        # 잠금에 따른 일시 지연과 저장소 오류를 구분해 갱신 중이라는 오해를 막는다.
        error_code = getattr(error, "sqlite_errorcode", 0)
        busy = error_code & 0xFF in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
        return {"status": "updating" if busy else "unavailable", "document_types": {}, "chunk_count": None}


@app.get("/knowledge-graph/status")
def knowledge_graph_status() -> dict[str, object]:
    """화면이 Neo4j의 설정 여부만 확인하고 연결 비밀값은 노출하지 않는다."""
    return {
        "configured": neo4j_settings() is not None,
        "mode": "neo4j" if neo4j_settings() is not None else "sqlite_relation_fallback",
    }


@app.get("/admin/chat-analytics")
def chat_analytics(_: None = Depends(require_admin)) -> dict[str, object]:
    """관리자가 반복 질문·계산 수요·자주 사용된 근거를 개인 식별 없이 확인한다."""
    initialize_chat_analytics()
    with sqlite3.connect(ANALYTICS_DB_PATH) as connection:
        rows = connection.execute(
            "SELECT question_text, question_hash, answer_summary, answer_text, answer_mode, calculation_used, evidence_articles_json, created_at FROM chat_events ORDER BY created_at DESC"
        ).fetchall()
    question_counts = Counter(row[1] for row in rows)
    representative_questions: dict[str, str] = {}
    article_counts: Counter[str] = Counter()
    keyword_counts: Counter[str] = Counter()
    for question, question_hash, _summary, _answer_text, _mode, _calculation, article_json, _created_at in rows:
        representative_questions.setdefault(str(question_hash), str(question))
        for article in json.loads(str(article_json)):
            article_counts[str(article)] += 1
        for keyword in re.findall(r"[가-힣A-Za-z0-9]{2,}", str(question)):
            if keyword not in ANALYTICS_STOPWORDS:
                keyword_counts[keyword] += 1
    frequent_questions = [
        {"question": representative_questions[key], "count": count}
        for key, count in question_counts.most_common(10)
    ]
    return {
        "event_count": len(rows),
        "calculation_count": sum(int(row[5]) for row in rows),
        "frequent_questions": frequent_questions,
        "frequent_keywords": [{"keyword": key, "count": count} for key, count in keyword_counts.most_common(15)],
        "frequent_articles": [{"article": key, "count": count} for key, count in article_counts.most_common(15)],
        "latest_events": [
            {"question": row[0], "answer_summary": row[2], "answer_text": row[3], "answer_mode": row[4], "calculation_used": bool(row[5]), "evidence_articles": json.loads(str(row[6])), "created_at": row[7]}
            for row in rows[:30]
        ],
    }


@app.get("/ledger/schema")
def ledger_schema() -> dict[str, object]:
    """SAP 원장 업로드 전 필요한 최소 컬럼을 반환한다."""
    return {"required_columns": REQUIRED_LEDGER_COLUMNS}


@app.post("/ledger/validate-csv")
def validate_csv(payload: CsvHeaderValidationRequest) -> dict[str, object]:
    """화면에서 전달한 CSV 헤더를 순수 원장 검증 로직으로 확인한다."""
    return validate_csv_headers(payload.csv_text)


@app.get("/ai-review/status")
def ai_review_status() -> dict[str, str]:
    """화면이 지정 모델과 키 설정 여부만 확인하도록 한다."""
    return {"model": MODEL_NAME, "status": "configured" if os.environ.get("OPENAI_API_KEY") else "not_configured"}


@app.post("/ai-review")
def ai_review(payload: AiReviewRequest) -> dict[str, object]:
    """승인된 근거 문서를 바탕으로 OpenAI 잠정 검토를 생성한다."""
    try:
        return review_with_openai(payload.transaction, [item.model_dump() for item in payload.evidence_documents])
    except AiReviewError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/ai-review/evidence-preview")
def ai_review_evidence_preview(payload: AutoAiReviewRequest) -> dict[str, object]:
    """AI 호출 전 자동 검색된 근거 문서와 검색어를 확인한다."""
    try:
        return search_local_evidence(payload.transaction, payload.issue_keywords, payload.evidence_limit)
    except EvidenceSearchError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/ai-review/with-auto-evidence")
def ai_review_with_auto_evidence(payload: AutoAiReviewRequest) -> dict[str, object]:
    """로컬 지식기반 검색 결과만 사용해 OpenAI 잠정 검토를 생성한다."""
    try:
        evidence_result = search_local_evidence(payload.transaction, payload.issue_keywords, payload.evidence_limit)
        review_result = review_with_openai(payload.transaction, evidence_result["evidence_documents"])
        return {**evidence_result, **review_result}
    except (EvidenceSearchError, AiReviewError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def legal_article_evidence(law_title: str, article_prefix: str) -> list[dict[str, object]]:
    """계산에 사용한 법령 조문을 제목·조문번호로 정확히 읽어 근거와 함께 반환한다."""
    with sqlite3.connect(f"file:{DEFAULT_DB_PATH.resolve()}?mode=ro", uri=True, timeout=2) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT c.chunk_id, c.content, c.law_article, c.hierarchy_path, c.metadata_json,
                      d.document_id, d.title, d.source, d.source_url, d.effective_date, d.version
               FROM document_chunks c JOIN documents d ON d.document_id = c.document_id
               WHERE d.title = ? AND c.law_article LIKE ? ORDER BY c.chunk_index LIMIT 4""",
            (law_title, f"{article_prefix}%"),
        ).fetchall()
    return [
        {
            "document_id": str(row["chunk_id"]),
            "title": str(row["title"]),
            "source": str(row["source"]),
            "source_url": row["source_url"],
            "effective_date_or_version": row["effective_date"] or row["version"],
            "article": row["law_article"],
            "hierarchy_path": row["hierarchy_path"],
            "excerpt": str(row["content"]),
            "metadata": {"calculation_evidence": True, **json.loads(str(row["metadata_json"]))},
            "relevance": 100,
        }
        for row in rows
    ]


def calculation_input_required(message: str, fields: list[str], evidence: list[dict[str, object]]) -> dict[str, object]:
    """입력값이 없을 때도 계산 불가 사유와 필요한 최소 항목을 구조화해 반환한다."""
    return {"status": "input_required", "message": message, "required_fields": fields, "evidence_documents": evidence}


@app.post("/tax-calculations")
def tax_calculation(payload: TaxCalculationRequest) -> dict[str, object]:
    """확인 가능한 법정 요율과 사용자가 준 수치만으로 세액·가산세를 계산한다."""
    if payload.calculation_type == "national_strategy_credit":
        evidence = legal_article_evidence("조세특례제한법", "제24조")
        if payload.amount is None or payload.enterprise_type is None or payload.tax_year is None:
            return calculation_input_required("공제액 계산에는 투자금액·기업유형·투자 과세연도가 필요합니다.", ["amount", "enterprise_type", "tax_year"], evidence)
        if payload.tax_year > 2029:
            return calculation_input_required("현재 보존된 제24조의 국가전략기술 시설 투자 적용기한은 2029년 12월 31일까지입니다. 해당 과세연도 법령 버전을 확인해 주세요.", ["tax_year"], evidence)
        rates = (
            {"small": 30, "graduating": 25, "other": 20}
            if payload.semiconductor
            else {"small": 25, "graduating": 20, "other": 15}
        )
        rate = rates[payload.enterprise_type]
        calculated = round(payload.amount * rate / 100)
        facility = "반도체 분야 국가전략기술 시설" if payload.semiconductor else "국가전략기술 시설"
        return {
            "status": "calculated",
            "calculation_type": payload.calculation_type,
            "result_amount": calculated,
            "result_label": "기본공제 예상액",
            "formula": f"투자금액 {payload.amount:,.0f}원 × {rate}% = {calculated:,.0f}원",
            "assumptions": [f"{facility}에 대한 기본공제만 계산했습니다.", "직전 3년 평균 투자액 초과분의 추가공제, 공제 한도 및 다른 적용요건은 별도로 검토해야 합니다."],
            "evidence_documents": evidence,
        }
    if payload.calculation_type == "unreported_penalty":
        evidence = legal_article_evidence("지방세기본법", "제53조")
        if payload.amount is None:
            return calculation_input_required("무신고가산세 계산에는 무신고납부세액 또는 과소신고세액이 필요합니다.", ["amount", "violation_type"], evidence)
        violation = payload.violation_type or "ordinary"
        rate = 40 if violation == "fraudulent" else 20
        calculated = round(payload.amount * rate / 100)
        return {
            "status": "calculated",
            "calculation_type": payload.calculation_type,
            "result_amount": calculated,
            "result_label": "무신고가산세 예상액",
            "formula": f"무신고납부세액 {payload.amount:,.0f}원 × {rate}% = {calculated:,.0f}원",
            "assumptions": ["부정행위 여부는 사용자가 입력한 구분을 사용했습니다.", "감면·중복 가산세 조정 등 개별 법정 예외는 포함하지 않았습니다."],
            "evidence_documents": evidence,
        }
    evidence = legal_article_evidence("지방세기본법", "제55조")
    required = ["amount", "statutory_due_date", "actual_payment_date", "daily_rate_percent"]
    if payload.amount is None or not payload.statutory_due_date or not payload.actual_payment_date or payload.daily_rate_percent is None:
        return calculation_input_required("납부지연가산세 계산에는 미납세액·법정납부기한·실제 납부일·적용 일일요율이 필요합니다.", required, evidence)
    try:
        overdue_days = max((date.fromisoformat(payload.actual_payment_date) - date.fromisoformat(payload.statutory_due_date)).days, 0)
    except ValueError as error:
        raise HTTPException(status_code=422, detail="날짜는 YYYY-MM-DD 형식으로 입력해 주세요.") from error
    calculated = round(payload.amount * payload.daily_rate_percent / 100 * overdue_days)
    return {
        "status": "calculated",
        "calculation_type": payload.calculation_type,
        "result_amount": calculated,
        "result_label": "납부지연가산세 예상액",
        "formula": f"미납세액 {payload.amount:,.0f}원 × 일일요율 {payload.daily_rate_percent}% × {overdue_days}일 = {calculated:,.0f}원",
        "assumptions": ["일일요율은 해당 납세의무에 적용되는 법령상 요율을 사용자가 확인해 입력한 값입니다.", "납부일이 법정납부기한보다 빠르거나 같으면 지연일수는 0일입니다."],
        "evidence_documents": evidence,
    }


def amount_from_korean_text(question: str) -> float | None:
    """자연어의 100억·1,000만원·1000000원 표기를 원화 숫자로 안전하게 변환한다."""
    match = re.search(r"(?P<amount>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>억원|억|만원|만|원)", question)
    if not match:
        return None
    number = float(match.group("amount").replace(",", ""))
    multiplier = {"억원": 100_000_000, "억": 100_000_000, "만원": 10_000, "만": 10_000, "원": 1}[match.group("unit")]
    return number * multiplier


def calculation_answer_from_question(question: str) -> dict[str, object] | None:
    """금액·유형·연도가 모두 드러난 계산형 질문만 결정적 산식으로 우선 처리한다."""
    if not any(term in question for term in ("얼마", "계산", "공제액", "가산세")):
        return None
    amount = amount_from_korean_text(question)
    if amount is None:
        return None
    if "국가전략기술" in question and "통합투자세액공제" in question:
        enterprise_type = "small" if "중소기업" in question else "graduating" if "졸업" in question else "other" if any(term in question for term in ("중견기업", "대기업", "그 밖")) else None
        year_match = re.search(r"(20\d{2})\s*년", question)
        if enterprise_type is None or year_match is None:
            return {
                "key_answer": "공제액 계산에는 투자금액 외에 기업유형과 투자 과세연도가 필요합니다.",
                "answer": "예: ‘2026년 국가전략기술 시설에 100억 원 투자한 중소기업의 공제액’처럼 입력해 주세요.",
                "evidence_ids": [], "limitations": [], "follow_up_questions": [], "highlight_terms": ["기업유형", "투자 과세연도"],
            }
        calculated = tax_calculation(TaxCalculationRequest(
            calculation_type="national_strategy_credit", amount=amount, enterprise_type=enterprise_type,
            tax_year=int(year_match.group(1)), semiconductor="반도체" in question,
        ))
        if calculated["status"] != "calculated":
            return None
        evidence = list(calculated["evidence_documents"])
        return {
            "key_answer": f"기본공제 예상액은 {int(calculated['result_amount']):,}원입니다.",
            "answer": f"{calculated['formula']}\n" + "\n".join(str(item) for item in calculated["assumptions"]),
            "evidence_ids": [str(item["document_id"]) for item in evidence[:1]],
            "limitations": list(calculated["assumptions"]),
            "follow_up_questions": ["추가공제까지 포함하면 예상 공제액은 얼마인가요?", "국가전략기술 시설 요건은 무엇인가요?"],
            "highlight_terms": [f"{int(calculated['result_amount']):,}원", "25%", "20%", "15%"],
            "calculation": calculated,
            "evidence_documents": evidence,
        }
    if "무신고" in question and "가산세" in question:
        calculated = tax_calculation(TaxCalculationRequest(
            calculation_type="unreported_penalty", amount=amount,
            violation_type="fraudulent" if "부정" in question else "ordinary",
        ))
        evidence = list(calculated["evidence_documents"])
        return {
            "key_answer": f"무신고가산세 예상액은 {int(calculated['result_amount']):,}원입니다.",
            "answer": f"{calculated['formula']}\n" + "\n".join(str(item) for item in calculated["assumptions"]),
            "evidence_ids": [str(item["document_id"]) for item in evidence[:1]],
            "limitations": list(calculated["assumptions"]),
            "follow_up_questions": ["납부지연가산세까지 함께 계산할까요?"],
            "highlight_terms": [f"{int(calculated['result_amount']):,}원", "20%" if "부정" not in question else "40%"],
            "calculation": calculated,
            "evidence_documents": evidence,
        }
    return None


def transaction_hint_from_question(question: str) -> dict[str, object] | None:
    """거래성 자연어에만 최소 검토 단서를 만들고, 단순 법령 검색에는 표시하지 않는다."""
    cues = ("거래", "매출", "매입", "구매", "판매", "계약", "투자", "취득", "처분", "비용", "자산", "차입", "대여", "수익")
    if not any(cue in question for cue in cues):
        return None
    issue_map = {
        "특수관계": "특수관계자 거래·시가 검토",
        "매출": "수익 인식·공급시기",
        "매입": "증빙·매입세액·손금 검토",
        "투자": "투자세액공제·자산 요건",
        "자산": "자본적 지출·감가상각·손상",
        "비용": "손금·증빙·기간귀속",
        "차입": "이자율·금융부채·특수관계자 자금거래",
        "대여": "이자수익·시가·특수관계자 자금거래",
        "계약": "거래 실질·권리의무·수익인식",
    }
    account_map = {
        "매출": "매출액·매출채권·부가세예수금",
        "매입": "매입·원재료·매입세액",
        "투자": "건설중인자산·기계장치·무형자산",
        "자산": "유형자산·무형자산·감가상각비",
        "비용": "지급수수료·소모품비·외주비",
        "차입": "차입금·이자비용",
        "대여": "대여금·미수수익·이자수익",
    }
    issues = [label for cue, label in issue_map.items() if cue in question]
    accounts = [label for cue, label in account_map.items() if cue in question]
    if "특수관계" in question:
        accounts.append("관계회사채권·채무·이자수익/비용")
    return {
        "summary": question[:240],
        "issue_keywords": list(dict.fromkeys(issues)) or ["거래 실질·증빙·기간귀속"],
        "related_accounts": list(dict.fromkeys(accounts)) or ["거래 성격에 따른 계정과목 확인 필요"],
        "minimum_facts": ["거래 상대방·금액·거래일", "계약서·세금계산서 등 증빙", "거래 목적과 대가 산정 근거"],
    }


EXPERT_REVIEW_TERMS = (
    "검토", "판단", "처리", "회계처리", "세무처리", "비용", "자산", "수익", "매출", "매입",
    "거래", "계약", "지출", "투자", "취득", "처분", "인식", "상각", "공제", "가산세",
)


def extract_review_facts(question: str, hint: dict[str, object] | None, attachments: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    """질문·첨부에서 확인된 사실만 전문가 검토 도구의 거래 입력으로 정리한다."""
    facts: dict[str, object] = {"사용자 질의": question}
    if hint:
        facts["거래 설명"] = hint.get("summary")
        facts["쟁점 키워드"] = hint.get("issue_keywords")
        facts["관련 회계계정 후보"] = hint.get("related_accounts")
    filenames = [str(item["filename"]) for item in attachments["text_documents"] + attachments["file_documents"] + attachments["image_documents"]]
    if filenames:
        facts["첨부자료"] = filenames
    return facts


def requires_expert_review(question: str, hint: dict[str, object] | None, attachments: dict[str, list[dict[str, str]]]) -> bool:
    """조문 단순 조회와 사실관계 판단형 회계·세무 검토를 구분한다."""
    normalized = question.replace(" ", "")
    has_review_intent = any(term.replace(" ", "") in normalized for term in EXPERT_REVIEW_TERMS)
    has_attachment = any(attachments.values())
    return bool(hint and has_review_intent) or has_attachment


def collect_review_evidence_ids(value: object, allowed_ids: set[str]) -> list[str]:
    """전문가 검토 JSON의 주장별 근거 ID를 순회해 사용자 답변 근거로 연결한다."""
    found: list[str] = []
    if isinstance(value, dict):
        ids = value.get("evidence_ids")
        if isinstance(ids, list):
            found.extend(str(item) for item in ids if str(item) in allowed_ids)
        for nested in value.values():
            found.extend(collect_review_evidence_ids(nested, allowed_ids))
    elif isinstance(value, list):
        for nested in value:
            found.extend(collect_review_evidence_ids(nested, allowed_ids))
    return list(dict.fromkeys(found))


def expert_review_chat_answer(
    question: str,
    internal_context: dict[str, object],
    evidence_documents: list[dict[str, object]],
    attachments: dict[str, list[dict[str, str]]],
    conversation: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """같은 Evidence Pack을 쓰되, 챗봇에 맞춘 짧은 전문가 검토를 생성한다."""
    answer = answer_natural_language_question(
        question, internal_context, evidence_documents, conversation or [], attachments, expert_mode=True,
    )
    answer["generation_mode"] = "expert_review"
    return answer


def run_chat_review_graph(
    question: str,
    internal_context: dict[str, object],
    evidence_documents: list[dict[str, object]],
    attachments: dict[str, list[dict[str, str]]],
    conversation: list[dict[str, object]] | None = None,
    expert_mode: bool = False,
) -> dict[str, object]:
    """챗봇 질의를 단계별 상태로 처리해 후속 질문 흐름을 확장 가능하게 만든다."""

    def prepare(state: dict[str, object]) -> dict[str, object]:
        # 향후 사실 추출·계산 검증·담당자 확인 단계를 이 지점에 연결한다.
        return {**state, "workflow_stage": "prepared"}

    def generate(state: dict[str, object]) -> dict[str, object]:
        # 기존 Evidence Pack과 AI 답변 로직은 유지하고 그래프가 실행 순서만 관리한다.
        if state["expert_mode"]:
            answer = expert_review_chat_answer(
                state["question"], state["internal_context"], state["evidence_documents"],
                state["attachments"], state["conversation"],
            )
        else:
            answer = answer_natural_language_question(
                state["question"], state["internal_context"], state["evidence_documents"],
                state["conversation"], state["attachments"],
            )
        return {**state, "answer": answer, "workflow_stage": "generated"}

    def validate(state: dict[str, object]) -> dict[str, object]:
        # 기존 화면 계약인 문자열 배열을 유지하면서 중복·과다 질문을 정리한다.
        answer = dict(state.get("answer", {}))
        questions = answer.get("follow_up_questions", [])
        if not isinstance(questions, list):
            questions = []
        answer["follow_up_questions"] = list(dict.fromkeys(
            str(item).strip() for item in questions if str(item).strip()
        ))[:3]
        answer["workflow_stage"] = "follow_up_required" if answer["follow_up_questions"] else "answered"
        return {**state, "answer": answer, "workflow_stage": "validated"}

    graph = StateGraph(dict)
    graph.add_node("prepare", prepare)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", END)
    result = graph.compile().invoke({
        "question": question,
        "internal_context": internal_context,
        "evidence_documents": evidence_documents,
        "attachments": attachments,
        "conversation": conversation or [],
        "expert_mode": expert_mode,
    })
    return result["answer"]


def grounded_evidence_fallback(question: str, evidence_documents: list[dict[str, object]]) -> dict[str, object]:
    """AI 연결이 일시적으로 실패해도 검색된 공식 근거를 버리지 않고 즉시 안내한다."""
    if not evidence_documents:
        return {
            "key_answer": "승인된 지식기반에서 질문과 직접 연결되는 근거를 찾지 못했습니다.",
            "answer": "검색어를 법령명·조문·세목 또는 회계기준 문단 중심으로 구체화해 다시 질문해 주세요.",
            "evidence_ids": [],
            "limitations": [],
            "follow_up_questions": [],
            "highlight_terms": [],
            "generation_mode": "evidence_fallback",
        }
    primary = evidence_documents[0]
    title = str(primary.get("title") or "근거 문서")
    article = str(primary.get("article") or "")
    excerpt = re.sub(r"\s+", " ", str(primary.get("excerpt") or "")).strip()
    citation = f"{title} {article}".strip()
    period = re.search(r"(\d{1,2}월\s*\d{1,2}일부터\s*\d{1,2}월\s*\d{1,2}일까지)", excerpt)
    is_business_resident_tax = all(term in question for term in ("주민세", "사업소분"))
    is_national_strategy_credit = all(term in question for term in ("국가전략기술", "통합투자세액공제"))
    related_citations = list(dict.fromkeys(
        f"{item.get('title')} {item.get('article')}".strip()
        for item in evidence_documents[1:]
        if item.get("relation_info") and item.get("article")
    ))[:4]
    if is_national_strategy_credit and "100분의 25" in excerpt:
        key_answer = "국가전략기술 시설 투자 기본공제율은 중소기업 25%, 중소기업 졸업 유예기업 20%, 그 밖의 기업 15%입니다."
        answer = f"[적용 기준]\n{citation} 제24조 제1항 제2호 가목 2)에 따른 2029년 12월 31일까지의 국가전략기술사업화시설·연구개발시설 투자 기준입니다.\n[검토 의견]\n반도체 분야 국가전략기술 시설은 같은 조 제3호에 따라 중소기업 30%, 졸업 유예기업 25%, 그 밖의 기업 20%가 적용됩니다."
        highlights = ["중소기업 25%", "중소기업 졸업 유예기업 20%", "그 밖의 기업 15%", citation]
    elif is_business_resident_tax and period:
        key_answer = f"주민세 사업소분은 매년 {period.group(1)} 신고·납부합니다."
        answer = f"[적용 기준]\n{citation}에 따르면 사업소분의 징수는 신고·납부 방식입니다.\n[검토 의견]\n납세의무자는 위 기간에 납세지를 관할하는 지방자치단체의 장에게 신고하고 납부해야 합니다."
        highlights = ["주민세 사업소분", period.group(1), citation]
    else:
        key_answer = f"AI 연결이 일시적으로 실패했지만, {citation}에서 질문과 직접 관련된 근거를 찾았습니다."
        answer = f"[적용 기준]\n{excerpt[:900]}"
        highlights = [citation]
    if related_citations:
        answer += "\n\n함께 확인할 연결 조문: " + ", ".join(related_citations) + "."
        highlights.extend(related_citations[:2])
    return {
        "key_answer": key_answer,
        "answer": answer,
        "evidence_ids": [str(item["document_id"]) for item in evidence_documents[:3]],
        "limitations": [],
        "follow_up_questions": [
            "납부서를 받은 경우에도 신고한 것으로 보나요?",
            "사업소분의 납세의무자는 누구인가요?",
            "신고·납부하지 않으면 어떤 가산세가 적용되나요?",
        ] if is_business_resident_tax else [],
        "highlight_terms": highlights,
        "generation_mode": "evidence_fallback",
    }


@app.get("/knowledge-source/{document_id}", include_in_schema=False)
def accounting_standard_source(document_id: str) -> FileResponse:
    """검색 근거로 사용한 회계기준 원문 PDF만 안전하게 열어 준다."""
    with sqlite3.connect(f"file:{DEFAULT_DB_PATH.resolve()}?mode=ro", uri=True, timeout=2) as connection:
        row = connection.execute(
            "SELECT document_type, local_path FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    if row is None or row[0] != "accounting_standard" or not row[1]:
        raise HTTPException(status_code=404, detail="회계기준 원문을 찾지 못했습니다.")
    source_path = Path(str(row[1])).resolve()
    allowed_directory = (PROJECT_ROOT / "ifrs").resolve()
    try:
        source_path.relative_to(allowed_directory)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="허용되지 않은 회계기준 원문 경로입니다.") from error
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="회계기준 원문 파일을 찾지 못했습니다.")
    return FileResponse(source_path, media_type="application/pdf", content_disposition_type="inline")


@app.post("/knowledge-chat")
def knowledge_chat(payload: NaturalLanguageQueryRequest) -> dict[str, object]:
    """자연어 질문에 대해 읽기 전용 내부 데이터와 승인 근거를 결합해 답변한다."""
    try:
        calculated_answer = calculation_answer_from_question(payload.question)
        if calculated_answer is not None:
            calculation_evidence = calculated_answer.pop("evidence_documents", [])
            response = {
                "answer": calculated_answer,
                "internal_context": {"scope": "계산형 질의 — 내부 거래 데이터 미조회", "analysis_runs": [], "risk_findings": [], "unavailable_data": []},
                "queries": [payload.question],
                "evidence_track": "세무",
                "evidence_documents": calculation_evidence,
                "transaction_hint": transaction_hint_from_question(payload.question),
            }
            record_chat_event(payload.question, calculated_answer, calculation_evidence)
            return response
        evidence = search_local_evidence({"전표적요": payload.question}, [payload.question], payload.evidence_limit)
        attachments = prepare_attachments([item.model_dump() for item in payload.attachments])
        try:
            internal_context = load_read_only_chat_context(payload.data_limit)
        except Exception:
            internal_context = {"scope": "PostgreSQL 미설정", "analysis_runs": [], "risk_findings": [], "unavailable_data": ["내부 거래·Risk Score·검토 이력·조치 현황"]}
        try:
            # 판단형 질의는 공통 Evidence Pack을 사용해 전문가 검토기를 호출하고,
            # 단순 조문·기한 조회는 짧고 직접적인 근거 답변으로 유지한다.
            conversation = [turn.model_dump() for turn in payload.conversation]
            if requires_expert_review(payload.question, transaction_hint_from_question(payload.question), attachments):
                answer = run_chat_review_graph(
                    payload.question, internal_context, evidence["evidence_documents"], attachments, conversation,
                    expert_mode=True,
                )
            else:
                answer = run_chat_review_graph(
                    payload.question,
                    internal_context,
                    evidence["evidence_documents"],
                    attachments,
                    conversation,
                )
        except AiReviewError:
            answer = grounded_evidence_fallback(payload.question, evidence["evidence_documents"])
        response = {"answer": answer, "internal_context": internal_context, "transaction_hint": transaction_hint_from_question(payload.question), **evidence}
        record_chat_event(payload.question, answer, evidence["evidence_documents"])
        return response
    except EvidenceSearchError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/risk-score/preview")
def risk_score_preview(payload: RiskScoreRequest) -> dict[str, object]:
    """실제 원장을 저장하지 않고 선택 월의 검토 후보와 점수를 계산한다."""
    try:
        records = parse_ledger_csv(payload.ledger_csv_text)
        related_parties = parse_related_parties(payload.related_party_csv_text)
        findings = score_records(records, related_parties, payload.analysis_year_month)
        risk_summary = {
            level: {
                "count": sum(1 for item in findings if item["risk_level"] == level),
                "amount": sum(item["amount"] for item in findings if item["risk_level"] == level),
            }
            for level in ("High", "Medium", "Low")
        }
        return {
            "analysis_year_month": payload.analysis_year_month,
            "ledger_record_count": len(records),
            "finding_count": len(findings),
            "risk_summary": risk_summary,
            "findings": findings,
        }
    except (ValueError, ArithmeticError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/risk-score/analyze-and-save")
def risk_score_analyze_and_save(payload: RiskScoreRequest) -> dict[str, object]:
    """선택 월 원장을 누적 저장하고 과거 활성 원장과 비교해 위험후보를 산정한다."""
    try:
        uploaded_records = parse_ledger_csv(payload.ledger_csv_text)
        year, month = map(int, payload.analysis_year_month.split("-"))
        current_records = [record for record in uploaded_records if (record["posting_date"].year, record["posting_date"].month) == (year, month)]
        if not current_records:
            raise ValueError("분석 대상 월에 해당하는 원장 행이 없습니다.")
        company_code = current_records[0]["company_code"] or "미지정"
        initialize_monthly_storage()
        history_records = load_active_history(company_code, payload.analysis_year_month)
        related_parties = parse_related_parties(payload.related_party_csv_text)
        findings = score_records([*history_records, *current_records], related_parties, payload.analysis_year_month)
        saved_run = save_monthly_analysis(current_records, findings, payload.analysis_year_month)
        comparison = compare_with_previous_month(company_code, payload.analysis_year_month, findings)
        risk_summary = {
            level: {"count": sum(1 for item in findings if item["risk_level"] == level), "amount": sum(item["amount"] for item in findings if item["risk_level"] == level)}
            for level in ("High", "Medium", "Low")
        }
        return {"analysis_year_month": payload.analysis_year_month, "ledger_record_count": len(current_records), "finding_count": len(findings), "risk_summary": risk_summary, "findings": findings, "storage": saved_run, "comparison": comparison}
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (ValueError, ArithmeticError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def expected_transaction_facts(payload: ExpectedTransactionRequest) -> dict[str, object]:
    """입력한 예정 거래를 기존 근거 검색·AI 검토에 맞는 사실 묶음으로 바꾼다."""
    return {
        "거래구분": "예상 거래",
        "법인명": payload.company_name,
        "예정거래일": payload.expected_date,
        "계정과목코드": payload.account_code,
        "계정과목명": payload.account_name,
        "거래처명": payload.counterparty_name,
        "특수관계자여부": "예" if payload.related_party else "아니오",
        "차대변구분자": payload.debit_credit,
        "검토 대상 거래금액": payload.amount,
        "거래설명": payload.description,
        "추가 사실관계": payload.facts,
    }


def expected_risk_assessment(payload: ExpectedTransactionRequest) -> dict[str, object]:
    """이력 없는 예상 거래에는 자동 판단만 제안하고 점수는 임의로 만들지 않는다."""
    return {
        "status": "산정 보류",
        "risk_score": None,
        "message": "과거 거래 이력 또는 담당자 승인 점수 규칙이 없어 Risk Score를 계산하지 않았습니다. AI가 별도의 검토 관점과 확인사항을 제안합니다.",
    }


@app.post("/expected-transaction/evidence-preview")
def expected_transaction_evidence_preview(payload: ExpectedTransactionRequest) -> dict[str, object]:
    """예상 거래의 사실관계로 로컬 지식기반 근거 후보를 검색한다."""
    try:
        transaction = expected_transaction_facts(payload)
        evidence_result = search_local_evidence(transaction, payload.issue_keywords, payload.evidence_limit)
        return {"transaction": transaction, "risk_assessment": expected_risk_assessment(payload), **evidence_result}
    except EvidenceSearchError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/expected-transaction/diagnose")
def expected_transaction_diagnose(payload: ExpectedTransactionRequest) -> dict[str, object]:
    """자동 검색한 근거만 사용해 예상 거래의 AI 잠정 검토를 생성한다."""
    try:
        transaction = expected_transaction_facts(payload)
        evidence_result = search_local_evidence(transaction, payload.issue_keywords, payload.evidence_limit)
        attachments = prepare_attachments([item.model_dump() for item in payload.attachments])
        review_result = review_with_openai(transaction, evidence_result["evidence_documents"], attachments)
        return {
            "transaction": transaction,
            "risk_assessment": expected_risk_assessment(payload),
            **evidence_result,
            **review_result,
        }
    except (EvidenceSearchError, AiReviewError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

