# AI 회계·세무 리스크 사전검증 PoC

현재 구현 범위는 SAP 거래 분석 전 단계인 외부 기준 데이터 지식 기반입니다.

- 국가법령정보 Open API에서 법인세법·부가가치세법·조세특례제한법과 각 시행령·시행규칙을 수동 갱신합니다.
- 관련 판례를 API로 검색해 수집합니다.
- `ifrs/` 폴더의 현재 시행 K-IFRS·일반기업회계기준 PDF 전체를 기준체계 태그와 함께 색인합니다.
- 읽기 전용 MCP 서버가 색인된 기준 데이터를 검색하고 원문 근거를 반환합니다.

## 준비

프로젝트별 가상환경을 만들고 필요한 라이브러리를 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

`.env.example`을 복사해 `.env`를 만들고, API 키는 `.env`에만 저장합니다. 실제 키는 채팅, 코드, `prd.md`에 적지 않습니다.

```powershell
Copy-Item .env.example .env
# .env 파일에 LAW_API_OC=발급받은_국가법령정보_API_인증값 형태로 입력
```

## 실행

아래 명령은 사용자가 원할 때만 실행합니다. 갱신은 법령·시행령·시행규칙·판례를 함께 처리합니다.

```powershell
.\.venv\Scripts\python.exe app.py refresh-law
.\.venv\Scripts\python.exe app.py index-ifrs
.\.venv\Scripts\python.exe app.py search "특수관계자 거래"
.\.venv\Scripts\python.exe app.py mcp
```

기본 데이터베이스 위치는 `data/knowledge.db`이며 Git에서 제외됩니다. `.venv/`, `.env`, `.env.*`, `data/`는 Git에서 제외됩니다. `refresh-law`은 `LAW_API_OC`가 없으면 실행을 중단합니다.

## 화면·백엔드 실행

Streamlit 화면과 FastAPI 백엔드는 분리돼 있으며, 화면은 HTTP API만 호출합니다.

```powershell
# 터미널 1: FastAPI 백엔드
.\.venv\Scripts\uvicorn.exe backend.main:app --reload

# 터미널 2: Streamlit 화면
.\.venv\Scripts\streamlit.exe run frontend/streamlit_app.py
```

PostgreSQL은 `.env`의 `POSTGRES_HOST`, `POSTGRES_DATABASE`, `POSTGRES_USER`, `POSTGRES_PASSWORD`를 모두 입력한 경우에만 연결을 시도합니다. 현재 단계에서는 DB를 생성하거나 SAP 거래 데이터를 저장하지 않습니다.

## OpenAI 근거 기반 검토

AI 검토 API는 지정 모델 `gpt-5.6-terra`를 사용합니다. `.env`에 `OPENAI_API_KEY`를 직접 입력한 경우에만 실제 호출을 수행합니다. 키는 코드·화면·로그·보고서에 표시하지 않습니다.

`POST /ai-review`는 거래 사실과 승인된 근거 문서를 받아 잠정 검토 결과를 반환합니다. AI 응답은 근거 문서 ID를 포함해야 하며, 제공되지 않은 문서 ID를 인용한 경우 별도로 표시합니다.

테스트 등에서 다른 데이터베이스를 MCP 서버에 연결하려면 현재 세션에서만 `KNOWLEDGE_DB_PATH`를 설정합니다.

```powershell
$env:KNOWLEDGE_DB_PATH = "data/knowledge.db"
```

## MCP 도구

`app.py mcp`는 표준 입력/출력 기반의 읽기 전용 MCP 서버입니다.

- `search_knowledge`: 키워드로 법령·판례·K-IFRS를 검색합니다.
- `get_document`: 검색 결과의 문서 ID로 원문과 메타데이터를 조회합니다.

검색 결과에는 원천, 문서 유형, 기준체계, 출처 URL, 시행일, 수집일, 버전이 포함됩니다. K-IFRS와 일반기업회계기준에 속하지 않는 PDF는 현재 PoC 검색 범위에서 제외됩니다. API 키는 MCP 응답에 포함되지 않습니다.

## 제한

- 예규·해석사례의 공식 수집 원천은 아직 확정되지 않아 자동 수집하지 않습니다.
- 법적 적용 여부는 자동으로 판단하지 않으며, 담당자 검토를 전제로 합니다.
