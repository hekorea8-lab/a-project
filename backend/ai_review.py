"""OpenAI를 이용한 근거 기반 회계·세무 잠정 검토 로직이다."""

import base64
import io
import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pypdf import PdfReader


load_dotenv()
MODEL_NAME = "gpt-5.6-terra"


class AiReviewError(RuntimeError):
    """AI 검토 설정·응답 형식·호출 오류다."""


MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024


def prepare_attachments(attachments: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """첨부 PDF의 텍스트와 이미지를 이번 AI 검토 요청용으로만 준비한다."""
    text_documents: list[dict[str, str]] = []
    file_documents: list[dict[str, str]] = []
    image_documents: list[dict[str, str]] = []
    for attachment in attachments:
        filename = attachment["filename"]
        content_type = attachment["content_type"]
        try:
            raw = base64.b64decode(attachment["content_base64"], validate=True)
        except ValueError as error:
            raise AiReviewError(f"첨부 파일 '{filename}'의 형식이 올바르지 않습니다.") from error
        if len(raw) > MAX_ATTACHMENT_BYTES:
            raise AiReviewError(f"첨부 파일 '{filename}'은 10MB 이하만 지원합니다.")
        if content_type == "application/pdf":
            try:
                reader = PdfReader(io.BytesIO(raw))
                text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            except Exception as error:
                raise AiReviewError(f"PDF '{filename}'의 텍스트를 읽지 못했습니다. 암호화 여부와 파일 상태를 확인하세요.") from error
            text_documents.append({"filename": filename, "text": text[:30000] or "텍스트를 추출하지 못했습니다."})
            file_documents.append({"filename": filename, "content_base64": attachment["content_base64"]})
        elif content_type in {"image/jpeg", "image/png"}:
            image_documents.append({"filename": filename, "data_url": f"data:{content_type};base64,{attachment['content_base64']}"})
        else:
            raise AiReviewError(f"첨부 파일 '{filename}'은 PDF, PNG, JPG 형식만 지원합니다.")
    return {"text_documents": text_documents, "file_documents": file_documents, "image_documents": image_documents}


def build_evidence_packet(evidence_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """AI에 전달할 근거를 문서 ID와 출처 메타데이터 중심으로 최소화한다."""
    return [
        {
            "document_id": document["document_id"],
            "title": document["title"],
            "source": document["source"],
            "source_url": document.get("source_url"),
            "effective_date_or_version": document.get("effective_date_or_version"),
            "excerpt": document["excerpt"],
        }
        for document in evidence_documents
    ]


def build_review_instructions(transaction: dict[str, Any], evidence_documents: list[dict[str, Any]], attachments: dict[str, list[dict[str, str]]]) -> str:
    """고연차 검토 관점을 근거·불확실성·반대 논리 중심의 출력 규칙으로 고정한다."""
    payload = {
        "transaction": transaction,
        "approved_evidence_documents": build_evidence_packet(evidence_documents),
        "user_attached_document_text": attachments["text_documents"],
    }
    return f"""
역할: 당신은 결산·감사·세무조사 대응 경험을 전제로 사고하는 회계·세무 검토 보조 AI입니다.
전문 자격 보유자라고 주장하지 않으며, 최종 회계·세무 판단이나 법적 결론을 확정하지 마세요.

목표: 입력된 거래와 승인된 근거 문서에 근거해 잠재 쟁점, 적용 논리, 반대 논리, 필요한 증빙과 담당자 조치를 구조화하세요.

근거 원칙:
- 제공된 거래 사실, 승인된 근거 문서, 사용자 첨부자료만 사용하세요.
- 제공되지 않은 법령·회계기준·판례·예규를 사실처럼 인용하거나 문서 ID를 만들어내지 마세요.
- 각 핵심 주장에는 제공된 document_id만 evidence_ids로 연결하세요. 근거가 없으면 빈 배열로 두고 uncertainty에 부족한 이유를 적으세요.
- 첨부자료는 사용자 제공 자료이므로 승인된 법령·기준 근거와 구분하고, 그 내용을 언급할 때는 파일명을 밝히세요.
- 근거가 충돌하거나 적용 요건이 불명확하면 우선순위를 단정하지 말고 충돌 내용과 확인 필요사항을 적으세요.

검토 원칙:
- 확인된 사실과 미확인 사항을 구분하세요. 탐지사유와 Risk Score는 내부 검토 우선순위이며 법령 위반 또는 회계오류의 확정 근거가 아닙니다.
- 검토 대상 거래금액과 회계·세무 영향 추정액을 구분하세요. 산식·입력값·적용 근거가 모두 없으면 영향 추정액을 만들지 마세요.
- 특수관계자 거래는 거래 목적, 정상가격·비교가능성, 계약 조건, 대가 산정근거, 실제 이행 여부를 우선 확인하세요.
- 증빙이 부족해도 분석을 중단하지 말고 잠정 의견과 최소 추가 증빙을 제시하세요.
- 결론을 바꿀 수 있는 반대 논리 또는 예외 요건을 하나 이상 검토하세요.
- `담당자 추가 확인 답변`이 입력된 경우, 이는 담당자가 새로 제공한 사실관계입니다. 답변 내용과 승인된 근거 문서를 구분하고, 새 답변이 잠정 결론을 어떻게 좁혔는지 설명하세요.

결론 기준:
- 적정 가능성: 현재 확보된 사실과 근거를 기준으로 적정 처리의 가능성이 더 높습니다.
- 비적정 가능성: 현재 확보된 사실과 근거를 기준으로 비적정 처리 또는 조정 필요의 가능성이 더 높습니다.
- 증빙·사실관계가 부족하거나 근거가 충돌해도 `추가 검토 필요`를 결론값으로 반환하지 마세요. 현재 자료에서 더 가능성 높은 방향을 선택하고, 신뢰 수준·제한사항·결론을 바꿀 조건을 함께 적으세요.
- Risk Score만으로 비적정 가능성을 선택하지 마세요.

추가 확인 질문:
- 현재 결론을 바꿀 가능성이 있는 확인 사항이 있으면, 결론과 무관하게 최대 5개의 질문을 제시하세요. 없으면 빈 배열로 반환하세요.
- 질문은 담당자가 사실 또는 증빙으로 답할 수 있게 구체적으로 작성하세요.
- 단순히 "추가 자료를 제출하세요"라고 쓰지 말고, 무엇을 왜 확인해야 하는지와 답변에 따라 달라지는 결론을 적으세요.

보고서 문안:
- `report_draft`에는 담당자가 바로 검토 보고서에 옮길 수 있는 한국어 보고서 문안을 작성하세요.
- `1. 주요 메시지`는 항상 첫 번째 최상위 문단으로 작성하세요. 그 뒤 최상위 문단의 제목·개수·순서는 해당 거래의 회계·세무 쟁점에 맞춰 정하고, 판단에 필요하지 않은 고정 목차는 만들지 마세요.
- 예를 들어 회계 인식·측정이 핵심이면 `회계 처리 검토`, 세무가 핵심이면 `세무상 쟁점`, 특수관계자 거래면 `거래 실질 및 대가 적정성`, 증빙·통제가 핵심이면 `증빙 및 통제 검토`, 실제 조치가 필요하면 `조치 제안`처럼 쟁점에 맞는 제목을 사용하세요. 이 예시를 기계적으로 모두 포함하지 마세요.
- 문단 계층은 최상위 `1. 제목`, 그 아래 쟁점별 핵심 판단 `○`, 세부 사실·근거·조치 `-` 순서로 작성하세요. 더 깊은 구분이 꼭 필요한 경우에만 `가.` 또는 `①`을 사용하세요.
- 비어 있거나 일반론적인 항목, 같은 내용을 반복하는 항목은 생략하고, 확인되지 않은 사항은 단정하지 마세요.
- 금액은 천 단위 구분 쉼표를 사용하고, 확인되지 않은 사항은 단정하지 마세요.
- 마크다운 표·코드블록·별표 목록 없이, 줄바꿈을 포함한 일반 텍스트 문단으로만 작성하세요.

전문가 검토 의견:
- `expert_opinion`에는 보고서 하단에 표시할 3~5문장 분량의 한국어 검토 의견을 작성하세요.
- 20년 이상 실무를 수행한 회계·세무 전문가의 검토 메모처럼, 확정된 핵심 사실과 적용 기준, 현재 더 가능성 높은 판단 및 판단의 한계를 연결해 설명하세요.
- 과장된 단정이나 법률 자문 확정 표현은 피하고, 근거가 부족한 부분은 어떤 사실·증빙이 결론을 바꿀 수 있는지 구체적으로 밝히세요.

반드시 아래 JSON 객체만 반환하세요.
각 핵심 주장에는 evidence_ids 배열을 넣고, 배열 값은 제공된 document_id 중에서만 선택하세요.
근거가 없으면 evidence_ids를 빈 배열로 하고 uncertainty에 이유를 적으세요.

{{
  "confirmed_facts": [{{"statement": "", "evidence_ids": []}}],
  "applicable_standards": [{{"issue_type": "", "statement": "", "evidence_ids": []}}],
  "reasoning": [{{"statement": "", "evidence_ids": []}}],
  "counterarguments": [{{"statement": "", "evidence_ids": []}}],
  "attached_document_findings": [{{"filename": "", "statement": ""}}],
  "suggested_review_focus": [""],
  "provisional_conclusion": {{"status": "적정 가능성|비적정 가능성", "confidence_level": "높음|보통|낮음", "statement": "", "evidence_ids": []}},
  "required_evidence": [""],
  "reviewer_actions": [""],
  "uncertainty": [""],
  "follow_up_questions": [{{"question_id": "FQ1", "question": "", "why_needed": "", "conclusion_impact": "", "priority": "높음|보통"}}],
  "refinement": {{"previous_status": "", "changed": false, "statement": "", "remaining_questions": [""]}},
  "report_draft": "",
  "expert_opinion": ""
}}

검토 입력:
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def parse_review_response(response_text: str, allowed_document_ids: set[str]) -> dict[str, Any]:
    """AI 응답을 JSON으로 읽고 허용되지 않은 근거 문서 ID를 별도로 표시한다."""
    cleaned = response_text.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[7:-3].strip()
    try:
        review = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise AiReviewError("AI 응답이 지정된 JSON 형식이 아닙니다.") from error

    invalid_ids: set[str] = set()

    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            evidence_ids = value.get("evidence_ids")
            if isinstance(evidence_ids, list):
                invalid_ids.update(str(item) for item in evidence_ids if str(item) not in allowed_document_ids)
            for nested in value.values():
                inspect(nested)
        elif isinstance(value, list):
            for nested in value:
                inspect(nested)

    inspect(review)
    return {"review": review, "invalid_evidence_ids": sorted(invalid_ids)}


def build_review_message(instructions: str, attachments: dict[str, list[dict[str, str]]]) -> HumanMessage:
    """첨부 자료를 보존한 LangChain 메시지를 만들어 Responses API 형식으로 전달한다."""
    content: list[dict[str, Any]] = [{"type": "text", "text": instructions}]
    for document in attachments["file_documents"]:
        content.append({"type": "file", "file": {"filename": document["filename"], "file_data": document["content_base64"]}})
    for image in attachments["image_documents"]:
        content.append({"type": "text", "text": f"사용자 첨부 이미지 파일명: {image['filename']}"})
        content.append({"type": "image_url", "image_url": {"url": image["data_url"], "detail": "auto"}})
    return HumanMessage(content=content)


def response_text_from_chain(response: Any) -> str:
    """LangChain 모델 응답에서 JSON 원문만 추출한다."""
    if isinstance(response.content, str):
        return response.content
    if isinstance(response.content, list):
        return "".join(item.get("text", "") for item in response.content if isinstance(item, dict))
    raise AiReviewError("LangChain AI 검토 응답을 텍스트로 읽지 못했습니다.")


def build_review_chain(api_key: str):
    """근거 입력을 메시지로 변환하고 AI 응답 텍스트만 반환하는 제한된 LangChain 체인이다."""
    model = ChatOpenAI(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0,
        timeout=120,
        max_retries=1,
        store=False,
        use_responses_api=True,
    )
    return RunnableLambda(lambda payload: [build_review_message(payload["instructions"], payload["attachments"])]) | model | RunnableLambda(response_text_from_chain)


def review_with_openai(transaction: dict[str, Any], evidence_documents: list[dict[str, Any]], attachments: dict[str, list[dict[str, str]]] | None = None) -> dict[str, Any]:
    """LangChain 체인으로 근거 기반 잠정 검토를 요청한다."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise AiReviewError("OPENAI_API_KEY가 비어 있습니다. .env에 직접 입력한 후 다시 실행하세요.")
    prepared_attachments = attachments or {"text_documents": [], "file_documents": [], "image_documents": []}
    instructions = build_review_instructions(transaction, evidence_documents, prepared_attachments)
    try:
        response_text = build_review_chain(api_key).invoke({"instructions": instructions, "attachments": prepared_attachments})
    except Exception as error:
        raise AiReviewError("LangChain AI 검토 요청에 실패했습니다.") from error
    allowed_document_ids = {document["document_id"] for document in evidence_documents}
    return parse_review_response(response_text, allowed_document_ids)
