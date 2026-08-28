import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .config import LAW_SEARCH_URL, LAW_SERVICE_URL, PRECEDENT_QUERIES, TAX_LAW_NAMES
from .store import upsert_document


class LawApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class LawApiClient:
    oc: str
    timeout_seconds: int = 30

    @classmethod
    def from_environment(cls) -> "LawApiClient":
        oc = os.environ.get("LAW_API_OC")
        if not oc:
            raise LawApiError("LAW_API_OC is required. Set it in the current shell before refreshing.")
        return cls(oc=oc)

    def request(self, endpoint: str, **parameters: str | int) -> ET.Element:
        query = urllib.parse.urlencode({"OC": self.oc, **parameters})
        request = urllib.request.Request(f"{endpoint}?{query}", headers={"User-Agent": "tax-risk-poc/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except OSError as error:
            raise LawApiError("국가법령정보 Open API request failed.") from error
        try:
            return ET.fromstring(body)
        except ET.ParseError as error:
            raise LawApiError("국가법령정보 Open API returned non-XML content.") from error

    def law_by_name(self, name: str) -> ET.Element:
        return self.request(LAW_SERVICE_URL, target="law", type="XML", LM=name)

    def precedent_page(self, query: str, page: int) -> ET.Element:
        return self.request(
            LAW_SEARCH_URL,
            target="prec",
            type="XML",
            search=2,
            query=query,
            display=100,
            page=page,
        )

    def precedent_by_id(self, precedent_id: str) -> ET.Element:
        return self.request(LAW_SERVICE_URL, target="prec", type="XML", ID=precedent_id)


def text_of(element: ET.Element, *names: str) -> str | None:
    for name in names:
        found = element.find(f".//{name}")
        if found is not None and found.text:
            return " ".join(found.itertext()).strip()
    return None


def flatten_xml(element: ET.Element) -> str:
    return "\n".join(part.strip() for part in element.itertext() if part and part.strip())


def collect_tax_laws(connection, client: LawApiClient) -> int:
    count = 0
    for name in TAX_LAW_NAMES:
        root = client.law_by_name(name)
        title = text_of(root, "법령명한글", "법령명") or name
        law_id = text_of(root, "법령ID", "법령일련번호") or title
        effective_date = text_of(root, "시행일자")
        proclamation_number = text_of(root, "공포번호")
        source_url = f"https://www.law.go.kr/법령/{urllib.parse.quote(title)}"
        upsert_document(
            connection,
            {
                "document_id": f"law:{law_id}",
                "source": "국가법령정보센터",
                "document_type": "law",
                "title": title,
                "content": flatten_xml(root),
                "source_url": source_url,
                "effective_date": effective_date,
                "version": proclamation_number,
                "local_path": None,
            },
        )
        count += 1
    return count


def precedent_ids(root: ET.Element) -> list[str]:
    ids: list[str] = []
    for item in root.findall(".//prec"):
        value = item.attrib.get("id") or text_of(item, "판례일련번호", "판례정보일련번호")
        if value:
            ids.append(value)
    return list(dict.fromkeys(ids))


def total_count(root: ET.Element) -> int:
    raw = text_of(root, "totalCnt", "총건수")
    return int(raw) if raw and raw.isdigit() else 0


def collect_precedents(connection, client: LawApiClient, pause_seconds: float = 0.1) -> int:
    seen: set[str] = set()
    count = 0
    for query in PRECEDENT_QUERIES:
        first_page = client.precedent_page(query, 1)
        pages = max(1, (total_count(first_page) + 99) // 100)
        page_roots = [first_page]
        for page in range(2, pages + 1):
            time.sleep(pause_seconds)
            page_roots.append(client.precedent_page(query, page))
        for page_root in page_roots:
            for precedent_id in precedent_ids(page_root):
                if precedent_id in seen:
                    continue
                seen.add(precedent_id)
                time.sleep(pause_seconds)
                root = client.precedent_by_id(precedent_id)
                title = text_of(root, "사건명") or f"판례 {precedent_id}"
                sentence_date = text_of(root, "선고일자")
                source_url = f"https://www.law.go.kr/LSW/precInfoP.do?precSeq={urllib.parse.quote(precedent_id)}"
                upsert_document(
                    connection,
                    {
                        "document_id": f"precedent:{precedent_id}",
                        "source": "국가법령정보센터",
                        "document_type": "precedent",
                        "title": title,
                        "content": flatten_xml(root),
                        "source_url": source_url,
                        "effective_date": sentence_date,
                        "version": sentence_date,
                        "local_path": None,
                    },
                )
                count += 1
    return count

