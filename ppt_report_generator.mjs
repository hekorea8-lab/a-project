import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const inputPath = process.argv[2];
const outputPath = process.argv[3];
if (!inputPath || !outputPath) throw new Error("입력·출력 경로가 필요합니다.");
const data = JSON.parse(await fs.readFile(inputPath, "utf8"));
const { SKILL_DIR } = process.env;
const { resolvePresentationFont } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);
const font = resolvePresentationFont({ fontFamily: "Malgun Gothic" });
const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const blue = "#0068B7", dark = "#17324D", pale = "#EAF3FA", line = "#C9D9E5";
const clean = (value, max = 700) => String(value || "").replace(/\r/g, "").trim().slice(0, max);
const bullets = (text) => clean(text, 1400).split(/\n+/).map(x => x.trim()).filter(Boolean).slice(0, 8);
const titleBox = (slide, title, page) => {
  const bar = slide.shapes.add({ geometry: "rect", position: { left: 0, top: 0, width: 1280, height: 62 }, fill: blue, line: { fill: blue, width: 0 } });
  const t = slide.shapes.add({ geometry: "textbox", position: { left: 34, top: 14, width: 820, height: 36 }, fill: "none", line: { fill: "none", width: 0 } });
  t.text = title; t.text.style = { typeface: font, fontSize: 24, bold: true, color: "#FFFFFF" };
  const m = slide.shapes.add({ geometry: "textbox", position: { left: 900, top: 12, width: 340, height: 40 }, fill: "none", line: { fill: "none", width: 0 } });
  m.text = `${data.date || "'26. 9. 7."}\n${data.organization || "회계세무그룹"}`; m.text.style = { typeface: font, fontSize: 13, color: "#FFFFFF", align: "right" };
  const pg = slide.shapes.add({ geometry: "textbox", position: { left: 1188, top: 675, width: 60, height: 22 }, fill: "none", line: { fill: "none", width: 0 } });
  pg.text = String(page); pg.text.style = { typeface: font, fontSize: 12, color: "#607489", align: "right" };
};
const textBox = (slide, text, left, top, width, height, size = 20, color = dark, bold = false) => {
  const box = slide.shapes.add({ geometry: "textbox", position: { left, top, width, height }, fill: "none", line: { fill: "none", width: 0 } });
  box.text = text; box.text.style = { typeface: font, fontSize: size, color, bold, breakLine: true, autoFit: "shrink" }; return box;
};
const section = (slide, label, text, top) => { textBox(slide, label, 48, top, 180, 30, 17, blue, true); textBox(slide, clean(text, 900), 48, top + 32, 1160, 105, 17, dark); };

let slide = p.slides.add(); slide.background.fill = "#FFFFFF"; titleBox(slide, clean(data.title || "회계·세무 검토보고서", 100), 1);
textBox(slide, "검토보고서", 62, 138, 1100, 64, 38, dark, true);
textBox(slide, `○ 목적/배경  ${clean(data.question, 180)}\n○ 현재 이슈  근거 문서와 사실관계의 연결 필요\n○ 결론       확인된 근거 범위에서 잠정 판단 및 추가 확인사항 제시`, 64, 245, 1120, 170, 22, dark);
textBox(slide, "회계세무그룹 · 지식 챗봇 기반 검토", 64, 610, 650, 26, 16, "#607489");

slide = p.slides.add(); slide.background.fill = "#FFFFFF"; titleBox(slide, "## 1 개요", 2);
section(slide, "검토 배경", `○ 자연어 질의: ${clean(data.question, 300)}\n- 회계·세무 지식영역을 분리해 관련 기준과 법령을 검색`, 105);
section(slide, "목적 및 범위", `○ 검색된 근거를 기준으로 사실관계·쟁점·검토의견을 정리\n- 공개 회사자료는 거래 맥락과 추가 쟁점 후보로만 사용`, 295);
section(slide, "진행 경과", "○ 질의 분석 → 키워드 확장 → 근거 검색 → 답변 생성 → 보고서 산출", 485);

slide = p.slides.add(); slide.background.fill = "#FFFFFF"; titleBox(slide, "## 2 검토사항", 3);
section(slide, "현황 및 쟁점", data.answer || "검색된 근거를 확인하고 담당자 검토가 필요합니다.", 100);
const rows = [["구분", "확인 내용", "근거"], ["질의", clean(data.question, 120), "사용자 입력"], ["핵심 판단", clean(data.key_answer, 180), "AI 답변"], ["추가 확인", bullets(data.limitations).join(" / ") || "사실관계 보완 필요", "검토 메모"]];
const table = slide.tables.add({ rows: rows.length, columns: 3, left: 48, top: 355, width: 1160, height: 210, values: rows, columnWidths: [170, 720, 270] });
table.getCell(0,0).fill = blue; table.getCell(0,1).fill = blue; table.getCell(0,2).fill = blue; table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 3 }).assign({ textStyle: { color: "#FFFFFF", bold: true, fontSize: 16, typeface: font } }); table.borders.assign({ style: "solid", fill: line, width: 1 });
for (let r = 1; r < rows.length; r++) table.cells.block({ row: r, column: 0, rowCount: 1, columnCount: 3 }).assign({ textStyle: { color: dark, fontSize: 14, typeface: font } });
textBox(slide, "※ 표의 근거는 검색된 문서와 사용자 입력을 구분하여 표시했습니다.", 48, 585, 1000, 24, 12, "#607489");

slide = p.slides.add(); slide.background.fill = "#FFFFFF"; titleBox(slide, "## 3 검토결과", 4);
section(slide, "결론", `○ ${clean(data.key_answer || "현재 근거 범위에서 잠정 판단을 정리합니다.", 360)}\n- 최종 회계·세무 처리는 원문과 거래 증빙을 확인한 후 판단 필요`, 105);
section(slide, "의사결정 요청사항", "○ 사실관계 보완자료 제출 여부\n○ 관련 부서 검토 및 증빙 확보 추진\n○ 적용 기준일과 회사 거래 해당성 확인 고려", 300);
section(slide, "향후계획", "○ (즉시) 계약서·세금계산서·원가명세 등 사실관계 확인\n○ (검토 후) 적용 기준과 세무 쟁점 재검색\n○ (확정 시) 회계처리 및 세무신고 대응방안 추진", 500);
const evidence = (data.evidence || []).map((x, i) => `[${i + 1}] ${clean(x.title, 100)}${x.article ? ` ${x.article}` : ""}`); if (evidence.length) textBox(slide, `근거 문서\n${evidence.join("\n")}`, 760, 500, 430, 130, 11, "#607489");

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await (await PresentationFile.exportPptx(p)).save(outputPath);
