import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
// 원문을 보존하며 분량에 따라 페이지를 늘린다. 새로운 판단이나 수치를 만들지 않는다.
const [input, output] = process.argv.slice(2);
const d=JSON.parse(await fs.readFile(input,'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:900}});
const blue='#16769A', navy='#144F79', ink='#202E39';
const clean=v=>String(v??'').replace(/\r/g,'').replace(/\*\*/g,'').replace(/\[([^\]]+)\]\(([^)]+)\)/g,'$1 ($2)').trim();
const list=v=>(Array.isArray(v)?v:[v]).map(clean).filter(Boolean);
function box(s,t,x,y,w,h,size=24,color=ink,bold=false){const b=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});b.text=t;b.text.style={typeface:'Malgun Gothic',fontSize:size,color,bold,autoFit:'shrink'};return b;}
function rect(s,x,y,w,h,fill){s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill,width:0}});}
const date=d.date||new Intl.DateTimeFormat('ko-KR',{timeZone:'Asia/Seoul',year:'2-digit',month:'numeric',day:'numeric'}).format(new Date());
function page(title){const s=p.slides.add();s.background.fill='#FFFFFF';rect(s,0,0,1280,70,blue);rect(s,980,0,300,70,'#DDEFF5');box(s,clean(title).replace(/^#+\s*/,''),28,15,940,44,30,'#FFFFFF',true);box(s,`${date}\n${d.organization||'회계세무그룹'}`,992,8,270,56,16,navy);rect(s,40,851,1200,1,'#CEDCE3');box(s,'포스코퓨처엠 | AI 검토의견 기반 · 검토용 초안',42,862,1050,25,13,'#61717F');box(s,String(p.slides.items.length),1188,860,50,26,14);return s;}
function heading(s,n,t,y){const nw=String(n).length>1?52:31;rect(s,42,y,nw,33,blue);box(s,n,46,y,nw-8,34,23,'#FFFFFF',true);box(s,t,42+nw+13,y-1,1110,38,27,navy,true);}
// 본문은 삭제하지 않고 줄을 나눠 다음 페이지로 전달한다.
function wrap(t,max=72){const out=[];for(const line of clean(t).split('\n')){if(!line.trim())continue;let r=line.trim();while(r.length>max){let at=r.lastIndexOf(' ',max);if(at<max/2)at=max;out.push(r.slice(0,at));r=r.slice(at).trimStart();}out.push(r);}return out;}
const first=(t,n=4)=>{const a=wrap(t,68);return a.slice(0,n).join('\n')+(a.length>n?' …':'');};
let bodySlide=null,bodyTitle='',bodyY=106;
function body(title,n,label,t){const a=wrap(t);for(let i=0;i<a.length;i+=15){const lines=a.slice(i,i+15),h=lines.length*35+90;if(bodyTitle!==title||bodyY+h>820){bodySlide=page(title);bodyTitle=title;bodyY=106;}heading(bodySlide,n,label+(i?' (계속)':''),bodyY);box(bodySlide,lines.join('\n'),58,bodyY+52,1164,lines.length*35+10);bodyY+=h;}}
const question=clean(d.question),answer=clean(d.answer),conclusion=clean(d.key_answer);
if(!answer)throw new Error('검토의견이 비어 있습니다.');
const sections=[];let current={title:'검토 의견',lines:[]};
for(const line of answer.split('\n')){const m=line.match(/^\s*(?:\[([^\]]+)\]|#{1,3}\s+(.+))\s*(.*)$/);if(m){if(current.lines.length)sections.push(current);current={title:m[1]||m[2],lines:m[3]?[m[3]]:[]};}else current.lines.push(line);}
if(current.lines.length)sections.push(current);
const limitations=list(d.limitations);
const entry=typeof d.accounting_entry==='object'&&d.accounting_entry?d.accounting_entry:{};
const entryStatus=clean(entry.status)||'추가 확인 필요';
const entryItems=value=>Array.isArray(value)?value.filter(item=>item&&typeof item==='object').slice(0,2):[];
const debit=entryItems(entry.debit),credit=entryItems(entry.credit);
const entryCell=item=>item?`${clean(item.account_name)||'계정과목 확인 필요'}${clean(item.note)?'\n'+clean(item.note):''}`:'-';
function cover(){const s=p.slides.add();s.background.fill='#FFFFFF';rect(s,0,0,1280,82,blue);rect(s,0,82,1280,7,'#B8DDE7');box(s,`${date}\n${d.organization||'회계세무그룹'}`,1010,18,230,50,16,navy);box(s,clean(d.title)||'회계·세무 검토보고서',70,185,1140,90,45,navy,true);box(s,'AI 검토의견 기반 검토보고서',72,290,1000,34,20,'#61717F');rect(s,70,414,1140,2,'#B8DDE7');box(s,`○ 목적/배경  ${first(question,2)}\n○ 현재 이슈  ${first(answer,2)}\n○ 결론/요청  ${first(conclusion||'검토결과 및 후속 조치 확인 필요',2)}`,78,485,1120,185,24,ink);box(s,'포스코퓨처엠 회계세무그룹',78,810,600,30,16,'#61717F');box(s,'1',1188,860,50,26,14,'#61717F');s.speakerNotes.textFrame.setText(`사용자 질문\n${question}\n검토의견 원문\n${answer}`);return s;}
let s=cover();
// 1. 개요: 검토 배경·목적 및 범위를 표준 목차로 고정한다.
s=page('## 1 개요');heading(s,'1','검토 배경',104);box(s,`○ ${first(question,3)}\n- 검토 기준일: ${date}`,58,156,1160,110,23);heading(s,'1.1','목적 및 범위',296);box(s,'○ 검색된 법령·기준서와 제공 사실을 연결해 검토의견을 작성\n- 대상: 사용자 질문에 포함된 거래·기간·금액\n- 회사 공개자료는 관련성 확인을 위한 보조 Context로 사용',58,348,1160,145,22);heading(s,'1.2','진행 경과',535);box(s,`○ (${date}) 질문·사실관계 정리\n○ (${date}) 적용 기준·근거 검색 및 검토\n○ (${date}) 결론·추가 확인사항 및 산출물 작성`,58,585,1160,145,22);
// 2. 검토사항: 현황·쟁점과 비교표를 같은 페이지에 둔다.
s=page('## 2 검토사항');heading(s,'2','현황 및 쟁점',104);box(s,`○ ${first(answer,4)}\n- 확인된 사실과 미확인 사실을 구분해 검토`,58,156,1160,125,22);heading(s,'2.1','리스크 및 검토 포인트',315);box(s,`○ ${first(limitations.join('\n')||'검색된 근거와 사실관계의 연결을 확인할 필요',3)}\n- 결론을 바꿀 수 있는 계약·증빙·적용시점을 우선 확인`,58,367,1160,115,22);heading(s,'2.2','As-Is / To-Be 비교 (단위: 질문·근거 기준)',505);const compare=[['구분','기존(As-Is)','개선(To-Be)','효과·비고'],['검토 기준',first(answer,1),'근거 문단·요건별 검토','적용 근거 추적'],['담당자 조치','추가 확인사항 미정','필요 증빙과 일정 확인','의사결정 지원']];const ct=s.tables.add({rows:3,columns:4,left:48,top:557,width:1184,height:178,values:compare,columnWidths:[150,370,390,274]});ct.borders.assign({style:'solid',fill:'#CEDCE3',width:1});for(let r=0;r<3;r++){ct.cells.block({row:r,column:0,rowCount:1,columnCount:4}).assign({textStyle:{typeface:'Malgun Gothic',fontSize:r?16:15,color:r?'#202E39':'#FFFFFF',bold:r===0}});for(let c=0;c<4;c++)ct.getCell(r,c).fill=r===0?blue:(c===0?'#F0F5F8':'#FFFFFF');}box(s,'1) As-Is는 사용자 제공 사실과 현재 검토의견, To-Be는 필요한 조치 기준입니다.',50,754,1160,23,13,'#61717F');
for(const sec of sections)body('2  검토사항','2',sec.title,sec.lines.join('\n'));
// 3. 검토결과: 결론과 의사결정 요청, 회계처리, 향후계획 순서다.
bodyTitle='';s=page('## 3 검토결과');heading(s,'3','결론 및 의사결정 요청',105);rect(s,44,164,1192,130,'#EDF2DF');box(s,`○ ${first(conclusion||answer,4)}\n- 결론 변경 가능 사실은 아래 확인사항 기준으로 판단`,62,184,1150,91,24,ink,true);heading(s,'3.1','의사결정 요청사항',330);box(s,`○ 현재 검토의견 기준 처리방향 승인 여부\n○ 추가 증빙 확보 및 담당자 지정 여부\n○ 적용 시점·금액 확인 후 최종 반영 여부`,58,380,1160,105,22);
heading(s,'3.2','검토 요약 및 확인사항',508);const rows=[['구분','검토의견 / 확인할 내용'],['핵심 판단',first(conclusion||answer,4)],['미확인사항',first(limitations.join('\n')||'검토의견에 별도 미확인사항이 기재되어 있지 않습니다.',4)]];const table=s.tables.add({rows:3,columns:2,left:44,top:552,width:1192,height:178,values:rows,columnWidths:[186,1006]});table.borders.assign({style:'solid',fill:'#CEDCE3',width:1});for(let r=0;r<3;r++){table.cells.block({row:r,column:0,rowCount:1,columnCount:2}).assign({textStyle:{typeface:'Malgun Gothic',fontSize:r?17:16,color:r?'#202E39':'#FFFFFF',bold:r===0}});for(let c=0;c<2;c++)table.getCell(r,c).fill=r===0?blue:(c===0?'#F0F5F8':'#FFFFFF');}box(s,'1) 근거 문서는 별첨에서 원문과 함께 확인합니다.',48,744,1100,20,13,'#61717F');
if(d.knowledge_track==='accounting'){
  s=page('## 3 검토결과');heading(s,'3.3','※ 회계처리(안)',105);box(s,entryStatus==='제안 가능'?`○ 검토의견 기준 분개 방향\n- ${clean(entry.basis)||'검토의견 및 사용자 제공 사실 기준'}`:`○ 분개안은 ${entryStatus} 상태입니다.\n- ${clean(entry.note)||'계정과목·금액·지급조건을 확인한 뒤 확정합니다.'}`,58,158,1160,84,22);const entryRows=[['차변 계정과목','금액','대변 계정과목','금액'],[entryCell(debit[0]),clean(debit[0]?.amount)||'미확정',entryCell(credit[0]),clean(credit[0]?.amount)||'미확정']];const entryTable=s.tables.add({rows:2,columns:4,left:49,top:286,width:1182,height:132,values:entryRows,columnWidths:[420,170,420,172]});entryTable.borders.assign({style:'solid',fill:'#CEDCE3',width:1});for(let r=0;r<2;r++){entryTable.cells.block({row:r,column:0,rowCount:1,columnCount:4}).assign({textStyle:{typeface:'Malgun Gothic',fontSize:r?18:16,color:r?ink:'#FFFFFF',bold:r===0}});for(let c=0;c<4;c++)entryTable.getCell(r,c).fill=r===0?blue:'#FFFFFF';}box(s,'※ 금액은 실제 전표·계약서·세금계산서 확인 후 확정합니다.',58,438,1150,23,14,'#61717F');heading(s,'3.4','향후계획',520);box(s,`○ (${date}) 담당자 사실관계·증빙 확인\n○ (확인 후) 회계처리 및 주석 공시 검토\n○ (확정 시) 결산 반영 및 후속 조치 추진`,58,570,1160,125,22);
}else{ s=page('## 3 검토결과');heading(s,'3.3','향후계획',105);box(s,`○ (${date}) 담당자 사실관계·고지서·신고자료 확인\n○ (확인 후) 적용 세목·기한·금액 재검토\n○ (확정 시) 신고·납부 및 후속 조치 추진`,58,160,1160,150,23); }
if(wrap(conclusion,68).length>5)body('3  검토결과','3','검토 결론 · 전문',conclusion);
if(wrap(limitations.join('\n'),68).length>5)body('3  검토결과','3','추가 확인사항 · 요청문구',limitations.map(x=>'○ '+x).join('\n'));
if(d.calculation&&Object.keys(d.calculation).length)body('# 첨부  계산 내역','+','검토에 사용된 계산값',typeof d.calculation==='string'?d.calculation:JSON.stringify(d.calculation,null,2));
const evidence=(d.evidence||[]).map((e,i)=>`[${i+1}] ${clean(e.title)} ${clean(e.article)}\n${clean(e.url||e.source_url)}${e.metadata?.effective_date?'\n시행일: '+e.metadata.effective_date:''}`);
if(evidence.length)body('# 첨부  근거 문서','+','검색된 근거 · 원문 확인',evidence.join('\n\n'));
await fs.mkdir(path.dirname(output),{recursive:true});await(await PresentationFile.exportPptx(p)).save(output);
if(process.env.PPT_PREVIEW_DIR){await fs.mkdir(process.env.PPT_PREVIEW_DIR,{recursive:true});for(const [i,slide]of p.slides.items.entries()){const png=await p.export({slide,format:'png',scale:1});await fs.writeFile(path.join(process.env.PPT_PREVIEW_DIR,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));}}
console.log(JSON.stringify({slides:p.slides.items.length,output}));
