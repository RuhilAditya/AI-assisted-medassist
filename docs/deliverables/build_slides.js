// Builds docs/deliverables/MedAssist_Presentation.pptx: structured deck (theme, layouts, sections),
// native charts, speaker notes. Numbers come from results/*.json.
// Usage: node docs/deliverables/build_slides.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require("./theme.js");

const ROOT = path.resolve(__dirname, "..", "..");
const FIG = path.join(ROOT, "docs", "figures");
const R = (n) => {
  const p = path.join(ROOT, "results", `${n}.json`);
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null;
};
const img = R("phase2_image_only"), qa = R("phase2_text_qa"), fusion = R("phase3_fusion");
const robust = R("phase5_robustness"), held = R("phase5_heldout_text"), answers = R("phase5_answers");
const retr = R("phase2_retrieval");
const ens = img.results.find((r) => r.variant.startsWith("ensemble"));
const f2 = (x) => Number(x).toFixed(2), f3 = (x) => Number(x).toFixed(3);
const pc = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

// ------------------------------------------------------------------ theme (Atlas plate palette)
const THEME = {
  name: "MedAssist Atlas",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "14223B", lt1: "FFFFFF", dk2: "3F4D66", lt2: "F3F6F9",
    accent1: "C93A1B", // image evidence (vermilion)
    accent2: "1F5AA6", // note evidence (blue)
    accent3: "2E8A5A", // fused (green)
    accent4: "A3172F", // emergency
    accent5: "A84D0A", // urgent
    accent6: "5E6B82", // muted
    hlink: "1F5AA6", folHlink: "5E6B82",
  },
};
const HEX = { ink: "14223B", ink2: "3F4D66", muted: "5E6B82", rule: "D9E0E8", image: "C93A1B", note: "1F5AA6", fused: "2E8A5A", wash: "F3F6F9" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "MedAssist: A Multimodal Medical Assistant";
pres.author = "[Student Name]";
const C = pres.SchemeColor;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.7, h: 1.5, fontSize: 48, bold: true, color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.7, w: 11.7, h: 1.2, fontSize: 20, color: "C8D3E3", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.2, fontSize: 40, bold: true, color: C.background1, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 1.0, fontSize: 18, color: "C8D3E3", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.4, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: "C8D3E3" },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.text1, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "MedAssist · B.Tech capstone", options: { x: 0.6, y: 7.0, w: 5, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 12.4, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: C.accent6 },
});

// ------------------------------------------------------------------ helpers
async function icon(Comp, hex) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: `#${hex}`, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function fitImage(slide, file, x, y, maxW, maxH, name) {
  const { w, h } = pngSize(file);
  const scale = Math.min(maxW / w, maxH / h);
  const W = w * scale, H = h * scale;
  slide.addImage({ path: file, x: x + (maxW - W) / 2, y: y + (maxH - H) / 2, w: W, h: H, objectName: name,
    altText: name });
}
const text = (slide, t, o) => slide.addText(t, { isTextBox: true, margin: 0, fontSize: 16, color: C.text2, valign: "top", ...o });
function card(slide, x, y, w, h, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 },
    line: { color: HEX.rule, width: 0.75 }, objectName: name });
}
function stat(slide, x, y, w, value, label, color, name) {
  text(slide, value, { x, y, w, h: 0.9, fontSize: 44, bold: true, color, fontFace: THEME.headFontFace, objectName: `${name} value` });
  text(slide, label, { x, y: y + 0.9, w, h: 0.7, fontSize: 14, color: C.text2, objectName: `${name} label` });
}
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  titleFontFace: "+mn-lt", legendFontFace: "+mn-lt", catAxisLabelColor: HEX.ink2, valAxisLabelColor: HEX.muted,
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 12, dataLabelColor: HEX.ink,
  valGridLine: { color: "E3E8EE", size: 0.75 }, catGridLine: { style: "none" } };

async function build() {
  const ic = {
    note: await icon(fa.FaNotesMedical, HEX.note), xray: await icon(fa.FaXRay, HEX.image),
    quote: await icon(fa.FaQuoteRight, HEX.ink), warn: await icon(fa.FaExclamationTriangle, "A3172F"),
    heart: await icon(fa.FaHeartbeat, "A3172F"), lock: await icon(fa.FaLock, HEX.ink),
    scale: await icon(fa.FaBalanceScale, HEX.fused), route: await icon(fa.FaRoute, HEX.note),
    shield: await icon(fa.FaShieldAlt, HEX.fused), steth: await icon(fa.FaStethoscope, HEX.ink),
  };
  const iconCircle = (slide, data, x, y, d, name) => {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: "FFFFFF" }, line: { color: HEX.rule, width: 0.75 }, objectName: `${name} circle` });
    slide.addImage({ data, x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56, objectName: `${name} icon`, altText: name });
  };

  // ---------------------------------------------------------------- 1. title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("MedAssist", { placeholder: "title" });
  s.addText([
    { text: "A multimodal medical assistant that reads clinical notes and chest X-rays together, and shows its evidence", options: { breakLine: true } },
    { text: "[Student Name] · [Roll Number] · Guide: [Guide Name] · [College] · [Month Year]", options: { fontSize: 14, color: "9FB0C8" } },
  ], { placeholder: "body" });
  s.addNotes("Introduce yourself and the project. One sentence: MedAssist is a local AI assistant that combines a patient's clinical note with their chest X-ray, answers questions with every claim cited, flags when the note and the image disagree, and gives rule-based triage.");

  // ---------------------------------------------------------------- 2. problem
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Doctors read note and image together; AI tools see one", { placeholder: "title" });
  const probs = [
    [ic.xray, "Single modality", "Image classifiers ignore the history; chatbots never see the film. Errors hide where the two disagree."],
    [ic.quote, "Unverifiable answers", "LLMs answer fluently but rarely show where a claim came from, so a clinician cannot check it quickly."],
    [ic.lock, "Privacy", "Cloud assistants send patient data off-site, which clashes with HIPAA and India's DPDP Act 2023."],
  ];
  probs.forEach(([icn, h, b], i) => {
    const y = 1.55 + i * 1.65;
    iconCircle(s, icn, 0.7, y, 0.95, h);
    text(s, h, { x: 1.95, y, w: 5.6, h: 0.45, fontSize: 20, bold: true, color: C.text1, objectName: `${h} heading` });
    text(s, b, { x: 1.95, y: y + 0.48, w: 5.6, h: 0.95, fontSize: 15, objectName: `${h} body` });
  });
  card(s, 8.1, 1.55, 4.6, 4.9, "goal card");
  text(s, "Project goal", { x: 8.4, y: 1.8, w: 4.0, h: 0.45, fontSize: 18, bold: true, color: C.accent3, objectName: "goal heading" });
  text(s, "Build an assistant that processes notes, images and reports together to support decision-making, triage and information retrieval, and evaluate honestly whether combining them helps, whether answers are faithful, and whether it is robust and usable.",
    { x: 8.4, y: 2.35, w: 4.0, h: 3.8, fontSize: 16, color: C.text1, objectName: "goal body" });
  s.addNotes("Motivate with the example of a note saying 'no effusion' next to a film that clearly shows one. That mismatch is exactly what a single-modality tool cannot catch.");

  // ---------------------------------------------------------------- 3. what it does
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("What MedAssist does for one case", { placeholder: "title" });
  const caps = [
    [ic.shield, "De-identifies first", "Names, IDs, phones, dates, hospitals removed before anything is stored; DICOM tags anonymised."],
    [ic.xray, "Reads the X-ray", "BiomedCLIP scores 9 findings with calibrated probabilities, heatmaps and similar prior cases."],
    [ic.note, "Reads the note", "Symptoms, medications, vitals; knows 'denies chest pain' is absent, not present."],
    [ic.warn, "Flags disagreements", "Note says no effusion but the image shows one: a HIGH conflict, surfaced first."],
    [ic.heart, "Triages by rules", "NEWS2 early-warning score + red flags: EMERGENCY, URGENT, SOON, ROUTINE."],
    [ic.quote, "Answers with citations", "Local Llama 3.1 explains the evidence; every claim links to its source."],
  ];
  caps.forEach(([icn, h, b], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + col * 4.1, y = 1.5 + row * 2.65;
    card(s, x, y, 3.85, 2.4, `${h} card`);
    iconCircle(s, icn, x + 0.25, y + 0.25, 0.75, h);
    text(s, h, { x: x + 1.15, y: y + 0.38, w: 2.55, h: 0.5, fontSize: 18, bold: true, color: C.text1, objectName: `${h} heading` });
    text(s, b, { x: x + 0.25, y: y + 1.15, w: 3.35, h: 1.1, fontSize: 14, objectName: `${h} body` });
  });
  s.addNotes("Walk through the six capabilities left to right. Emphasise that triage and conflicts are rules, not the LLM.");

  // ---------------------------------------------------------------- 4. architecture
  pres.addSection({ title: "Design" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Design" });
  s.addText("Architecture: rules decide, the LLM explains", { placeholder: "title" });
  fitImage(s, path.join(FIG, "architecture.png"), 0.6, 1.35, 12.1, 5.45, "Architecture diagram");
  s.addNotes("Text path in blue, image path in vermilion, fusion in green. Each modality is processed by its own specialist model and combined in an explicit evidence bundle. That keeps every stage testable and explainable, unlike an end-to-end black box. Everything runs on a 6 GB laptop GPU.");

  // ---------------------------------------------------------------- 5. principles
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Design" });
  s.addText("Four design principles", { placeholder: "title" });
  const pr = [
    [ic.scale, "Rules decide, the LLM explains", "Triage and conflicts are deterministic and unit-tested; the model only explains them."],
    [ic.route, "Evidence before answers", "Findings, conflicts and triage in ~0.1 s; the written answer streams in afterwards."],
    [ic.steth, "Honest uncertainty", "Calibrated probabilities shown as unlikely / possible / likely; never 'ruled out'."],
    [ic.lock, "Private by design", "Everything local; raw uploads never written to disk; audit log without content."],
  ];
  pr.forEach(([icn, h, b], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 6.15, y = 1.55 + row * 2.6;
    iconCircle(s, icn, x, y, 0.9, h);
    text(s, h, { x: x + 1.15, y: y + 0.05, w: 4.7, h: 0.5, fontSize: 20, bold: true, color: C.text1, objectName: `${h} heading` });
    text(s, b, { x: x + 1.15, y: y + 0.6, w: 4.7, h: 1.4, fontSize: 16, objectName: `${h} body` });
  });
  s.addNotes("These principles came out of testing. For example, Presidio once deleted the word 'No' from 'No night sweats', turning an absent symptom into a present one, which is why safety-relevant logic stays in testable rules.");

  // ---------------------------------------------------------------- 6. interface
  pres.addSection({ title: "Interface" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Interface" });
  s.addText("The case workspace", { placeholder: "title" });
  fitImage(s, path.join(FIG, "ui", "workspace_pneumonia.png"), 0.6, 1.35, 8.6, 5.45, "Case workspace screenshot");
  const callouts = [
    ["Triage band", "Colour and NEWS2 at a glance"],
    ["X-ray + heatmap", "Where the model looked, per finding"],
    ["De-identified note", "Present / absent / history underlined"],
    ["Findings as bands", "Image score and note-adjusted score"],
    ["Cited answer", "Streams in; every claim cited"],
  ];
  callouts.forEach(([h, b], i) => {
    const y = 1.45 + i * 1.06;
    s.addShape(pres.shapes.OVAL, { x: 9.55, y: y + 0.05, w: 0.38, h: 0.38, fill: { color: C.text1 }, line: { color: HEX.ink, width: 0 }, objectName: `callout ${i + 1} dot` });
    text(s, String(i + 1), { x: 9.55, y: y + 0.05, w: 0.38, h: 0.38, fontSize: 13, bold: true, color: C.background1, align: "center", valign: "middle", objectName: `callout ${i + 1} number` });
    text(s, h, { x: 10.1, y, w: 2.7, h: 0.4, fontSize: 16, bold: true, color: C.text1, objectName: `callout ${i + 1} heading` });
    text(s, b, { x: 10.1, y: y + 0.4, w: 2.7, h: 0.55, fontSize: 13, objectName: `callout ${i + 1} body` });
  });
  s.addNotes("Designed with the Impeccable design skill as an 'anatomy atlas plate': light theme for projectors, Atkinson Hyperlegible type, vermilion for image evidence and blue for note evidence.");

  // ---------------------------------------------------------------- 7. signature
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Interface" });
  s.addText("Every claim traceable: leader lines to the evidence", { placeholder: "title" });
  fitImage(s, path.join(FIG, "ui", "conflict_case.png"), 0.6, 1.35, 8.6, 5.45, "Conflict case with leader lines");
  text(s, "Hover a citation or a finding and a line is drawn to its exact source: the X-ray in vermilion, the note phrase in blue.", { x: 9.55, y: 1.5, w: 3.2, h: 1.6, fontSize: 16, color: C.text1, objectName: "signature body" });
  text(s, "Here the HIGH conflict sits first: the image suggests pleural effusion (0.98) while the note says 'no pleural effusion'.", { x: 9.55, y: 3.3, w: 3.2, h: 1.8, fontSize: 16, objectName: "conflict body" });
  s.addNotes("This is the most important slide for 'explainability'. In the live demo, hover the Pleural effusion finding to show the three lines.");

  // ---------------------------------------------------------------- 8. results divider
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Results" });
  s.addText("Results", { placeholder: "title" });
  s.addText("IU X-Ray (435 test studies) · PubMedQA · 120 held-out notes · 28 real answers", { placeholder: "body" });

  // ---------------------------------------------------------------- 9. image findings
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Chest X-ray findings: strong on large findings", { placeholder: "title" });
  const findings = Object.entries(ens.per_finding).sort((a, b) => a[1].auroc - b[1].auroc);
  s.addChart(pres.charts.BAR, [{ name: "AUROC", labels: findings.map(([k]) => k), values: findings.map(([, v]) => v.auroc) }], {
    x: 0.6, y: 1.35, w: 7.8, h: 5.45, barDir: "bar", chartColors: [HEX.image], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", valAxisMinVal: 0.4, valAxisMaxVal: 1.0, valAxisLabelFormatCode: "0.0", showLegend: false,
    showTitle: true, title: "Per-finding AUROC, calibrated ensemble (n = 435)", titleFontSize: 14, titleColor: HEX.ink, ...chartText,
    objectName: "Per-finding AUROC chart",
  });
  stat(s, 8.9, 1.5, 3.8, f3(ens.summary.macro_auroc), "macro AUROC over 9 findings (zero-shot alone 0.710)", C.accent1, "macro auroc");
  stat(s, 8.9, 3.25, 3.8, f3(ens.summary.abnormal_vs_normal_auroc), "abnormal vs normal AUROC", C.text1, "abnormal auroc");
  stat(s, 8.9, 5.0, 3.8, String(ens.calibration.ece_calibrated.toFixed(3)), `calibration error (from ${ens.calibration.ece_raw.toFixed(3)}), so 0.9 means ~90%`, C.accent3, "ece");
  s.addNotes("Zero-shot BiomedCLIP plus a linear probe, averaged and Platt-calibrated. Effusion, cardiomegaly and edema near 0.9; small focal findings like nodules are weak at 224-pixel resolution, an honest limitation.");

  // ---------------------------------------------------------------- 10. RAG
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Retrieval lifts answer accuracy by 23 points", { placeholder: "title" });
  const qaOrder = [["majority-class baseline", "Majority class"], ["closed-book", "Llama closed-book"], ["RAG (top-3)", "Llama + RAG"], ["oracle context", "Llama + gold abstract"]];
  s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: qaOrder.map(([, l]) => l), values: qaOrder.map(([k]) => qa.qa[k].accuracy) }], {
    x: 0.6, y: 1.35, w: 7.8, h: 5.45, barDir: "col", chartColors: [HEX.note], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0", showLegend: false,
    showTitle: true, title: "PubMedQA accuracy (150 questions, Llama 3.1 8B)", titleFontSize: 14, titleColor: HEX.ink, ...chartText,
    objectName: "PubMedQA accuracy chart",
  });
  stat(s, 8.9, 1.5, 3.8, `${f2(qa.qa["closed-book"].accuracy)} → ${f2(qa.qa["RAG (top-3)"].accuracy)}`, "accuracy without vs with retrieval (gold-context ceiling 0.82)", C.accent2, "rag");
  stat(s, 8.9, 3.4, 3.8, f3(qa.retrieval["PubMedBERT (MS-MARCO)"]["recall@1"]), "Recall@1 of the chosen retriever over 1,000 abstracts", C.text1, "recall");
  text(s, "Raw ClinicalBERT: Recall@1 only 0.159. Domain pre-training is not retrieval ability.", { x: 8.9, y: 5.3, w: 3.8, h: 1.2, fontSize: 15, objectName: "clinicalbert note" });
  s.addNotes("Closed-book the 8B model is below the majority baseline: it answers confidently but wrongly. With retrieved evidence it nearly reaches the gold-context ceiling. That is the argument for evidence-grounded answers.");

  // ---------------------------------------------------------------- 11. fusion
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Fusion helps when the text carries information", { placeholder: "title" });
  const fA = fusion.A_real_indication.macro_auroc, fB = fusion.B_simulation.macro_auroc;
  s.addChart(pres.charts.BAR, [
    { name: "Image only", labels: ["Real data (IU indication)", "Controlled simulation"], values: [fA.image_only, fB.image_only] },
    { name: "Text only", labels: ["Real data (IU indication)", "Controlled simulation"], values: [fA.text_only, fB.text_only] },
    { name: "Fused", labels: ["Real data (IU indication)", "Controlled simulation"], values: [fA.fused, fB.fused] },
  ], {
    x: 0.6, y: 1.35, w: 7.8, h: 5.45, barDir: "col", barGrouping: "clustered", chartColors: [HEX.image, HEX.note, HEX.fused],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0.4, valAxisMaxVal: 0.9, valAxisMajorUnit: 0.1,
    valAxisLabelFormatCode: "0.0", showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: HEX.ink2,
    showTitle: true, title: "Macro AUROC, 435 test studies", titleFontSize: 14, titleColor: HEX.ink, ...chartText,
    objectName: "Fusion ablation chart",
  });
  text(s, [
    { text: "Real data: ", options: { bold: true, color: C.text1 } },
    { text: `IU indications are a few words ("chest pain"), so fusion adds a small but significant ${f3(fA.image_only)} → ${f3(fA.fused)}.`, options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Simulation: ", options: { bold: true, color: C.text1 } },
    { text: `with informative notes, fused (${f3(fB.fused)}) beats both image and text alone.`, options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Conflicts: ", options: { bold: true, color: C.text1 } },
    { text: `rules caught ${pc(fusion.C_conflicts.detection_rate_when_image_agrees_with_truth, 0)} of injected disagreements the image model could see.` },
  ], { x: 8.9, y: 1.5, w: 3.8, h: 5.1, fontSize: 16, objectName: "fusion commentary" });
  s.addNotes("Be upfront: on real data the gain is small because the text is thin. The simulation shows the mechanism works when the note is informative; the text strength there is set by us, so it is a sanity check, not clinical evidence.");

  // ---------------------------------------------------------------- 12. robustness & privacy
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Robust to realistic noise; private by default", { placeholder: "title" });
  fitImage(s, path.join(FIG, "robustness.png"), 0.6, 1.35, 7.6, 5.4, "Robustness chart");
  const r0 = held.runs[0], r10 = held.runs[held.runs.length - 1];
  stat(s, 8.7, 1.45, 4.0, "≤ 0.017", "AUROC drop at realistic noise, blur, rotation, JPEG", C.accent1, "robust");
  stat(s, 8.7, 3.2, 4.0, pc(r0.phi_recall), `identifiers removed in 120 held-out notes (${pc(r10.phi_recall)} with 10% typos)`, C.accent3, "deid");
  stat(s, 8.7, 4.95, 4.0, f2(r0.negation_f1), `negation F1; finding recall ${pc(r0.active_recall)}`, C.accent2, "negation");
  s.addNotes("Robustness: only extreme corruption (heavy noise, JPEG quality 5) costs about 0.10 AUROC. De-identification: the first held-out set exposed that hospital names were never removed; after the fix it was re-measured on a fresh set, so the fix is not scored on the data that revealed it.");

  // ---------------------------------------------------------------- 13. faithfulness
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Answers stay faithful to the evidence", { placeholder: "title" });
  const a = answers.summary;
  const grid = [
    [pc(a.citation_validity), "citations point to real evidence", C.accent3],
    [pc(a.ungrounded_number_rate), "numbers not found in the evidence", C.accent3],
    [pc(a.triage_level_stated, 0), "triage answers state the rule-computed level", C.text1],
    [pc(a.conflict_disclosure), "answers disclose a HIGH/MEDIUM conflict", C.text1],
    [String(a.answers_with_unsafe_phrases), "answers with 'ruled out' / 'definitely'", C.text1],
    [pc(a.citation_coverage, 0), "claims carrying a citation: the weak spot", C.accent1],
  ];
  grid.forEach(([v, l, col], i) => {
    const x = 0.6 + (i % 3) * 4.1, y = 1.5 + Math.floor(i / 3) * 2.55;
    card(s, x, y, 3.85, 2.25, `metric ${i + 1} card`);
    stat(s, x + 0.3, y + 0.3, 3.3, v, l, col, `metric ${i + 1}`);
  });
  text(s, `28 real Llama answers (7 demo cases × 4 question types), scored automatically, no LLM-as-judge. Median latency: evidence ${a.latency_s.evidence_median} s, written answer ${a.latency_s.llm_median} s.`,
    { x: 0.6, y: 6.55, w: 12.1, h: 0.4, fontSize: 13, color: C.accent6, objectName: "faithfulness footnote" });
  s.addNotes("Mention the one real slip: an answer called 37.9 C 'normal temperature', which is why the interface keeps the source evidence visible. Citation coverage is the main thing to improve.");

  // ---------------------------------------------------------------- 14. demo
  pres.addSection({ title: "Demo and conclusion" });
  s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Demo and conclusion" });
  s.addText("Live demo", { placeholder: "title" });
  s.addText("Pneumonia (EMERGENCY) → missed effusion (HIGH conflict, leader lines) → wrong image (modality gate) → ask a question", { placeholder: "body" });
  s.addNotes("Demo script: 1) Open 'Fever, productive cough, hypoxia': point at the EMERGENCY band and NEWS2 cells, turn on the effusion heatmap. 2) Open 'Note says no effusion': show the HIGH conflict, hover the Pleural effusion finding for leader lines. 3) Open 'brain MRI uploaded': the classifier refuses. 4) Ask 'How urgent is this case, and why?' and hover a citation while it streams.");

  // ---------------------------------------------------------------- 15. limitations & future
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demo and conclusion" });
  s.addText("Limitations and next steps", { placeholder: "title" });
  card(s, 0.6, 1.5, 5.9, 4.3, "limitations card");
  text(s, "Limitations", { x: 0.9, y: 1.75, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: C.accent1, objectName: "limitations heading" });
  text(s, [
    { text: "Small focal findings (nodules, fractures) weak at 224 px", options: { bullet: true, breakLine: true } },
    { text: "No public paired notes: rich-text fusion shown in simulation", options: { bullet: true, breakLine: true } },
    { text: "De-ID tested on generated, not real, notes", options: { bullet: true, breakLine: true } },
    { text: "Hand-set fusion weights; 50% citation coverage", options: { bullet: true, breakLine: true } },
    { text: "Research prototype, not a medical device", options: { bullet: true } },
  ], { x: 0.9, y: 2.4, w: 5.3, h: 3.9, fontSize: 16, paraSpaceAfter: 10, objectName: "limitations list" });
  card(s, 6.8, 1.5, 5.9, 4.3, "future card");
  text(s, "Next steps", { x: 7.1, y: 1.75, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: C.accent3, objectName: "future heading" });
  text(s, [
    { text: "Higher-resolution fine-tuning on CheXpert / MIMIC-CXR", options: { bullet: true, breakLine: true } },
    { text: "Learn fusion weights from MIMIC notes linked to images", options: { bullet: true, breakLine: true } },
    { text: "Per-sentence citation enforcement (verifier step)", options: { bullet: true, breakLine: true } },
    { text: "CT and more modalities; MedQuAD knowledge base", options: { bullet: true, breakLine: true } },
    { text: "Clinician usability study with ethics approval", options: { bullet: true } },
  ], { x: 7.1, y: 2.4, w: 5.3, h: 3.9, fontSize: 16, paraSpaceAfter: 10, objectName: "future list" });
  s.addNotes("If the user study has been run, add the SUS score here.");

  // ---------------------------------------------------------------- 16. close
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Demo and conclusion" });
  s.addText("Thank you", { placeholder: "title" });
  s.addText([
    { text: "MedAssist: notes and X-rays together, every claim traceable, safety in rules, fully local", options: { breakLine: true } },
    { text: "Questions?", options: { fontSize: 16, color: "9FB0C8" } },
  ], { placeholder: "body" });
  s.addNotes("Close with the one-line summary and invite questions. Likely questions: why late fusion instead of end-to-end; why rules for triage; how de-identification was validated; why the real-data fusion gain is small.");

  const out = path.join(__dirname, "MedAssist_Presentation.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log(`Wrote ${out}`);
}

build().catch((e) => { console.error(e); process.exit(1); });
