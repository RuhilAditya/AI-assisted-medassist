// Builds docs/deliverables/MedAssist_Project_Report.docx (A4, B.Tech report format).
// Results tables are read from results/*.json so the report always matches the experiments.
// Usage: node docs/deliverables/build_report.js     (then open in Word and press F9 / accept "update fields" for the TOC)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, PageBreak, Footer, PageNumber, NumberFormat,
  TableOfContents, LevelFormat, TabStopType, TabStopPosition,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const FIG = path.join(ROOT, "docs", "figures");
const R = (name) => {
  const p = path.join(ROOT, "results", `${name}.json`);
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null;
};
const res = {
  image: R("phase2_image_only"), retrieval: R("phase2_retrieval"), qa: R("phase2_text_qa"),
  textCxr: R("phase2_text_only_cxr"), fusion: R("phase3_fusion"), robust: R("phase5_robustness"),
  held: R("phase5_heldout_text"), heldPre: R("phase5_heldout_text_discovery_prefix"),
  answers: R("phase5_answers"), study: R("phase5_user_study"),
};

// ------------------------------------------------------------------ styling helpers
const FONT = "Times New Roman";
const BODY = 24; // 12 pt
const PAGE_W = 11906, MARGIN_L = 2160, MARGIN_R = 1440; // A4, 1.5" binding margin, 1" right
const CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R; // 8306 DXA

let figNo = { ch: 0, n: 0 }, tabNo = { ch: 0, n: 0 };

function runs(text, base = {}) {
  // **bold**, *italic*, `code`
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: BODY, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, size: BODY, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: BODY - 3, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, size: BODY, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: BODY, ...base }));
  return out;
}
const P = (text, opts = {}) =>
  new Paragraph({ children: runs(text), alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: 360 }, ...opts });
const C = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), alignment: AlignmentType.CENTER, spacing: { after: 120 }, ...opts.para });
const bullet = (text) => new Paragraph({ children: runs(text), numbering: { reference: "bullets", level: 0 }, spacing: { after: 80, line: 340 }, alignment: AlignmentType.JUSTIFIED });
const numbered = (text, instance = 1) => new Paragraph({ children: runs(text), numbering: { reference: "numbers", level: 0, instance }, spacing: { after: 80, line: 340 } });
const numbered2 = (text) => numbered(text, 2);
const blank = () => new Paragraph({ children: [] });

function H1(text, ch, label) {
  const key = label || ch;
  figNo = { ch: key, n: 0 }; tabNo = { ch: key, n: 0 };
  const heading = new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: ch ? `Chapter ${ch}: ${text}` : text })] });
  // the body section already starts on a new page, so chapter 1 needs no break
  return ch === 1 ? [heading] : [new Paragraph({ children: [new PageBreak()] }), heading];
}
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text })] });
const H3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text })] });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}
function figure(file, caption, widthIn = 6.0) {
  const full = path.join(FIG, file);
  if (!fs.existsSync(full)) return [P(`[Missing figure: ${file}]`)];
  const { w, h, data } = pngSize(full);
  const width = Math.round(Math.min(widthIn, 5.75) * 96), height = Math.round((width * h) / w);
  figNo.n += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data, transformation: { width, height },
        altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
      children: [new TextRun({ text: `Figure ${figNo.ch}.${figNo.n}: `, bold: true, font: FONT, size: 22 }), new TextRun({ text: caption, font: FONT, size: 22 })] }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "8C96A3" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(caption, header, rows, weights) {
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor((CONTENT_W * w) / total));
  widths[widths.length - 1] = CONTENT_W - widths.slice(0, -1).reduce((a, b) => a + b, 0);
  const cell = (text, i, head) =>
    new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      shading: head ? { fill: "E3E8EE", type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ children: runs(String(text), { size: 20, bold: !!head }), alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })],
    });
  tabNo.n += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 80 }, keepNext: true,
      children: [new TextRun({ text: `Table ${tabNo.ch}.${tabNo.n}: `, bold: true, font: FONT, size: 22 }), new TextRun({ text: caption, font: FONT, size: 22 })] }),
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
        ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
    }),
    blank(),
  ];
}
const f3 = (x) => (typeof x === "number" ? x.toFixed(3) : String(x ?? "n/a"));
const pct = (x, d = 1) => (typeof x === "number" ? `${(x * 100).toFixed(d)}%` : "n/a");

// ------------------------------------------------------------------ front matter
const title = [
  blank(), blank(),
  C("MEDASSIST: A MULTIMODAL MEDICAL ASSISTANT FOR CLINICAL DECISION SUPPORT", { run: { bold: true, size: 36 } }),
  C("Combining clinical notes and chest X-rays with explainable, evidence-cited reasoning", { run: { italics: true, size: 26 } }),
  blank(),
  C("A Project Report", { run: { size: 26 } }),
  C("submitted in partial fulfilment of the requirements for the award of the degree of", { run: { size: 24 } }),
  C("BACHELOR OF TECHNOLOGY", { run: { bold: true, size: 28 } }),
  C("in", { run: { size: 24 } }),
  C("[Computer Science and Engineering]", { run: { bold: true, size: 26 } }),
  blank(),
  C("Submitted by", { run: { size: 24 } }),
  C("[Student Name] ([Roll Number])", { run: { bold: true, size: 26 } }),
  blank(),
  C("Under the guidance of", { run: { size: 24 } }),
  C("[Guide Name], [Designation]", { run: { bold: true, size: 26 } }),
  blank(), blank(),
  C("[Department of Computer Science and Engineering]", { run: { size: 24 } }),
  C("[College Name], [City]", { run: { bold: true, size: 26 } }),
  C("[University Name]", { run: { size: 24 } }),
  C("[Month Year]", { run: { size: 24 } }),
];

const certificate = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Certificate")] }),
  P("This is to certify that the project report entitled **“MedAssist: A Multimodal Medical Assistant for Clinical Decision Support”** submitted by **[Student Name] ([Roll Number])** in partial fulfilment of the requirements for the award of the degree of Bachelor of Technology in [Computer Science and Engineering] is a record of bona fide work carried out under my supervision and guidance during the academic year [YYYY–YY]."),
  blank(), blank(),
  new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
    children: [new TextRun({ text: "[Guide Name]\t[Head of Department]", font: FONT, size: BODY, bold: true })] }),
  new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
    children: [new TextRun({ text: "Project Guide\tHead of Department", font: FONT, size: BODY })] }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Declaration")] }),
  P("I hereby declare that the work presented in this report is my own and has not been submitted elsewhere for the award of any degree or diploma. Wherever the work of others has been used, it has been duly acknowledged and referenced. Software tools and AI coding assistants used during development are acknowledged in the Acknowledgement and in Chapter 3."),
  blank(),
  new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
    children: [new TextRun({ text: "Place: [City]\t[Student Name]", font: FONT, size: BODY })] }),
  new Paragraph({ children: [new TextRun({ text: "Date: [DD/MM/YYYY]", font: FONT, size: BODY })] }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Acknowledgement")] }),
  P("[Thank your guide, head of department, faculty, user-study participants, friends and family here.] I also acknowledge the open-source projects this work builds on, including Meta Llama, Microsoft BiomedCLIP and Presidio, scispaCy, medspaCy, LangChain and Ollama, and the providers of the Indiana University chest X-ray collection and PubMedQA datasets. An AI coding assistant (Claude Code) was used to help implement and document parts of the software; all design decisions, experiments and results were reviewed by the author."),
  new Paragraph({ children: [new PageBreak()] }),
];

const abstractText = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Abstract")] }),
  P("Clinicians routinely reason across several kinds of evidence at once: what the patient reports, the vital signs, and what an image shows. Most AI tools see only one of these, and most chat-style assistants give answers without showing where they came from. This project presents **MedAssist**, a locally-run multimodal clinical decision-support assistant that combines de-identified clinical notes with chest X-rays, answers questions with every claim cited to its evidence, flags disagreements between the note and the image, and gives rule-based triage guidance."),
  P("Notes are de-identified with Microsoft Presidio extended with Indian identifier formats, then structured by scispaCy and medspaCy (negation, history and family context). Images pass a modality gate before a BiomedCLIP-based classifier scores nine chest X-ray findings; an ensemble of zero-shot prompting and a linear probe, Platt-calibrated, reaches a macro AUROC of " +
    `**${f3(res.image?.results?.at(-1)?.summary?.macro_auroc)}** on 435 held-out IU X-Ray studies, with occlusion heatmaps and similar-case retrieval for explanation. A late-fusion layer shifts image probabilities by the log-likelihood of the note evidence, detects cross-modal conflicts, and computes the NEWS2 early-warning score. A local Llama 3.1 8B model, orchestrated with LangChain, explains these rule-computed results and cites each claim; a React interface draws “leader lines” from each claim to its source.`),
  P(`Retrieval-augmented generation raised PubMedQA accuracy from ${f3(res.qa?.qa?.["closed-book"]?.accuracy)} to **${f3(res.qa?.qa?.["RAG (top-3)"]?.accuracy)}**. Fusing image and text improved macro AUROC on real data by a small but significant margin (${f3(res.fusion?.A_real_indication?.macro_auroc?.image_only)} → ${f3(res.fusion?.A_real_indication?.macro_auroc?.fused)}) and substantially in a controlled simulation with informative notes (→ ${f3(res.fusion?.B_simulation?.macro_auroc?.fused)}); the conflict rules detected every injected disagreement the image model could see. On 120 held-out notes, de-identification removed ${pct(res.held?.runs?.[0]?.phi_recall)} of planted identifiers while preserving ${pct(res.held?.runs?.[0]?.clinical_text_kept)} of clinical text. Across 28 real answers, ${pct(res.answers?.summary?.citation_validity)} of citations were valid and only ${pct(res.answers?.summary?.ungrounded_number_rate)} of numbers were ungrounded. The system is a research prototype and not a medical device.`),
  P("**Keywords:** multimodal AI, clinical decision support, BiomedCLIP, retrieval-augmented generation, Llama, de-identification, NEWS2, explainable AI, chest X-ray."),
];

const toc = [
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Table of Contents")] }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
];

// ------------------------------------------------------------------ chapters
const img = res.image;
const ens = img?.results?.find((r) => r.variant.startsWith("ensemble"));
const zs = img?.results?.find((r) => r.variant === "zero-shot");
const calib = ens?.calibration;
const fA = res.fusion?.A_real_indication, fB = res.fusion?.B_simulation, fC = res.fusion?.C_conflicts;
const held0 = res.held?.runs?.[0];
const ans = res.answers?.summary;

const ch1 = [
  ...H1("Introduction", 1),
  H2("1.1 Background"),
  P("Clinical decisions are rarely made from a single source of information. A doctor assessing a breathless patient reads the referral note, checks the vital signs, looks at the chest X-ray and mentally cross-checks them: does the image explain the symptoms, and does anything not fit? Errors often hide in exactly those mismatches, for example a note that says “no effusion” next to a film that clearly shows one."),
  P("Recent foundation models make it possible to build assistants that read both text and images. Vision–language models such as BiomedCLIP [2] align medical images with text, and large language models (LLMs) such as Llama 3 [16] can explain findings in fluent language. However, most published systems handle one modality, produce answers without traceable evidence, or require sending patient data to cloud services, which conflicts with privacy obligations such as India's Digital Personal Data Protection Act [26]."),
  H2("1.2 Problem statement"),
  P("Design and develop an AI assistant that processes and understands multimodal medical data (clinical text, images and reports) to support clinical decision-making, triage and information retrieval, and that evaluates rigorously whether combining modalities helps, whether its answers are faithful to the evidence, and whether it is robust and usable."),
  H2("1.3 Objectives"),
  numbered("Ingest clinical notes (TXT/PDF) and medical images (DICOM/PNG/JPEG) and **de-identify** them before storage or processing."),
  numbered("Extract and normalise clinical entities from text, distinguishing present, negated, historical and family findings."),
  numbered("Detect chest X-ray findings with **BiomedCLIP**, with calibrated probabilities, heatmap explanations and similar-case retrieval."),
  numbered("Answer questions with **retrieval-augmented generation** using a local Llama model orchestrated by LangChain, citing every claim."),
  numbered("**Fuse** text and image evidence, detect cross-modal conflicts and provide rule-based triage guidance (NEWS2)."),
  numbered("Provide a clinician-facing interface that makes the reasoning visible."),
  numbered("Evaluate diagnostic accuracy, the multimodal-versus-unimodal gain, robustness, explanation faithfulness, latency and usability."),
  H2("1.4 Scope and assumptions"),
  P("Chest X-ray is the primary benchmarked image domain, because public datasets pair chest radiographs with reports (IU X-Ray [4]). Other modalities are recognised by a modality gate but not analysed. All processing runs locally on a consumer laptop (RTX 4050, 6 GB VRAM). The system is a **research prototype for decision support**: it never states a diagnosis, and it is not validated for clinical use."),
  H2("1.5 Organisation of the report"),
  P("Chapter 2 reviews related work. Chapter 3 presents requirements and architecture. Chapter 4 describes the implementation of each module and the user interface. Chapter 5 describes datasets and evaluation methodology, and Chapter 6 reports and discusses results. Chapter 7 covers safety, privacy and ethics, and Chapter 8 concludes with limitations and future work."),
];

const ch2 = [
  ...H1("Literature Review", 2),
  H2("2.1 Vision–language models for medical images"),
  P("CLIP [1] learns a joint embedding space for images and text from image–caption pairs, enabling zero-shot classification by comparing an image with textual prompts. BiomedCLIP [2] adapts this idea to biomedicine using 15 million figure–caption pairs from PubMed Central, with a PubMedBERT text encoder and a ViT-B/16 image encoder; it outperforms general CLIP on radiology and pathology retrieval and classification. CheXzero [3] showed that self-supervised image–report training can reach expert-level detection of several chest X-ray pathologies without labels. Large labelled datasets such as CheXpert [5] and ChestX-ray14 [6] are commonly used to benchmark supervised models."),
  H2("2.2 Clinical NLP and de-identification"),
  P("BioBERT [10], ClinicalBERT [11] and PubMedBERT [12] adapt BERT to biomedical and clinical text. Sentence-BERT [13] fine-tunes such encoders to produce sentence embeddings suitable for semantic search. scispaCy [7] provides fast biomedical NER pipelines, and medspaCy [8] implements clinical text tools including the ConText algorithm [9], which determines whether a mentioned condition is negated, historical, hypothetical or experienced by someone other than the patient. Microsoft Presidio [24] is an open-source framework for detecting and anonymising personal data; HIPAA Safe Harbor [23] lists the identifier types to remove, and DICOM PS3.15 [25] defines attribute-level de-identification profiles for images."),
  H2("2.3 Large language models and retrieval-augmented generation"),
  P("LLMs encode substantial clinical knowledge [29] but can state incorrect facts fluently. Retrieval-augmented generation (RAG) [14] grounds generation in retrieved passages, improving factuality and enabling citations. PubMedQA [15] is a standard benchmark of yes/no/maybe questions over PubMed abstracts. Open-weight models such as Llama 3 [16] can run locally, which keeps patient data on the device. Multimodal assistants such as LLaVA-Med [30] connect an image encoder directly to an LLM; this end-to-end design is powerful but harder to audit and evaluate per modality."),
  H2("2.4 Triage scores"),
  P("The National Early Warning Score 2 (NEWS2) [17] aggregates six physiological parameters (respiratory rate, oxygen saturation, systolic blood pressure, pulse, consciousness and temperature) into a validated risk score with defined escalation thresholds. It is deterministic and widely used, which makes it a natural, auditable basis for triage guidance instead of letting an LLM decide urgency."),
  H2("2.5 Explainability, calibration and robustness"),
  P("Occlusion sensitivity [18] explains a model by masking image regions and measuring the change in its output; it is model-agnostic. Modern neural networks are often poorly calibrated [20]; Platt scaling [19] is a simple post-hoc fix. Hendrycks and Dietterich [27] proposed benchmarking robustness under common corruptions such as noise, blur and compression."),
  H2("2.6 Research gap"),
  P("Existing work tends to evaluate image models, text models or chatbots in isolation. Few systems (a) keep safety-relevant decisions such as triage and conflict detection in transparent rules, (b) make every LLM claim traceable to a specific note phrase or image region in the interface, and (c) measure honestly whether adding text to images helps on real data. MedAssist addresses these three points with a late-fusion design that can be tested stage by stage."),
];

const archWords = [
  ["Ingestion & privacy", "FastAPI upload endpoints, pydicom, OpenCV, Presidio + custom recognisers, DICOM anonymiser"],
  ["Clinical NLP", "scispaCy `en_ner_bc5cdr_md`, medspaCy target rules + ConText + sectionizer"],
  ["Image analysis", "BiomedCLIP (open_clip), zero-shot + linear probe ensemble, Platt calibration, occlusion heatmaps"],
  ["Retrieval", "Sentence Transformers (PubMedBERT MS-MARCO), cosine vector index"],
  ["Reasoning", "Llama 3.1 8B via Ollama, LangChain prompt templates, streaming"],
  ["Interface", "React 19 + Vite + TypeScript, server-sent events, SVG leader lines"],
  ["Evaluation", "scikit-learn, matplotlib, pytest (80 tests)"],
];

const ch3 = [
  ...H1("System Requirements and Design", 3),
  H2("3.1 Functional requirements"),
  bullet("Upload notes and images; de-identify before anything is stored; reject unsupported or oversized files."),
  bullet("Extract problems, medications, tests and vital signs with negation and history context."),
  bullet("Classify chest X-ray findings with calibrated probabilities, heatmaps and similar prior cases; refuse to classify non-chest images."),
  bullet("Answer free-text questions about a case with cited evidence; show triage level, its reasons and any note–image conflicts."),
  bullet("Suggest follow-up questions and information that would improve the assessment."),
  H2("3.2 Non-functional requirements"),
  bullet("**Privacy:** all inference local; raw uploads never written to disk; audit log without content."),
  bullet("**Transparency:** safety-relevant outputs computed by rules; every claim citable."),
  bullet("**Latency:** findings, conflicts and triage within about one second; written answers streamed."),
  bullet("**Hardware:** run on a 6 GB laptop GPU."),
  bullet("**Reproducibility:** every reported number regenerated by one script from saved results."),
  H2("3.3 Architecture"),
  P("MedAssist uses **late fusion**: each modality is processed by a specialist model, and their outputs are combined in an explicit evidence bundle rather than inside one neural network. This keeps each stage testable and lets the system explain exactly which evidence drove each output (Figure 3.1)."),
  ...figure("architecture.png", "System architecture of MedAssist", 6.2),
  H2("3.4 Design principles"),
  numbered2("**Rules decide, the LLM explains.** Triage levels and conflicts are computed deterministically and passed to the LLM as facts, so safety-relevant outputs are reproducible and unit-testable."),
  numbered2("**Evidence before answers.** The interface shows findings, conflicts and triage within a second; the LLM's written answer streams in afterwards."),
  numbered2("**Honest uncertainty.** Probabilities are calibrated and shown as bands (unlikely / possible / likely); the assistant says “not detected by the classifier”, never “ruled out”."),
  numbered2("**No label leakage.** Model inputs never include the radiology report from which evaluation labels are derived."),
  H2("3.5 Technology stack"),
  ...table("Technology stack by layer", ["Layer", "Technologies"], archWords, [1, 3]),
  H2("3.6 Hardware and environment"),
  P("Development and all experiments ran on a laptop with an NVIDIA RTX 4050 (6 GB VRAM), 16 GB RAM, Windows 11, Python 3.11, PyTorch 2.5.1 with CUDA 12.4, and Ollama serving Llama 3.1 8B (4-bit). With a 6 GB GPU about 20% of the LLM runs on the CPU, which dominates answer latency."),
];

const ch4 = [
  ...H1("Implementation", 4),
  H2("4.1 Ingestion and privacy"),
  P("Text de-identification uses Microsoft Presidio [24] with spaCy `en_core_web_lg`, extended with custom recognisers for medical record numbers and UHIDs, Aadhaar numbers, Indian mobile numbers, ages over 89 and healthcare facility names, plus an allow-list of medical eponyms (e.g. Parkinson, Murphy sign). Placeholders such as `[PERSON]` or `[MRN]` replace identifiers, and only the identifier *type* is logged."),
  P("**A safety finding from testing.** Presidio's generic date detector redacted the word “No” in “No night sweats”, deleting the negation and turning an absent symptom into a present one; it also removed ages and a drug name. Redaction was therefore restricted to calendar dates, which is what HIPAA Safe Harbor actually requires [23], with regression tests. A later held-out evaluation exposed that hospital names were never removed; a case-sensitive facility recogniser fixed this without over-redacting phrases such as “admitted to hospital”."),
  P("DICOM files are anonymised following a subset of the PS3.15 basic profile [25]: patient, staff and institution tags are removed, dates are reduced to the year, UIDs are replaced, private tags are stripped, and a warning is raised when burned-in text is likely. Pixel data are converted with the modality and VOI look-up tables, so 12- and 16-bit radiographs are windowed correctly."),
  H2("4.2 Clinical NLP"),
  P("Entities are extracted with scispaCy's BC5CDR model [7] (diseases and chemicals), extended with medspaCy target rules [8] for symptoms, tests, procedures and 30 radiology terms that BC5CDR misses (e.g. “lung opacity”, “nodule”). The ConText algorithm [9] marks each entity as negated, historical, hypothetical or family. A custom pipeline component trims cue words that the NER model sometimes absorbs into entities (“Denies chest pain” tagged as one disease span), which would otherwise hide the negation. Vital signs are extracted with regular expressions and normalised."),
  H2("4.3 Image analysis with BiomedCLIP"),
  P("A **modality gate** first runs BiomedCLIP zero-shot over nine modality labels; the chest X-ray classifier runs only when the image is recognised as a chest X-ray. Images are letterboxed to a square so that lung edges are never cropped."),
  P("For each of nine findings (cardiomegaly, lung opacity, atelectasis, pleural effusion, granuloma, nodule, emphysema, pulmonary edema, fracture), **zero-shot** scoring compares the image embedding with an ensemble of positive and negative prompts. A **linear probe** (logistic regression per finding on frozen image features) is trained on the training split. The deployed **ensemble** averages the two logits with equal weight and is **Platt-calibrated** [19] per finding on out-of-fold training predictions, so that a displayed 0.9 means roughly 90%."),
  P("Explanations use **occlusion sensitivity** [18]: about 170 masked copies of the image are scored in batches on the GPU, and regions whose masking lowers the finding score are drawn as a single-hue vermilion overlay. **Similar-case retrieval** finds the most similar training X-rays by image embedding and shows their reports as context."),
  H2("4.4 Retrieval-augmented question answering"),
  P("Notes are split into sentence-aware chunks of about 600 characters with one sentence of overlap, embedded with a Sentence Transformer and stored in a cosine-similarity index. Four encoders were compared on PubMedQA retrieval (Chapter 6); PubMedBERT fine-tuned on MS-MARCO was selected. Retrieved passages are numbered and passed to Llama through a LangChain `prompt | ChatOllama | parser` chain that requires bracketed citations."),
  H2("4.5 Fusion, conflicts and triage"),
  P("For every finding the image probability is updated with text evidence as a log-odds shift:"),
  C("p_fused = sigmoid( logit(p_image) + Σ LLR_text )", { run: { italics: true } }),
  P("The log-likelihood ratios are hand-set and documented: +2.0 when the note asserts the finding, −1.5 when it negates it, and +0.4 per supporting symptom or abnormal vital sign (capped at +1.2). A documented history of a chronic cause (heart failure, COPD, tuberculosis) counts as an assertion for chronic findings only."),
  P("**Conflict detection** compares the two modalities: the note asserts a finding that the image does not show; the image shows a finding the note explicitly denies (high severity); strong clinical signs without an image correlate; a confident image finding the note never mentions; disagreement between notes; and an image of the wrong modality."),
  P("**Triage** combines NEWS2 [17] (computed when at least three vitals are recorded) with red-flag rules (stroke signs, reduced consciousness, haemoptysis, syncope, new confusion, chest pain with breathlessness) and an escalation rule for an acute X-ray finding with SpO₂ below 94%. The most severe level wins: EMERGENCY, URGENT, SOON or ROUTINE."),
  ...table("NEWS2 scoring bands implemented (room air)", ["Parameter", "3", "2", "1", "0", "1", "2", "3"], [
    ["Respiratory rate (/min)", "≤8", "", "9–11", "12–20", "", "21–24", "≥25"],
    ["SpO₂ (%)", "≤91", "92–93", "94–95", "≥96", "", "", ""],
    ["Systolic BP (mmHg)", "≤90", "91–100", "101–110", "111–219", "", "", "≥220"],
    ["Pulse (/min)", "≤40", "", "41–50", "51–90", "91–110", "111–130", "≥131"],
    ["Consciousness", "", "", "", "Alert", "", "", "New confusion"],
    ["Temperature (°C)", "≤35.0", "", "35.1–36.0", "36.1–38.0", "38.1–39.0", "≥39.1", ""],
  ], [2.4, 1, 1, 1, 1, 1, 1, 1.2]),
  H2("4.6 LLM orchestration"),
  P("A keyword router classifies each question as text, image, multimodal or triage, with a fallback when a modality is missing. The orchestrator de-identifies the question, gathers the evidence bundle, applies the rules, and fills a route-specific prompt. A shared system prompt requires the model to use only the evidence block, cite every claim, separate observations from interpretations, report the computed triage level unchanged and never state a diagnosis. Answers are streamed to the client as server-sent events: the evidence arrives first, then tokens, then the parsed answer with its citations. If the LLM is unavailable, the deterministic results are still returned."),
  H2("4.7 Backend API"),
  P("FastAPI exposes endpoints to create and list cases, upload notes and images, analyse an image, render a heatmap, fetch a deterministic case overview, ask a question (with or without streaming) and retrieve the de-identified chat history (Appendix B). The same server serves the built web interface, so the demo starts with a single command."),
  H2("4.8 User interface"),
  P("The interface was designed with an explicit direction (an anatomy-atlas “plate”) after a short product interview: light theme for projectors in lit rooms, the Atkinson Hyperlegible typeface for legibility, and a colour code for evidence provenance (vermilion for image evidence, blue for note evidence). The case band takes the triage colour; conflicts sit directly below the triage reasons; findings are shown as reference-band bars with the image and note-adjusted probabilities. The signature interaction is the **leader line**: hovering a citation or a finding draws a line to the exact evidence (the image, a note phrase, the vitals, a similar case or a conflict)."),
  ...figure(path.join("ui", "workspace_pneumonia.png"), "Case workspace: triage band, X-ray with effusion heatmap, de-identified note, findings and a cited answer", 6.3),
  ...figure(path.join("ui", "conflict_case.png"), "A HIGH note–image conflict, with leader lines from the pleural-effusion finding to the image and to the note phrases", 6.3),
  H2("4.9 Testing"),
  P("The code base has 80 automated tests (pytest) covering de-identification regressions, DICOM anonymisation, negation handling, NEWS2 against hand calculation, every conflict type, fusion arithmetic, the streaming protocol, the API and the evaluation tooling itself (for example SUS scoring and the answer-faithfulness scorer), so that reported numbers cannot silently be wrong."),
];

// ---------- Chapter 5
const ch5 = [
  ...H1("Datasets and Experimental Setup", 5),
  H2("5.1 IU X-Ray"),
  P("The Indiana University Chest X-ray Collection [4] pairs frontal and lateral radiographs with radiology reports and NLM-assigned MeSH terms. One frontal image per study was used. MeSH terms were mapped to nine findings plus “normal”, and studies marked as technically unsatisfactory were excluded. A stratified subset of 1,500 studies (guaranteeing rare findings) was split 70/30 with a fixed seed into 1,065 training studies (probe training and retrieval knowledge base) and **435 test studies**."),
  P("**Leakage rule.** Model inputs may use the image and the short clinical *indication* only. The findings and impression sections are the source of the labels, so they are used only as the retrieval corpus of *other* patients' reports and never as input for the same study."),
  H2("5.2 PubMedQA"),
  P("The expert-labelled subset of PubMedQA [15] (1,000 questions, each with an abstract and a yes/no/maybe answer) was used to compare retrieval encoders over all 1,000 abstracts and to measure answer accuracy for a random sample of 150 questions with closed-book, retrieved-context and gold-context prompts."),
  H2("5.3 Synthetic clinical notes"),
  P("No public dataset pairs realistic clinical notes containing personal identifiers with images. Three synthetic sets were therefore used, all with fictional people: six hand-written notes used during development; a **held-out set of 120 generated notes** created after the rules were frozen, with different templates, name lists and identifier formats; and seven demo cases pairing synthetic notes with real test-set X-rays, used only for demonstration and for the answer-faithfulness study."),
  H2("5.4 Metrics"),
  ...table("Evaluation metrics", ["Aspect", "Metric"], [
    ["Image findings", "Per-finding and macro AUROC, macro F1 (thresholds from training data only), abnormal-vs-normal AUROC, ECE and Brier score"],
    ["Retrieval", "Recall@K (exact), label precision@K for similar cases vs random"],
    ["Question answering", "Accuracy and macro F1 (yes / no / maybe)"],
    ["Fusion", "Macro AUROC of image-only, text-only and fused scores; paired bootstrap 95% CI (1,000 resamples)"],
    ["Conflicts", "Detection rate of injected discordances; alert rate without injection"],
    ["Robustness", "Macro AUROC under 6 corruption types × 3 severities [27]; typo robustness of text pipeline"],
    ["Privacy", "PHI recall per identifier type, token leak rate, clinical text preserved"],
    ["Explanation faithfulness", "Citation coverage and validity, ungrounded-number rate, triage consistency, conflict disclosure, unsafe certainty phrases"],
    ["Latency / usability", "Seconds per stage; System Usability Scale [21] with adjective rating [22]"],
  ], [1.2, 3]),
];

// ---------- Chapter 6 (data-driven)
const ch6 = [...H1("Results and Discussion", 6)];
if (img) {
  ch6.push(H2("6.1 Chest X-ray finding classification"),
    ...table("Image-only classification on IU X-Ray test (n = 435)", ["Variant", "Macro AUROC", "Macro F1", "Abnormal vs normal AUROC"],
      img.results.map((r) => [r.variant, f3(r.summary.macro_auroc), f3(r.summary.macro_f1), f3(r.summary.abnormal_vs_normal_auroc)]), [2.2, 1, 1, 1.4]),
    P(`The calibrated ensemble is the best variant overall (macro AUROC ${f3(ens?.summary?.macro_auroc)}). Two findings are useful for the discussion: OpenCV contrast enhancement (CLAHE) slightly **hurts** zero-shot BiomedCLIP, which was trained on unenhanced figures, so it is disabled; and the zero-shot and probe models have **complementary strengths**: the probe is far better on emphysema (${f3(zs?.per_finding?.emphysema?.auroc)} → ${f3(ens?.per_finding?.emphysema?.auroc)} in the ensemble) but overfits rare findings, which is why their average beats both.` +
      (calib ? ` Platt scaling reduced the expected calibration error from ${calib.ece_raw} to **${calib.ece_calibrated}** (Brier ${calib.brier_raw} → ${calib.brier_calibrated}) without changing AUROC.` : "")),
    ...figure("per_finding_auroc.png", "Per-finding AUROC, zero-shot versus calibrated ensemble", 6.0),
    P("Large findings (effusion, cardiomegaly, edema) reach AUROC 0.87–0.91, whereas small focal findings (nodule, granuloma, fracture) remain weak at 0.56–0.62, which is expected at 224-pixel input resolution and is reported as a limitation."));
}
if (res.retrieval) {
  const s = res.retrieval.similar_case_label_precision, e = res.retrieval.exact_report_retrieval;
  ch6.push(H2("6.2 Similar-case retrieval"),
    ...table("Similar-case retrieval: share of retrieved cases sharing a finding with the query", ["Route", "P@1", "P@5", "P@10"],
      [...Object.entries(s).filter(([, v]) => typeof v === "object").map(([k, v]) => [k, f3(v["label_precision@1"]), f3(v["label_precision@5"]), f3(v["label_precision@10"])]),
        ["random", f3(s.random), f3(s.random), f3(s.random)]], [2, 1, 1, 1]),
    P(`Image-to-image retrieval returns a case sharing a finding ${pct(s["image->image"]["label_precision@1"], 0)} of the time against ${pct(s.random, 0)} for random retrieval. Retrieving a study's exact report is near chance (Recall@10 ${f3(e.model["recall@10"])} vs ${f3(e.random["recall@10"])}) because IU reports are highly templated (about 40% are near-identical normal reports), so label precision is the meaningful metric.`));
}
if (res.qa) {
  const q = res.qa;
  ch6.push(H2("6.3 Text retrieval and question answering"),
    ...table("PubMedQA retrieval over 1,000 abstracts", ["Embedding model", "R@1", "R@3", "R@10"],
      Object.entries(q.retrieval).map(([k, v]) => [k, f3(v["recall@1"]), f3(v["recall@3"]), f3(v["recall@10"])]), [2.4, 1, 1, 1]),
    P("Domain pre-training alone does not produce a good retriever: raw ClinicalBERT is nearly useless for search, while models fine-tuned for retrieval (PubMedBERT MS-MARCO and even a small general-domain model) are best. PubMedBERT MS-MARCO was therefore selected."),
    ...table("PubMedQA answer accuracy with Llama 3.1 8B (150 questions)", ["Mode", "Accuracy", "Macro F1"],
      Object.entries(q.qa).map(([k, v]) => [k, f3(v.accuracy), f3(v.macro_f1)]), [2.4, 1, 1]),
    ...figure("rag_accuracy.png", "Retrieval-augmented generation versus closed-book answering", 5.6),
    P(`Retrieval raises accuracy by ${((q.qa["RAG (top-3)"].accuracy - q.qa["closed-book"].accuracy) * 100).toFixed(1)} points and recovers most of the gap to the gold-context upper bound. Closed-book accuracy is below the majority-class baseline: without evidence, the model answers confidently but often wrongly, which directly motivates evidence-grounded answers.`));
}
if (res.fusion) {
  ch6.push(H2("6.4 Multimodal fusion versus single modalities"),
    ...table("Macro AUROC of image-only, text-only and fused scores (IU X-Ray test, n = 435)", ["Setting", "Image only", "Text only", "Fused", "Fused − image (95% CI)"], [
      ["A. Real indication text", f3(fA.macro_auroc.image_only), f3(fA.macro_auroc.text_only), f3(fA.macro_auroc.fused), `${fA.fused_minus_image_ci95[0]} to ${fA.fused_minus_image_ci95[1]}`],
      ["B. Controlled simulation", f3(fB.macro_auroc.image_only), f3(fB.macro_auroc.text_only), f3(fB.macro_auroc.fused), `${fB.fused_minus_image_ci95[0]} to ${fB.fused_minus_image_ci95[1]}`],
    ], [2.2, 1, 1, 1, 1.6]),
    ...figure("fusion_ablation.png", "Fusion ablation on real data and in a controlled simulation", 5.8),
    P(`On **real data**, the text is each study's actual indication (“chest pain”, “preop”): it carries little information (text-only AUROC ${f3(fA.macro_auroc.text_only)}${res.textCxr ? `; an LLM reading the same text reached ${f3(res.textCxr.summary.macro_auroc)}` : ""}), so fusion yields a small but statistically significant gain. In a **controlled simulation**, each study received a synthetic note generated from its labels by a documented noisy symptom model; there the fused score beats *both* single modalities. Because the strength of the simulated text is set by the experimenter, B demonstrates that the fusion mechanism combines complementary evidence correctly; it is not clinical evidence.`),
    H2("6.5 Conflict detection"),
    P(`Discordances were injected into 15% of simulated notes (“no X” where the image shows X, “known X” where it does not). The rules detected ${fC.detected}/${fC.injected} overall, but **${pct(fC.detection_rate_when_image_agrees_with_truth, 0)}** of those where the image classifier itself agreed with the ground truth: every miss was caused by the image model not seeing a subtle finding, not by the conflict logic. Notes without injected conflicts triggered an alert ${pct(fC.alert_rate_without_injection, 0)} of the time, mostly true findings that the classifier missed.`));
}
if (res.robust) {
  const rb = res.robust;
  ch6.push(H2("6.6 Robustness of the image classifier"),
    ...table("Macro AUROC under image corruptions (clean = " + f3(rb.clean.macro_auroc) + ")", ["Corruption", "Level 1", "Level 2", "Level 3"],
      Object.entries(rb.corruptions).map(([k, lv]) => [k, ...lv.map((x) => `${f3(x.macro_auroc)} (${x.severity})`)]), [1.4, 1.4, 1.4, 1.4]),
    ...figure("robustness.png", "Classifier performance as corruption severity increases", 6.2),
    P("At realistic levels (moderate noise, slight blur, rotation up to 10°, halved contrast, JPEG quality 20, 128-pixel resolution) macro AUROC drops by at most 0.017. Performance degrades substantially only at extremes: heavy noise and JPEG quality 5 lose about 0.10, which suggests warning users when an uploaded image is very low quality."));
}
if (res.held) {
  ch6.push(H2("6.7 De-identification and clinical NLP on held-out notes"),
    ...table("De-identification recall by identifier type (120 held-out notes)", ["Identifier type", "n", "Recall"],
      Object.entries(held0.phi_recall_by_type).map(([k, v]) => [k, v.n, f3(v.recall)]), [2, 1, 1]),
    ...table("Text pipeline under injected typos", ["Words with typo", "PHI recall", "Token leak", "Clinical text kept", "Finding recall", "Negation F1"],
      res.held.runs.map((r) => [`${(r.typo_rate * 100).toFixed(0)}%`, f3(r.phi_recall), f3(r.phi_token_leak_rate), f3(r.clinical_text_kept), f3(r.active_recall), f3(r.negation_f1)]), [1.2, 1, 1, 1.2, 1, 1]),
    ...figure("text_typo_robustness.png", "Text pipeline metrics as typos are injected", 5.8),
    P(`On the held-out set, ${pct(held0.phi_recall)} of planted identifiers were removed and ${pct(held0.clinical_text_kept)} of clinical text was preserved; negation was detected perfectly. ` +
      (res.heldPre ? `**Methodological note:** the first held-out (discovery) set revealed that hospital names were never removed (${pct(res.heldPre.runs[0].phi_recall_by_type.facility?.recall ?? 0, 0)}; overall ${pct(res.heldPre.runs[0].phi_recall)}). After the fix, the numbers above were measured on a **freshly generated** set, so the fix is not scored on the data that exposed the problem. ` : "") +
      "Generated notes are more regular than real ones, so real-world recall is likely lower; a larger real de-identified corpus is future work."));
}
if (ans) {
  ch6.push(H2("6.8 Answer faithfulness and latency"),
    P("Twenty-eight real answers (seven demo cases × four question types) were scored automatically. No LLM-as-judge was used, to avoid the model grading itself."),
    ...table("Faithfulness of generated answers", ["Metric", "Value"], [
      ["Routing accuracy", pct(ans.routing_accuracy, 0)], ["Format compliance", pct(ans.format_compliance, 0)],
      ["Citation validity (cited ids exist)", pct(ans.citation_validity)], ["Citation coverage (claims with a citation)", pct(ans.citation_coverage)],
      ["Ungrounded numbers", pct(ans.ungrounded_number_rate)], ["Triage level stated (triage questions)", pct(ans.triage_level_stated, 0)],
      ["Triage contradictions", ans.triage_contradictions], ["Conflicts disclosed (cases with a high/medium conflict)", pct(ans.conflict_disclosure)],
      ["Answers with unsafe certainty phrases", ans.answers_with_unsafe_phrases],
      ["Median latency: evidence / LLM (s)", `${ans.latency_s.evidence_median} / ${ans.latency_s.llm_median}`],
    ], [3, 1.2]),
    P("Answers almost never invent numbers and always report the rule-computed triage level. The main weakness is **citation coverage**: only about half of claims carry a citation (the metric is strict, because a citation at the end of a multi-sentence bullet credits only its last sentence). One answer described a temperature of 37.9 °C as “normal”, a small factual slip that illustrates why the interface keeps the source evidence visible. Written answers take a median of about 16 seconds on the 6 GB GPU (text questions about 11 s, multimodal about 27 s), while findings, conflicts and triage are available almost instantly."));
}
ch6.push(H2("6.9 Usability"));
if (res.study) {
  const a = res.study.all.sus;
  ch6.push(P(`The System Usability Scale score was **${a.mean}** (SD ${a.sd}, n = ${a.n}), rated “${a.adjective}” on the Bangor scale [22].`));
} else {
  ch6.push(P("[To be completed after the user study: follow `docs/user_study/protocol.md` with 10–15 participants, run `python -m scripts.evaluate.user_study` and `python -m scripts.report.make_report`, then rebuild this report. Report n, the mean SUS score with its 95% confidence interval and adjective rating, task success rates, and the explanation-quality items.]"));
}
ch6.push(H2("6.10 Discussion"),
  P("Three results stand out. First, **grounding matters more than model size** for a local 8B model: retrieval lifted accuracy by over 20 points, and the faithfulness study shows answers stay close to the evidence they are given. Second, **multimodal gains depend on informative text**: real IU indications are too short for fusion to add much, yet with informative notes fusion clearly beats both single modalities, which argues for richer clinical context in future datasets rather than against fusion. Third, keeping **safety-relevant logic in rules** made it measurable: conflict rules were perfect whenever the image model was right, so improving the image model directly improves conflict detection."));

const ch7 = [
  ...H1("Safety, Privacy and Ethics", 7),
  bullet("**Local processing:** all models run on the user's machine; no patient data is sent to external services."),
  bullet("**De-identification before storage:** raw uploads exist only in memory; original filenames are discarded; the audit log records actions and identifier types, never content."),
  bullet("**Clinical safety:** triage and conflict detection are deterministic and tested; the LLM may only explain them. Every answer carries a decision-support disclaimer, and the prompt forbids diagnostic certainty (no unsafe phrases occurred in the evaluation)."),
  bullet("**Modality gate:** the chest X-ray classifier refuses non-chest images instead of producing meaningless findings."),
  bullet("**Honest reporting:** caveats are stated next to every result: synthetic notes, small samples and simulation versus real data."),
  bullet("**Data and licences:** only public de-identified images and fictional notes were used; datasets and models are credited in the references."),
  bullet("**Not a medical device:** the system is not validated for clinical use and must not be used for real patient care."),
];

const ch8 = [
  ...H1("Conclusion and Future Work", 8),
  H2("8.1 Conclusion"),
  P("MedAssist demonstrates that a multimodal clinical assistant can run entirely on a laptop while remaining transparent. It de-identifies inputs, combines a calibrated BiomedCLIP image classifier with structured clinical text, detects disagreements between the two, computes triage with NEWS2, and uses a local Llama model only to explain the evidence, with every claim traceable in the interface. The evaluation shows strong performance on large chest X-ray findings, a large benefit from retrieval, faithful answers, and robustness to realistic image and text noise, together with clearly reported limitations."),
  H2("8.2 Limitations"),
  bullet("Weak detection of small focal findings (nodules, granulomas, fractures) at 224-pixel resolution."),
  bullet("Real paired notes are unavailable, so the fusion benefit on rich clinical text is shown only in simulation."),
  bullet("De-identification is evaluated on generated notes, which are more regular than real records."),
  bullet("Fusion weights are hand-set rather than learned; citation coverage of answers is about 50%."),
  bullet("Answer latency is bounded by an 8B model on a 6 GB GPU."),
  H2("8.3 Future work"),
  bullet("Fine-tune the image encoder at higher resolution on larger datasets (CheXpert, MIMIC-CXR) for small findings."),
  bullet("Learn fusion weights from paired notes and images (e.g. MIMIC-IV notes linked to MIMIC-CXR, with credentialed access)."),
  bullet("Enforce citations per sentence with constrained decoding or a verifier step."),
  bullet("Add CT and other modalities, and a medical knowledge base (e.g. MedQuAD) for general questions."),
  bullet("Conduct a clinician user study and prospective evaluation with institutional approval."),
];

const refs = [
  "A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proc. ICML, 2021.",
  "S. Zhang et al., “BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs,” arXiv:2303.00915, 2023.",
  "E. Tiu et al., “Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning,” Nature Biomedical Engineering, vol. 6, pp. 1399–1406, 2022.",
  "D. Demner-Fushman et al., “Preparing a collection of radiology examinations for distribution and retrieval,” J. Am. Med. Inform. Assoc., vol. 23, no. 2, pp. 304–310, 2016.",
  "J. Irvin et al., “CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison,” in Proc. AAAI, 2019.",
  "X. Wang et al., “ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases,” in Proc. CVPR, 2017.",
  "M. Neumann, D. King, I. Beltagy and W. Ammar, “ScispaCy: Fast and robust models for biomedical natural language processing,” in Proc. BioNLP Workshop, 2019.",
  "H. Eyre et al., “Launching into clinical space with medspaCy: a new clinical text processing toolkit in Python,” in AMIA Annual Symposium Proceedings, 2021.",
  "H. Harkema, J. N. Dowling, T. Thornblade and W. W. Chapman, “ConText: An algorithm for determining negation, experiencer, and temporal status from clinical reports,” J. Biomed. Inform., vol. 42, no. 5, pp. 839–851, 2009.",
  "J. Lee et al., “BioBERT: a pre-trained biomedical language representation model for biomedical text mining,” Bioinformatics, vol. 36, no. 4, pp. 1234–1240, 2020.",
  "E. Alsentzer et al., “Publicly available clinical BERT embeddings,” in Proc. Clinical NLP Workshop, 2019.",
  "Y. Gu et al., “Domain-specific language model pretraining for biomedical natural language processing,” ACM Trans. Comput. Healthcare, vol. 3, no. 1, 2021.",
  "N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in Proc. EMNLP-IJCNLP, 2019.",
  "P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in Proc. NeurIPS, 2020.",
  "Q. Jin et al., “PubMedQA: A dataset for biomedical research question answering,” in Proc. EMNLP-IJCNLP, 2019.",
  "A. Dubey et al. (Llama Team, Meta AI), “The Llama 3 herd of models,” arXiv:2407.21783, 2024.",
  "Royal College of Physicians, National Early Warning Score (NEWS) 2: Standardising the assessment of acute-illness severity in the NHS. London: RCP, 2017.",
  "M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional networks,” in Proc. ECCV, 2014.",
  "J. Platt, “Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,” in Advances in Large Margin Classifiers, MIT Press, 1999.",
  "C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, 2017.",
  "J. Brooke, “SUS: A ‘quick and dirty’ usability scale,” in Usability Evaluation in Industry, Taylor & Francis, 1996.",
  "A. Bangor, P. Kortum and J. Miller, “Determining what individual SUS scores mean: Adding an adjective rating scale,” J. Usability Studies, vol. 4, no. 3, pp. 114–123, 2009.",
  "U.S. Department of Health and Human Services, “Guidance regarding methods for de-identification of protected health information in accordance with the HIPAA Privacy Rule,” 2012.",
  "Microsoft, “Presidio: Data protection and de-identification SDK,” open-source software, https://github.com/microsoft/presidio.",
  "NEMA, “Digital Imaging and Communications in Medicine (DICOM) PS3.15: Security and System Management Profiles, Annex E: Attribute Confidentiality Profiles.”",
  "Government of India, The Digital Personal Data Protection Act, 2023.",
  "D. Hendrycks and T. Dietterich, “Benchmarking neural network robustness to common corruptions and perturbations,” in Proc. ICLR, 2019.",
  "LangChain, open-source framework, https://github.com/langchain-ai/langchain; Ollama, https://ollama.com.",
  "K. Singhal et al., “Large language models encode clinical knowledge,” Nature, vol. 620, pp. 172–180, 2023.",
  "C. Li et al., “LLaVA-Med: Training a large language-and-vision assistant for biomedicine in one day,” in Proc. NeurIPS Datasets and Benchmarks, 2023.",
];
const references = [
  ...H1("References", 0),
  ...refs.map((r, i) => new Paragraph({ children: runs(`[${i + 1}] ${r}`, { size: 22 }), spacing: { after: 100 }, indent: { left: 567, hanging: 567 } })),
];

const appendix = [
  ...H1("Appendix A: Running the Project", 0, "A"),
  P("Requirements: Python 3.11, an NVIDIA GPU (recommended), Ollama, Node 18+ (to build the interface once). From the repository root:"),
  ...[
    "py -3.11 -m venv .venv && .venv\\Scripts\\activate",
    "pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124",
    "pip install -r requirements.txt && ollama pull llama3.1:8b",
    "python -m scripts.data.prepare_iu_xray --n 1500",
    "python -m scripts.evaluate.image --train-probe && python -m scripts.evaluate.retrieval",
    "python -m scripts.data.load_demo_cases --analyze --reset",
    "npm --prefix ui install && npm --prefix ui run build",
    "uvicorn api.main:app --port 8000   (open http://127.0.0.1:8000)",
  ].map((t) => new Paragraph({ children: [new TextRun({ text: t, font: "Consolas", size: 19 })], spacing: { after: 60 } })),
  P("All experiments: `evaluate.image`, `evaluate.retrieval`, `evaluate.text_qa`, `evaluate.text_only_cxr`, `evaluate.fusion`, `evaluate.robustness`, `evaluate.heldout_text`, `evaluate.answers`; then `python -m scripts.report.make_report` regenerates every table and figure. `pytest -q` runs the 80 tests."),
  ...H1("Appendix B: API Endpoints", 0, "B"),
  ...table("REST API", ["Method and path", "Purpose"], [
    ["GET /cases", "List cases"], ["POST /cases", "Create a case"], ["GET /cases/{id}", "Case with de-identified notes and images"],
    ["POST /cases/{id}/notes", "Upload or paste a note (de-identified, entities extracted)"], ["POST /cases/{id}/images", "Upload DICOM/PNG/JPEG (anonymised)"],
    ["POST /cases/{id}/images/{image_id}/analyze", "Modality, findings, similar cases"], ["GET /cases/{id}/images/{image_id}/heatmap", "Occlusion heatmap for one finding"],
    ["GET /cases/{id}/overview", "Fused findings, conflicts, triage without the LLM"], ["POST /cases/{id}/chat", "Full multimodal answer"],
    ["POST /cases/{id}/chat/stream", "Same, streamed as server-sent events"], ["GET /cases/{id}/chat", "De-identified conversation history"],
  ], [2.2, 3]),
  ...H1("Appendix C: Demo Cases", 0, "C"),
  ...table("Demo cases (synthetic notes on real IU X-Ray test images)", ["Case", "What it demonstrates"], [
    ["Fever, productive cough, hypoxia", "Concordant pneumonia with effusion; EMERGENCY (NEWS2 7)"],
    ["Breathless with leg swelling", "Cardiomegaly and edema supported by heart-failure history; URGENT"],
    ["Note says no effusion, X-ray disagrees", "HIGH conflict between note and image"],
    ["Note says pneumonia, X-ray looks normal", "Conflict: asserted finding not seen on the image"],
    ["COPD flare with low oxygen", "Emphysema plus COPD history; EMERGENCY (NEWS2 8)"],
    ["Pre-employment check", "All normal; ROUTINE"],
    ["Chest symptoms but a brain MRI uploaded", "Modality gate and mismatch warning"],
  ], [2, 3]),
];

// ------------------------------------------------------------------ assemble
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] });
const pageProps = (fmt, start) => ({
  page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 1440, bottom: 1440, left: MARGIN_L, right: MARGIN_R },
    pageNumbers: { start, formatType: fmt } },
});

const doc = new Document({
  creator: "[Student Name]",
  title: "MedAssist: A Multimodal Medical Assistant for Clinical Decision Support",
  description: "B.Tech capstone project report",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true, color: "14223B" },
        paragraph: { spacing: { before: 0, after: 360 }, alignment: AlignmentType.CENTER, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: "14223B" },
        paragraph: { spacing: { before: 300, after: 140 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true },
        paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 567, hanging: 283 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 567, hanging: 340 } } } }] },
    ],
  },
  sections: [
    { properties: pageProps(NumberFormat.LOWER_ROMAN, 1), children: title },
    { properties: pageProps(NumberFormat.LOWER_ROMAN, 2), footers: { default: footer },
      children: [...certificate, ...abstractText, ...toc] },
    { properties: pageProps(NumberFormat.DECIMAL, 1), footers: { default: footer },
      children: [...ch1, ...ch2, ...ch3, ...ch4, ...ch5, ...ch6, ...ch7, ...ch8, ...references, ...appendix] },
  ],
});

const out = path.join(__dirname, "MedAssist_Project_Report.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(out, buf);
  console.log(`Wrote ${out} (${(buf.length / 1024).toFixed(0)} KB)`);
});
