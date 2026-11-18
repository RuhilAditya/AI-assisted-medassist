# Surface brief: case workspace (ui/)

Scope: the whole MedAssist web app (case list + case workspace + new-case flow). Mode: Operate.
Audience/job: a clinician reviewing one case (note + chest X-ray) under time pressure; examiners watch it on a projector in a lit room and on a laptop. Task: see urgency, image findings, text/image agreement, ask questions, verify claims.
Constraints: LLM answers take ~30 s; deterministic evidence arrives in ~1 s. Light ground (lit rooms, projector). Ruled out: generic AI dashboard, toy/flashy, cold grey EHR, hidden reasoning.

## Direction contract

THESIS: An anatomy-atlas plate for one patient. Every claim is labelled the way a plate labels anatomy, with a fine leader line from label to structure. Refuses the category default of a chatbot pane beside stat cards.

OWN-WORLD: Cool plate-white ground (never cream), ink-navy type, a cool blue-grey second neutral for the rail. Two annotation inks taken from the atlas convention (arterial red, venous blue): vermilion marks image evidence, plate blue marks note evidence. Triage owns one committed colour field (raised from the Triage-tag direction): the case header band takes the triage hue. Probabilities are drawn as reference-band bars, unlikely | possible | likely, with tabular numerals (raised from the Lab-report direction). One workhorse family, Atkinson Hyperlegible Next (built for legibility: projector, low vision), and its Mono only for measured values. Hairline rules, no cards-in-cards, no gradients or glass.

STORY: The clinician sees how urgent the case is and why, then what the image suggests and whether the note agrees, then asks; each answer sentence can be traced to its source.

FIRST VIEWPORT: Left rail with the case list (title, triage dot). Header band in the triage colour with the case title, level and NEWS2. Three work columns: the X-ray plate with heatmap toggle and the de-identified note beneath it; the evidence column with findings as band bars, conflicts, triage reasons; the conversation with the cited answer, follow-up chips and the ask box at the bottom.

FORM: Operate product shell (standard rail + header + columns + standard controls), 1st of 4 derived directions. seed: none (unavailable). Impeccable engine not run; directions derived manually and chosen by the user through the structured tool. Signature interaction: hovering a citation chip or a finding's evidence draws a leader line (SVG cubic, drawn in 180 ms) in the evidence's ink to the exact target: heatmap, note phrase, conflict, similar case. Motion grammar: state only, 150-250 ms, exponential ease-out.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
