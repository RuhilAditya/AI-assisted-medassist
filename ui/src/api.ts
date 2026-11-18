// Typed client for the MedAssist FastAPI backend.

export type TriageLevel = "EMERGENCY" | "URGENT" | "SOON" | "ROUTINE";
export type Level = "likely" | "possible" | "unlikely";

export interface CaseSummary {
  case_id: string;
  created: string;
  title: string | null;
  tags: { demo?: string; shows?: string };
  notes: number;
  images: number;
}

export interface Entity {
  text: string;
  canonical: string;
  label: string;
  start: number;
  end: number;
  negated: boolean;
  historical: boolean;
  hypothetical: boolean;
  family: boolean;
}

export interface NoteRecord {
  note_id: string;
  text: string;
  deid: { counts: Record<string, number> };
  extraction: { entities: Entity[]; vitals: Record<string, string> };
  summary: Record<string, unknown>;
}

export interface ImageRecord {
  image_id: string;
  source_format: string;
  shape: number[];
  metadata: Record<string, string>;
  warnings: string[];
}

export interface CaseDetail {
  case_id: string;
  title: string | null;
  tags: CaseSummary["tags"];
  notes: NoteRecord[];
  images: ImageRecord[];
}

export interface TextSignal {
  term: string;
  llr: number;
  kind: "asserted" | "negated" | "supporting" | "vital";
  source: string;
}

export interface FusedFinding {
  finding: string;
  image_probability: number;
  image_positive: boolean;
  image_level: Level;
  text_llr: number;
  fused_probability: number;
  signals: TextSignal[];
}

export interface SimilarCase {
  uid: string;
  score: number;
  report: string;
  labels: string[];
}

export interface Evidence {
  image_id: string | null;
  image_analysis: {
    modality: { modality: string; confidence: number; is_chest_xray: boolean };
    similar_cases?: SimilarCase[];
    warnings?: string[];
    classifier_mode?: string;
  } | null;
  fused_findings: FusedFinding[];
  passages: { text: string; source: string; score: number }[];
  vitals: Record<string, string>;
}

export interface Conflict {
  kind: string;
  severity: "high" | "medium" | "info";
  finding: string | null;
  message: string;
  evidence: string[];
}

export interface Triage {
  level: TriageLevel;
  news2_score: number | null;
  news2_components: Record<string, number>;
  reasons: string[];
  limitations: string[];
}

export interface Overview {
  evidence: Evidence;
  conflicts: Conflict[];
  triage: Triage;
  follow_ups: { requests: string[]; next_questions: string[] };
  disclaimer: string;
}

export interface ChatResult extends Overview {
  question: string;
  route: { name: string; reason: string; notice: string | null };
  summary: string;
  details: string;
  citations: string[];
  llm_error: string | null;
  timings_s: Record<string, number>;
}

export interface HistoryTurn {
  time: string;
  question: string;
  summary: string;
  route: string;
  triage_level: TriageLevel;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export const api = {
  listCases: () => fetch("/cases").then((r) => json<CaseSummary[]>(r)),
  getCase: (id: string) => fetch(`/cases/${id}`).then((r) => json<CaseDetail>(r)),
  overview: (id: string) => fetch(`/cases/${id}/overview`).then((r) => json<Overview>(r)),
  history: (id: string) => fetch(`/cases/${id}/chat`).then((r) => json<HistoryTurn[]>(r)),
  imageUrl: (id: string, imageId: string) => `/cases/${id}/images/${imageId}`,
  heatmapUrl: (id: string, imageId: string, finding: string) =>
    `/cases/${id}/images/${imageId}/heatmap?finding=${encodeURIComponent(finding)}`,

  createCase: (title: string) =>
    fetch("/cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: title || null }),
    }).then((r) => json<{ case_id: string }>(r)),

  addNote: (id: string, note: { text?: string; file?: File }) => {
    const form = new FormData();
    if (note.file) form.append("file", note.file);
    else form.append("text", note.text ?? "");
    return fetch(`/cases/${id}/notes`, { method: "POST", body: form }).then((r) => json<NoteRecord>(r));
  },

  addImage: (id: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`/cases/${id}/images`, { method: "POST", body: form }).then((r) => json<ImageRecord>(r));
  },
};

export type StreamEvent =
  | { type: "evidence"; data: Omit<ChatResult, "summary" | "details" | "citations" | "llm_error"> }
  | { type: "token"; data: { text: string } }
  | { type: "done"; data: ChatResult }
  | { type: "error"; data: { detail: string } };

/** POST a question and yield server-sent events as they arrive. */
export async function* streamChat(caseId: string, question: string, signal?: AbortSignal): AsyncGenerator<StreamEvent> {
  const res = await fetch(`/cases/${caseId}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!res.ok || !res.body) {
    await json(res);
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let split: number;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const type = /^event: (.+)$/m.exec(block)?.[1];
      const data = /^data: (.+)$/m.exec(block)?.[1];
      if (type && data) yield { type, data: JSON.parse(data) } as StreamEvent;
    }
  }
}
