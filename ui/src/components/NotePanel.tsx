import { Fragment } from "react";
import { FileText, ShieldCheck } from "lucide-react";
import type { Entity, NoteRecord, Triage } from "../api";
import { useTarget } from "../leader";

const VITAL_LABELS: Record<string, [string, string]> = {
  respiratory_rate: ["Resp. rate", "/min"],
  spo2: ["SpO₂", "%"],
  blood_pressure: ["BP", ""],
  heart_rate: ["Heart rate", "/min"],
  temperature: ["Temp", ""],
};
const NEWS_KEY: Record<string, string> = { blood_pressure: "systolic_bp" };
const PHI_RE = /(\[(?:PERSON|PHONE|EMAIL|LOCATION|DATE|ID|MRN|URL|IP|AGE>89)\])/g;
const PHI_PART = /^\[(?:PERSON|PHONE|EMAIL|LOCATION|DATE|ID|MRN|URL|IP|AGE>89)\]$/; // non-global: safe for .test

export function termKey(term: string) {
  return `term:${term.toLowerCase().replace(/^history of /, "").trim()}`;
}

function EntitySpan({ e, children }: { e: Entity; children: string }) {
  const ref = useTarget(termKey(e.canonical));
  const cls = e.negated ? "ent ent-negated" : e.historical || e.family ? "ent ent-history" : "ent ent-active";
  const meaning = e.negated ? "absent (negated)" : e.family ? "family history" : e.historical ? "past history" : "present";
  return (
    <span ref={ref} className={cls} title={`${e.label.toLowerCase()} · ${meaning}`}>
      {children}
    </span>
  );
}

function withPhi(text: string) {
  return text.split(PHI_RE).map((part, i) =>
    PHI_PART.test(part) ? (
      <span key={i} className="phi" title="Removed during de-identification">
        {part.slice(1, -1)}
      </span>
    ) : (
      <Fragment key={i}>{part}</Fragment>
    ),
  );
}

function NoteText({ note }: { note: NoteRecord }) {
  const ents = [...note.extraction.entities]
    .filter((e) => ["PROBLEM", "MEDICATION", "TEST", "PROCEDURE"].includes(e.label))
    .sort((a, b) => a.start - b.start);
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  ents.forEach((e, i) => {
    if (e.start < cursor) return; // overlapping span: keep the first
    parts.push(<Fragment key={`t${i}`}>{withPhi(note.text.slice(cursor, e.start))}</Fragment>);
    parts.push(
      <EntitySpan key={`e${i}`} e={e}>
        {note.text.slice(e.start, e.end)}
      </EntitySpan>,
    );
    cursor = e.end;
  });
  parts.push(<Fragment key="end">{withPhi(note.text.slice(cursor))}</Fragment>);
  return <div className="note-body">{parts}</div>;
}

export function Vitals({ vitals, triage }: { vitals: Record<string, string>; triage?: Triage }) {
  const ref = useTarget("vitals");
  const entries = Object.entries(VITAL_LABELS).filter(([k]) => vitals[k]);
  if (!entries.length) return null;
  return (
    <dl className="vitals" ref={ref}>
      {entries.map(([k, [label, unit]]) => (
        <div className="vital" key={k} data-score={triage?.news2_components?.[NEWS_KEY[k] ?? k] ?? 0}>
          <dt>{label}</dt>
          <dd className="num">
            {vitals[k]}
            {unit && <small style={{ fontSize: "0.7em", fontWeight: 400 }}>{unit}</small>}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function NoteBlock({ note, triage }: { note: NoteRecord; triage?: Triage }) {
  const ref = useTarget(note.note_id);
  const removed = Object.values(note.deid?.counts ?? {}).reduce((a, b) => a + b, 0);
  return (
    <div ref={ref} style={{ borderRadius: 6 }}>
      <Vitals vitals={note.extraction.vitals} triage={triage} />
      <NoteText note={note} />
      <div className="legend">
        <span>
          <i className="ent ent-active">underlined</i> present
        </span>
        <span>
          <i className="ent ent-negated">dotted</i> absent
        </span>
        <span>
          <i className="ent ent-history">dashed</i> history
        </span>
        <span>
          <ShieldCheck size={12} aria-hidden="true" style={{ verticalAlign: -2 }} /> {removed} identifier{removed === 1 ? "" : "s"} removed
        </span>
      </div>
    </div>
  );
}

export function NotePanel({ notes, triage }: { notes: NoteRecord[]; triage?: Triage }) {
  return (
    <section className="section" aria-labelledby="note-h">
      <div className="section-head">
        <h2 id="note-h">
          <FileText size={16} aria-hidden="true" /> Clinical note
        </h2>
        <span className="section-sub">de-identified · {notes.length || "no"} note{notes.length === 1 ? "" : "s"}</span>
      </div>
      {notes.length === 0 && <p className="empty-line">No clinical note for this case.</p>}
      {notes.map((n) => (
        <NoteBlock key={n.note_id} note={n} triage={triage} />
      ))}
    </section>
  );
}
