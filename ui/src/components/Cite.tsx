import { createContext, useContext } from "react";
import type { Ink } from "../leader";
import { useTraceProps } from "../leader";

export interface ResolvedCite {
  target: string;
  ink: Ink;
  label: string;
  title: string;
}

/** Maps evidence ids used by the LLM to on-screen targets. */
export type CiteResolver = (id: string) => ResolvedCite | null;

export const CiteContext = createContext<CiteResolver>(() => null);

export function makeResolver(passageSources: string[]): CiteResolver {
  return (raw) => {
    const id = raw.trim();
    let m: RegExpMatchArray | null;
    if (id === "image") return { target: "image", ink: "image", label: "Image", title: "Image findings (automated classifier)" };
    if ((m = id.match(/^note_(\d+)$/))) return { target: id, ink: "note", label: `Note ${m[1]}`, title: `Clinical note ${m[1]} (de-identified)` };
    if ((m = id.match(/^case-(\d+)$/))) return { target: id, ink: "neutral", label: `Case ${m[1]}`, title: "Similar prior case (different patient)" };
    if ((m = id.match(/^ref-(\d+)$/))) {
      const source = passageSources[Number(m[1]) - 1] ?? "note_1";
      return { target: source, ink: "note", label: `Ref ${m[1]}`, title: `Passage retrieved from ${source.replace("_", " ")}` };
    }
    if (id === "triage") return { target: "triage", ink: "neutral", label: "Triage", title: "Rule-based triage (NEWS2 + red flags)" };
    if (id === "conflict") return { target: "conflicts", ink: "neutral", label: "Conflict", title: "Detected text/image conflicts" };
    return null;
  };
}

export function Cite({ id }: { id: string }) {
  const resolve = useContext(CiteContext);
  const r = resolve(id);
  const trace = useTraceProps(r ? [{ id: r.target, ink: r.ink }] : []);
  if (!r) return <span>[{id}]</span>;
  return (
    <button type="button" className="cite" data-ink={r.ink} title={r.title} aria-label={`Evidence: ${r.title}`} {...trace}>
      <span>{r.label}</span>
    </button>
  );
}
