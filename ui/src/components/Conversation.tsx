import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { ArrowUp, ChevronDown, ChevronUp, CircleAlert, Loader2, ShieldAlert } from "lucide-react";
import type { ChatResult, HistoryTurn } from "../api";
import { Cite } from "./Cite";

export interface LiveTurn {
  id: number;
  question: string;
  status: "preparing" | "writing" | "done" | "error";
  text: string;
  result?: ChatResult;
  error?: string;
  startedAt: number;
}

const CITE_ID = /^(note_\d+|image|case-\d+|ref-\d+|triage|conflict)$/;

/** Turn "[image, note_1]" into markdown links the renderer maps to Cite chips. */
function prepare(md: string) {
  return md
    .replace(/^(\s*)[•●▪]\s+/gm, "$1- ")
    .replace(/\[([^\]\n]{1,80})\]/g, (whole, inner: string) => {
      const ids = inner.split(",").map((s) => s.trim());
      if (!ids.every((id) => CITE_ID.test(id))) return whole;
      return ids.map((id) => `[${id}](#cite-${id})`).join(" ");
    });
}

function stripSections(raw: string) {
  return raw.replace(/\*{0,2}SUMMARY:?\*{0,2}\s*/i, "").replace(/\*{0,2}DETAILS:?\*{0,2}\s*/i, "\n\n");
}

export function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown
      components={{
        a: ({ href, children }) =>
          href?.startsWith("#cite-") ? <Cite id={href.slice(6)} /> : <a href={href}>{children}</a>,
      }}
    >
      {prepare(text)}
    </ReactMarkdown>
  );
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, []);
  return <span className="num">{Math.round((now - since) / 1000)} s</span>;
}

function LiveAnswer({ turn }: { turn: LiveTurn }) {
  const [open, setOpen] = useState(true);
  const r = turn.result;
  return (
    <article className="turn" aria-live="polite">
      <h3 className="turn-q">{turn.question}</h3>
      {turn.status === "preparing" && (
        <p className="turn-status">
          <Loader2 size={14} className="spin" aria-hidden="true" /> Analysing note and image…
        </p>
      )}
      {turn.status === "writing" && (
        <>
          <div className="streaming">
            {turn.text ? <Markdown text={stripSections(turn.text)} /> : null}
            <span className="caret" aria-hidden="true" />
          </div>
          <p className="turn-status">
            <Loader2 size={14} className="spin" aria-hidden="true" /> Evidence ready · writing answer <Elapsed since={turn.startedAt} />
          </p>
        </>
      )}
      {turn.status === "error" && (
        <p className="error-line">
          <CircleAlert size={14} aria-hidden="true" /> {turn.error}
        </p>
      )}
      {turn.status === "done" && r && (
        <>
          <div className="turn-summary">
            <Markdown text={r.summary} />
          </div>
          {r.details && (
            <>
              {open && (
                <div className="turn-details">
                  <Markdown text={r.details} />
                </div>
              )}
              <button className="btn btn-ghost btn-sm turn-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
                {open ? <ChevronUp size={14} aria-hidden="true" /> : <ChevronDown size={14} aria-hidden="true" />}
                {open ? "Hide details" : "Show details"}
              </button>
            </>
          )}
          {r.llm_error && <p className="error-line">Language model unavailable: findings and triage were still computed.</p>}
          <p className="turn-route">
            Answered from {r.route.name === "multimodal" ? "note + image" : r.route.name} evidence
            {r.route.notice ? ` · ${r.route.notice}` : ""} · {r.timings_s.llm?.toFixed(0)} s
          </p>
        </>
      )}
    </article>
  );
}

function PastTurn({ turn }: { turn: HistoryTurn }) {
  return (
    <article className="turn">
      <h3 className="turn-q">
        {turn.question}
        <small>{new Date(turn.time).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}</small>
      </h3>
      <div className="turn-summary">
        <Markdown text={turn.summary} />
      </div>
    </article>
  );
}

interface Props {
  history: HistoryTurn[];
  live: LiveTurn[];
  suggestions: string[];
  requests: string[];
  busy: boolean;
  disclaimer?: string;
  onAsk: (q: string) => void;
}

const STARTERS = [
  "Interpret this X-ray and correlate it with the symptoms.",
  "How urgent is this case, and why?",
  "What is the differential diagnosis?",
];

export function Conversation({ history, live, suggestions, requests, busy, disclaimer, onAsk }: Props) {
  const [draft, setDraft] = useState("");
  const end = useRef<HTMLDivElement>(null);
  const empty = !history.length && !live.length;
  const chips = (suggestions.length ? suggestions : STARTERS).slice(0, 3);

  useEffect(() => {
    end.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [live]);

  const submit = (q: string) => {
    const question = q.trim();
    if (!question || busy) return;
    onAsk(question);
    setDraft("");
  };

  return (
    <>
      <div className="thread">
        {empty && (
          <div className="intro">
            <h2>Ask about this case</h2>
            <p>
              Answers combine the de-identified note and the image. Every claim carries a citation: hover it to see the
              evidence it rests on. Triage and conflicts are computed by rules, not by the language model.
            </p>
          </div>
        )}
        {history.map((t, i) => (
          <PastTurn key={`h${i}`} turn={t} />
        ))}
        {live.map((t) => (
          <LiveAnswer key={t.id} turn={t} />
        ))}
        <div ref={end} />
      </div>
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          submit(draft);
        }}
      >
        {requests.length > 0 && (
          <ul className="requests" aria-label="Information that would improve the assessment">
            {requests.map((r) => (
              <li key={r}>
                <CircleAlert size={13} aria-hidden="true" style={{ flex: "none", marginTop: 2 }} /> {r}
              </li>
            ))}
          </ul>
        )}
        <div className="suggestions" aria-label="Suggested questions">
          {chips.map((s) => (
            <button type="button" key={s} className="suggestion" onClick={() => submit(s)} disabled={busy}>
              {s}
            </button>
          ))}
        </div>
        <div className="ask">
          <label className="visually-hidden" htmlFor="ask-input">
            Ask a question about this case
          </label>
          <textarea
            id="ask-input"
            rows={1}
            value={draft}
            placeholder="Ask about this case…"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit(draft);
              }
            }}
          />
          <button className="btn btn-primary" type="submit" disabled={busy || !draft.trim()} aria-label="Ask">
            {busy ? <Loader2 size={16} className="spin" aria-hidden="true" /> : <ArrowUp size={16} aria-hidden="true" />}
            Ask
          </button>
        </div>
        <p className="disclaimer">
          <ShieldAlert size={13} aria-hidden="true" style={{ flex: "none", marginTop: 2 }} />
          {disclaimer ?? "Decision support only; not a diagnosis."}
        </p>
      </form>
    </>
  );
}
