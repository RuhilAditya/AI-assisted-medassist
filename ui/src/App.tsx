import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, Plus, ShieldCheck, Siren } from "lucide-react";
import { api, streamChat } from "./api";
import type { CaseDetail, CaseSummary, HistoryTurn, Overview, TriageLevel } from "./api";
import { LeaderProvider, useTarget } from "./leader";
import { CiteContext, makeResolver } from "./components/Cite";
import { Mark } from "./components/Mark";
import { ImagePlate } from "./components/ImagePlate";
import { NotePanel } from "./components/NotePanel";
import { Conflicts, Findings, SimilarCases, TriagePanel } from "./components/Evidence";
import { Conversation } from "./components/Conversation";
import type { LiveTurn } from "./components/Conversation";
import { NewCase } from "./components/NewCase";

const LEVEL_COLOR: Record<TriageLevel, string> = {
  EMERGENCY: "var(--emergency)",
  URGENT: "var(--urgent)",
  SOON: "var(--soon)",
  ROUTINE: "var(--routine)",
};

function CaseBand({ detail, overview }: { detail: CaseDetail; overview: Overview | undefined }) {
  const triageRef = useTarget("triage");
  const t = overview?.triage;
  const removed = detail.notes.reduce((n, note) => n + Object.values(note.deid?.counts ?? {}).reduce((a, b) => a + b, 0), 0);
  return (
    <header className="case-band" data-level={t?.level}>
      <div>
        <h1 className="band-title">{detail.title || "Untitled case"}</h1>
        <div className="band-meta">
          <span>
            Case <span className="num">{detail.case_id}</span>
          </span>
          <span>
            <ShieldCheck size={14} aria-hidden="true" /> {removed} identifier{removed === 1 ? "" : "s"} removed
          </span>
          {detail.tags?.demo && <span>Demo case · synthetic note, real X-ray</span>}
        </div>
      </div>
      <div className="band-triage" ref={triageRef} aria-label="Triage level">
        <span className="band-level">
          <Siren size={20} aria-hidden="true" />
          {t ? t.level : "…"}
        </span>
        <span className="band-news">
          {t?.news2_score !== null && t?.news2_score !== undefined ? (
            <>
              NEWS2 <span className="num">{t.news2_score}</span>
            </>
          ) : (
            "NEWS2 not computable"
          )}
        </span>
      </div>
    </header>
  );
}

type Tab = "image" | "evidence" | "ask";

function Workspace({ caseId, onTriage }: { caseId: string; onTriage: (id: string, level: TriageLevel) => void }) {
  const [detail, setDetail] = useState<CaseDetail>();
  const [overview, setOverview] = useState<Overview>();
  const [history, setHistory] = useState<HistoryTurn[]>([]);
  const [live, setLive] = useState<LiveTurn[]>([]);
  const [heatFinding, setHeatFinding] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("evidence");
  const turnId = useRef(0);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    setDetail(undefined);
    setOverview(undefined);
    setLive([]);
    setHeatFinding(null);
    setError(null);
    api.getCase(caseId).then((d) => !cancelled && setDetail(d)).catch((e) => setError(String(e.message ?? e)));
    api.history(caseId).then((h) => !cancelled && setHistory(h)).catch(() => setHistory([]));
    api
      .overview(caseId)
      .then((o) => {
        if (cancelled) return;
        setOverview(o);
        onTriage(caseId, o.triage.level);
      })
      .catch((e) => setError(String(e.message ?? e)));
    return () => {
      cancelled = true;
      abort.current?.abort();
    };
  }, [caseId, onTriage]);

  const ask = useCallback(
    async (question: string) => {
      const id = ++turnId.current;
      const update = (patch: Partial<LiveTurn>) => setLive((ts) => ts.map((t) => (t.id === id ? { ...t, ...patch } : t)));
      setLive((ts) => [...ts, { id, question, status: "preparing", text: "", startedAt: Date.now() }]);
      setTab("ask");
      abort.current = new AbortController();
      try {
        let text = "";
        for await (const ev of streamChat(caseId, question, abort.current.signal)) {
          if (ev.type === "evidence") {
            setOverview((o) => (o ? { ...o, ...ev.data } : (ev.data as Overview)));
            update({ status: "writing", startedAt: Date.now() });
          } else if (ev.type === "token") {
            text += ev.data.text;
            update({ text });
          } else if (ev.type === "done") {
            update({ status: "done", result: ev.data });
            setOverview((o) => (o ? { ...o, follow_ups: ev.data.follow_ups } : o));
          } else if (ev.type === "error") {
            update({ status: "error", error: ev.data.detail });
          }
        }
      } catch (e) {
        if ((e as Error).name !== "AbortError") update({ status: "error", error: (e as Error).message });
      }
    },
    [caseId],
  );

  const busy = live.some((t) => t.status === "preparing" || t.status === "writing");
  const resolver = useMemo(
    () => makeResolver((overview?.evidence.passages ?? []).map((p) => p.source)),
    [overview?.evidence.passages],
  );
  const image = detail?.images[detail.images.length - 1];

  if (error) {
    return (
      <main className="workspace">
        <p className="error-line" style={{ padding: 32 }}>
          Could not load this case: {error}. Is the backend running (uvicorn api.main:app)?
        </p>
      </main>
    );
  }

  return (
    <CiteContext.Provider value={resolver}>
      <main className="workspace">
        {detail ? <CaseBand detail={detail} overview={overview} /> : <header className="case-band"><div className="skeleton" style={{ height: 40, width: 320, opacity: 0.2 }} /></header>}
        <nav className="mobile-tabs" role="tablist" aria-label="Case sections">
          {(["image", "evidence", "ask"] as Tab[]).map((t) => (
            <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>
              {t === "image" ? "Image & note" : t === "evidence" ? "Evidence" : "Ask"}
            </button>
          ))}
        </nav>
        <div className="columns">
          <div className="col" data-active={tab === "image"}>
            <ImagePlate caseId={caseId} image={image} evidence={overview?.evidence} heatFinding={heatFinding} onHeatFinding={setHeatFinding} />
            <NotePanel notes={detail?.notes ?? []} triage={overview?.triage} />
          </div>
          <div className="col" data-active={tab === "evidence"}>
            <TriagePanel triage={overview?.triage} />
            <Conflicts conflicts={overview?.conflicts} />
            <Findings evidence={overview?.evidence} heatFinding={heatFinding} onHeatFinding={setHeatFinding} />
            <SimilarCases evidence={overview?.evidence} />
          </div>
          <div className="col col-chat" data-active={tab === "ask"}>
            <Conversation
              history={history}
              live={live}
              suggestions={overview?.follow_ups.next_questions ?? []}
              requests={overview?.follow_ups.requests ?? []}
              busy={busy}
              disclaimer={overview?.disclaimer}
              onAsk={ask}
            />
          </div>
        </div>
      </main>
    </CiteContext.Provider>
  );
}

export default function App() {
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [levels, setLevels] = useState<Record<string, TriageLevel>>({});
  const [loadError, setLoadError] = useState<string | null>(null);

  const refresh = useCallback(async (select?: string) => {
    try {
      const list = await api.listCases();
      // demo cases first, in the order they were defined; then everything else newest first
      list.sort((a, b) => Number(!!b.tags?.demo) - Number(!!a.tags?.demo) || (a.tags?.demo ? a.created.localeCompare(b.created) : b.created.localeCompare(a.created)));
      setCases(list);
      setSelected((cur) => select ?? cur ?? list[0]?.case_id ?? null);
      setLoadError(null);
    } catch (e) {
      setLoadError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const onTriage = useCallback((id: string, level: TriageLevel) => setLevels((l) => (l[id] === level ? l : { ...l, [id]: level })), []);

  return (
    <LeaderProvider>
      <div className="app">
        <aside className="rail" aria-label="Cases">
          <div className="brand">
            <Mark />
            <div>
              <div className="brand-name">MedAssist</div>
              <div className="brand-tag">Multimodal decision support</div>
            </div>
          </div>
          <div className="rail-section">
            <span>Cases</span>
            <button className="btn btn-ghost btn-sm" onClick={() => setCreating(true)}>
              <Plus size={14} aria-hidden="true" /> New
            </button>
          </div>
          <ul className="case-list">
            {cases.map((c) => (
              <li key={c.case_id}>
                <button
                  className="case-item"
                  aria-current={!creating && selected === c.case_id}
                  onClick={() => {
                    setCreating(false);
                    setSelected(c.case_id);
                  }}
                >
                  <span className="case-dot" style={levels[c.case_id] ? { background: LEVEL_COLOR[levels[c.case_id]] } : undefined} aria-hidden="true" />
                  <span className="case-title">{c.title || `Case ${c.case_id}`}</span>
                  {c.tags?.shows && <span className="case-shows">{c.tags.shows}</span>}
                </button>
              </li>
            ))}
            {loadError && (
              <li className="empty-line" style={{ padding: 12 }}>
                Backend not reachable: {loadError}
              </li>
            )}
          </ul>
          <div className="rail-mobile">
            <label className="visually-hidden" htmlFor="case-select">
              Case
            </label>
            <select id="case-select" value={selected ?? ""} onChange={(e) => { setCreating(false); setSelected(e.target.value); }}>
              {cases.map((c) => (
                <option key={c.case_id} value={c.case_id}>
                  {c.title || c.case_id}
                </option>
              ))}
            </select>
            <button className="btn btn-sm" onClick={() => setCreating(true)} aria-label="New case">
              <Plus size={14} aria-hidden="true" />
            </button>
          </div>
          <div className="rail-foot">
            <Activity size={12} aria-hidden="true" style={{ verticalAlign: -1 }} /> Research prototype. Runs locally; no data
            leaves this machine. Not a medical device.
          </div>
        </aside>
        {creating ? (
          <main className="workspace" style={{ display: "block", overflowY: "auto" }}>
            <NewCase
              onCancel={() => setCreating(false)}
              onCreated={(id) => {
                setCreating(false);
                refresh(id);
              }}
            />
          </main>
        ) : selected ? (
          <Workspace key={selected} caseId={selected} onTriage={onTriage} />
        ) : (
          <main className="workspace" style={{ display: "grid", placeItems: "center" }}>
            <div className="intro" style={{ textAlign: "center" }}>
              <h2>No cases yet</h2>
              <p>
                Create a case, or load the demo cases with <code>python -m scripts.load_demo_cases --analyze</code>.
              </p>
              <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={() => setCreating(true)}>
                <Plus size={16} aria-hidden="true" /> New case
              </button>
            </div>
          </main>
        )}
      </div>
    </LeaderProvider>
  );
}
