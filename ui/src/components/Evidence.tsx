import { AlertTriangle, CheckCircle2, Info, Microscope, Siren, Images } from "lucide-react";
import type { Conflict, Evidence, FusedFinding, Triage } from "../api";
import type { Ink } from "../leader";
import { useTarget, useTraceProps } from "../leader";
import { Cite } from "./Cite";
import { termKey } from "./NotePanel";

/* --------------------------------------------------------------- findings */

const pct = (p: number) => `${Math.max(0, Math.min(100, p * 100))}%`;

function FindingRow({ f, selected, onSelect }: { f: FusedFinding; selected: boolean; onSelect: () => void }) {
  const targets: { id: string; ink: Ink }[] = [{ id: "image", ink: "image" }];
  f.signals.forEach((s) => {
    if (s.kind === "vital") targets.push({ id: "vitals", ink: "note" });
    else targets.push({ id: termKey(s.term), ink: "note" });
  });
  const trace = useTraceProps(targets);
  const shift = f.fused_probability - f.image_probability;
  const supports = f.signals.filter((s) => s.llr > 0).map((s) => s.term).slice(0, 3);
  const against = f.signals.filter((s) => s.llr < 0).map((s) => `no ${s.term}`);
  return (
    <button
      type="button"
      className="finding"
      aria-pressed={selected}
      onClick={onSelect}
      title="Show heatmap for this finding"
      {...trace}
    >
      <span className="finding-name">{f.finding}</span>
      <span className="finding-level" data-level={f.image_level}>
        {f.image_level}
      </span>
      <span className="band" aria-hidden="true">
        <span />
        <span />
        <span />
        <i className="mark mark-image" style={{ left: pct(f.image_probability) }} />
        {Math.abs(shift) >= 0.02 && <i className="mark mark-fused" style={{ left: pct(f.fused_probability) }} />}
      </span>
      <span className="finding-nums">
        <span>
          image <b className="num">{f.image_probability.toFixed(2)}</b>
        </span>
        {Math.abs(shift) >= 0.02 && (
          <span>
            with note <b className="num">{f.fused_probability.toFixed(2)}</b>
          </span>
        )}
      </span>
      {(supports.length > 0 || against.length > 0) && (
        <span className="finding-why">
          {supports.length > 0 && (
            <>
              <b>Note supports:</b> {supports.join(", ")}
            </>
          )}
          {supports.length > 0 && against.length > 0 && " · "}
          {against.length > 0 && (
            <span className="against">
              <b>Note against:</b> {against.join(", ")}
            </span>
          )}
        </span>
      )}
    </button>
  );
}

export function Findings({
  evidence,
  heatFinding,
  onHeatFinding,
}: {
  evidence: Evidence | undefined;
  heatFinding: string | null;
  onHeatFinding: (f: string | null) => void;
}) {
  if (!evidence) return <Skeleton rows={4} title="Image findings" />;
  const analysis = evidence.image_analysis;
  if (!analysis) {
    return (
      <section className="section">
        <div className="section-head">
          <h2>
            <Microscope size={16} aria-hidden="true" /> Image findings
          </h2>
        </div>
        <p className="empty-line">Upload a chest X-ray to see findings.</p>
      </section>
    );
  }
  if (!analysis.modality.is_chest_xray) {
    return (
      <section className="section">
        <div className="section-head">
          <h2>
            <Microscope size={16} aria-hidden="true" /> Image findings
          </h2>
        </div>
        <p className="empty-line">
          <Info size={14} aria-hidden="true" /> Not analysed: the image looks like a {analysis.modality.modality}, not a chest X-ray.
        </p>
      </section>
    );
  }
  const shown = evidence.fused_findings
    .filter((f) => f.image_level !== "unlikely" || f.signals.some((s) => s.kind === "asserted" || s.kind === "negated"))
    .sort((a, b) => b.fused_probability - a.fused_probability);
  const hidden = evidence.fused_findings.filter((f) => !shown.includes(f));
  return (
    <section className="section" aria-labelledby="find-h">
      <div className="section-head">
        <h2 id="find-h">
          <Microscope size={16} aria-hidden="true" /> Image findings
        </h2>
        <span className="section-sub">calibrated probability · BiomedCLIP</span>
      </div>
      <div className="scale-key" aria-hidden="true">
        <span>
          <i className="key-swatch" /> image
        </span>
        <span>
          <i className="key-dot" /> with note context
        </span>
        <span>bands: unlikely · possible · likely</span>
      </div>
      <div className="findings">
        {shown.map((f) => (
          <FindingRow
            key={f.finding}
            f={f}
            selected={heatFinding === f.finding}
            onSelect={() => onHeatFinding(heatFinding === f.finding ? null : f.finding)}
          />
        ))}
      </div>
      {hidden.length > 0 && (
        <p className="not-detected">
          Not detected (&lt; 0.2): {hidden.map((f) => f.finding).join(", ")}
        </p>
      )}
    </section>
  );
}

/* --------------------------------------------------------------- conflicts */

const SEV_ICON = { high: AlertTriangle, medium: AlertTriangle, info: Info };

export function Conflicts({ conflicts }: { conflicts: Conflict[] | undefined }) {
  const ref = useTarget("conflicts");
  if (!conflicts) return <Skeleton rows={2} title="Note vs image" />;
  return (
    <section className="section" aria-labelledby="conf-h" ref={ref} style={{ borderRadius: 6 }}>
      <div className="section-head">
        <h2 id="conf-h">
          <AlertTriangle size={16} aria-hidden="true" /> Note vs image
        </h2>
        <span className="section-sub">
          {conflicts.length ? `${conflicts.length} disagreement${conflicts.length > 1 ? "s" : ""}` : "rules-based check"}
        </span>
      </div>
      {conflicts.length === 0 ? (
        <p className="empty-line">
          <CheckCircle2 size={15} aria-hidden="true" style={{ color: "var(--routine)" }} /> No disagreement between the note and the image.
        </p>
      ) : (
        conflicts.map((c, i) => {
          const Icon = SEV_ICON[c.severity];
          return (
            <div className="conflict" data-severity={c.severity} key={i}>
              <Icon size={16} aria-hidden="true" />
              <span className="conflict-sev">{c.severity.toUpperCase()}</span>
              <p className="conflict-msg">{c.message}</p>
              <span className="conflict-cites">
                {c.evidence.map((e) => (
                  <Cite key={e} id={e} />
                ))}
              </span>
            </div>
          );
        })
      )}
    </section>
  );
}

/* --------------------------------------------------------------- triage */

const NEWS_LABELS: Record<string, string> = {
  respiratory_rate: "Resp. rate",
  spo2: "SpO₂",
  systolic_bp: "Systolic BP",
  heart_rate: "Heart rate",
  temperature: "Temp",
  consciousness: "Consciousness",
};

export function TriagePanel({ triage }: { triage: Triage | undefined }) {
  if (!triage) return <Skeleton rows={2} title="Why this triage level" />;
  return (
    <section className="section" aria-labelledby="tri-h">
      <div className="section-head">
        <h2 id="tri-h">
          <Siren size={16} aria-hidden="true" /> Why {triage.level.toLowerCase()}
        </h2>
        <span className="section-sub">NEWS2 + red-flag rules</span>
      </div>
      {triage.reasons.length ? (
        <ul className="reasons">
          {triage.reasons.map((r, i) => {
            const [level, ...rest] = r.split(": ");
            return (
              <li key={i}>
                <span className="reason-level" data-level={level}>
                  {level}
                </span>
                <span>{rest.join(": ")}</span>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="empty-line">No risk factors in the recorded vitals or symptoms.</p>
      )}
      {triage.news2_score !== null && (
        <div className="news" aria-label="NEWS2 components">
          {Object.entries(triage.news2_components).map(([k, v]) => (
            <div className="news-cell" key={k} data-score={v}>
              <b>{v}</b>
              {NEWS_LABELS[k] ?? k}
            </div>
          ))}
        </div>
      )}
      {triage.limitations.map((l, i) => (
        <p className="limitation" key={i}>
          {l}
        </p>
      ))}
    </section>
  );
}

/* --------------------------------------------------------------- similar cases */

function SimilarItem({ i, report, labels, score }: { i: number; report: string; labels: string[]; score: number }) {
  const ref = useTarget(`case-${i}`);
  return (
    <div className="similar-item" ref={ref}>
      <b>
        {labels.join(", ")} <span className="num" style={{ fontWeight: 400, color: "var(--ink-3)" }}>· similarity {score.toFixed(2)}</span>
      </b>
      “{report.length > 180 ? report.slice(0, 180) + "…" : report}”
    </div>
  );
}

export function SimilarCases({ evidence }: { evidence: Evidence | undefined }) {
  const cases = evidence?.image_analysis?.similar_cases ?? [];
  if (!cases.length) return null;
  return (
    <section className="section" aria-labelledby="sim-h">
      <div className="section-head">
        <h2 id="sim-h">
          <Images size={16} aria-hidden="true" /> Similar prior X-rays
        </h2>
        <span className="section-sub">other patients · context only</span>
      </div>
      <div className="similar">
        {cases.map((c, i) => (
          <SimilarItem key={c.uid} i={i + 1} report={c.report} labels={c.labels} score={c.score} />
        ))}
      </div>
    </section>
  );
}

function Skeleton({ rows, title }: { rows: number; title: string }) {
  return (
    <section className="section" aria-busy="true">
      <div className="section-head">
        <h2>{title}</h2>
      </div>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="skeleton" style={{ height: 30, marginBottom: 10 }} />
      ))}
    </section>
  );
}
