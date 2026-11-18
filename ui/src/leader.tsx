// Leader lines: the atlas-plate signature. Hovering a citation (or a finding's evidence)
// draws a fine line from the claim to the exact evidence it rests on, in that evidence's ink.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

export type Ink = "image" | "note" | "neutral";

interface Line {
  key: string;
  d: string;
  end: { x: number; y: number };
  ink: Ink;
}

interface LeaderApi {
  register: (id: string, el: HTMLElement | null) => void;
  trace: (from: HTMLElement, targets: { id: string; ink: Ink }[]) => void;
  clear: () => void;
  has: (id: string) => boolean;
}

const LeaderContext = createContext<LeaderApi | null>(null);

function anchor(rect: DOMRect, toward: { x: number; y: number }) {
  // nearest point on the rectangle's edge to the source point, pulled 4px outside
  const x = Math.min(Math.max(toward.x, rect.left), rect.right);
  const y = Math.min(Math.max(toward.y, rect.top), rect.bottom);
  const inside = toward.x > rect.left && toward.x < rect.right && toward.y > rect.top && toward.y < rect.bottom;
  if (!inside) return { x, y };
  return { x: toward.x < rect.left + rect.width / 2 ? rect.left - 4 : rect.right + 4, y: rect.top + rect.height / 2 };
}

function visibleRect(el: HTMLElement): DOMRect | null {
  const r = el.getBoundingClientRect();
  if (r.width === 0 && r.height === 0) return null;
  return r;
}

export function LeaderProvider({ children }: { children: ReactNode }) {
  const targets = useRef(new Map<string, HTMLElement>());
  const active = useRef<{ from: HTMLElement; targets: { id: string; ink: Ink }[] } | null>(null);
  const traced = useRef<HTMLElement[]>([]);
  const [lines, setLines] = useState<Line[]>([]);

  const compute = useCallback(() => {
    const a = active.current;
    if (!a) return setLines([]);
    const fromRect = visibleRect(a.from);
    if (!fromRect) return setLines([]);
    const next: Line[] = [];
    a.targets.forEach(({ id, ink }, i) => {
      const el = targets.current.get(id);
      const r = el && visibleRect(el);
      if (!r) return;
      const toward = { x: r.left + r.width / 2, y: r.top + r.height / 2 };
      const start = {
        x: toward.x < fromRect.left ? fromRect.left : toward.x > fromRect.right ? fromRect.right : fromRect.left + fromRect.width / 2,
        y: fromRect.top + fromRect.height / 2,
      };
      const end = anchor(r, start);
      const dx = Math.max(40, Math.abs(end.x - start.x) * 0.45) * (end.x < start.x ? -1 : 1);
      const d = `M ${start.x} ${start.y} C ${start.x + dx} ${start.y}, ${end.x - dx} ${end.y}, ${end.x} ${end.y}`;
      next.push({ key: `${id}-${i}`, d, end, ink });
    });
    setLines(next);
  }, []);

  const clear = useCallback(() => {
    active.current = null;
    traced.current.forEach((el) => delete el.dataset.traced);
    traced.current = [];
    setLines([]);
  }, []);

  const trace = useCallback(
    (from: HTMLElement, list: { id: string; ink: Ink }[]) => {
      clear();
      active.current = { from, targets: list };
      list.forEach(({ id, ink }) => {
        const el = targets.current.get(id);
        if (el) {
          el.dataset.traced = ink;
          traced.current.push(el);
        }
      });
      compute();
    },
    [clear, compute],
  );

  const register = useCallback((id: string, el: HTMLElement | null) => {
    if (el) targets.current.set(id, el);
    else targets.current.delete(id);
  }, []);

  const has = useCallback((id: string) => targets.current.has(id), []);

  useEffect(() => {
    const onMove = () => active.current && compute();
    window.addEventListener("scroll", onMove, true);
    window.addEventListener("resize", onMove);
    return () => {
      window.removeEventListener("scroll", onMove, true);
      window.removeEventListener("resize", onMove);
    };
  }, [compute]);

  const value = useMemo(() => ({ register, trace, clear, has }), [register, trace, clear, has]);

  return (
    <LeaderContext.Provider value={value}>
      {children}
      <svg className="leaders" aria-hidden="true">
        {lines.map((l) => (
          <g key={l.key}>
            <path d={l.d} pathLength={1} data-ink={l.ink} />
            <circle cx={l.end.x} cy={l.end.y} r={3.5} data-ink={l.ink} />
          </g>
        ))}
      </svg>
    </LeaderContext.Provider>
  );
}

export function useLeader() {
  const ctx = useContext(LeaderContext);
  if (!ctx) throw new Error("useLeader outside LeaderProvider");
  return ctx;
}

/** Ref callback that registers an element as a leader-line target. */
export function useTarget(id: string) {
  const { register } = useLeader();
  return useCallback((el: HTMLElement | null) => register(id, el), [id, register]);
}

/** Props that make an element trace lines to targets on hover/focus. */
export function useTraceProps(targets: { id: string; ink: Ink }[]) {
  const { trace, clear } = useLeader();
  const key = targets.map((t) => t.id + t.ink).join("|");
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const list = useMemo(() => targets, [key]);
  return useMemo(
    () => ({
      onMouseEnter: (e: React.MouseEvent<HTMLElement>) => trace(e.currentTarget, list),
      onMouseLeave: clear,
      onFocus: (e: React.FocusEvent<HTMLElement>) => trace(e.currentTarget, list),
      onBlur: clear,
    }),
    [list, trace, clear],
  );
}
