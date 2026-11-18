---
name: MedAssist "Atlas plate"
mode: operate
source: ui/src/styles.css
---

# Design system: Atlas plate

Recorded from the built UI (`ui/src/styles.css`, components in `ui/src/components/`). The
world comes from anatomy-atlas plates: a clean page, ink type, and labels joined to
structures by fine leader lines. It enters only through type, palette, density and one
signature move; layout, navigation and controls are standard product UI.

## Colour

Strategy: Restrained, with one committed field (the triage band). Light ground, chosen
from the use scene: wards and seminar rooms under bright light, often projected.

| Token | Value | Role |
|---|---|---|
| `--ground` | `#f6f8fa` | cool plate-white page (never cream) |
| `--surface` | `#ffffff` | answer column, inputs, cards that hold data |
| `--rail` / `--rail-hover` | `#edf1f5` / `#e3e9f0` | second neutral layer: case rail, empty bands |
| `--ink` | `#14223b` | primary text, primary button, focus of hierarchy |
| `--ink-2` / `--ink-3` | `#3f4d66` / `#5e6b82` | secondary / tertiary text (navy-tinted, never grey) |
| `--rule` / `--rule-strong` | `#dde3ea` / `#c3ccd7` | hairlines, control borders |
| `--image-ink` / `--image-wash` | `#c93a1b` / `#fbe9e4` | **image evidence** (atlas "arterial red"): image marks, image citations, heatmap overlay (`HEATMAP_INK` in `cxr.py`), likely band |
| `--note-ink` / `--note-wash` | `#1f5aa6` / `#e5eef9` | **note evidence** (atlas "venous blue"): note citations, entity underlines, "with note" marks, focus ring |
| `--emergency` | `#a3172f` | triage EMERGENCY band, NEWS2 parameter = 3 |
| `--urgent` | `#a84d0a` | URGENT band, NEWS2 parameter = 2 |
| `--soon` | `#7d6100` | SOON band |
| `--routine` | `#276f46` | ROUTINE band, "no disagreement" check |
| `--plate-well` | `#0b0f16` | the image well (radiographs are read on black) |

Rules: the two evidence inks are reserved for evidence provenance and never decorate.
Triage colour appears only on the case band, the rail dot, reason labels and NEWS2
cells, and is always paired with a text label (never colour alone).

## Type

One family, Atkinson Hyperlegible Next (variable, self-hosted via Fontsource), chosen for
legibility on projectors and for low-vision users; Atkinson Hyperlegible Mono only for
measured values (probabilities, vitals, NEWS2, case ids) with tabular numerals (`.num`).

Fixed rem scale, ratio ~1.2: `--t-xs` 0.75, `--t-sm` 0.8125, `--t-md` 0.9375 (body),
`--t-lg` 1.125, `--t-xl` 1.375, `--t-2xl` 1.625. Headings 700, triage level 800 with
0.04em tracking. Prose capped at 70–72ch.

## Space, shape, depth, motion

Spacing scale 4 / 8 / 12 / 16 / 24 / 32 (`--s-1` … `--s-6`). Radius 6px; pills fully
rounded. One shadow (`--shadow`: 1px + 4px soft, navy-tinted) for the selected rail item
only; everything else is separated by hairline rules, not cards. Motion conveys state only:
150–250 ms, `--ease` = cubic-bezier(0.16, 1, 0.3, 1); `prefers-reduced-motion` disables it.

## Layout

Rail (264px; 228px under 1360px) + workspace. Workspace = triage band + three scrolling
columns: **image & note** | **evidence** (triage reasons, conflicts, findings, similar
cases) | **conversation** (answer log + composer pinned at bottom). Under 1100px the
conversation spans the bottom row; under 760px the rail becomes a case select and the
columns become three tabs (Image & note / Evidence / Ask).

## Components

- **Case band**: background = triage colour; title, case id (mono), identifiers removed,
  level + NEWS2 on the right (a leader-line target, `triage`).
- **Finding row**: name + level word; reference-band bar (unlikely 0–0.2 | possible
  0.2–0.5 | likely 0.5–1); vermilion tick = image probability, blue dot = probability with
  note context; "Note supports / Note against" line. Click toggles the heatmap.
- **Conflict**: severity word + icon, message, citation chips; HIGH gets a warm wash.
- **NEWS2 cells**: score per parameter, cell tint by points (1/2/3).
- **Vitals strip**: label + mono value; value coloured by its NEWS2 points.
- **Note text**: entity underlines (solid blue = present, dotted = absent, dashed blue =
  history); PHI placeholders shown as mono redaction tags.
- **Citation chip**: pill in the evidence's ink with a dot; fills on hover/focus.
- **Answer log** (not chat bubbles): question as a bold line, summary at body size, details
  collapsible, route + latency footnote. Streaming shows a caret and elapsed seconds.
- Buttons: primary (ink), default (outlined), ghost; pills for toggles; suggestion chips.

## Signature: leader lines

`ui/src/leader.tsx`. Any element can register as a target (`useTarget(id)`); any trigger
(`useTraceProps([...])`) draws SVG cubic curves from itself to its targets on hover or
focus, in the target evidence's ink, with a 3.5px end dot, drawn in 180 ms; the target
gets a 2px outline in the same ink. Targets: `image`, `note_N`, `term:<canonical>` (note
phrases), `vitals`, `case-N`, `triage`, `conflicts`. Citation chips trace one line; a
finding row traces to the image and to every note phrase that moved its score.

## Browser surfaces

Selection = note wash; focus ring 2px `--focus` with 2px offset; thin scrollbars in
`--rule-strong`; favicon is the brand mark (ring + vermilion leader + blue label dot).
