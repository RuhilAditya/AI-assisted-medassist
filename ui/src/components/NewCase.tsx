import { useState } from "react";
import { Check, FileUp, ImageUp, Loader2 } from "lucide-react";
import { api } from "../api";

type Step = { label: string; state: "todo" | "doing" | "done" };

function FilePick({
  id,
  accept,
  file,
  onFile,
  icon,
  prompt,
}: {
  id: string;
  accept: string;
  file: File | null;
  onFile: (f: File | null) => void;
  icon: React.ReactNode;
  prompt: string;
}) {
  const [over, setOver] = useState(false);
  return (
    <label
      className="drop"
      htmlFor={id}
      data-over={over}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        onFile(e.dataTransfer.files[0] ?? null);
      }}
    >
      {icon}
      <span>{file ? <b>{file.name}</b> : prompt}</span>
      <input id={id} type="file" accept={accept} className="visually-hidden" onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
    </label>
  );
}

export function NewCase({ onCreated, onCancel }: { onCreated: (id: string) => void; onCancel: () => void }) {
  const [title, setTitle] = useState("");
  const [noteText, setNoteText] = useState("");
  const [noteFile, setNoteFile] = useState<File | null>(null);
  const [image, setImage] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [steps, setSteps] = useState<Step[] | null>(null);
  const busy = steps !== null && !error;

  const set = (i: number, state: Step["state"]) => setSteps((s) => s && s.map((x, j) => (j === i ? { ...x, state } : x)));

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!noteText.trim() && !noteFile && !image) {
      setError("Add a clinical note or an image to create a case.");
      return;
    }
    const plan: Step[] = [{ label: "Create case", state: "todo" }];
    if (noteText.trim() || noteFile) plan.push({ label: "De-identify note and extract entities", state: "todo" });
    if (image) plan.push({ label: "Anonymise and process image", state: "todo" });
    setSteps(plan);
    try {
      set(0, "doing");
      const { case_id } = await api.createCase(title.trim());
      set(0, "done");
      let i = 1;
      if (noteText.trim() || noteFile) {
        set(i, "doing");
        await api.addNote(case_id, noteFile ? { file: noteFile } : { text: noteText });
        set(i++, "done");
      }
      if (image) {
        set(i, "doing");
        await api.addImage(case_id, image);
        set(i, "done");
      }
      onCreated(case_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <form className="new-case" onSubmit={submit}>
      <h1>New case</h1>
      <p>
        Identifiers in the note (names, phone numbers, IDs, dates) are removed before anything is stored. DICOM tags are
        anonymised. Use synthetic or already de-identified data for testing.
      </p>
      <div className="field">
        <label htmlFor="nc-title">Case title</label>
        <input id="nc-title" type="text" value={title} maxLength={120} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Fever and breathlessness, 3 days" />
      </div>
      <div className="field">
        <label htmlFor="nc-note">Clinical note</label>
        <span className="hint">Paste the note, or upload a TXT/PDF file instead.</span>
        <textarea
          id="nc-note"
          value={noteText}
          onChange={(e) => setNoteText(e.target.value)}
          disabled={!!noteFile}
          placeholder={"Presenting complaint, history, examination, vitals (e.g. RR 24, SpO2 90%, HR 108, BP 124/78, Temp 38.8 C)…"}
        />
        <FilePick id="nc-note-file" accept=".txt,.pdf,.md" file={noteFile} onFile={setNoteFile} icon={<FileUp size={18} aria-hidden="true" />} prompt="Or drop a TXT / PDF note here" />
      </div>
      <div className="field">
        <label htmlFor="nc-image">Image</label>
        <span className="hint">Chest X-ray as DICOM, PNG or JPEG (max 20 MB). Other modalities are accepted but not analysed.</span>
        <FilePick id="nc-image" accept=".dcm,.dicom,.png,.jpg,.jpeg" file={image} onFile={setImage} icon={<ImageUp size={18} aria-hidden="true" />} prompt="Drop an image here, or click to choose" />
      </div>
      {steps && (
        <ul className="progress-steps" aria-live="polite">
          {steps.map((s) => (
            <li key={s.label}>
              {s.state === "done" ? <Check size={15} aria-hidden="true" style={{ color: "var(--routine)" }} /> : s.state === "doing" ? <Loader2 size={15} className="spin" aria-hidden="true" /> : <span style={{ width: 15 }} />}
              {s.label}
            </li>
          ))}
        </ul>
      )}
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="form-actions">
        <button className="btn btn-primary" type="submit" disabled={busy}>
          {busy ? <Loader2 size={16} className="spin" aria-hidden="true" /> : null} Create case
        </button>
        <button className="btn btn-ghost" type="button" onClick={onCancel} disabled={busy}>
          Cancel
        </button>
      </div>
    </form>
  );
}
