import { useEffect, useState } from "react";
import { Loader2, ScanLine } from "lucide-react";
import { api } from "../api";
import type { Evidence, ImageRecord } from "../api";
import { useTarget } from "../leader";

interface Props {
  caseId: string;
  image: ImageRecord | undefined;
  evidence: Evidence | undefined;
  heatFinding: string | null;
  onHeatFinding: (finding: string | null) => void;
}

export function ImagePlate({ caseId, image, evidence, heatFinding, onHeatFinding }: Props) {
  const ref = useTarget("image");
  const [loadedHeat, setLoadedHeat] = useState<string | null>(null);
  const [heatError, setHeatError] = useState<string | null>(null);
  const analysis = evidence?.image_analysis;
  const isCxr = analysis?.modality.is_chest_xray ?? false;
  const candidates = (evidence?.fused_findings ?? [])
    .filter((f) => f.image_level !== "unlikely")
    .sort((a, b) => b.image_probability - a.image_probability)
    .slice(0, 4);
  const heatUrl = image && heatFinding ? api.heatmapUrl(caseId, image.image_id, heatFinding) : null;

  useEffect(() => {
    setLoadedHeat(null);
    setHeatError(null);
  }, [heatUrl]);

  return (
    <section className="section" aria-labelledby="plate-h">
      <div className="section-head">
        <h2 id="plate-h">
          <ScanLine size={16} aria-hidden="true" /> Image
        </h2>
        {image && <span className="section-sub">{image.source_format.toUpperCase()} · {image.shape?.slice(0, 2).join(" × ")} px</span>}
      </div>

      <figure className="plate" ref={ref} style={{ margin: 0 }}>
        {image ? (
          <>
            <img src={api.imageUrl(caseId, image.image_id)} alt="Uploaded de-identified medical image" />
            {heatUrl && (
              <img
                key={heatUrl}
                className="heat"
                src={heatUrl}
                alt={`Occlusion heatmap for ${heatFinding}`}
                data-shown={loadedHeat === heatUrl}
                onLoad={() => setLoadedHeat(heatUrl)}
                onError={() => setHeatError("Heatmap unavailable for this image.")}
              />
            )}
            {heatFinding && loadedHeat === heatUrl && <span className="plate-label">Where the model looks: {heatFinding}</span>}
            {heatUrl && loadedHeat !== heatUrl && !heatError && (
              <span className="plate-loading">
                <Loader2 size={14} className="spin" aria-hidden="true" /> Computing heatmap for {heatFinding}…
              </span>
            )}
            {heatError && <span className="plate-loading">{heatError}</span>}
          </>
        ) : (
          <div className="plate-empty">No image uploaded for this case.</div>
        )}
      </figure>

      {isCxr && candidates.length > 0 && (
        <div className="heat-controls" role="group" aria-label="Show heatmap for finding">
          <span className="heat-label">Heatmap</span>
          <button className="pill pill-off" aria-pressed={heatFinding === null} onClick={() => onHeatFinding(null)}>
            Off
          </button>
          {candidates.map((f) => (
            <button
              key={f.finding}
              className="pill"
              aria-pressed={heatFinding === f.finding}
              onClick={() => onHeatFinding(heatFinding === f.finding ? null : f.finding)}
            >
              {f.finding}
            </button>
          ))}
        </div>
      )}
      {analysis && (
        <p className="modality">
          Modality check: <b>{analysis.modality.modality}</b> ({Math.round(analysis.modality.confidence * 100)}%)
          {isCxr ? " · chest X-ray classifier ran" : " · chest X-ray classifier not run"}
        </p>
      )}
    </section>
  );
}
