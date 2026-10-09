import type { Consequence } from "@/lib/splice/scorer";

/** Predicted transcript: neighbouring exons with the edited exon extended, shortened, or skipped. Heuristic. */
export function TranscriptDiagram({ c }: { c: Consequence }) {
  if (c.kind === "gain") return (<><div className="sv-eyebrow">Predicted transcript</div><div className="mt-0.5 text-[13px]">A new {c.site} site may compete with the canonical one.</div></>);
  const y = 30, P = [16, 66], X = [118, 184], N = [236, 282], dim = "#636366", lost = "rgba(255,69,58,.8)";
  const box = (x0: number, x1: number, fill: string, key: string) => <rect key={key} x={x0} y={y - 9} width={x1 - x0} height={18} rx={4} fill={fill} />;
  const dashed = (x0: number, w: number, key: string) => <rect key={key} x={x0} y={y - 9} width={w} height={18} rx={3} fill="none" stroke={lost} strokeDasharray="3 3" />;
  const arc = (x0: number, x1: number, h: number, key: string, color = "rgba(235,235,245,.55)") =>
    <path key={key} d={`M${x0} ${y - 10} C ${x0 + (x1 - x0) * 0.3} ${y - h}, ${x1 - (x1 - x0) * 0.3} ${y - h}, ${x1} ${y - 10}`} fill="none" stroke={color} strokeWidth={1.4} />;
  const parts = [<line key="l" x1={0} y1={y} x2={292} y2={y} stroke="rgba(235,235,245,.25)" strokeWidth={1.5} />, box(P[0], P[1], dim, "p"), box(N[0], N[1], dim, "n")];
  let caption: string, detail: string;
  if (c.kind === "skip") {
    parts.push(dashed(X[0], X[1] - X[0], "x"), arc(P[1], N[0], 34, "a", "#fff"));
    caption = `Exon ${c.exonNumber} skipped`; detail = `${c.frameshift ? "frameshift" : "in frame"} (${c.exonLength} nt)`;
  } else {
    const px = Math.min(40, Math.max(6, Math.abs(c.nt) * 1.8)), ext = c.nt > 0;
    let x0 = X[0], x1 = X[1];
    if (c.side === "donor") { if (ext) { parts.push(box(X[1], X[1] + px, "#FF9F0A", "e")); x1 = X[1] + px; } else { parts.push(dashed(X[1] - px, px, "e")); x1 = X[1] - px; } }
    else { if (ext) { parts.push(box(X[0] - px, X[0], "#FF9F0A", "e")); x0 = X[0] - px; } else { parts.push(dashed(X[0], px, "e")); x0 = X[0] + px; } }
    const bx0 = c.side === "donor" ? X[0] : ext ? X[0] : x0, bx1 = c.side === "donor" ? (ext ? X[1] : x1) : X[1];
    parts.push(box(bx0, bx1, "#d1d1d6", "x"), arc(P[1], x0, 22, "a1"), arc(x1, N[0], 22, "a2"));
    caption = `Exon ${c.exonNumber} ${ext ? "extended" : "shortened"} by ${Math.abs(c.nt)} nt`; detail = c.frameshift ? "frameshift" : "in frame";
  }
  const label = (x: number, text: string, strong = false) => <text x={x} y={58} fill={strong ? "#f5f5f7" : "rgba(235,235,245,.4)"} fontSize={10} fontWeight={strong ? 600 : 400} textAnchor="middle">{text}</text>;
  const n = c.exonNumber;
  return (
    <div className="mt-2.5">
      <div className="sv-eyebrow">Predicted transcript</div>
      <svg viewBox="0 0 292 64" className="mt-1.5 block h-auto w-full" fontFamily="var(--sys)" role="img" aria-label={`${caption}, ${detail}`}>
        {parts}
        {label((P[0] + P[1]) / 2, `exon ${n - 1}`)}{label((X[0] + X[1]) / 2, `exon ${n}`, true)}{label((N[0] + N[1]) / 2, `exon ${n + 1}`)}
      </svg>
      <div className="mt-0.5 text-[13px]" data-testid="consequence">{caption} · <span className="text-[var(--label2)]">{detail}</span></div>
    </div>
  );
}
