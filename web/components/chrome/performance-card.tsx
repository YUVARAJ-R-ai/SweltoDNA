"use client";
import { useExplorer, useUI } from "@/lib/explorer-context";

/** #9 telemetry HUD. Reads per-response telemetry; labelled "Simulated" unless the server says it measured it. */
export function PerformanceCard() {
  const open = useUI((s) => s.perfOpen);
  const t = useExplorer((s) => s.telemetry);
  const latency = useExplorer((s) => s.latencyMs);
  const measured = t?.source === "measured";
  const svelto = t ? t.vanillaLatencyMs / t.speedup : null;
  return (
    <section aria-label="Performance" data-off={!open}
      className="sv-mat sv-regular sv-panel fixed bottom-[188px] left-4 z-[5] w-[272px] origin-[20px_100%] rounded-[20px] px-4 py-3.5 data-[off=true]:translate-y-2.5 data-[off=true]:scale-[.97]">
      <h4 className="text-[13px] font-semibold">Speculative verification</h4>
      <div className="text-[11px] text-[var(--label2)]" data-testid="perf-source">{t ? (measured ? "Measured · updates on each edit" : "Simulated · updates on each edit") : "Edit a base to see figures"}</div>
      <div className="sv-num my-2.5 text-[34px] font-semibold leading-none tracking-[-0.03em]" style={{ color: t && t.speedup > 2 ? "var(--green)" : "var(--label)" }}>
        {t ? `${t.speedup.toFixed(1)}×` : "—"}
      </div>
      <Row label="Latency" value={svelto ? `${svelto.toFixed(0)} ms` : "—"} extra={t ? `vs ${t.vanillaLatencyMs.toFixed(0)} ms` : ""} />
      <Row label="Draft acceptance" value={t ? `${(t.acceptanceRate * 100).toFixed(0)}%` : "—"} />
      <Row label="FLOPs saved" value={t ? `${(t.flopsSaved * 100).toFixed(0)}%` : "—"} />
      <div className="sv-hr my-2" />
      {measured && t?.basis && <p className="mb-1 text-[11px] leading-snug text-[var(--label3)]" data-testid="perf-basis">{t.basis}</p>}
      <Row label="Scorer time · measured" value={latency === null ? "—" : latency < 1 ? `${Math.max(1, Math.round(latency * 1000))} µs` : `${latency.toFixed(2)} ms`} />
    </section>
  );
}
function Row({ label, value, extra }: { label: string; value: string; extra?: string }) {
  return (
    <div className="flex h-7 items-baseline justify-between text-[13px] text-[var(--label2)]">
      <span>{label}</span>
      <span><b className="sv-num font-semibold text-[var(--label)]">{value}</b>{extra ? <span className="sv-num"> {extra}</span> : null}</span>
    </div>
  );
}
