"use client";
import dynamic from "next/dynamic";
import { useEffect, useState, type ReactNode } from "react";
import type { Region } from "@/lib/splice/sequence";
import type { SpliceClient } from "@/lib/splice/client";

/** three.js and WebGL need the browser; render the explorer client-side only. */
const Explorer = dynamic(() => import("./explorer"), { ssr: false, loading: () => <div className="fixed inset-0 bg-black" /> });

type State = "local" | "connecting" | { error: string } | { region: Region; client: SpliceClient };

/** With NEXT_PUBLIC_SPLICE_WS set, connect to the #6 server first and use the region it announces. */
export default function ExplorerLoader() {
  const url = process.env.NEXT_PUBLIC_SPLICE_WS;
  const [state, setState] = useState<State>(url ? "connecting" : "local");
  useEffect(() => {
    if (!url) return;
    let alive = true;
    (async () => {
      const { WebSocketClient } = await import("@/lib/splice/client");
      const client = new WebSocketClient(url, crypto.randomUUID());
      const timeout = new Promise<never>((_, rej) => setTimeout(() => rej(new Error("no reply after 8 s")), 8000));
      try {
        const h = await Promise.race([client.hello, timeout]);
        if (alive) setState({ region: h.region, client });
      } catch (e) {
        client.close();
        if (alive) setState({ error: `Could not reach the scoring server at ${url} (${(e as Error).message}).` });
      }
    })();
    return () => { alive = false; };
  }, [url]);

  if (state === "local") return <Explorer />;
  if (state === "connecting") return <Status>Connecting to the scoring server…</Status>;
  if ("error" in state) return (
    <Status>
      <p>{state.error}</p>
      <button className="mt-4 h-9 rounded-full bg-[#0A84FF] px-4 text-sm font-medium text-white" onClick={() => setState("local")}>Use the built-in heuristic instead</button>
    </Status>
  );
  return <Explorer region={state.region} client={state.client} />;
}

function Status({ children }: { children: ReactNode }) {
  return <div role="status" className="fixed inset-0 flex flex-col items-center justify-center bg-black px-6 text-center text-sm text-[rgba(235,235,245,.64)]">{children}</div>;
}
