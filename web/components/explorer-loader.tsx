"use client";
import dynamic from "next/dynamic";

/** three.js and WebGL need the browser; render the explorer client-side only. */
const Explorer = dynamic(() => import("./explorer"), { ssr: false, loading: () => <div className="fixed inset-0 bg-black" /> });
export default function ExplorerLoader() { return <Explorer />; }
