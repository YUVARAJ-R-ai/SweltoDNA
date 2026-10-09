"use client";
import { useUI } from "@/lib/explorer-context";
export function Hint() {
  const gone = useUI((s) => s.hintGone);
  return (
    <div className="fixed bottom-7 left-1/2 z-[4] -translate-x-1/2 whitespace-nowrap text-[13px] text-[var(--label2)] transition-opacity duration-500" style={{ opacity: gone ? 0 : 1 }}>
      Scroll or pinch to zoom · drag to look around · <kbd className="sv-kbd">P</kbd> guided tour
    </div>
  );
}
