import { useEffect, useState, type ImgHTMLAttributes } from "react";
import { mediaUrl, onMediaToken } from "../api/client";
import { Badge, cn } from "./ui";

export const FLAG_HELP: Record<string, string> = {
  MASK: "Face mask or lower face covered",
  FULL_COVER: "Face covered except the eyes (e.g. niqab); not used for search",
  SUNGLASSES: "Sunglasses",
  CAP_HAT: "Cap or hat",
  HELMET: "Helmet",
  HOOD_SCARF: "Hood, scarf or hijab (hair covering; informational)",
  PARTIAL_FACE: "Face cut by the frame edge or head out of frame",
  EXTREME_POSE: "Head turned or tilted strongly",
  LOW_RES: "Low resolution (inter-ocular distance < 25 px)",
  MOTION_BLUR: "Blurred",
  LOW_LIGHT: "Dark face",
  BACKLIT: "Backlit: dark face against a bright background",
  ENHANCED: "Enhanced image, not original pixels",
};

const SEVERE = new Set(["FULL_COVER", "MASK", "HELMET", "PARTIAL_FACE", "ENHANCED"]);
const INFO = new Set(["HOOD_SCARF"]);

export function FlagChips({ flags, className }: { flags: string[]; className?: string }) {
  if (!flags?.length) return <span className="text-[11px] text-muted">no flags</span>;
  return (
    <div className={cn("flex flex-wrap gap-1", className)}>
      {flags.map((f) => (
        <Badge key={f} tone={SEVERE.has(f) ? "red" : INFO.has(f) ? "accent" : "amber"} title={FLAG_HELP[f]}>
          {f}
        </Badge>
      ))}
    </div>
  );
}

export function GradeBadge({ grade, total }: { grade: string; total?: number }) {
  const tone = grade === "A" ? "bg-ok/15 text-ok ring-ok/40" : grade === "B" ? "bg-warn/15 text-warn ring-warn/40" : "bg-err/15 text-err ring-err/40";
  return (
    <span title={`Quality grade ${grade}`} className={cn("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-bold ring-1 ring-inset", tone)}>
      {grade}
      {total !== undefined && <span className="font-mono font-normal">{total.toFixed(0)}</span>}
    </span>
  );
}

const SUBS = ["fiqa", "sharpness", "pose", "illumination", "occlusion", "resolution"];
export function SubScoreBars({ sub }: { sub: Record<string, number> }) {
  return (
    <div className="space-y-1">
      {SUBS.map((k) => {
        const v = Number(sub?.[k] ?? 0);
        return (
          <div key={k} className="flex items-center gap-2 text-[10px] text-muted">
            <span className="w-16 shrink-0 font-mono">{k}</span>
            <div className="h-1.5 flex-1 rounded-full bg-raised">
              <div className={cn("h-full rounded-full", v >= 0.66 ? "bg-ok" : v >= 0.33 ? "bg-warn" : "bg-err")} style={{ width: `${v * 100}%` }} />
            </div>
            <span className="w-7 text-right font-mono tabular-nums">{v.toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}

/** <img> for an authorised media path; re-renders when the media token is refreshed. */
export function AuthImg({ path, extra, enhanced, className, ...p }: { path?: string | null; extra?: Record<string, string | number>; enhanced?: boolean } & Omit<ImgHTMLAttributes<HTMLImageElement>, "src">) {
  const [, bump] = useState(0);
  useEffect(() => {
    const off = onMediaToken(() => bump((x) => x + 1));
    return () => {
      off();
    };
  }, []);
  const src = mediaUrl(path, extra);
  if (!src) return <div className={cn("bg-raised", className)} />;
  const img = <img {...p} src={src} className={cn(enhanced && "enhanced-frame", className)} loading="lazy" />;
  if (!enhanced) return img;
  return (
    <div className="relative inline-block">
      {img}
      <span className="absolute left-0 top-0 bg-err px-1 text-[9px] font-bold text-white">ENHANCED · NOT ORIGINAL</span>
    </div>
  );
}

type Tone = "neutral" | "accent" | "green" | "red";
/** Plain-language status labels; the raw backend value stays available as a tooltip. */
const STATUS: Record<string, [string, Tone]> = {
  uploaded: ["Uploaded", "neutral"],
  ingesting: ["Preparing video", "neutral"],
  ingested: ["Ready for setup", "accent"],
  tracking: ["Finding people", "neutral"],
  tracked: ["People found", "green"],
  indexing: ["Indexing for search", "neutral"],
  indexed: ["Searchable", "green"],
  pending: ["Waiting to rank", "neutral"],
  linked: ["Candidates ready", "accent"],
  confirmed: ["Customer confirmed", "green"],
  analysed: ["Faces extracted", "green"],
  queued: ["Queued", "neutral"],
  running: ["Running", "neutral"],
  done: ["Done", "green"],
  cancel_requested: ["Cancelling", "neutral"],
  cancelled: ["Cancelled", "neutral"],
  failed: ["Failed", "red"],
  blocked: ["Blocked", "red"],
  open: ["Open", "accent"],
};

export function statusLabel(status: string) {
  return STATUS[status]?.[0] ?? status;
}

export function StatusChip({ status }: { status: string }) {
  const [label, tone] = STATUS[status] ?? [status, "neutral"];
  return (
    <Badge tone={tone} title={`status: ${status}`}>
      {label}
    </Badge>
  );
}

export function fmtTime(s: number | null | undefined) {
  if (s === null || s === undefined || Number.isNaN(s)) return "-";
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r.toFixed(1).padStart(4, "0")}`;
}

export function fmtBytes(n: number) {
  if (n > 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n > 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${(n / 1e3).toFixed(0)} kB`;
}
