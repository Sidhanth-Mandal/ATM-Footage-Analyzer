// Small shadcn-style primitives built on Tailwind and the design tokens in index.css.
import clsx from "clsx";
import { useEffect, useId, useState, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes } from "react";

export const cn = clsx;

const focusRing = "focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg";

type BtnVariant = "primary" | "secondary" | "ghost" | "danger" | "outline";

/** Button look, also used for router links that act as buttons. */
export function buttonClass(variant: BtnVariant = "primary", size: "sm" | "md" = "md", className?: string) {
  return cn(
    "inline-flex items-center justify-center gap-1.5 rounded-control font-medium whitespace-nowrap transition-colors disabled:pointer-events-none disabled:opacity-40",
    focusRing,
    size === "sm" ? "h-8 px-3 text-xs" : "h-9 px-4 text-sm",
    variant === "primary" && "bg-accent text-white hover:bg-accent-deep",
    variant === "secondary" && "bg-raised text-fg hover:bg-raised/70",
    variant === "outline" && "border border-line text-fg hover:bg-raised",
    variant === "ghost" && "text-muted hover:bg-raised hover:text-fg",
    variant === "danger" && "border border-err/40 bg-err/15 text-err hover:bg-err/25",
    className,
  );
}

export function Button({ variant = "primary", size = "md", className, ...p }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: BtnVariant; size?: "sm" | "md" }) {
  return <button {...p} className={buttonClass(variant, size, className)} />;
}

export function Card({ title, description, actions, children, className }: { title?: ReactNode; description?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn("rounded-card border border-line bg-surface", className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 px-5 pt-4">
          <div className="min-w-0">
            {title && <h2 className="text-sm font-semibold text-fg">{title}</h2>}
            {description && <p className="mt-0.5 text-xs text-muted">{description}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

type Tone = "neutral" | "accent" | "green" | "amber" | "red" | "slate" | "sky" | "violet";
const TONES: Record<Tone, string> = {
  neutral: "bg-raised text-muted ring-line",
  slate: "bg-raised text-muted ring-line",
  accent: "bg-accent/10 text-accent ring-accent/25",
  sky: "bg-accent/10 text-accent ring-accent/25",
  violet: "bg-accent/10 text-accent ring-accent/25",
  green: "bg-ok/10 text-ok ring-ok/30",
  amber: "bg-warn/10 text-warn ring-warn/30",
  red: "bg-err/10 text-err ring-err/30",
};

export function Badge({ children, tone = "neutral", title, className }: { children: ReactNode; tone?: Tone; title?: string; className?: string }) {
  return (
    <span title={title} className={cn("inline-flex items-center whitespace-nowrap rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset", TONES[tone], className)}>
      {children}
    </span>
  );
}

const field = "rounded-control border border-line bg-surface text-sm text-fg placeholder:text-muted focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent disabled:opacity-50";

export function Input(p: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...p} className={cn("h-9 w-full px-3", field, p.className)} />;
}

export function Select(p: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...p} className={cn("h-9 px-2", field, p.className)} />;
}

export function Label({ children, hint }: { children: ReactNode; hint?: ReactNode }) {
  return (
    <label className="mb-1.5 block font-mono text-[11px] uppercase tracking-wide text-muted">
      {children}
      {hint && <span className="ml-1.5 font-sans normal-case tracking-normal text-muted">{hint}</span>}
    </label>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <span role="status" aria-label="loading" className={cn("inline-block h-4 w-4 animate-spin rounded-full border-2 border-raised border-t-accent", className)} />;
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return <div className="mt-2 rounded-control border border-err/40 bg-err/10 px-3 py-2 text-sm text-err">{msg}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="py-6 text-center text-sm text-muted">{children}</div>;
}

function CloseBtn({ onClose }: { onClose: () => void }) {
  return (
    <button className={cn("rounded-control px-2 py-1 text-muted hover:bg-raised hover:text-fg", focusRing)} onClick={onClose} aria-label="close">
      ✕
    </button>
  );
}

export function Dialog({ open, onClose, title, children, wide }: { open: boolean; onClose: () => void; title: ReactNode; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    if (open) window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/40 p-6" onMouseDown={onClose}>
      <div role="dialog" aria-modal="true" className={cn("w-full rounded-card border border-line bg-surface shadow-2xl shadow-black/20", wide ? "max-w-5xl" : "max-w-lg")} onMouseDown={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-5 pt-4">
          <h3 className="font-semibold">{title}</h3>
          <CloseBtn onClose={onClose} />
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}

export function Drawer({ open, onClose, title, children }: { open: boolean; onClose: () => void; title: ReactNode; children: ReactNode }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/30" onMouseDown={onClose}>
      <aside role="dialog" aria-modal="true" className="h-full w-full max-w-2xl overflow-y-auto border-l border-line bg-surface" onMouseDown={(e) => e.stopPropagation()}>
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-line bg-surface px-5 py-3">
          <h3 className="font-semibold">{title}</h3>
          <CloseBtn onClose={onClose} />
        </div>
        <div className="p-5">{children}</div>
      </aside>
    </div>
  );
}

export function Progress({ value, className }: { value: number; className?: string }) {
  return (
    <div className={cn("h-1.5 w-full overflow-hidden rounded-full bg-raised", className)}>
      <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%` }} />
    </div>
  );
}

export function KV({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[max-content_1fr] items-baseline gap-x-4 gap-y-1.5 text-sm">
      {items.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="font-mono text-[11px] uppercase tracking-wide text-muted">{k}</dt>
          <dd className="min-w-0 break-all text-fg">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Mono({ children }: { children: ReactNode }) {
  return <code className="break-all font-mono text-[11px] text-muted">{children}</code>;
}

/** Collapsed-by-default disclosure for forensic detail. */
export function Details({ children, label = "details", className }: { children: ReactNode; label?: string; className?: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <div className={className}>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
        className={cn("inline-flex items-center gap-1 rounded-control py-0.5 text-xs text-accent hover:underline", focusRing)}
      >
        <span aria-hidden className={cn("inline-block transition-transform", open && "rotate-90")}>
          ›
        </span>
        {open ? "Hide" : "Show"} {label}
      </button>
      {open && (
        <div id={id} role="region" className="mt-2 rounded-control border border-line bg-bg p-3 text-xs" onClick={(e) => e.stopPropagation()}>
          {children}
        </div>
      )}
    </div>
  );
}

/** Page title and one-sentence purpose. */
export function PageHeader({ title, description, meta, actions }: { title: ReactNode; description: ReactNode; meta?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0 space-y-1">
        <h1 className="flex flex-wrap items-center gap-2 text-2xl font-semibold tracking-tight">{title}</h1>
        <p className="max-w-3xl text-sm text-muted">{description}</p>
        {meta && <div className="text-sm text-fg">{meta}</div>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

/** Labeled group of cards inside a page. */
export function Section({ title, description, children, id }: { title: string; description?: ReactNode; children: ReactNode; id?: string }) {
  return (
    <section id={id} className="scroll-mt-6 space-y-3">
      <div>
        <h2 className="font-mono text-xs uppercase tracking-widest text-accent">{title}</h2>
        {description && <p className="mt-1 text-sm text-muted">{description}</p>}
      </div>
      {children}
    </section>
  );
}

/** Shared table header cell and row classes. */
export const th = "pb-2 pr-3 text-left font-mono text-[11px] font-normal uppercase tracking-wide text-muted";
export const tr = "border-t border-line";
export const link = "text-accent hover:underline";
