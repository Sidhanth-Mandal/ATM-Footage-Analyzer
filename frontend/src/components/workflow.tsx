// Guided case workflow: step progress derived from existing case data, plus the stepper, hint and page frame.
import { useQuery } from "@tanstack/react-query";
import { useEffect, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { get } from "../api/client";
import type { CaseOut, ClipOut, ExportOut, SearchOut, TxnOut } from "../api/types";
import { Button, buttonClass, cn } from "./ui";

export type StepId = 1 | 2 | 3 | 4 | 5 | 6 | 7;
export type StepState = "complete" | "available" | "locked";

export interface StepInfo {
  id: StepId;
  label: string;
  state: StepState;
  to?: string;
}

export interface CaseProgress {
  steps: StepInfo[];
  firstIncomplete: StepId | null;
  latest: { txnClip?: ClipOut; txn?: TxnOut; confirmedTxn?: TxnOut; search?: SearchOut };
}

export const STEP_LABELS: Record<StepId, string> = {
  1: "Upload footage",
  2: "Set up clip",
  3: "Add transaction",
  4: "Pick the customer",
  5: "Review faces",
  6: "Search other clips",
  7: "Export",
};

/** Default "what to do next" sentence per step, used when a page has no more specific hint. */
export const STEP_HINT: Record<StepId, string> = {
  1: "Upload the ATM video for this case.",
  2: "Open the transaction clip and set its time mapping.",
  3: "Enter the journal time of the withdrawal so the system can rank who was at the machine.",
  4: "Watch the candidates and confirm which person made the withdrawal.",
  5: "Choose the clearest faces of the customer.",
  6: "Search the other clips in this case for the same person.",
  7: "Create the PDF evidence sheet.",
};

const byDate = <T extends { created_at: string }>(a: T, b: T) => b.created_at.localeCompare(a.created_at);

/** Pure: works out each step's state from case data the frontend already loads. */
export function deriveProgress(caseId: string, d: { clips: ClipOut[]; txns: TxnOut[]; searches: SearchOut[]; exports: ExportOut[] }): CaseProgress {
  const clips = [...d.clips].sort(byDate);
  const txnClips = clips.filter((c) => c.kind === "transaction");
  const txnClip = txnClips[0] ?? clips[0];
  const txns = [...d.txns].sort((a, b) => b.id - a.id);
  const txn = txns[0];
  const confirmedTxn = txns.find((t) => t.chosen_track_id != null);
  const search = [...d.searches].sort(byDate)[0];

  const facesRoute = confirmedTxn ? `/tracks/${confirmedTxn.chosen_track_id}/faces` : undefined;
  const rules: [StepId, boolean, boolean, string | undefined][] = [
    // id, available, done, route
    [1, true, clips.length > 0, `/cases/${caseId}#footage`],
    [2, clips.length > 0, txnClips.some((c) => c.timemap != null), txnClip && `/clips/${txnClip.id}/setup`],
    [3, txnClips.length > 0, txns.length > 0, txnClips[0] && `/clips/${txnClips[0].id}/setup#transactions`],
    [4, txns.length > 0, !!confirmedTxn, txn && `/transactions/${txn.id}`],
    [5, !!confirmedTxn, d.searches.length > 0 || d.exports.length > 0, facesRoute],
    [6, !!confirmedTxn, d.searches.length > 0, search ? `/search/${search.search_id}` : facesRoute],
    [7, !!confirmedTxn, d.exports.some((e) => e.status === "done"), `/cases/${caseId}/export`],
  ];
  const steps = rules.map(([id, available, done, to]): StepInfo => {
    const state: StepState = !available ? "locked" : done ? "complete" : "available";
    return { id, label: STEP_LABELS[id], state, to: state === "locked" ? undefined : to };
  });
  const firstIncomplete = steps.find((s) => s.state !== "complete")?.id ?? null;
  return { steps, firstIncomplete, latest: { txnClip, txn, confirmedTxn, search } };
}

/** Runs the existing case-level queries (same keys and endpoints as CaseDetail) and returns the progress. */
export function useCaseProgress(caseId?: string) {
  const on = !!caseId;
  const clips = useQuery({ queryKey: ["clips", caseId], queryFn: () => get<ClipOut[]>(`/cases/${caseId}/clips`), enabled: on });
  const txns = useQuery({ queryKey: ["txns", caseId], queryFn: () => get<TxnOut[]>(`/cases/${caseId}/transactions`), enabled: on });
  const searches = useQuery({ queryKey: ["searches", caseId], queryFn: () => get<SearchOut[]>(`/cases/${caseId}/searches`), enabled: on });
  const exports = useQuery({ queryKey: ["exports", caseId], queryFn: () => get<ExportOut[]>(`/cases/${caseId}/exports`), enabled: on });
  const progress =
    caseId && clips.data && txns.data && searches.data && exports.data
      ? deriveProgress(caseId, { clips: clips.data, txns: txns.data, searches: searches.data, exports: exports.data })
      : undefined;
  return {
    progress,
    isLoading: clips.isLoading || txns.isLoading || searches.isLoading || exports.isLoading,
    error: clips.error || txns.error || searches.error || exports.error,
  };
}

export function CaseStepper({ progress, current }: { progress?: CaseProgress; current?: StepId }) {
  const steps: StepInfo[] = progress?.steps ?? (Object.keys(STEP_LABELS).map(Number) as StepId[]).map((id) => ({ id, label: STEP_LABELS[id], state: "available" }));
  return (
    <ol aria-label="Case progress" className="flex items-start">
      {steps.map((s, i) => {
        const isCurrent = s.id === current;
        const done = s.state === "complete";
        const locked = s.state === "locked";
        const circle = (
          <span
            className={cn(
              "flex h-7 w-7 shrink-0 items-center justify-center rounded-full font-mono text-xs",
              done && "bg-accent text-white",
              !done && isCurrent && "text-accent ring-2 ring-accent",
              !done && !isCurrent && "border border-line text-muted",
              locked && "opacity-50",
            )}
          >
            {done ? "✓" : s.id}
          </span>
        );
        const label = <span className={cn("mt-1.5 text-center text-xs leading-tight", isCurrent ? "font-semibold text-fg" : done ? "text-fg" : "text-muted", locked && "opacity-50")}>{s.label}</span>;
        const inner = (
          <>
            {circle}
            {label}
          </>
        );
        const nextDone = done && steps[i + 1]?.state === "complete";
        return (
          <li key={s.id} className="relative flex min-w-24 flex-1 flex-col items-center" aria-current={isCurrent ? "step" : undefined}>
            {i < steps.length - 1 && <span aria-hidden className={cn("absolute left-[calc(50%+20px)] right-[calc(-50%+20px)] top-[18px] h-px", nextDone ? "bg-accent" : "bg-line")} />}
            {s.to && !locked ? (
              <Link to={s.to} className="relative flex flex-col items-center rounded-control px-1 py-1 hover:bg-raised focus:outline-none focus-visible:ring-2 focus-visible:ring-accent" title={`Step ${s.id}: ${s.label}`}>
                {inner}
              </Link>
            ) : (
              <span className="relative flex flex-col items-center px-1 py-1" aria-disabled={locked || undefined} title={locked ? `Step ${s.id}: finish the earlier steps first` : `Step ${s.id}: ${s.label}`}>
                {inner}
              </span>
            )}
          </li>
        );
      })}
    </ol>
  );
}

export interface HintAction {
  label: string;
  to?: string;
  onClick?: () => void;
  disabled?: boolean;
}
export interface Hint {
  text: ReactNode;
  action?: HintAction;
}

export function NextStepHint({ text, action }: Hint) {
  return (
    <div className="flex flex-wrap items-center gap-4 rounded-card border border-accent/60 bg-accent/10 px-5 py-4">
      <span className="font-mono text-[11px] uppercase tracking-widest text-accent">Next</span>
      <p className="min-w-0 flex-1 text-sm text-fg">{text}</p>
      {action &&
        (action.to ? (
          <Link to={action.to} className={buttonClass("primary")}>
            {action.label}
          </Link>
        ) : (
          <Button onClick={action.onClick} disabled={action.disabled}>
            {action.label}
          </Button>
        ))}
    </div>
  );
}

/** Shared frame for every page that belongs to a case: back link, stepper, hint, content. */
export function CaseFrame({ caseId, current, hint, progress, children }: { caseId?: string; current?: StepId; hint?: Hint; progress?: CaseProgress; children: ReactNode }) {
  const caseQ = useQuery({ queryKey: ["case", caseId], queryFn: () => get<CaseOut>(`/cases/${caseId}`), enabled: !!caseId });
  const own = useCaseProgress(progress ? undefined : caseId);
  const p = progress ?? own.progress;
  const { hash } = useLocation();
  useEffect(() => {
    if (hash) document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: "smooth" });
  }, [hash]);
  return (
    <div className="space-y-6">
      <div className="space-y-4">
        {caseId && caseQ.data ? (
          <Link to={`/cases/${caseId}`} className="inline-flex items-center gap-2 text-sm text-muted hover:text-fg">
            <span aria-hidden>←</span>
            <span className="font-mono text-xs text-accent">{caseQ.data.ref_no}</span>
            <span>{caseQ.data.title}</span>
          </Link>
        ) : (
          <div className="h-5" />
        )}
        <div className="overflow-x-auto rounded-card border border-line bg-surface px-4 py-4">
          <CaseStepper progress={p} current={current} />
        </div>
        {hint && <NextStepHint {...hint} />}
      </div>
      {children}
    </div>
  );
}
