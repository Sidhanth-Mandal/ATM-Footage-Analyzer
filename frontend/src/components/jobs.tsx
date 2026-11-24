import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { get, post } from "../api/client";
import type { JobOut } from "../api/types";
import { useAuth } from "../auth";
import { StatusChip } from "./face";
import { Button, Progress } from "./ui";

const ACTIVE = new Set(["queued", "running", "cancel_requested"]);

export function useCaseJobs(caseId?: string) {
  return useQuery({
    queryKey: ["jobs", caseId],
    queryFn: () => get<JobOut[]>(`/cases/${caseId}/jobs`),
    enabled: !!caseId,
    refetchInterval: (q) => ((q.state.data || []).some((j) => ACTIVE.has(j.status)) ? 1000 : 5000),
  });
}

/** Wait for one job to finish (1 s polling). */
export function useJob(id?: number | null) {
  return useQuery({
    queryKey: ["job", id],
    queryFn: () => get<JobOut>(`/jobs/${id}`),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data && !ACTIVE.has(q.state.data.status) ? false : 1000),
  });
}

/** Floating list of active jobs for a case; invalidates case data whenever a job finishes. */
export function JobToasts({ caseId }: { caseId?: string }) {
  const { data } = useCaseJobs(caseId);
  const qc = useQueryClient();
  const { can } = useAuth();
  const seen = useRef<Record<number, string>>({});
  useEffect(() => {
    let changed = false;
    for (const j of data || []) {
      const prev = seen.current[j.id];
      if (prev && prev !== j.status && !ACTIVE.has(j.status)) changed = true;
      seen.current[j.id] = j.status;
    }
    if (changed) qc.invalidateQueries({ predicate: (q) => q.queryKey[0] !== "jobs" });
  }, [data, qc]);
  const active = (data || []).filter((j) => ACTIVE.has(j.status));
  const failed = (data || []).filter((j) => j.status === "failed").slice(0, 2);
  if (!active.length && !failed.length) return null;
  return (
    <div className="fixed bottom-4 right-4 z-30 w-80 space-y-2" aria-live="polite">
      {active.map((j) => (
        <div key={j.id} className="rounded-card border border-line bg-surface p-3 text-xs shadow-lg shadow-black/10">
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="font-semibold">
              {j.type} <span className="font-mono font-normal text-muted">#{j.id}</span>
            </span>
            <StatusChip status={j.status} />
          </div>
          <Progress value={j.progress} />
          <div className="mt-1.5 flex items-center justify-between gap-2 text-muted">
            <span className="truncate">{j.message || "waiting for the GPU worker"}</span>
            {can("investigator") && j.status !== "cancel_requested" && (
              <Button size="sm" variant="ghost" onClick={() => post(`/jobs/${j.id}/cancel`).then(() => qc.invalidateQueries({ queryKey: ["jobs"] }))}>
                Cancel
              </Button>
            )}
          </div>
        </div>
      ))}
      {failed.map((j) => (
        <div key={j.id} className="rounded-card border border-err/40 bg-surface p-3 text-xs text-err shadow-lg shadow-black/10">
          <b>{j.type}</b> #{j.id} failed: {j.message}
        </div>
      ))}
    </div>
  );
}
