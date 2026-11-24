import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { fmtErr, get, mediaUrl, post } from "../api/client";
import type { ExportOut, FaceOut, SearchOut, TxnOut } from "../api/types";
import { useAuth } from "../auth";
import { AuthImg, FlagChips, GradeBadge, StatusChip } from "../components/face";
import { JobToasts, useJob } from "../components/jobs";
import { Badge, Button, buttonClass, Card, cn, Details, Empty, ErrorBox, KV, Mono, PageHeader, Spinner } from "../components/ui";
import { CaseFrame, type Hint } from "../components/workflow";

function ExportBuilder({ caseId, onSelection }: { caseId: string; onSelection: (n: number) => void }) {
  const [params] = useSearchParams();
  const txns = useQuery({ queryKey: ["txns", caseId], queryFn: () => get<TxnOut[]>(`/cases/${caseId}/transactions`) });
  const chosen = (txns.data || []).filter((t) => t.chosen_track_id);
  const faceQs = useQueries({
    queries: chosen.map((t) => ({ queryKey: ["faces", String(t.chosen_track_id), false], queryFn: () => get<FaceOut[]>(`/tracks/${t.chosen_track_id}/faces?selected=true`) })),
  });
  const searches = useQuery({ queryKey: ["searches", caseId], queryFn: () => get<SearchOut[]>(`/cases/${caseId}/searches`) });
  const allFaces = useMemo(() => faceQs.flatMap((q) => q.data || []), [faceQs]);
  const hits = useMemo(() => (searches.data || []).flatMap((s) => s.hits.filter((h) => h.review === "confirmed")), [searches.data]);
  const [faces, setFaces] = useState<Set<number>>(new Set());
  const [hitSel, setHitSel] = useState<Set<number>>(new Set());
  const [enh, setEnh] = useState(false);
  const [jobId, setJobId] = useState<number | null>(null);
  const job = useJob(jobId);
  const qc = useQueryClient();
  useEffect(() => {
    const pre = params.get("faces");
    if (pre) setFaces(new Set(pre.split(",").map(Number)));
    else if (allFaces.length && !faces.size) setFaces(new Set(allFaces.map((f) => f.id)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params, allFaces.length]);
  useEffect(() => {
    if (hits.length && !hitSel.size) setHitSel(new Set(hits.map((h) => h.id)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hits.length]);
  useEffect(() => {
    if (job.data?.status === "done") qc.invalidateQueries({ queryKey: ["exports", caseId] });
  }, [job.data?.status, qc, caseId]);
  useEffect(() => onSelection(faces.size + hitSel.size), [faces.size, hitSel.size, onSelection]);
  const m = useMutation({
    mutationFn: () => post<ExportOut>(`/cases/${caseId}/exports`, { face_ids: [...faces], search_hit_ids: [...hitSel], include_enhanced: enh }),
    onSuccess: (e) => setJobId(e.job_id ?? null),
  });
  const tog = (s: Set<number>, set: (x: Set<number>) => void, id: number) => {
    const n = new Set(s);
    n.has(id) ? n.delete(id) : n.add(id);
    set(n);
  };
  return (
    <Card title="Build evidence sheet" description="Everything selected here goes into the PDF.">
      <div className="mb-2 font-mono text-[11px] uppercase tracking-wide text-muted">Best frames of confirmed customers</div>
      {!allFaces.length ? (
        <Empty>No confirmed transaction with extracted faces yet.</Empty>
      ) : (
        <div className="flex flex-wrap gap-2">
          {allFaces.map((f) => (
            <label key={f.id} className={cn("w-32 cursor-pointer rounded-control border p-2 text-[10px] focus-within:ring-2 focus-within:ring-accent", faces.has(f.id) ? "border-accent bg-accent/10" : "border-line hover:bg-raised/50")}>
              <div className="flex items-center justify-between">
                <input type="checkbox" checked={faces.has(f.id)} onChange={() => tog(faces, setFaces, f.id)} aria-label={`include face ${f.id}`} />
                <GradeBadge grade={f.grade} total={f.total} />
              </div>
              <AuthImg path={f.urls.crop} className="my-1.5 h-28 w-full rounded-control object-cover" />
              <Details label="flags">
                <FlagChips flags={f.flags as string[]} />
              </Details>
            </label>
          ))}
        </div>
      )}
      <div className="mb-2 mt-5 font-mono text-[11px] uppercase tracking-wide text-muted">Confirmed possible matches</div>
      {!hits.length ? (
        <Empty>No reviewer-confirmed matches. Unreviewed or rejected matches are never exported.</Empty>
      ) : (
        <div className="space-y-1.5">
          {hits.map((h) => (
            <label key={h.id} className="flex flex-wrap items-center gap-2 text-sm">
              <input type="checkbox" checked={hitSel.has(h.id)} onChange={() => tog(hitSel, setHitSel, h.id)} />
              {h.clip_name} · person {h.track_no} <Badge tone="green">{h.band}</Badge> <span className="font-mono text-xs text-muted">{h.score.toFixed(3)}</span> <span className="text-xs text-muted">{h.note}</span>
            </label>
          ))}
        </div>
      )}
      <label className="mt-5 flex items-center gap-2 text-sm">
        <input type="checkbox" checked={enh} onChange={(e) => setEnh(e.target.checked)} /> Include enhanced images (separate, labelled section)
      </label>
      <div className="mt-4 flex items-center gap-3">
        <Button onClick={() => m.mutate()} disabled={m.isPending || (!faces.size && !hitSel.size) || (!!jobId && job.data?.status !== "done" && job.data?.status !== "failed")}>
          Generate PDF
        </Button>
        {job.data && <span className="text-sm text-muted">{job.data.status === "done" ? "Done. Download it below." : `${job.data.status}…`}</span>}
      </div>
      {m.error && <ErrorBox error={fmtErr(m.error)} />}
      {job.data?.status === "failed" && <ErrorBox error={job.data.message} />}
    </Card>
  );
}

function ExportList({ caseId }: { caseId: string }) {
  const q = useQuery({ queryKey: ["exports", caseId], queryFn: () => get<ExportOut[]>(`/cases/${caseId}/exports`), refetchInterval: 4000 });
  if (q.isLoading) return <Spinner />;
  return (
    <Card title="Exports">
      {!q.data?.length ? (
        <Empty>No exports yet.</Empty>
      ) : (
        <ul className="divide-y divide-line">
          {q.data.map((e) => (
            <li key={e.id} className="py-3 first:pt-0 last:pb-0">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
                <span className="font-mono">#{e.id}</span>
                <span className="text-muted">{new Date(e.created_at).toLocaleString()}</span>
                <StatusChip status={e.status} />
                <span className="flex-1 text-xs text-muted">
                  {(e.selection as any).face_ids?.length || 0} faces, {(e.selection as any).search_hit_ids?.length || 0} matches{(e.selection as any).include_enhanced ? ", enhanced" : ""}
                </span>
                {e.status === "done" && (
                  <a href={mediaUrl(`/exports/${e.id}/download`)} target="_blank" rel="noreferrer" className={buttonClass("outline", "sm")}>
                    Download PDF
                  </a>
                )}
              </div>
              {e.sha256 && (
                <Details>
                  <KV items={[["SHA-256", <Mono>{e.sha256}</Mono>]]} />
                </Details>
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

export default function ExportPage() {
  const { id = "" } = useParams();
  const { can } = useAuth();
  const [selected, setSelected] = useState(0);
  const exports = useQuery({ queryKey: ["exports", id], queryFn: () => get<ExportOut[]>(`/cases/${id}/exports`), refetchInterval: 4000 });
  const ready = (exports.data || []).some((e) => e.status === "done");
  const hint: Hint = ready
    ? { text: "The PDF is ready. Download it from the list below." }
    : can("investigator") && !selected
      ? { text: "Choose the faces and confirmed matches to include." }
      : { text: can("investigator") ? "Generate the PDF evidence sheet below." : "Exports created by investigators appear below." };
  return (
    <CaseFrame caseId={id} current={7} hint={hint}>
      <PageHeader title="Export evidence" description="Create a PDF evidence sheet from selected faces and confirmed matches." />
      {can("investigator") && <ExportBuilder caseId={id} onSelection={setSelected} />}
      <ExportList caseId={id} />
      <JobToasts caseId={id} />
    </CaseFrame>
  );
}
