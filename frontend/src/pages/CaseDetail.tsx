import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fmtErr, get, post, upload } from "../api/client";
import type { AtmOut, CaseOut, ClipOut, ExportOut, JobOut, SearchOut, TxnOut } from "../api/types";
import { useAuth } from "../auth";
import { fmtBytes, StatusChip } from "../components/face";
import { JobToasts } from "../components/jobs";
import { Badge, Button, buttonClass, Card, Details, Empty, ErrorBox, KV, Label, link, Mono, PageHeader, Progress, Section, Select, Spinner } from "../components/ui";
import { CaseFrame, STEP_HINT, STEP_LABELS, useCaseProgress, type CaseProgress, type Hint, type StepId } from "../components/workflow";

interface UploadItem {
  name: string;
  progress: number;
  error?: string;
  done?: boolean;
}

function ClipUploader({ caseId, atms }: { caseId: string; atms: AtmOut[] }) {
  const [items, setItems] = useState<UploadItem[]>([]);
  const [kind, setKind] = useState("transaction");
  const [atmId, setAtmId] = useState<string>("");
  const [drag, setDrag] = useState(false);
  const qc = useQueryClient();

  const send = useCallback(
    async (files: FileList | File[]) => {
      const list = Array.from(files);
      setItems((x) => [...list.map((f) => ({ name: f.name, progress: 0 })), ...x]);
      for (const f of list) {
        const fd = new FormData();
        fd.append("file", f);
        fd.append("kind", kind);
        if (atmId) fd.append("atm_id", atmId);
        const set = (p: Partial<UploadItem>) => setItems((x) => x.map((i) => (i.name === f.name && !i.done && !i.error ? { ...i, ...p } : i)));
        try {
          await upload<ClipOut>(`/cases/${caseId}/clips`, fd, (p) => set({ progress: p }));
          set({ progress: 1, done: true });
          qc.invalidateQueries({ queryKey: ["clips", caseId] });
          qc.invalidateQueries({ queryKey: ["jobs", caseId] });
        } catch (e) {
          set({ error: fmtErr(e) });
        }
      }
    },
    [caseId, kind, atmId, qc],
  );

  return (
    <Card title="Upload footage" description="Use a transaction clip for the video of the withdrawal. Use gallery clips for other footage you want to search.">
      <div className="mb-4 flex flex-wrap items-end gap-4">
        <div>
          <Label>Clip type</Label>
          <Select value={kind} onChange={(e) => setKind(e.target.value)}>
            <option value="transaction">Transaction clip (link a withdrawal)</option>
            <option value="gallery">Gallery clip (other footage to search)</option>
          </Select>
        </div>
        <div>
          <Label hint="camera type, regions, time zone">ATM</Label>
          <Select value={atmId} onChange={(e) => setAtmId(e.target.value)}>
            <option value="">(none: fascia defaults, UTC)</option>
            {atms.map((a) => (
              <option key={a.id} value={a.id}>
                {a.atm_code} · {a.camera_type}
              </option>
            ))}
          </Select>
        </div>
      </div>
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          send(e.dataTransfer.files);
        }}
        className={`flex cursor-pointer flex-col items-center justify-center gap-1 rounded-card border border-dashed px-4 py-8 text-sm transition-colors focus-within:ring-2 focus-within:ring-accent ${drag ? "border-accent bg-accent/10" : "border-line hover:bg-raised/40"}`}
      >
        <span className="font-medium text-fg">Drop video files here or click to choose</span>
        <span className="text-xs text-muted">MP4, AVI, MOV, MKV, WMV, FLV, MPG, raw .264/.265, Dahua .dav. Originals are hashed and kept read-only.</span>
        <input type="file" multiple className="sr-only" onChange={(e) => e.target.files && send(e.target.files)} />
      </label>
      {!!items.length && (
        <ul className="mt-4 space-y-2 text-xs">
          {items.map((i, n) => (
            <li key={n} className="flex items-center gap-3">
              <span className="w-56 truncate">{i.name}</span>
              {i.error ? <span className="text-err">{i.error}</span> : <Progress value={i.progress} className="flex-1" />}
              {i.done && <Badge tone="green">Uploaded</Badge>}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function ErrorPanel({ error }: { error: any }) {
  return (
    <div className="mt-2 rounded-control border border-err/40 bg-err/10 p-3 text-xs text-err">
      <div>
        <b>{error.code}</b>: {error.message}
        {error.vendor && <> (vendor: {error.vendor})</>}
      </div>
      {!!error.instructions?.length && (
        <ol className="mt-1 list-decimal pl-4">
          {error.instructions.map((s: string, i: number) => (
            <li key={i}>{s}</li>
          ))}
        </ol>
      )}
    </div>
  );
}

function ClipList({ caseId, clips, atms }: { caseId: string; clips: ClipOut[]; atms: AtmOut[] }) {
  const { can } = useAuth();
  const qc = useQueryClient();
  const reindex = useMutation({
    mutationFn: (ids: string[]) => post<JobOut[]>(`/cases/${caseId}/gallery/index`, { clip_ids: ids }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs", caseId] }),
  });
  if (!clips.length) return <Empty>No clips uploaded yet.</Empty>;
  return (
    <ul className="divide-y divide-line">
      {clips.map((c) => {
        const p: any = c.probe || {};
        const tm: any = c.timemap;
        const atm = atms.find((a) => a.id === c.atm_id);
        return (
          <li key={c.id} className="py-3 first:pt-0 last:pb-0">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
              <div className="min-w-0 flex-1">
                <div className="truncate font-medium">{c.orig_name}</div>
                <div className="text-xs text-muted">{c.kind === "transaction" ? "Transaction clip" : "Gallery clip"}</div>
              </div>
              <StatusChip status={c.status} />
              {tm ? <Badge tone="green">Time set</Badge> : <Badge tone="amber">Time not set</Badge>}
              <div className="flex items-center gap-1">
                <Link to={`/clips/${c.id}/setup`} className={buttonClass("outline", "sm")}>
                  {c.kind === "transaction" ? "Set up" : "Open"}
                </Link>
                {can("investigator") && ["tracked", "indexed"].includes(c.status) && (
                  <Button size="sm" variant="ghost" onClick={() => reindex.mutate([c.id])} title="Index every person in this clip for face search">
                    Index for search
                  </Button>
                )}
              </div>
            </div>
            {c.error && <ErrorPanel error={c.error} />}
            <Details className="mt-1.5">
              <KV
                items={[
                  ["SHA-256", <Mono>{c.orig_sha256}</Mono>],
                  ["Size", fmtBytes(c.size)],
                  ["ATM", atm ? `${atm.atm_code} · ${atm.camera_type}` : "none"],
                  ["Video", p.video_codec ? `${p.video_codec} ${p.width}×${p.height}` : "-"],
                  ["Frame rate", p.video_codec ? `${Number(p.avg_fps).toFixed(2)} fps · ${Number(p.duration_s).toFixed(1)} s${p.is_vfr ? " · VFR" : ""}${p.is_interlaced ? " · interlaced" : ""}` : "-"],
                  ["Normalised", c.normalised ? ((c.ingest as any)?.decisions || []).join("; ") || "yes" : "no"],
                  ["Time map", tm ? `${tm.method} · start ${c.start_wall?.replace("T", " ") || "-"}` : "not set"],
                  ["Clock offset", `${c.clock_offset_s > 0 ? "+" : ""}${c.clock_offset_s} s`],
                  ["People tracked", c.n_tracks],
                ]}
              />
            </Details>
          </li>
        );
      })}
    </ul>
  );
}

function caseHint(caseId: string, p: CaseProgress | undefined, canUpload: boolean): Hint | undefined {
  if (!p) return undefined;
  const fi = p.firstIncomplete;
  const step = (id: StepId) => p.steps[id - 1];
  if (fi === null) return { text: "All steps are done. You can review results or export again.", action: { label: "Export", to: `/cases/${caseId}/export` } };
  if (fi === 1)
    return {
      text: STEP_HINT[1],
      action: canUpload ? { label: "Upload footage", onClick: () => document.getElementById("footage")?.scrollIntoView({ behavior: "smooth" }) } : undefined,
    };
  if (fi === 2) {
    const clip = p.latest.txnClip;
    if (clip?.kind !== "transaction") return { text: "This case has only gallery clips. Upload a transaction clip of the withdrawal to continue." };
    if (!clip.analysis_sha256 && !clip.error) return { text: "The video is still being prepared. You can set it up once it shows Ready for setup.", action: { label: "Set up clip", to: step(2).to } };
    return { text: STEP_HINT[2], action: { label: "Set up clip", to: step(2).to } };
  }
  return { text: STEP_HINT[fi], action: step(fi).to ? { label: STEP_LABELS[fi], to: step(fi).to } : undefined };
}

export default function CaseDetail() {
  const { id = "" } = useParams();
  const caseQ = useQuery({ queryKey: ["case", id], queryFn: () => get<CaseOut>(`/cases/${id}`) });
  const clips = useQuery({ queryKey: ["clips", id], queryFn: () => get<ClipOut[]>(`/cases/${id}/clips`), refetchInterval: 3000 });
  const txns = useQuery({ queryKey: ["txns", id], queryFn: () => get<TxnOut[]>(`/cases/${id}/transactions`), refetchInterval: 3000 });
  const atms = useQuery({ queryKey: ["atms"], queryFn: () => get<AtmOut[]>("/atms") });
  const searches = useQuery({ queryKey: ["searches", id], queryFn: () => get<SearchOut[]>(`/cases/${id}/searches`) });
  const exports = useQuery({ queryKey: ["exports", id], queryFn: () => get<ExportOut[]>(`/cases/${id}/exports`) });
  const { progress } = useCaseProgress(id);
  const { can } = useAuth();

  if (caseQ.isLoading) return <Spinner />;
  if (caseQ.error) return <ErrorBox error={caseQ.error} />;
  const c = caseQ.data!;
  const fi = progress?.firstIncomplete;
  const current = fi && fi <= 3 ? fi : undefined;
  return (
    <CaseFrame caseId={id} current={current} progress={progress} hint={caseHint(id, progress, can("investigator"))}>
      <PageHeader
        title={<span className="font-mono">{c.ref_no}</span>}
        description={c.title}
        actions={
          <Link to={`/cases/${id}/export`} className={buttonClass("outline")}>
            Evidence export
          </Link>
        }
      />
      <Section id="footage" title="Footage" description="Upload the ATM video, then set up each clip.">
        {can("investigator") && <ClipUploader caseId={id} atms={atms.data || []} />}
        <Card title="Clips">{clips.data ? <ClipList caseId={id} clips={clips.data} atms={atms.data || []} /> : <Spinner />}</Card>
      </Section>
      <Section title="Transactions" description="Withdrawals linked to a transaction clip. Open one to pick the customer.">
        <Card>
          {!txns.data?.length ? (
            <Empty>No transactions yet. Open a transaction clip to add the journal time.</Empty>
          ) : (
            <ul className="divide-y divide-line">
              {txns.data.map((t) => (
                <li key={t.id} className="flex flex-wrap items-center gap-x-4 gap-y-1 py-3 first:pt-0 last:pb-0">
                  <div className="min-w-0 flex-1">
                    <Link className={`${link} font-medium`} to={`/transactions/${t.id}`}>
                      {t.txn_ref || `#${t.id}`}
                    </Link>
                    <div className="truncate text-xs text-muted">{t.clip_name}</div>
                  </div>
                  <span className="font-mono text-xs">{t.txn_time_journal.replace("T", " ")}</span>
                  <span className="w-20 text-right font-mono text-xs">{t.amount != null ? t.amount.toLocaleString() : ""}</span>
                  <StatusChip status={t.status} />
                </li>
              ))}
            </ul>
          )}
        </Card>
      </Section>
      <Section title="Results" description="Face searches and evidence exports for this case.">
        <Card>
          <div className="space-y-3 text-sm">
            {(searches.data || []).map((s) => (
              <div key={s.search_id} className="flex flex-wrap items-baseline gap-x-3">
                <Link className={link} to={`/search/${s.search_id}`}>
                  Search #{s.search_id}
                </Link>
                <span className="text-xs text-muted">
                  {s.hits.length} possible matches · {s.hits.filter((h) => h.review === "confirmed").length} confirmed · {new Date(s.created_at).toLocaleString()}
                </span>
              </div>
            ))}
            {(exports.data || []).map((e) => (
              <div key={e.id}>
                <div className="flex flex-wrap items-center gap-3">
                  <span>Export #{e.id}</span>
                  <StatusChip status={e.status} />
                </div>
                {e.sha256 && (
                  <Details>
                    <KV items={[["SHA-256", <Mono>{e.sha256}</Mono>]]} />
                  </Details>
                )}
              </div>
            ))}
            {!searches.data?.length && !exports.data?.length && <Empty>Nothing yet.</Empty>}
          </div>
        </Card>
      </Section>
      <JobToasts caseId={id} />
    </CaseFrame>
  );
}
