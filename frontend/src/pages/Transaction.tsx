import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { fmtErr, get, mediaUrl, post } from "../api/client";
import type { CandidateOut, ClipOut, JobOut, Timeline, TrackSamplesOut, TxnOut } from "../api/types";
import { useAuth } from "../auth";
import { AuthImg, fmtTime, StatusChip } from "../components/face";
import { JobToasts, useJob } from "../components/jobs";
import { Badge, Button, Card, cn, Details, Dialog, Empty, ErrorBox, Input, KV, Label, PageHeader, Progress, Spinner } from "../components/ui";
import { CaseFrame, type Hint } from "../components/workflow";
import { cssColor, OVERLAY_BOX } from "../theme";

function VideoPlayer({ clip, seekTo, samples, onTime }: { clip: ClipOut; seekTo: { t: number; n: number } | null; samples?: TrackSamplesOut; onTime: (t: number) => void }) {
  const v = useRef<HTMLVideoElement>(null);
  const [t, setT] = useState(0);
  const start = Number((clip.probe as any)?.analysis_start_time || 0);
  const W = Number((clip.probe as any)?.analysis_width || 1920);
  const H = Number((clip.probe as any)?.analysis_height || 1080);
  useEffect(() => {
    if (seekTo && v.current) {
      v.current.currentTime = Math.max(0, seekTo.t - start);
      v.current.pause();
    }
  }, [seekTo, start]);
  let box: number[] | null = null;
  if (samples?.pts.length) {
    const pts = t + start;
    let best = -1;
    let bd = 1e9;
    samples.pts.forEach((p, i) => {
      const d = Math.abs(p - pts);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    if (best >= 0 && bd < 0.5) box = samples.bbox[best];
  }
  return (
    <div className="relative w-full overflow-hidden rounded-control bg-black" style={{ aspectRatio: `${W} / ${H}` }}>
      <video
        ref={v}
        src={mediaUrl(clip.proxy_url)}
        controls
        preload="metadata"
        className="absolute inset-0 h-full w-full"
        onTimeUpdate={(e) => {
          const ct = (e.target as HTMLVideoElement).currentTime;
          setT(ct);
          onTime(ct + start);
        }}
        onSeeked={(e) => setT((e.target as HTMLVideoElement).currentTime)}
      />
      {box && (
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-[calc(100%-40px)] w-full">
          <rect x={box[0]} y={box[1]} width={box[2] - box[0]} height={box[3] - box[1]} fill="none" style={{ stroke: OVERLAY_BOX }} strokeWidth={W / 300} />
          <text x={box[0] + 6} y={box[1] + W / 50} style={{ fill: OVERLAY_BOX }} fontSize={W / 55}>
            person {samples!.track_no}
          </text>
        </svg>
      )}
    </div>
  );
}

function TrackTimeline({ tl, txn, cands, selected, onSelect, now, onSeek }: { tl: Timeline; txn: TxnOut; cands: CandidateOut[]; selected?: number; onSelect: (id: number) => void; now: number; onSeek: (t: number) => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const tp = txn.t_txn_pts ?? 0;
  const dur = tl.duration || Math.max(...tl.tracks.map((t) => t.t_end), 1);
  // Show the part of the clip that overlaps the window (plus a margin); the window may be much longer than the clip.
  const w0 = Math.max(0, tp - txn.window_before_s);
  const w1 = Math.min(dur, tp + txn.window_after_s);
  const pad = Math.max(1, (w1 - w0) * 0.05);
  const [lo, hi] = w1 > w0 ? [Math.max(0, w0 - pad) - 0.2, Math.min(dur, w1 + pad) + 0.2] : [-0.2, dur + 0.2];
  const candIds = new Set(cands.map((c) => c.track_id));
  const rows = [...tl.tracks].sort((a, b) => Number(candIds.has(b.track_id)) - Number(candIds.has(a.track_id)) || a.t_start - b.t_start).slice(0, 40);
  const RH = 14;
  const LEFT = 70;
  const height = 28 + rows.length * RH;
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const col = {
      accent: cssColor("--color-accent"),
      accentSoft: cssColor("--color-accent"),
      accentMid: cssColor("--color-accent-mid"),
      line: cssColor("--color-line"),
      raised: cssColor("--color-raised"),
      fg: cssColor("--color-fg"),
      muted: cssColor("--color-muted"),
      warn: cssColor("--color-warn"),
      font: cssColor("--font-sans"),
    };
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth;
    c.width = w * dpr;
    c.height = height * dpr;
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    const x = (s: number) => LEFT + ((s - lo) / (hi - lo)) * (w - LEFT - 8);
    g.clearRect(0, 0, w, height);
    g.globalAlpha = 0.25;
    g.fillStyle = col.accent;
    g.fillRect(x(tp - txn.window_before_s), 0, x(tp + txn.window_after_s) - x(tp - txn.window_before_s), height);
    g.globalAlpha = 1;
    g.fillStyle = col.raised;
    g.fillRect(x(0), 16, x(dur) - x(0), 4);
    g.font = `10px ${col.font}`;
    rows.forEach((r, i) => {
      const y = 26 + i * RH;
      const sel = r.track_id === selected;
      g.fillStyle = sel ? col.fg : col.muted;
      g.globalAlpha = sel || candIds.has(r.track_id) ? 1 : 0.5;
      g.fillText(`${r.source === "face" ? "face " : "person "}${r.track_no}`, 4, y + 8);
      g.globalAlpha = sel ? 0.6 : 1;
      g.fillStyle = sel ? col.fg : col.line;
      g.fillRect(x(r.t_start), y + 2, Math.max(2, x(r.t_end) - x(r.t_start)), 6);
      g.globalAlpha = 1;
      g.fillStyle = sel ? col.accentSoft : col.accentMid;
      r.near.forEach(([a, b]) => g.fillRect(x(a), y, Math.max(2, x(b) - x(a)), 10));
    });
    g.strokeStyle = col.warn;
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(x(tp), 0);
    g.lineTo(x(tp), height);
    g.stroke();
    g.fillStyle = col.warn;
    g.fillText("transaction", x(tp) + 3, 10);
    g.strokeStyle = col.fg;
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(x(now), 12);
    g.lineTo(x(now), height);
    g.stroke();
  }, [tl, txn, cands, selected, now, lo, hi, dur, tp, height, rows]);
  return (
    <div>
      <canvas
        ref={canvas}
        style={{ width: "100%", height }}
        className="cursor-pointer"
        onClick={(e) => {
          const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
          const s = lo + ((e.clientX - r.left - LEFT) / (r.width - LEFT - 8)) * (hi - lo);
          const row = Math.floor((e.clientY - r.top - 26) / RH);
          if (rows[row] && e.clientX - r.left < LEFT) onSelect(rows[row].track_id);
          else onSeek(Math.max(0, Math.min(dur, s)));
        }}
      />
      <div className="mt-2 flex flex-wrap gap-4 text-[11px] text-muted">
        <span>
          <span className="mr-1.5 inline-block h-2 w-3 bg-accent/25" />
          window −{txn.window_before_s}/+{txn.window_after_s} s
        </span>
        <span>
          <span className="mr-1.5 inline-block h-2 w-3 bg-accent-mid" />
          near the machine
        </span>
        <span>
          <span className="mr-1.5 inline-block h-2 w-0.5 bg-warn" />
          transaction
        </span>
        <span>click a bar to seek, a label to select</span>
      </div>
    </div>
  );
}

function ConfirmDialog({ open, onClose, txn, cand }: { open: boolean; onClose: () => void; txn: TxnOut; cand?: CandidateOut }) {
  const [note, setNote] = useState("");
  const [jobId, setJobId] = useState<number | null>(null);
  const job = useJob(jobId);
  const qc = useQueryClient();
  const nav = useNavigate();
  const m = useMutation({
    mutationFn: () => post<JobOut>(`/transactions/${txn.id}/confirm`, { track_id: cand!.track_id, note }),
    onSuccess: (j) => {
      setJobId(j.id);
      qc.invalidateQueries({ queryKey: ["txn", txn.id] });
    },
  });
  useEffect(() => {
    if (job.data?.status === "done" && cand) nav(`/tracks/${cand.track_id}/faces`);
  }, [job.data?.status, cand, nav]);
  return (
    <Dialog open={open} onClose={onClose} title="Confirm the customer">
      {cand && (
        <div className="space-y-4 text-sm">
          <p>
            You confirm that <b>person {cand.track_no}</b> is the person who performed transaction <b>{txn.txn_ref || txn.id}</b>. This is recorded in the audit log under your name. The system then extracts and scores this person's faces.
          </p>
          <div>
            <Label>Reason / note (optional)</Label>
            <Input value={note} onChange={(e) => setNote(e.target.value)} placeholder="e.g. only person at the keypad during the dispense" />
          </div>
          {m.error && <ErrorBox error={fmtErr(m.error)} />}
          {job.data && (
            <div>
              <div className="mb-1 text-xs text-muted">
                {job.data.status}: {job.data.message}
              </div>
              <Progress value={job.data.progress} />
              {job.data.status === "failed" && <ErrorBox error={job.data.message} />}
            </div>
          )}
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
            <Button onClick={() => m.mutate()} disabled={m.isPending || !!jobId}>
              Confirm and extract faces
            </Button>
          </div>
        </div>
      )}
    </Dialog>
  );
}

function txnHint(txn: TxnOut, cands?: CandidateOut[]): Hint {
  if (txn.chosen_track_id) return { text: "Next: review the customer's faces and pick the clearest ones.", action: { label: "Review faces", to: `/tracks/${txn.chosen_track_id}/faces` } };
  if (txn.status === "pending") return { text: "Ranking candidates. This needs tracking and a time mapping, and updates by itself." };
  if (cands && !cands.length) return { text: "No one was near the machine during the window. Check the time mapping, clock offset and window on the clip setup page." };
  return { text: "Watch the candidates, then confirm which person made the withdrawal." };
}

export default function TransactionPage() {
  const { id = "" } = useParams();
  const { can } = useAuth();
  const txnQ = useQuery({ queryKey: ["txn", Number(id)], queryFn: () => get<TxnOut>(`/transactions/${id}`), refetchInterval: 3000 });
  const clipId = txnQ.data?.clip_id;
  const clipQ = useQuery({ queryKey: ["clip", clipId], queryFn: () => get<ClipOut>(`/clips/${clipId}`), enabled: !!clipId });
  const cands = useQuery({ queryKey: ["cands", id, txnQ.data?.status], queryFn: () => get<CandidateOut[]>(`/transactions/${id}/candidates`), enabled: !!txnQ.data });
  const tl = useQuery({ queryKey: ["timeline", clipId], queryFn: () => get<Timeline>(`/clips/${clipId}/timeline`), enabled: !!clipId });
  const [sel, setSel] = useState<number | undefined>();
  const [seek, setSeek] = useState<{ t: number; n: number } | null>(null);
  const [now, setNow] = useState(0);
  const [confirm, setConfirm] = useState(false);
  const samples = useQuery({ queryKey: ["samples", sel], queryFn: () => get<TrackSamplesOut>(`/tracks/${sel}/samples`), enabled: !!sel });
  useEffect(() => {
    if (!sel && cands.data?.length) setSel(cands.data[0].track_id);
  }, [cands.data, sel]);
  if (txnQ.isLoading || !txnQ.data) return <Spinner />;
  const txn = txnQ.data;
  const clip = clipQ.data;
  const cand = cands.data?.find((c) => c.track_id === sel);
  const go = (t: number) => setSeek({ t, n: Date.now() });
  return (
    <CaseFrame caseId={txn.case_id} current={4} hint={txnHint(txn, cands.data)}>
      <PageHeader
        title={
          <>
            Pick the customer <StatusChip status={txn.status} />
          </>
        }
        description="The people below were near the machine around the transaction time, ranked by how likely they made it. You decide who it was."
        meta={
          <div className="space-y-1">
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              <span>
                <span className="text-muted">Transaction </span>
                <span className="font-mono">{txn.txn_ref || `#${txn.id}`}</span>
              </span>
              <span>
                <span className="text-muted">Journal time </span>
                <span className="font-mono">{txn.txn_time_journal.replace("T", " ")}</span>
              </span>
              {txn.amount != null && (
                <span>
                  <span className="text-muted">Amount </span>
                  <span className="font-mono">{txn.amount.toLocaleString()}</span>
                </span>
              )}
            </div>
            {(txn.notes as string[]).map((n, i) => (
              <div key={i} className="text-sm text-warn">
                {n}
              </div>
            ))}
            <Details>
              <KV
                items={[
                  ["Clip", txn.clip_name || txn.clip_id],
                  ["Video time", fmtTime(txn.t_txn_pts)],
                  ["Window", `−${txn.window_before_s} / +${txn.window_after_s} s`],
                ]}
              />
            </Details>
          </div>
        }
      />
      <div className="grid gap-4 xl:grid-cols-[3fr_2fr]">
        <div className="space-y-4">
          <Card title="Video">{clip?.proxy_url ? <VideoPlayer clip={clip} seekTo={seek} samples={samples.data} onTime={setNow} /> : <Spinner />}</Card>
          <Card title="Who was on screen" description="Each row is one person. Bars show when they were visible and near the machine.">
            {tl.data ? <TrackTimeline tl={tl.data} txn={txn} cands={cands.data || []} selected={sel} onSelect={setSel} now={now} onSeek={go} /> : <Spinner />}
          </Card>
        </div>
        <Card title="Candidates" description="Select a person to play their best view. Requires human confirmation.">
          {txn.status === "pending" && <Empty>Ranking candidates… (needs tracking and a time mapping)</Empty>}
          {cands.data && !cands.data.length && txn.status !== "pending" && <Empty>No person was near the machine during the window. Check the time mapping, clock offset and window.</Empty>}
          <div className="space-y-2">
            {(cands.data || []).map((c) => {
              const reasons = c.reasons as string[];
              const ambiguous = reasons.some((r) => r.startsWith("ambiguous"));
              return (
                <div
                  key={c.id}
                  role="button"
                  tabIndex={0}
                  aria-pressed={sel === c.track_id}
                  onClick={() => {
                    setSel(c.track_id);
                    go(c.best_view_pts);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setSel(c.track_id);
                      go(c.best_view_pts);
                    }
                  }}
                  className={cn(
                    "flex cursor-pointer gap-3 rounded-control border p-2.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent",
                    sel === c.track_id ? "border-accent bg-raised" : "border-line hover:bg-raised/50",
                  )}
                >
                  <AuthImg path={c.thumb_url} className="h-28 w-24 shrink-0 rounded-control object-cover" />
                  <div className="min-w-0 flex-1 text-xs">
                    <div className="mb-1 flex flex-wrap items-center gap-2">
                      <span className="font-mono text-sm font-semibold text-accent">#{c.rank}</span>
                      <span className="text-sm">
                        {c.source === "face" ? "Face track" : "Person"} {c.track_no}
                      </span>
                      {txn.chosen_track_id === c.track_id && <Badge tone="green">Confirmed</Badge>}
                      {ambiguous && <Badge tone="amber">Ambiguous</Badge>}
                    </div>
                    <div className="text-muted">
                      on screen <span className="font-mono">{fmtTime(c.t_start)}–{fmtTime(c.t_end)}</span> · best view <span className="font-mono">{fmtTime(c.best_view_pts)}</span>
                    </div>
                    <Details label="score" className="mt-1">
                      <div className="mb-2 font-mono">score {c.score.toFixed(3)}</div>
                      {Object.entries(c.parts).map(([k, v]) => (
                        <div key={k} className="flex items-center gap-2">
                          <span className="w-16 font-mono text-muted">{k}</span>
                          <div className="h-1.5 flex-1 rounded-full bg-raised">
                            <div className="h-full rounded-full bg-accent" style={{ width: `${Number(v) * 100}%` }} />
                          </div>
                          <span className="w-8 text-right font-mono tabular-nums">{Number(v).toFixed(2)}</span>
                        </div>
                      ))}
                      <ul className="mt-2 list-disc pl-4 text-muted">
                        {reasons.map((r, i) => (
                          <li key={i} className={r.startsWith("ambiguous") ? "text-warn" : ""}>
                            {r}
                          </li>
                        ))}
                      </ul>
                    </Details>
                  </div>
                </div>
              );
            })}
          </div>
          {can("investigator") && cand && (
            <div className="mt-4 flex justify-end">
              <Button onClick={() => setConfirm(true)}>Confirm person {cand.track_no} as the customer</Button>
            </div>
          )}
        </Card>
      </div>
      <ConfirmDialog open={confirm} onClose={() => setConfirm(false)} txn={txn} cand={cand} />
      <JobToasts caseId={txn.case_id} />
    </CaseFrame>
  );
}
