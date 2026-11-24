import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { z } from "zod";
import { fmtErr, get, put, post } from "../api/client";
import type { AtmIn, AtmOut, ClipOut, TxnOut } from "../api/types";
import { useAuth } from "../auth";
import { AuthImg, fmtTime, StatusChip } from "../components/face";
import { JobToasts } from "../components/jobs";
import { Badge, Button, Card, cn, Details, ErrorBox, Input, KV, Label, link, Mono, PageHeader, Section, Select, Spinner } from "../components/ui";
import { CaseFrame, type Hint } from "../components/workflow";
import { ROI_COLOR } from "../theme";

type Pt = [number, number];
const ROI_NAMES = ["keypad", "cash_slot", "card_slot", "atm_zone"] as const;
const ROI_LABEL: Record<string, string> = { keypad: "keypad", cash_slot: "cash slot", card_slot: "card slot", atm_zone: "ATM zone", ocr: "timestamp" };

/** Frame image with SVG overlay for polygon ROIs (normalised coords) and the OCR rectangle. */
function RoiEditor({ clip, pts, rois, setRois, ocr, setOcr, editable }: {
  clip: ClipOut; pts: number; rois: Record<string, Pt[]>; setRois: (r: Record<string, Pt[]>) => void;
  ocr: number[] | null; setOcr: (r: number[] | null) => void; editable: boolean;
}) {
  const [active, setActive] = useState<string>("keypad");
  const [drag, setDrag] = useState<Pt | null>(null);
  const [cursor, setCursor] = useState<Pt | null>(null);
  const svg = useRef<SVGSVGElement>(null);
  const p: any = clip.probe || {};
  const W = p.analysis_width || p.width || 16;
  const H = p.analysis_height || p.height || 9;

  const norm = (e: React.MouseEvent): Pt => {
    const r = svg.current!.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
  };
  const onDown = (e: React.MouseEvent) => {
    if (!editable) return;
    const q = norm(e);
    if (active === "ocr") setDrag(q);
    else setRois({ ...rois, [active]: [...(rois[active] || []), q] });
  };
  const onUp = (e: React.MouseEvent) => {
    if (active === "ocr" && drag) {
      const q = norm(e);
      const r = [Math.min(drag[0], q[0]), Math.min(drag[1], q[1]), Math.max(drag[0], q[0]), Math.max(drag[1], q[1])];
      if (r[2] - r[0] > 0.01 && r[3] - r[1] > 0.005) setOcr(r);
      setDrag(null);
    }
  };
  // Text labels sit in a pixel-space overlay so they are not stretched by the 0-1 viewBox.
  const labels: { name: string; at: Pt }[] = [
    ...ROI_NAMES.filter((n) => rois[n]?.length).map((n) => ({ name: n, at: rois[n][0] })),
    ...(ocr ? [{ name: "ocr", at: [ocr[0], ocr[1]] as Pt }] : []),
  ];
  return (
    <div>
      {editable && (
        <div className="mb-3 flex flex-wrap items-center gap-1.5 text-xs">
          <span className="mr-1 font-mono uppercase tracking-wide text-muted">Draw</span>
          {[...ROI_NAMES, "ocr"].map((n) => (
            <button
              key={n}
              onClick={() => setActive(n)}
              aria-pressed={active === n}
              className={cn(
                "rounded-control border px-2.5 py-1 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent",
                active === n ? "border-accent bg-accent/10 text-accent" : "border-line text-muted hover:bg-raised hover:text-fg",
              )}
            >
              <span className="mr-1.5 inline-block h-2 w-2 rounded-full ring-1 ring-line" style={{ background: ROI_COLOR[n] }} />
              {n === "ocr" ? "timestamp overlay (drag a box)" : ROI_LABEL[n]}
              {n !== "ocr" && rois[n]?.length ? ` (${rois[n].length})` : ""}
            </button>
          ))}
          {active !== "ocr" && (
            <>
              <Button size="sm" variant="ghost" onClick={() => setRois({ ...rois, [active]: (rois[active] || []).slice(0, -1) })}>
                Undo point
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setRois({ ...rois, [active]: [] })}>
                Clear {ROI_LABEL[active]}
              </Button>
            </>
          )}
          {active === "ocr" && ocr && (
            <Button size="sm" variant="ghost" onClick={() => setOcr(null)}>
              Clear box
            </Button>
          )}
        </div>
      )}
      <div className="relative w-full select-none overflow-hidden rounded-control bg-black" style={{ aspectRatio: `${W} / ${H}` }}>
        <AuthImg path={`/clips/${clip.id}/frame`} extra={{ pts: pts.toFixed(2), max_w: 1280 }} className="absolute inset-0 h-full w-full object-contain" draggable={false} />
        <svg
          ref={svg}
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
          className={`absolute inset-0 h-full w-full ${editable ? "cursor-crosshair" : ""}`}
          onMouseDown={onDown}
          onMouseUp={onUp}
          onMouseMove={(e) => setCursor(norm(e))}
          onMouseLeave={() => setCursor(null)}
        >
          {ROI_NAMES.map((n) =>
            rois[n]?.length ? (
              <g key={n}>
                <polygon points={rois[n].map((q) => q.join(",")).join(" ")} style={{ fill: ROI_COLOR[n], stroke: ROI_COLOR[n] }} fillOpacity={0.18} strokeWidth={0.003} />
                {rois[n].map((q, i) => (
                  <circle key={i} cx={q[0]} cy={q[1]} r={0.004} style={{ fill: ROI_COLOR[n] }} />
                ))}
              </g>
            ) : null,
          )}
          {ocr && <rect x={ocr[0]} y={ocr[1]} width={ocr[2] - ocr[0]} height={ocr[3] - ocr[1]} fill="none" style={{ stroke: ROI_COLOR.ocr }} strokeWidth={0.003} strokeDasharray="0.01 0.005" />}
          {drag && cursor && (
            <rect x={Math.min(drag[0], cursor[0])} y={Math.min(drag[1], cursor[1])} width={Math.abs(cursor[0] - drag[0])} height={Math.abs(cursor[1] - drag[1])} fill="none" style={{ stroke: ROI_COLOR.ocr }} strokeWidth={0.003} />
          )}
        </svg>
        {labels.map((l) => (
          <span
            key={l.name}
            className="pointer-events-none absolute -translate-y-full rounded-sm bg-black/70 px-1 font-mono text-[10px]"
            style={{ left: `${l.at[0] * 100}%`, top: `${l.at[1] * 100}%`, color: ROI_COLOR[l.name] }}
          >
            {ROI_LABEL[l.name]}
          </span>
        ))}
      </div>
    </div>
  );
}

function AtmPanel({ clip, atm, pts }: { clip: ClipOut; atm: AtmOut | undefined; pts: number }) {
  const { can } = useAuth();
  const qc = useQueryClient();
  const [rois, setRois] = useState<Record<string, Pt[]>>({});
  const [ocr, setOcr] = useState<number[] | null>(null);
  const [cam, setCam] = useState("fascia");
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => {
    if (atm) {
      setRois((atm.rois as any) || {});
      setOcr(atm.ocr_roi || null);
      setCam(atm.camera_type);
    }
  }, [atm]);
  const save = useMutation({
    mutationFn: () => {
      const body: AtmIn = { atm_code: atm!.atm_code, location: atm!.location, camera_type: cam as any, rois: rois as any, ocr_roi: ocr, date_order: atm!.date_order as any, tz: atm!.tz, metadata_is_end: atm!.metadata_is_end };
      return put<AtmOut>(`/atms/${atm!.id}`, body);
    },
    onSuccess: () => {
      setMsg("Saved. ROIs apply to every clip from this ATM; re-run linking by re-adding or re-mapping time.");
      qc.invalidateQueries({ queryKey: ["atms"] });
    },
    onError: (e) => setMsg(fmtErr(e)),
  });
  const runOcr = useMutation({
    mutationFn: () => put<ClipOut>(`/clips/${clip.id}/timemap`, { method: "ocr", roi: ocr }),
    onSuccess: () => {
      setMsg("OCR time mapping queued.");
      qc.invalidateQueries({ queryKey: ["jobs", clip.case_id] });
    },
    onError: (e) => setMsg(fmtErr(e)),
  });
  const editable = !!atm && can("investigator");
  return (
    <Card
      title={atm ? `ATM ${atm.atm_code}` : "Regions of interest"}
      actions={
        editable && (
          <>
            <Select value={cam} onChange={(e) => setCam(e.target.value)} title="Camera type">
              <option value="fascia">fascia camera (looks at the user)</option>
              <option value="lobby">lobby / ceiling camera</option>
            </Select>
            <Button onClick={() => save.mutate()} disabled={save.isPending}>
              Save ATM setup
            </Button>
          </>
        )
      }
    >
      {!atm && <p className="mb-3 text-xs text-warn">This clip has no ATM assigned, so fascia defaults are used and ROIs cannot be stored. Choose an ATM at upload to use ROIs.</p>}
      <RoiEditor clip={clip} pts={pts} rois={rois} setRois={setRois} ocr={ocr} setOcr={setOcr} editable={editable} />
      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
        {editable && ocr && (
          <Button size="sm" variant="outline" onClick={() => runOcr.mutate()}>
            Run OCR on timestamp box
          </Button>
        )}
        {msg && <span className="text-muted">{msg}</span>}
      </div>
      <Details label="how regions are used" className="mt-2">
        <p className="text-muted">
          Lobby cameras use the ATM zone (torso inside) and keypad / cash slot / card slot (wrist inside). Fascia cameras rank by near-field presence. Coordinates are stored normalised (0-1).
        </p>
      </Details>
    </Card>
  );
}

function TimeMapPanel({ clip, tz }: { clip: ClipOut; tz: string }) {
  const { can } = useAuth();
  const qc = useQueryClient();
  const tm: any = clip.timemap;
  const [start, setStart] = useState("");
  const [offset, setOffset] = useState(String(clip.clock_offset_s));
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => setOffset(String(clip.clock_offset_s)), [clip.clock_offset_s]);
  const m = useMutation({
    mutationFn: (b: object) => put<ClipOut>(`/clips/${clip.id}/timemap`, b),
    onSuccess: () => {
      setErr(null);
      qc.invalidateQueries({ queryKey: ["clip", clip.id] });
      qc.invalidateQueries({ queryKey: ["jobs", clip.case_id] });
    },
    onError: (e) => setErr(fmtErr(e)),
  });
  const reads = (clip.timemap_reads || []) as any[];
  const method = tm ? (tm.method === "ocr" ? "Set by OCR" : tm.method === "manual" ? "Set manually" : `Set from ${tm.method}`) : null;
  return (
    <Card title="Video time to bank time" description="The DVR clock in the video must line up with the bank journal clock.">
      <div className="mb-4 flex flex-wrap items-center gap-3 text-sm">
        {method ? <Badge tone="green">{method}</Badge> : <Badge tone="amber">Not set</Badge>}
        <span className="text-muted">Clip starts at</span>
        <span className="font-mono">{clip.start_wall?.replace("T", " ") || "-"}</span>
      </div>
      {can("investigator") && (
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <Label hint="DVR clock at the first frame">Manual start time</Label>
            <div className="flex gap-2">
              <Input type="datetime-local" step="1" value={start} onChange={(e) => setStart(e.target.value)} />
              <Button disabled={!start || m.isPending} onClick={() => m.mutate({ method: "manual", start })}>
                Set
              </Button>
            </div>
          </div>
          <div>
            <Label hint="DVR clock − bank journal clock, seconds">Clock offset</Label>
            <div className="flex gap-2">
              <Input type="number" step="0.1" value={offset} onChange={(e) => setOffset(e.target.value)} />
              <Button variant="outline" disabled={m.isPending || Number.isNaN(Number(offset))} onClick={() => m.mutate({ clock_offset_s: Number(offset) })}>
                Save
              </Button>
            </div>
            <p className="mt-1.5 text-[11px] text-muted">Example: DVR shows 10:03:00 when the journal says 10:01:00 → +120.</p>
          </div>
        </div>
      )}
      {err && <ErrorBox error={err} />}
      <Details className="mt-3">
        <KV
          items={[
            ["Method", tm?.method || "not set"],
            ["Model", tm ? <Mono>wall = {tm.a.toFixed(3)} + {tm.b.toFixed(6)} × pts</Mono> : "-"],
            ["OCR inliers / residual", tm ? `${tm.inliers} / ${tm.residual_s.toFixed(2)} s` : "-"],
            ["Notes", tm?.notes?.join("; ") || "-"],
            ["Time zone", tz],
          ]}
        />
        {!!reads.length && (
          <div className="mt-3">
            <div className="mb-1 font-mono text-[11px] uppercase tracking-wide text-muted">
              OCR preview ({reads.filter((r) => r.parsed).length}/{reads.length} parsed)
            </div>
            <table className="w-full">
              <tbody>
                {reads.slice(0, 40).map((r, i) => (
                  <tr key={i} className="border-t border-line">
                    <td className="py-0.5 pr-2 font-mono">{fmtTime(r.pts)}</td>
                    <td className="pr-2 font-mono">{r.text || "∅"}</td>
                    <td className="pr-2 font-mono">{r.conf?.toFixed(2)}</td>
                    <td>{r.parsed || <span className="text-err">unparsed</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Details>
    </Card>
  );
}

const txnSchema = z.object({
  txn_ref: z.string().max(64),
  txn_time: z.string().min(1, "journal time required"),
  amount: z.number().nonnegative().nullable(),
  window_before_s: z.number().min(0).max(3600),
  window_after_s: z.number().min(0).max(3600),
});

function TransactionForm({ clip }: { clip: ClipOut }) {
  const qc = useQueryClient();
  const [f, setF] = useState({ txn_ref: "", txn_time: "", amount: "", window_before_s: "90", window_after_s: "30" });
  const [err, setErr] = useState<string | null>(null);
  const m = useMutation({
    mutationFn: (b: object) => post<TxnOut[]>(`/clips/${clip.id}/transactions`, [b]),
    onSuccess: () => {
      setF({ ...f, txn_ref: "", txn_time: "", amount: "" });
      setErr(null);
      qc.invalidateQueries({ queryKey: ["clip-txns", clip.id] });
      qc.invalidateQueries({ queryKey: ["jobs", clip.case_id] });
    },
    onError: (e) => setErr(fmtErr(e)),
  });
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const v = txnSchema.safeParse({
      txn_ref: f.txn_ref,
      txn_time: f.txn_time,
      amount: f.amount === "" ? null : Number(f.amount),
      window_before_s: Number(f.window_before_s),
      window_after_s: Number(f.window_after_s),
    });
    if (!v.success) return setErr(v.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; "));
    m.mutate(v.data);
  };
  return (
    <form onSubmit={submit} className="grid grid-cols-2 gap-4 lg:grid-cols-6">
      <div>
        <Label>Reference</Label>
        <Input value={f.txn_ref} onChange={(e) => setF({ ...f, txn_ref: e.target.value })} placeholder="journal ref" />
      </div>
      <div className="col-span-2">
        <Label hint="bank journal clock, ATM time zone">Transaction time</Label>
        <Input type="datetime-local" step="1" value={f.txn_time} onChange={(e) => setF({ ...f, txn_time: e.target.value })} />
      </div>
      <div>
        <Label>Amount</Label>
        <Input type="number" step="0.01" value={f.amount} onChange={(e) => setF({ ...f, amount: e.target.value })} />
      </div>
      <div>
        <Label hint="s">Window before</Label>
        <Input type="number" value={f.window_before_s} onChange={(e) => setF({ ...f, window_before_s: e.target.value })} />
      </div>
      <div>
        <Label hint="s">Window after</Label>
        <Input type="number" value={f.window_after_s} onChange={(e) => setF({ ...f, window_after_s: e.target.value })} />
      </div>
      <div className="col-span-2 flex items-center gap-3 lg:col-span-6">
        <Button disabled={m.isPending}>Add transaction and rank candidates</Button>
        {err && <span className="text-sm text-err">{err}</span>}
      </div>
    </form>
  );
}

function clipHint(clip: ClipOut, ready: boolean, txns: TxnOut[]): Hint {
  if (!ready) return { text: clip.error ? "The video could not be prepared. See the error below." : "The video is still being prepared. This page updates by itself." };
  if (!clip.timemap) return { text: "Set the time mapping below so video time matches the bank journal." };
  if (clip.kind !== "transaction") return { text: "This gallery clip is set up. Index it for search from the case page." };
  if (!txns.length) return { text: "Add the transaction below. The system then ranks who was at the machine." };
  const latest = [...txns].sort((a, b) => b.id - a.id)[0];
  return { text: "Next: pick the customer for this transaction.", action: { label: "Pick the customer", to: `/transactions/${latest.id}` } };
}

export default function ClipSetup() {
  const { id = "" } = useParams();
  const { can } = useAuth();
  const clipQ = useQuery({ queryKey: ["clip", id], queryFn: () => get<ClipOut>(`/clips/${id}`), refetchInterval: 3000 });
  const atms = useQuery({ queryKey: ["atms"], queryFn: () => get<AtmOut[]>("/atms") });
  const txns = useQuery({ queryKey: ["clip-txns", id], queryFn: () => get<TxnOut[]>(`/clips/${id}/transactions`), refetchInterval: 3000 });
  const [pts, setPts] = useState(0);
  if (clipQ.isLoading) return <Spinner />;
  if (clipQ.error) return <ErrorBox error={clipQ.error} />;
  const clip = clipQ.data!;
  const atm = atms.data?.find((a) => a.id === clip.atm_id);
  const dur = Number((clip.probe as any)?.analysis_duration || (clip.probe as any)?.duration_s || 0);
  const ready = !!clip.analysis_sha256;
  return (
    <CaseFrame caseId={clip.case_id} current={clip.timemap ? 3 : 2} hint={clipHint(clip, ready, txns.data || [])}>
      <PageHeader
        title={
          <>
            <span className="break-all">{clip.orig_name}</span> <StatusChip status={clip.status} />
          </>
        }
        description="Set up this clip so the system can find who was at the machine."
        meta={
          <Details label="clip details">
            <KV
              items={[
                ["Original SHA-256", <Mono>{clip.orig_sha256}</Mono>],
                ["Analysis SHA-256", <Mono>{clip.analysis_sha256}</Mono>],
                ["Proxy SHA-256", <Mono>{clip.proxy_sha256}</Mono>],
                ["Decisions", ((clip.ingest as any)?.decisions || []).join("; ") || "-"],
                ["Problems", ((clip.probe as any)?.problems || []).join("; ") || "none"],
                ["Tracks", clip.n_tracks],
              ]}
            />
          </Details>
        }
      />
      {!ready ? (
        <Card title="Preparing video">
          {clip.error ? <ErrorBox error={`${(clip.error as any).code}: ${(clip.error as any).message}`} /> : <p className="text-sm text-muted">Waiting for ingest (probe, validation, normalisation, proxy)…</p>}
        </Card>
      ) : (
        <>
          <Section title="1 · Regions of interest" description="Mark the keypad, cash slot and timestamp area. These are saved per ATM, so this is usually done once.">
            <AtmPanel clip={clip} atm={atm} pts={pts} />
            <div className="flex items-center gap-3 text-xs">
              <span className="font-mono text-muted">Frame at {fmtTime(pts)}</span>
              <input type="range" aria-label="frame position" min={0} max={Math.max(0.1, dur - 0.05)} step={0.1} value={pts} onChange={(e) => setPts(Number(e.target.value))} className="flex-1" />
            </div>
          </Section>
          <Section title="2 · Time mapping" description="Tell the system what the DVR clock showed, so a bank journal time can be found in the video.">
            <TimeMapPanel clip={clip} tz={atm?.tz || "UTC"} />
          </Section>
        </>
      )}
      {clip.kind === "transaction" && (
        <Section id="transactions" title="3 · Transactions" description="Enter the withdrawal from the bank journal. The system ranks the people near the machine at that time.">
          <Card>
            {can("investigator") && <TransactionForm clip={clip} />}
            {!!txns.data?.length && (
              <ul className={cn("divide-y divide-line", can("investigator") && "mt-5 border-t border-line")}>
                {txns.data.map((t) => (
                  <li key={t.id} className="py-3 last:pb-0">
                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
                      <Link className={`${link} font-medium`} to={`/transactions/${t.id}`}>
                        {t.txn_ref || `#${t.id}`}
                      </Link>
                      <span className="font-mono text-xs">{t.txn_time_journal.replace("T", " ")}</span>
                      <StatusChip status={t.status} />
                    </div>
                    {!!(t.notes as string[]).length && <div className="mt-1 text-xs text-warn">{(t.notes as string[]).join("; ")}</div>}
                    <Details>
                      <KV
                        items={[
                          ["Video time", fmtTime(t.t_txn_pts)],
                          ["Window", `−${t.window_before_s} / +${t.window_after_s} s`],
                        ]}
                      />
                    </Details>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </Section>
      )}
      <JobToasts caseId={clip.case_id} />
    </CaseFrame>
  );
}
