import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { fmtErr, get, mediaUrl, post } from "../api/client";
import type { ClipOut, EnhancementOut, FaceOut, SearchOut, TrackOut } from "../api/types";
import { useAuth } from "../auth";
import { AuthImg, FLAG_HELP, FlagChips, fmtTime, GradeBadge, SubScoreBars } from "../components/face";
import { JobToasts, useJob } from "../components/jobs";
import { Badge, Button, Card, cn, Details, Dialog, Drawer, Empty, ErrorBox, Input, KV, Label, Mono, PageHeader, Select, Spinner } from "../components/ui";
import { CaseFrame, type Hint } from "../components/workflow";
import { OVERLAY_BOX } from "../theme";

function FrameWithBox({ face, clip }: { face: FaceOut; clip?: ClipOut }) {
  const W = Number((clip?.probe as any)?.analysis_width || 1920);
  const H = Number((clip?.probe as any)?.analysis_height || 1080);
  const b = face.bbox as number[];
  return (
    <div className="relative w-full overflow-hidden rounded-control bg-black" style={{ aspectRatio: `${W} / ${H}` }}>
      <AuthImg path={face.urls.frame} className="absolute inset-0 h-full w-full object-contain" />
      <svg viewBox={`0 0 ${W} ${H}`} className="absolute inset-0 h-full w-full" preserveAspectRatio="none">
        <rect x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill="none" style={{ stroke: OVERLAY_BOX }} strokeWidth={W / 400} />
      </svg>
    </div>
  );
}

function JumpVideo({ clip, pts }: { clip: ClipOut; pts: number }) {
  const v = useRef<HTMLVideoElement>(null);
  const start = Number((clip.probe as any)?.analysis_start_time || 0);
  return (
    <video
      ref={v}
      src={mediaUrl(clip.proxy_url)}
      controls
      preload="metadata"
      className="w-full rounded-control bg-black"
      onLoadedMetadata={() => {
        if (v.current) v.current.currentTime = Math.max(0, pts - start);
      }}
    />
  );
}

function FaceDetailDrawer({ face, clip, onClose }: { face: FaceOut | null; clip?: ClipOut; onClose: () => void }) {
  const [video, setVideo] = useState(false);
  useEffect(() => setVideo(false), [face?.id]);
  if (!face) return null;
  const raw = face.raw as Record<string, number>;
  return (
    <Drawer open={!!face} onClose={onClose} title={`Face ${face.id}`}>
      <div className="space-y-5">
        <FrameWithBox face={face} clip={clip} />
        <div className="flex gap-4">
          <AuthImg path={face.urls.crop} className="max-h-56 rounded-control border border-line" />
          <AuthImg path={face.urls.aligned} className="h-28 w-28 rounded-control border border-line" />
          <div className="flex-1 space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <GradeBadge grade={face.grade} total={face.total} />
              {face.searchable ? <Badge tone="green">Searchable</Badge> : <Badge tone="red">Not used for search</Badge>}
              {face.detview_corrected && (
                <Badge tone="accent" title="detector saw a brightness-corrected copy; evidence pixels are original">
                  detected on corrected view
                </Badge>
              )}
            </div>
            <FlagChips flags={face.flags as string[]} />
            {!!(face.caps as string[]).length && <div className="text-xs text-err">Capped at grade C: {(face.caps as string[]).join("; ")}</div>}
            <SubScoreBars sub={face.sub as any} />
          </div>
        </div>
        {clip?.proxy_url &&
          (video ? (
            <JumpVideo clip={clip} pts={face.pts} />
          ) : (
            <Button variant="outline" onClick={() => setVideo(true)}>
              Jump to video at {fmtTime(face.pts)}
            </Button>
          ))}
        {!!(face.flags as string[]).length && (
          <p className="text-xs text-muted">
            {(face.flags as string[]).map((f) => (
              <span key={f} className="mr-2">
                <b className="text-fg">{f}</b>: {FLAG_HELP[f]}.
              </span>
            ))}
          </p>
        )}
        <Details label="measurements and hashes">
          <KV
            items={[
              ["PTS / frame", `${face.pts.toFixed(3)} s / ${face.frame_idx}`],
              ["Wall time", face.wall_time || "-"],
              ["IOD", `${raw.iod_px?.toFixed(1)} px`],
              ["Pose (yaw, pitch, roll)", `${raw.yaw?.toFixed(1)}°, ${raw.pitch?.toFixed(1)}°, ${raw.roll?.toFixed(1)}°`],
              ["Laplacian variance", raw.lap_var?.toFixed(1)],
              ["Face luma / clipped / L-R ratio", `${raw.mean_luma?.toFixed(2)} / ${raw.clip_frac?.toFixed(3)} / ${raw.lr_ratio?.toFixed(2)}`],
              ["Visible fraction", `${raw.visible_frac?.toFixed(2)}${raw.parse_unreliable ? " (parsing unreliable, neutral value)" : ""}`],
              ["eDifFIQA raw", raw.fiqa_raw?.toFixed(4)],
              ["Attributes", Object.entries(face.attrib || {}).map(([k, v]) => `${k} ${typeof v === "number" ? v.toFixed(2) : JSON.stringify(v)}`).join(", ")],
              ["Frame SHA-256", <Mono>{face.frame_sha256}</Mono>],
              ["Crop SHA-256", <Mono>{face.crop_sha256}</Mono>],
              ["Aligned SHA-256", <Mono>{face.aligned_sha256}</Mono>],
            ]}
          />
        </Details>
      </div>
    </Drawer>
  );
}

function EnhanceDialog({ face, onClose }: { face: FaceOut | null; onClose: () => void }) {
  const [method, setMethod] = useState("clahe");
  const [param, setParam] = useState("0.7");
  const [enh, setEnh] = useState<EnhancementOut | null>(null);
  const job = useJob(enh?.job_id);
  const qc = useQueryClient();
  const m = useMutation({
    mutationFn: () => post<EnhancementOut>(`/faces/${face!.id}/enhance`, { method, params: method === "gamma" ? { gamma: Number(param) } : method === "clahe" ? { clip: 2.0 } : {} }),
    onSuccess: setEnh,
  });
  const done = job.data?.status === "done";
  useEffect(() => {
    if (done) qc.invalidateQueries({ queryKey: ["faces"] });
  }, [done, qc]);
  useEffect(() => setEnh(null), [face?.id]);
  if (!face) return null;
  return (
    <Dialog open={!!face} onClose={onClose} title={`Enhance face ${face.id} (labelled, never used for matching)`} wide>
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <div>
          <Label>Method</Label>
          <Select value={method} onChange={(e) => setMethod(e.target.value)}>
            <option value="clahe">CLAHE (local contrast)</option>
            <option value="gamma">Gamma</option>
            <option value="gfpgan_v1.4_onnx">GFPGAN v1.4 (generative: may alter identity)</option>
          </Select>
        </div>
        {method === "gamma" && (
          <div>
            <Label>Gamma</Label>
            <Input type="number" step="0.05" min="0.2" max="3" value={param} onChange={(e) => setParam(e.target.value)} className="w-24" />
          </div>
        )}
        <Button onClick={() => m.mutate()} disabled={m.isPending || (!!enh && !done && job.data?.status !== "failed")}>
          Create enhanced copy
        </Button>
      </div>
      {method === "gfpgan_v1.4_onnx" && <p className="mb-3 text-xs text-err">Generative restoration invents detail. Use it only for orientation; it is excluded from search and shown in a separate section of the evidence sheet.</p>}
      {m.error && <ErrorBox error={fmtErr(m.error)} />}
      {job.data?.status === "failed" && <ErrorBox error={job.data.message} />}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="mb-1.5 font-mono text-[11px] uppercase tracking-wide text-muted">Original (evidence pixels)</div>
          <AuthImg path={face.urls.crop} className="max-h-96 rounded-control border border-line" />
        </div>
        <div>
          <div className="mb-1.5 font-mono text-[11px] uppercase tracking-wide text-err">Enhanced copy</div>
          {done && enh ? <AuthImg path={`/media/enhanced/${enh.id}`} enhanced className="max-h-96 rounded-control" /> : enh ? <Spinner /> : <Empty>Choose a method and create a copy.</Empty>}
        </div>
      </div>
      {!!face.enhancements?.length && (
        <div className="mt-5">
          <div className="mb-1.5 font-mono text-[11px] uppercase tracking-wide text-muted">Earlier enhancements</div>
          <div className="flex flex-wrap gap-2">
            {(face.enhancements as any[]).filter((e) => e.url).map((e) => (
              <div key={e.id} className="text-[10px] text-muted">
                <AuthImg path={e.url} enhanced className="h-28 rounded-control" />
                <div>
                  {e.method} {JSON.stringify(e.params)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </Dialog>
  );
}

export default function TrackFaces() {
  const { id = "" } = useParams();
  const { can } = useAuth();
  const nav = useNavigate();
  const [showAll, setShowAll] = useState(false);
  const trackQ = useQuery({ queryKey: ["track", id], queryFn: () => get<TrackOut>(`/tracks/${id}`), refetchInterval: (q) => (q.state.data?.dense_status === "done" ? false : 2000) });
  const faces = useQuery({ queryKey: ["faces", id, showAll], queryFn: () => get<FaceOut[]>(`/tracks/${id}/faces${showAll ? "" : "?selected=true"}`), enabled: !!trackQ.data });
  const clipQ = useQuery({ queryKey: ["clip", trackQ.data?.clip_id], queryFn: () => get<ClipOut>(`/clips/${trackQ.data!.clip_id}`), enabled: !!trackQ.data });
  const [sel, setSel] = useState<Set<number>>(new Set());
  const [detail, setDetail] = useState<FaceOut | null>(null);
  const [enh, setEnh] = useState<FaceOut | null>(null);
  const search = useMutation({
    mutationFn: () => post<SearchOut>("/search", { case_id: clipQ.data!.case_id, face_ids: [...sel] }),
    onSuccess: (s) => nav(`/search/${s.search_id}`),
  });
  if (trackQ.isLoading) return <Spinner />;
  if (trackQ.error) return <ErrorBox error={trackQ.error} />;
  const t = trackQ.data!;
  const list = faces.data || [];
  const selFaces = list.filter((f) => sel.has(f.id));
  const nonSearchableSel = selFaces.filter((f) => !f.searchable);
  const toggle = (fid: number) =>
    setSel((s) => {
      const n = new Set(s);
      n.has(fid) ? n.delete(fid) : n.add(fid);
      return n;
    });
  const hint: Hint =
    t.dense_status !== "done"
      ? { text: "Extracting faces for this person. This page updates by itself." }
      : !sel.size
        ? { text: "Select the clearest faces. Grade A and B faces work best for search." }
        : {
            text: nonSearchableSel.length ? "Remove the faces that are not searchable from your selection to search other clips." : `Search the other clips in this case with the ${sel.size} selected face${sel.size === 1 ? "" : "s"}, or export them.`,
            action: can("investigator")
              ? { label: `Search other clips with ${sel.size} selected`, onClick: () => search.mutate(), disabled: !sel.size || !!nonSearchableSel.length || search.isPending || !clipQ.data }
              : undefined,
          };
  return (
    <CaseFrame caseId={clipQ.data?.case_id} current={5} hint={hint}>
      <PageHeader
        title={`Review faces · Person ${t.track_no}`}
        description={
          <>
            The best frames of the confirmed customer. Open a face for quality details.
            {t.face_visible_frac != null && <> The face is visible in {(t.face_visible_frac * 100).toFixed(0)}% of this person's frames.</>}
          </>
        }
        meta={
          t.stats ? (
            <Details label="scoring summary">
              <KV
                items={[
                  ["Faces scored", (t.stats as any).faces_scored],
                  ["Grades", <Mono>{JSON.stringify((t.stats as any).grades || {})}</Mono>],
                ]}
              />
            </Details>
          ) : undefined
        }
      />
      <div className="flex flex-wrap items-center gap-4 rounded-card border border-line bg-surface px-5 py-3">
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} /> Show alternates
        </label>
        <span className="text-sm text-muted">{sel.size} selected</span>
        {can("investigator") && (
          <Button variant="outline" className="ml-auto" disabled={!sel.size} onClick={() => nav(`/cases/${clipQ.data?.case_id}/export?faces=${[...sel].join(",")}`)}>
            Export selected
          </Button>
        )}
      </div>
      {search.error && <ErrorBox error={fmtErr(search.error)} />}
      {!!nonSearchableSel.length && <div className="text-sm text-warn">Faces {nonSearchableSel.map((f) => f.id).join(", ")} are not searchable (grade C, covered face or low visibility) and cannot be used as a search query.</div>}
      {t.dense_status !== "done" && (
        <Card>
          <div className="flex items-center gap-3 text-sm text-muted">
            <Spinner /> Extracting faces for this track…
          </div>
        </Card>
      )}
      {t.dense_status === "done" && !list.length && <Empty>No face was detected on this track.</Empty>}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
        {list.map((f) => (
          <div key={f.id} className={cn("rounded-card border bg-surface p-4", sel.has(f.id) ? "border-accent ring-1 ring-accent" : "border-line")}>
            <div className="flex gap-3">
              <button onClick={() => setDetail(f)} className="shrink-0 rounded-control focus:outline-none focus-visible:ring-2 focus-visible:ring-accent" aria-label={`Open face ${f.id}`}>
                <AuthImg path={f.urls.crop} className="h-36 w-28 rounded-control border border-line object-cover" />
              </button>
              <div className="min-w-0 flex-1 space-y-2">
                <div className="flex items-center justify-between">
                  <GradeBadge grade={f.grade} total={f.total} />
                  <input type="checkbox" checked={sel.has(f.id)} onChange={() => toggle(f.id)} aria-label={`select face ${f.id}`} className="h-4 w-4" />
                </div>
                <div className="font-mono text-[11px] text-muted">
                  {fmtTime(f.pts)} · {f.wall_time?.slice(11, 23) || "-"}
                  {f.selected ? "" : " · alternate"}
                </div>
                {f.searchable ? <Badge tone="green">Searchable</Badge> : <Badge tone="red">Not searchable</Badge>}
                {!!f.enhancements?.length && (
                  <div>
                    <Badge tone="red">{f.enhancements.length} enhanced</Badge>
                  </div>
                )}
              </div>
            </div>
            <div className="mt-3 flex gap-1">
              <Button size="sm" variant="ghost" onClick={() => setDetail(f)}>
                Details
              </Button>
              {can("investigator") && (
                <Button size="sm" variant="ghost" onClick={() => setEnh(f)}>
                  Enhance…
                </Button>
              )}
            </div>
          </div>
        ))}
      </div>
      <FaceDetailDrawer face={detail} clip={clipQ.data} onClose={() => setDetail(null)} />
      <EnhanceDialog face={enh} onClose={() => setEnh(null)} />
      <JobToasts caseId={clipQ.data?.case_id} />
    </CaseFrame>
  );
}
