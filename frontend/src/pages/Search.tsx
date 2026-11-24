import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useParams } from "react-router-dom";
import { fmtErr, get, mediaUrl, post } from "../api/client";
import type { FaceBrief, HitOut, SearchOut } from "../api/types";
import { useAuth } from "../auth";
import { AuthImg, FlagChips, fmtTime, GradeBadge } from "../components/face";
import { Badge, Button, Card, Details, Dialog, Empty, ErrorBox, Input, KV, PageHeader, Spinner } from "../components/ui";
import { CaseFrame, type Hint } from "../components/workflow";

const BANDS = {
  strong: { title: "Strong matches", text: "Very similar faces. Still check each one by eye.", tone: "accent" as const },
  possible: { title: "Possible matches", text: "Similar faces that need careful checking.", tone: "neutral" as const },
  weak: { title: "Weak matches", text: "Low similarity. Usually a different person.", tone: "neutral" as const },
};

function QueryStrip({ faces }: { faces: FaceBrief[] }) {
  return (
    <div className="flex flex-wrap gap-3">
      {faces.map((f) => (
        <div key={f.id} className="space-y-1 text-center">
          <AuthImg path={f.crop_url} className="h-24 w-20 rounded-control border border-line object-cover" />
          <GradeBadge grade={f.grade} total={f.total} />
        </div>
      ))}
    </div>
  );
}

function ReviewButtons({ hit }: { hit: HitOut }) {
  const qc = useQueryClient();
  const { can } = useAuth();
  const [note, setNote] = useState(hit.note || "");
  const m = useMutation({
    mutationFn: (decision: string) => post<HitOut>(`/search_hits/${hit.id}/review`, { decision, note }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["search"] }),
  });
  if (!can("investigator"))
    return hit.review ? <Badge tone={hit.review === "confirmed" ? "green" : "neutral"}>{hit.review === "confirmed" ? "Confirmed" : "Rejected"}</Badge> : <Badge tone="amber">Not reviewed</Badge>;
  return (
    <div className="space-y-2">
      <div>{hit.review ? <Badge tone={hit.review === "confirmed" ? "green" : "neutral"}>{hit.review === "confirmed" ? "Confirmed" : "Rejected"}</Badge> : <Badge tone="amber">Not reviewed</Badge>}</div>
      <Input value={note} onChange={(e) => setNote(e.target.value)} placeholder="review note" aria-label="review note" className="h-8 text-xs" />
      <div className="flex gap-1">
        <Button size="sm" variant={hit.review === "confirmed" ? "primary" : "outline"} onClick={() => m.mutate("confirmed")}>
          Possible match: confirm
        </Button>
        <Button size="sm" variant={hit.review === "rejected" ? "secondary" : "ghost"} onClick={() => m.mutate("rejected")}>
          Reject
        </Button>
      </div>
      {m.error && <ErrorBox error={fmtErr(m.error)} />}
    </div>
  );
}

function PlayAt({ hit, onClose }: { hit: HitOut | null; onClose: () => void }) {
  const f = hit?.faces?.[0] as FaceBrief | undefined;
  return (
    <Dialog open={!!hit} onClose={onClose} title={hit ? `${hit.clip_name} at ${fmtTime(f?.pts)}` : ""} wide>
      {hit && f && <video src={mediaUrl(`/media/proxy/${hit.clip_id}`)} controls autoPlay className="w-full rounded-control bg-black" onLoadedMetadata={(e) => ((e.target as HTMLVideoElement).currentTime = Math.max(0, f.pts - 1))} />}
    </Dialog>
  );
}

function HitCard({ hit, query, onPlay }: { hit: HitOut; query: FaceBrief[]; onPlay: () => void }) {
  const faces = hit.faces as FaceBrief[];
  const band = BANDS[hit.band as keyof typeof BANDS];
  return (
    <div className="flex flex-wrap gap-4 rounded-control border border-line p-3">
      <div className="flex gap-1.5">
        <div className="text-center font-mono text-[10px] text-muted">
          <AuthImg path={query[0]?.crop_url} className="h-28 w-22 rounded-control border border-line object-cover" />
          query
        </div>
        {faces.map((f) => (
          <div key={f.id} className="text-center font-mono text-[10px] text-muted">
            <AuthImg path={f.crop_url} className="h-28 w-22 rounded-control border border-line object-cover" />
            {f.grade} {f.total.toFixed(0)}
          </div>
        ))}
      </div>
      <div className="min-w-0 flex-1 space-y-1.5 text-xs">
        <div className="flex items-center gap-2">
          <Badge tone={band?.tone ?? "neutral"} className={hit.band === "weak" ? "opacity-70" : ""}>
            {hit.band}
          </Badge>
        </div>
        <div className="text-sm">
          {hit.clip_name} · person {hit.track_no}
        </div>
        {faces[0] && <div className="font-mono text-muted">at {fmtTime(faces[0].pts)}</div>}
        <Button size="sm" variant="ghost" onClick={onPlay}>
          Play at moment
        </Button>
        <Details>
          <KV
            items={[
              ["Score", hit.score.toFixed(3)],
              ["Template similarity", Number((hit.parts as any).template_sim ?? NaN).toFixed(3)],
              ["Top-3 faces", Number((hit.parts as any).top3_mean ?? NaN).toFixed(3)],
              ["Wall time", faces[0]?.wall_time?.replace("T", " ").slice(0, 19) || "-"],
              ["Flags", faces[0] ? <FlagChips flags={faces[0].flags} /> : "-"],
            ]}
          />
        </Details>
      </div>
      <div className="w-64">
        <ReviewButtons hit={hit} />
      </div>
    </div>
  );
}

function searchHint(s: SearchOut): Hint {
  const open = s.hits.filter((h) => !h.review && h.band !== "weak").length;
  if (open) return { text: `Review each possible match (${open} not reviewed yet). Confirm only those you have checked by eye.` };
  return { text: "Next: export the confirmed matches with the customer's faces.", action: { label: "Export confirmed matches", to: `/cases/${s.case_id}/export` } };
}

export default function SearchPage() {
  const { id = "" } = useParams();
  const q = useQuery({ queryKey: ["search", id], queryFn: () => get<SearchOut>(`/search/${id}`) });
  const [play, setPlay] = useState<HitOut | null>(null);
  if (q.isLoading) return <Spinner />;
  if (q.error) return <ErrorBox error={q.error} />;
  const s = q.data!;
  const query = s.query_faces as unknown as FaceBrief[];
  const groups = ["strong", "possible", "weak"] as const;
  return (
    <CaseFrame caseId={s.case_id} current={6} hint={searchHint(s)}>
      <PageHeader
        title="Search results"
        description="Similarity of face embeddings. A high score is a lead for human verification, never an identification."
        meta={!s.calibrated && <span className="text-sm text-warn">Bands use uncalibrated default thresholds (run scripts/calibrate.py).</span>}
      />
      <Card title="Searched with">
        <QueryStrip faces={query} />
        <Details label="band thresholds" className="mt-3">
          <KV
            items={[
              ["Strong", `≥ ${s.bands.strong?.toFixed(3)}`],
              ["Possible", `≥ ${s.bands.possible?.toFixed(3)}`],
              ["Weak", `≥ ${s.bands.weak?.toFixed(3)}`],
            ]}
          />
        </Details>
      </Card>
      {groups.map((g) => {
        const hits = s.hits.filter((h) => h.band === g);
        return (
          <Card key={g} title={`${BANDS[g].title} (${hits.length})`} description={BANDS[g].text}>
            {!hits.length ? <Empty>None.</Empty> : <div className="space-y-2">{hits.map((h) => <HitCard key={h.id} hit={h} query={query} onPlay={() => setPlay(h)} />)}</div>}
          </Card>
        );
      })}
      <Card title={`Not comparable (${s.not_comparable.length})`} description="These people were seen in indexed clips, but no face was good enough to compare (covered, too small or too poor). They are listed so that no one is silently excluded.">
        <Details label="people who could not be compared">
          <div className="flex flex-wrap gap-2">
            {(s.not_comparable as any[]).slice(0, 60).map((n) => (
              <div key={n.track_id} className="w-28 text-[10px] text-muted">
                {n.face ? <AuthImg path={n.face.crop_url} className="h-24 w-24 rounded-control border border-line object-cover" /> : <div className="h-24 w-24 rounded-control bg-raised" />}
                <div className="truncate text-fg">{n.clip_name}</div>
                <div>
                  person {n.track_no}: {n.reason.replace("not comparable: ", "")}
                </div>
              </div>
            ))}
          </div>
        </Details>
      </Card>
      <PlayAt hit={play} onClose={() => setPlay(null)} />
    </CaseFrame>
  );
}
