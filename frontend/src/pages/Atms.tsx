import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { z } from "zod";
import { fmtErr, get, post } from "../api/client";
import type { AtmOut } from "../api/types";
import { useAuth } from "../auth";
import { Badge, Button, Card, Empty, ErrorBox, Input, Label, PageHeader, Select, th, tr } from "../components/ui";

const atmSchema = z.object({
  atm_code: z.string().trim().min(1, "ATM code required").max(64),
  location: z.string().max(200),
  camera_type: z.enum(["fascia", "lobby"]),
  date_order: z.enum(["DMY", "MDY", "YMD"]),
  tz: z.string().min(1),
  metadata_is_end: z.boolean(),
});

export default function Atms() {
  const { can } = useAuth();
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["atms"], queryFn: () => get<AtmOut[]>("/atms") });
  const [f, setF] = useState({ atm_code: "", location: "", camera_type: "fascia", date_order: "DMY", tz: "Asia/Kolkata", metadata_is_end: false });
  const [err, setErr] = useState<string | null>(null);
  const m = useMutation({
    mutationFn: (b: object) => post<AtmOut>("/atms", b),
    onSuccess: () => {
      setErr(null);
      setF({ ...f, atm_code: "", location: "" });
      qc.invalidateQueries({ queryKey: ["atms"] });
    },
    onError: (e) => setErr(fmtErr(e)),
  });
  return (
    <div className="space-y-6">
      <PageHeader title="ATMs" description="Each ATM stores its camera type, regions and clock settings. Set these once and every clip from that ATM uses them." />
      <Card>
        {!q.data?.length ? (
          <Empty>No ATMs yet.</Empty>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  <th className={th}>Code</th>
                  <th className={th}>Location</th>
                  <th className={th}>Camera</th>
                  <th className={th}>Regions</th>
                  <th className={th}>Timestamp box</th>
                  <th className={th}>Date order</th>
                  <th className={th}>Time zone</th>
                  <th className={th}>File time is</th>
                </tr>
              </thead>
              <tbody>
                {q.data.map((a) => (
                  <tr key={a.id} className={tr}>
                    <td className="py-2.5 pr-3 font-mono font-medium">{a.atm_code}</td>
                    <td className="pr-3">{a.location}</td>
                    <td className="pr-3">
                      <Badge tone="neutral">{a.camera_type}</Badge>
                    </td>
                    <td className="pr-3 text-xs text-muted">{Object.keys(a.rois || {}).join(", ") || "-"}</td>
                    <td className="pr-3 text-xs">{a.ocr_roi ? "set" : <span className="text-muted">-</span>}</td>
                    <td className="pr-3 font-mono text-xs">{a.date_order}</td>
                    <td className="pr-3 font-mono text-xs">{a.tz}</td>
                    <td className="text-xs text-muted">{a.metadata_is_end ? "end of recording" : "start of recording"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="mt-3 text-xs text-muted">Draw regions and the timestamp box on a frame from the clip setup page. ATMs can also be defined in config/atms/&lt;code&gt;.yaml.</p>
      </Card>
      {can("investigator") && (
        <Card title="Add ATM">
          <form
            className="grid grid-cols-2 items-end gap-3 lg:grid-cols-7"
            onSubmit={(e) => {
              e.preventDefault();
              const v = atmSchema.safeParse(f);
              if (!v.success) return setErr(v.error.issues[0].message);
              m.mutate({ ...v.data, rois: {}, ocr_roi: null });
            }}
          >
            <div>
              <Label>Code</Label>
              <Input value={f.atm_code} onChange={(e) => setF({ ...f, atm_code: e.target.value })} />
            </div>
            <div className="lg:col-span-2">
              <Label>Location</Label>
              <Input value={f.location} onChange={(e) => setF({ ...f, location: e.target.value })} />
            </div>
            <div>
              <Label>Camera</Label>
              <Select value={f.camera_type} onChange={(e) => setF({ ...f, camera_type: e.target.value })} className="w-full">
                <option value="fascia">fascia</option>
                <option value="lobby">lobby</option>
              </Select>
            </div>
            <div>
              <Label>Date order</Label>
              <Select value={f.date_order} onChange={(e) => setF({ ...f, date_order: e.target.value })} className="w-full">
                <option>DMY</option>
                <option>MDY</option>
                <option>YMD</option>
              </Select>
            </div>
            <div>
              <Label>Time zone</Label>
              <Input value={f.tz} onChange={(e) => setF({ ...f, tz: e.target.value })} />
            </div>
            <label className="flex h-9 items-center gap-2 text-xs text-muted">
              <input type="checkbox" checked={f.metadata_is_end} onChange={(e) => setF({ ...f, metadata_is_end: e.target.checked })} /> creation_time = end
            </label>
            <div className="col-span-2 lg:col-span-7">
              <Button disabled={m.isPending}>Add ATM</Button>
            </div>
          </form>
          {err && <ErrorBox error={err} />}
        </Card>
      )}
    </div>
  );
}
