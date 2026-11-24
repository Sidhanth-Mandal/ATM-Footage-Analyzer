import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { z } from "zod";
import { fmtErr, get, post } from "../api/client";
import type { CaseOut } from "../api/types";
import { useAuth } from "../auth";
import { StatusChip } from "../components/face";
import { Button, Card, Dialog, Empty, ErrorBox, Input, Label, link, PageHeader, Spinner, th, tr } from "../components/ui";

const caseSchema = z.object({ ref_no: z.string().trim().min(1, "reference required").max(64), title: z.string().trim().min(1, "title required").max(200) });

function NewCaseDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [ref_no, setRef] = useState("");
  const [title, setTitle] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const qc = useQueryClient();
  const nav = useNavigate();
  const m = useMutation({
    mutationFn: (b: { ref_no: string; title: string }) => post<CaseOut>("/cases", b),
    onSuccess: (c) => {
      qc.invalidateQueries({ queryKey: ["cases"] });
      onClose();
      nav(`/cases/${c.id}`);
    },
    onError: (e) => setErr(fmtErr(e)),
  });
  return (
    <Dialog open={open} onClose={onClose} title="New case">
      <form
        className="space-y-3"
        onSubmit={(e) => {
          e.preventDefault();
          const v = caseSchema.safeParse({ ref_no, title });
          if (!v.success) return setErr(v.error.issues[0].message);
          m.mutate(v.data);
        }}
      >
        <div>
          <Label>Case reference</Label>
          <Input value={ref_no} onChange={(e) => setRef(e.target.value)} placeholder="e.g. FIR-2026-0412" />
        </div>
        <div>
          <Label>Title</Label>
          <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Disputed withdrawal, ATM 01" />
        </div>
        {err && <ErrorBox error={err} />}
        <div className="flex justify-end gap-2">
          <Button type="button" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button disabled={m.isPending}>Create</Button>
        </div>
      </form>
    </Dialog>
  );
}

export default function Cases() {
  const { can } = useAuth();
  const [open, setOpen] = useState(false);
  const q = useQuery({ queryKey: ["cases"], queryFn: () => get<CaseOut[]>("/cases") });
  return (
    <div className="space-y-6">
      <PageHeader
        title="Cases"
        description="Each case holds the footage, transactions and evidence for one incident. Open a case to continue where you left off."
        actions={can("investigator") && <Button onClick={() => setOpen(true)}>New case</Button>}
      />
      <Card>
        {q.isLoading && <Spinner />}
        {q.error && <ErrorBox error={q.error} />}
        {q.data && !q.data.length && <Empty>{can("investigator") ? "No cases yet. Create one to upload footage." : "You are not a member of any case yet."}</Empty>}
        {!!q.data?.length && (
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className={th}>Reference</th>
                <th className={th}>Title</th>
                <th className={th}>Clips</th>
                <th className={th}>Status</th>
                <th className={th}>Created</th>
              </tr>
            </thead>
            <tbody>
              {q.data.map((c) => (
                <tr key={c.id} className={`${tr} hover:bg-raised/50`}>
                  <td className="py-3 pr-3 font-medium">
                    <Link className={`${link} font-mono`} to={`/cases/${c.id}`}>
                      {c.ref_no}
                    </Link>
                  </td>
                  <td className="pr-3">
                    <div>{c.title}</div>
                    <div className="text-xs text-muted">
                      {c.member_ids?.length || 0} member{c.member_ids?.length === 1 ? "" : "s"}
                    </div>
                  </td>
                  <td className="pr-3">{c.n_clips}</td>
                  <td className="pr-3">
                    <StatusChip status={c.status} />
                  </td>
                  <td className="text-muted">{new Date(c.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
      <NewCaseDialog open={open} onClose={() => setOpen(false)} />
    </div>
  );
}
