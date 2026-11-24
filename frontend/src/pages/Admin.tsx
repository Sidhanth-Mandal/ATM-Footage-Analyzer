import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { z } from "zod";
import { fmtErr, get, patch, post, put } from "../api/client";
import type { AuditOut, CaseOut, UserOut, VerifyOut } from "../api/types";
import { Badge, Button, Card, ErrorBox, Input, Label, Mono, PageHeader, Section, Select, Spinner, th, tr } from "../components/ui";

const userSchema = z.object({
  username: z.string().regex(/^[A-Za-z0-9_.-]{3,64}$/, "3-64 letters, digits, _ . -"),
  password: z.string().min(10, "at least 10 characters"),
  role: z.enum(["viewer", "investigator", "admin"]),
});

function UserTable() {
  const qc = useQueryClient();
  const users = useQuery({ queryKey: ["users"], queryFn: () => get<UserOut[]>("/users") });
  const [f, setF] = useState({ username: "", password: "", role: "investigator" });
  const [err, setErr] = useState<string | null>(null);
  const create = useMutation({
    mutationFn: (b: object) => post<UserOut>("/users", b),
    onSuccess: () => {
      setF({ username: "", password: "", role: "investigator" });
      setErr(null);
      qc.invalidateQueries({ queryKey: ["users"] });
    },
    onError: (e) => setErr(fmtErr(e)),
  });
  const upd = useMutation({
    mutationFn: ({ id, body }: { id: number; body: object }) => patch<UserOut>(`/users/${id}`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["users"] }),
    onError: (e) => setErr(fmtErr(e)),
  });
  return (
    <Card title="Users">
      <table className="mb-4 w-full text-sm">
        <tbody>
          {(users.data || []).map((u) => (
            <tr key={u.id} className={tr}>
              <td className="py-2 pr-3">{u.username}</td>
              <td>
                <Select value={u.role} onChange={(e) => upd.mutate({ id: u.id, body: { role: e.target.value } })}>
                  <option>viewer</option>
                  <option>investigator</option>
                  <option>admin</option>
                </Select>
              </td>
              <td className="pr-3">{u.is_active ? <Badge tone="green">Active</Badge> : <Badge tone="red">Disabled</Badge>}</td>
              <td className="text-right">
                <Button size="sm" variant="ghost" onClick={() => upd.mutate({ id: u.id, body: { is_active: !u.is_active } })}>
                  {u.is_active ? "Disable" : "Enable"}
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <form
        className="grid grid-cols-4 items-end gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          const v = userSchema.safeParse(f);
          if (!v.success) return setErr(v.error.issues[0].message);
          create.mutate(v.data);
        }}
      >
        <div>
          <Label>Username</Label>
          <Input value={f.username} onChange={(e) => setF({ ...f, username: e.target.value })} />
        </div>
        <div>
          <Label>Password</Label>
          <Input type="password" value={f.password} onChange={(e) => setF({ ...f, password: e.target.value })} />
        </div>
        <div>
          <Label>Role</Label>
          <Select value={f.role} onChange={(e) => setF({ ...f, role: e.target.value })} className="w-full">
            <option>viewer</option>
            <option>investigator</option>
            <option>admin</option>
          </Select>
        </div>
        <Button>Add user</Button>
      </form>
      {err && <ErrorBox error={err} />}
    </Card>
  );
}

function MembershipEditor() {
  const qc = useQueryClient();
  const cases = useQuery({ queryKey: ["cases"], queryFn: () => get<CaseOut[]>("/cases") });
  const users = useQuery({ queryKey: ["users"], queryFn: () => get<UserOut[]>("/users") });
  const [caseId, setCaseId] = useState("");
  const c = cases.data?.find((x) => x.id === caseId);
  const m = useMutation({
    mutationFn: (ids: number[]) => put<CaseOut>(`/cases/${caseId}/members`, { user_ids: ids }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["cases"] }),
  });
  return (
    <Card title="Case access" description="Choose a case, then tick the users who may open it.">
      <Select value={caseId} onChange={(e) => setCaseId(e.target.value)} className="mb-3 w-full">
        <option value="">choose a case…</option>
        {(cases.data || []).map((x) => (
          <option key={x.id} value={x.id}>
            {x.ref_no}: {x.title}
          </option>
        ))}
      </Select>
      {c && (
        <div className="space-y-1">
          {(users.data || []).map((u) => {
            const on = (c.member_ids || []).includes(u.id);
            return (
              <label key={u.id} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={on}
                  onChange={() => m.mutate(on ? c.member_ids!.filter((x) => x !== u.id) : [...(c.member_ids || []), u.id])}
                />
                {u.username} <span className="font-mono text-xs text-muted">{u.role}</span>
              </label>
            );
          })}
        </div>
      )}
      {m.error && <ErrorBox error={fmtErr(m.error)} />}
    </Card>
  );
}

function AuditTable() {
  const [action, setAction] = useState("");
  const q = useQuery({ queryKey: ["audit", action], queryFn: () => get<AuditOut[]>(`/audit?limit=300${action ? `&action=${encodeURIComponent(action)}` : ""}`) });
  const [v, setV] = useState<VerifyOut | null>(null);
  const verify = useMutation({ mutationFn: () => get<VerifyOut>("/audit/verify"), onSuccess: setV });
  return (
    <Card
      title="Audit log (hash-chained)"
      actions={
        <>
          <Input placeholder="filter action, e.g. login" aria-label="filter by action" value={action} onChange={(e) => setAction(e.target.value)} className="h-8 w-56 text-xs" />
          <Button size="sm" onClick={() => verify.mutate()} disabled={verify.isPending}>
            Verify chain
          </Button>
        </>
      }
    >
      {v && (
        <div className={`mb-3 rounded-control border px-3 py-2 text-sm ${v.ok ? "border-ok/40 bg-ok/10 text-ok" : "border-err/40 bg-err/10 text-err"}`}>
          {v.ok ? `Chain intact: ${v.checked} events verified. Head ${v.head?.slice(0, 16)}…` : `Chain broken at event ${v.first_bad_id}: ${v.reason}`}
        </div>
      )}
      {q.isLoading ? (
        <Spinner />
      ) : (
        <div className="max-h-[480px] overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-surface">
              <tr>
                <th className={th}>#</th>
                <th className={th}>Time (UTC)</th>
                <th className={th}>User</th>
                <th className={th}>Action</th>
                <th className={th}>Object</th>
                <th className={th}>Details</th>
                <th className={th}>Hash</th>
              </tr>
            </thead>
            <tbody>
              {(q.data || []).map((e) => (
                <tr key={e.id} className={`${tr} align-top`}>
                  <td className="py-1.5 pr-3 font-mono text-muted">{e.id}</td>
                  <td className="whitespace-nowrap pr-3 font-mono">{e.ts.slice(0, 19).replace("T", " ")}</td>
                  <td className="pr-3">
                    {e.user_id ?? "-"} <span className="text-muted">{e.role}</span>
                  </td>
                  <td className="pr-3 font-medium">{e.action}</td>
                  <td className="pr-3">
                    {e.object_type} <span className="font-mono text-muted">{e.object_id?.slice(0, 12)}</span>
                  </td>
                  <td className="max-w-md truncate pr-3 text-muted" title={JSON.stringify(e.details)}>
                    {JSON.stringify(e.details)}
                  </td>
                  <td>
                    <Mono>{e.hash.slice(0, 10)}</Mono>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

export default function Admin() {
  return (
    <div className="space-y-6">
      <PageHeader title="Admin" description="Manage users, case access and the audit log." />
      <Section title="Users and case access">
        <div className="grid gap-4 lg:grid-cols-2">
          <UserTable />
          <MembershipEditor />
        </div>
      </Section>
      <Section title="Audit log" description="Every action is recorded in a hash chain. Verify the chain to check that nothing was changed.">
        <AuditTable />
      </Section>
    </div>
  );
}
