import { useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { z } from "zod";
import { fmtErr } from "../api/client";
import { useAuth } from "../auth";
import { Button, ErrorBox, Input, Label } from "../components/ui";

const schema = z.object({ username: z.string().min(1, "username required"), password: z.string().min(1, "password required") });

export default function Login() {
  const { login, user } = useAuth();
  const nav = useNavigate();
  const loc = useLocation() as { state?: { from?: string } };
  const [username, setU] = useState("");
  const [password, setP] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  if (user) return <Navigate to={loc.state?.from || "/cases"} replace />;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const v = schema.safeParse({ username, password });
    if (!v.success) return setErr(v.error.issues[0].message);
    setBusy(true);
    setErr(null);
    try {
      await login(username, password);
      nav(loc.state?.from || "/cases", { replace: true });
    } catch (e) {
      setErr(fmtErr(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-full bg-bg">
      <div className="hidden flex-1 flex-col justify-end bg-accent p-12 lg:flex">
        <div className="text-3xl font-semibold tracking-tight text-white">ATM CCTV</div>
        <div className="mt-1 max-w-sm text-sm text-white/80">Find and document the customer at an ATM transaction. Results are candidate leads and require human verification.</div>
      </div>
      <div className="flex flex-1 items-center justify-center p-6">
      <form onSubmit={submit} className="w-full max-w-sm space-y-5 rounded-card border border-line bg-surface p-8">
        <div>
          <h1 className="text-xl font-semibold">Sign in</h1>
          <p className="mt-1 text-sm text-muted">Local investigator tool. All actions are audited.</p>
        </div>
        <div>
          <Label>Username</Label>
          <Input autoFocus value={username} onChange={(e) => setU(e.target.value)} autoComplete="username" />
        </div>
        <div>
          <Label>Password</Label>
          <Input type="password" value={password} onChange={(e) => setP(e.target.value)} autoComplete="current-password" />
        </div>
        {err && <ErrorBox error={err} />}
        <Button className="w-full" disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </Button>
      </form>
      </div>
    </div>
  );
}
