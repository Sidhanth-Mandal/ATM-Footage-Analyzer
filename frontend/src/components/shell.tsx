import type { ReactNode } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { useAuth } from "../auth";
import { cn } from "./ui";

/** Routes that belong to a case; they keep the "Cases" link highlighted. */
const CASE_PREFIXES = ["/cases", "/clips", "/transactions", "/tracks", "/search"];

const focus = "focus:outline-none focus-visible:ring-2 focus-visible:ring-accent";

function NavItem({ to, active, children }: { to: string; active: boolean; children: ReactNode }) {
  return (
    <NavLink
      to={to}
      aria-current={active ? "page" : undefined}
      className={cn("relative flex items-center rounded-control px-3 py-2 text-sm transition-colors", focus, active ? "bg-raised font-medium text-accent" : "text-muted hover:bg-raised/60 hover:text-fg")}
    >
      {active && <span aria-hidden className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-accent" />}
      {children}
    </NavLink>
  );
}

export function Sidebar() {
  const { user, logout, can } = useAuth();
  const { pathname } = useLocation();
  const inCase = CASE_PREFIXES.some((p) => pathname.startsWith(p));
  return (
    <aside className="fixed inset-y-0 left-0 z-20 flex w-56 flex-col border-r border-line bg-surface">
      <Link to="/cases" className={cn("block px-5 pb-6 pt-6", focus)}>
        <div className="text-base font-semibold tracking-tight text-fg">ATM CCTV</div>
        <div className="font-mono text-[11px] uppercase tracking-widest text-muted">Face leads</div>
      </Link>
      <nav aria-label="Main" className="flex flex-col gap-1 px-3">
        <NavItem to="/cases" active={inCase}>
          Cases
        </NavItem>
        <NavItem to="/atms" active={pathname.startsWith("/atms")}>
          ATMs
        </NavItem>
        {can("admin") && (
          <NavItem to="/admin" active={pathname.startsWith("/admin")}>
            Admin
          </NavItem>
        )}
      </nav>
      <div className="mt-auto space-y-3 border-t border-line p-4">
        <p role="note" className="rounded-control border border-warn/30 bg-warn/10 p-3 text-xs leading-snug text-warn">
          Results are candidate leads and require human verification.
        </p>
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <div className="truncate text-sm font-medium">{user?.username}</div>
            <div className="font-mono text-[11px] uppercase tracking-wide text-muted">{user?.role}</div>
          </div>
          <button onClick={logout} className={cn("rounded-control px-2.5 py-1.5 text-xs text-muted hover:bg-raised hover:text-fg", focus)}>
            Log out
          </button>
        </div>
      </div>
    </aside>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-full bg-bg">
      <Sidebar />
      <main className="ml-56 min-h-full">
        <div className="mx-auto w-full max-w-[1400px] px-8 py-6">{children}</div>
      </main>
    </div>
  );
}
