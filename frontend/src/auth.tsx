import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { get, mediaTokenValid, onMediaToken, post, refreshMediaToken, setUnauthorizedHandler, tokenStore } from "./api/client";
import type { Role, TokenOut, UserOut } from "./api/types";

interface Auth {
  user: UserOut | null;
  ready: boolean;
  mediaReady: boolean;
  login: (u: string, p: string) => Promise<void>;
  logout: () => void;
  can: (min: Role) => boolean;
}

const Ctx = createContext<Auth>(null as unknown as Auth);
const RANK: Record<Role, number> = { viewer: 0, investigator: 1, admin: 2 };

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserOut | null>(null);
  const [ready, setReady] = useState(false);
  const [mediaReady, setMediaReady] = useState(mediaTokenValid());
  const qc = useQueryClient();

  const logout = useCallback(() => {
    tokenStore.clear();
    setUser(null);
    qc.clear();
  }, [qc]);

  useEffect(() => {
    setUnauthorizedHandler(() => logout());
    const off = onMediaToken(() => setMediaReady(true));
    if (tokenStore.get()) {
      get<UserOut>("/auth/me")
        .then(async (u) => {
          setUser(u);
          if (!mediaTokenValid()) await refreshMediaToken();
          setMediaReady(true);
        })
        .catch(() => tokenStore.clear())
        .finally(() => setReady(true));
    } else setReady(true);
    return () => {
      off();
    };
  }, [logout]);

  // refresh the short-lived media token every 20 minutes
  useEffect(() => {
    if (!user) return;
    const id = setInterval(() => refreshMediaToken().catch(() => {}), 20 * 60 * 1000);
    return () => clearInterval(id);
  }, [user]);

  const login = async (username: string, password: string) => {
    const t = await post<TokenOut>("/auth/login", { username, password });
    tokenStore.set(t.access_token);
    const me = await get<UserOut>("/auth/me");
    await refreshMediaToken();
    setMediaReady(true);
    setUser(me);
  };

  const can = (min: Role) => !!user && RANK[user.role as Role] >= RANK[min];
  return <Ctx.Provider value={{ user, ready, mediaReady, login, logout, can }}>{children}</Ctx.Provider>;
}

export const useAuth = () => useContext(Ctx);
