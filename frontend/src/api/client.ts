// Thin fetch wrapper. The JWT lives in sessionStorage (cleared when the tab closes).
// Media (<img>, <video>) cannot send headers, so a short-lived media token is appended as ?t=.

const BASE = "/api";
const TOKEN_KEY = "atmcctv.token";
const MEDIA_KEY = "atmcctv.media";

export class ApiError extends Error {
  status: number;
  code: string;
  details: unknown;
  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message);
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export const tokenStore = {
  get: () => sessionStorage.getItem(TOKEN_KEY),
  set: (t: string) => sessionStorage.setItem(TOKEN_KEY, t),
  clear: () => {
    sessionStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(MEDIA_KEY);
  },
};

let onUnauthorized: () => void = () => {};
export function setUnauthorizedHandler(fn: () => void) {
  onUnauthorized = fn;
}

async function parse(res: Response) {
  if (res.status === 204) return null;
  const ct = res.headers.get("content-type") || "";
  const body = ct.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) {
    const b = (typeof body === "object" && body) || {};
    const err = new ApiError(res.status, b.code || `HTTP_${res.status}`, b.message || String(body) || res.statusText, b.details);
    if (res.status === 401) onUnauthorized();
    throw err;
  }
  return body;
}

export async function api<T = unknown>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const t = tokenStore.get();
  if (t) headers.set("Authorization", `Bearer ${t}`);
  let body = init.body;
  if (init.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(init.json);
  }
  const res = await fetch(BASE + path, { ...init, headers, body });
  return parse(res) as Promise<T>;
}

export const get = <T,>(p: string) => api<T>(p);
export const post = <T,>(p: string, json?: unknown) => api<T>(p, { method: "POST", json });
export const put = <T,>(p: string, json?: unknown) => api<T>(p, { method: "PUT", json });
export const patch = <T,>(p: string, json?: unknown) => api<T>(p, { method: "PATCH", json });

/** Upload with progress (fetch has no upload progress). */
export function upload<T>(path: string, form: FormData, onProgress: (f: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", BASE + path);
    const t = tokenStore.get();
    if (t) xhr.setRequestHeader("Authorization", `Bearer ${t}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let b: any = null;
      try {
        b = JSON.parse(xhr.responseText);
      } catch {
        /* not JSON */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(b as T);
      else reject(new ApiError(xhr.status, b?.code || `HTTP_${xhr.status}`, b?.message || xhr.statusText, b?.details));
    };
    xhr.onerror = () => reject(new ApiError(0, "NETWORK", "network error"));
    xhr.send(form);
  });
}

// ---- media token
let mediaToken: { token: string; exp: number } | null = (() => {
  try {
    return JSON.parse(sessionStorage.getItem(MEDIA_KEY) || "null");
  } catch {
    return null;
  }
})();
const listeners = new Set<() => void>();

export async function refreshMediaToken(): Promise<void> {
  if (!tokenStore.get()) return;
  const r = await post<{ token: string; expires_at: string }>("/auth/media-token");
  mediaToken = { token: r.token, exp: new Date(r.expires_at).getTime() };
  sessionStorage.setItem(MEDIA_KEY, JSON.stringify(mediaToken));
  listeners.forEach((l) => l());
}

export function mediaTokenValid() {
  return !!mediaToken && mediaToken.exp - Date.now() > 60_000;
}

export function onMediaToken(fn: () => void) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Build an authorised media URL from an API-relative path like /media/face/1/crop. */
export function mediaUrl(path: string | null | undefined, extra: Record<string, string | number> = {}): string | undefined {
  if (!path || !mediaToken) return undefined;
  const u = new URL(BASE + path, window.location.origin);
  u.searchParams.set("t", mediaToken.token);
  for (const [k, v] of Object.entries(extra)) u.searchParams.set(k, String(v));
  return u.pathname + u.search;
}

export function fmtErr(e: unknown): string {
  if (e instanceof ApiError) {
    const errs = (e.details as any)?.errors;
    if (Array.isArray(errs) && errs.length) return `${e.message}: ${errs.map((x: any) => `${x.loc?.slice(1).join(".")} ${x.msg}`).join("; ")}`;
    return e.message;
  }
  return e instanceof Error ? e.message : String(e);
}
