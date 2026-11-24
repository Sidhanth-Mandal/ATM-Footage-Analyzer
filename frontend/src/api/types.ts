// Aliases over the types generated from FastAPI's OpenAPI schema (npm run gen:api).
import type { components } from "./schema";

type S = components["schemas"];
export type TokenOut = S["TokenOut"];
export type UserOut = S["UserOut"];
export type CaseOut = S["CaseOut"];
export type AtmOut = S["AtmOut"];
export type AtmIn = S["AtmIn"];
export type JobOut = S["JobOut"];
export type ClipOut = S["ClipOut"];
export type TxnOut = S["TxnOut"];
export type TxnIn = S["TxnIn"];
export type CandidateOut = S["CandidateOut"];
export type TrackOut = S["TrackOut"];
export type TrackSamplesOut = S["TrackSamplesOut"];
export type FaceOut = S["FaceOut"];
export type EnhancementOut = S["EnhancementOut"];
export type SearchOut = S["SearchOut"];
export type HitOut = S["HitOut"];
export type BehaviourOut = S["BehaviourOut"];
export type ExportOut = S["ExportOut"];
export type AuditOut = S["AuditOut"];
export type VerifyOut = S["VerifyOut"];

export type Role = "viewer" | "investigator" | "admin";

export interface Timeline {
  duration: number | null;
  width?: number;
  height?: number;
  tracks: { track_id: number; track_no: number; source: string; t_start: number; t_end: number; near: [number, number][] }[];
}

export type FaceBrief = S["FaceBriefOut"];
