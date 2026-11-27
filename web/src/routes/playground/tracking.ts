// Pure helpers for Playground live ByteTrack tracking: option mapping, label formatting, bounded motion trails,
// bounded frame-indexed track history recording, live session summaries, and JSON/CSV trajectory export builders.
// Never duplicates ByteTrack math — all association and Kalman state live in `folo-wasm`'s `FoloTracker`.
import type { ViewBox } from '$lib/detect/DetectionView';
import type { FoloTrackedBox, FoloTrackOptions } from '$lib/inference/browser';
import type { CsvCell } from '$lib/tableExport';
import { score } from './predict';

export interface TrackTrailPoint {
  x: number;
  y: number;
}

export interface TrackTrail {
  track_id: number;
  class: number;
  points: TrackTrailPoint[];
}

export interface TrackObservation {
  frame: number;
  timestamp_ms: number;
  track_id: number;
  class_id: number;
  class_name: string;
  confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface TrackClassCount {
  class_id: number;
  class_name: string;
  unique_tracks: number;
  observations: number;
}

export interface TrackSessionSummary {
  active_tracks: number;
  unique_tracks: number;
  frames_recorded: number;
  observations_recorded: number;
  observations_dropped: number;
  truncated: boolean;
  by_class: TrackClassCount[];
}

export interface TrackExportPayload {
  source: string;
  mode: 'video' | 'webcam';
  model_id: string | null;
  model_name: string | null;
  summary: TrackSessionSummary;
  observations: TrackObservation[];
}

/** Maximum recent box-center points retained per active track for short motion trails. */
export const MAX_TRAIL_POINTS = 20;
/** Frames a missing track's trail fades out before pruning. */
const TRAIL_STALE_FRAMES = 3;
/** Maximum frame-indexed track observations retained in memory for trajectory export. */
export const MAX_TRACK_OBSERVATIONS = 10_000;
/** Maximum unique track IDs tracked in session metadata to keep long webcam runs strictly bounded. */
export const MAX_TRACK_SUMMARIES = 5_000;

export const TRACK_CSV_HEADERS = [
  'frame',
  'timestamp_ms',
  'track_id',
  'class_id',
  'class_name',
  'confidence',
  'x1',
  'y1',
  'x2',
  'y2',
] as const;

export const EMPTY_TRACK_SUMMARY: TrackSessionSummary = {
  active_tracks: 0,
  unique_tracks: 0,
  frames_recorded: 0,
  observations_recorded: 0,
  observations_dropped: 0,
  truncated: false,
  by_class: [],
};

/** Builds valid `FoloTrackOptions` aligned with Playground's active confidence thresholds (`0 <= lowThresh <= highThresh <= newTrackThresh <= 1`). */
export function trackerOptionsForConf(conf: number): FoloTrackOptions {
  const high = Math.max(0, Math.min(1, Number.isFinite(conf) ? conf : 0.25));
  const low = Math.min(0.1, high);
  // Omit `classes` so per-frame `seen_names` are cleared cleanly on `reset()` when switching models.
  return {
    highThresh: high,
    lowThresh: low,
    newTrackThresh: high,
    minHits: 1,
    classAware: true,
  };
}

/** Formats one canvas/export chip label, prefixing `#<track_id>` when a detection carries a ByteTrack identity. */
export function formatBoxChipText(box: Pick<ViewBox, 'class' | 'score' | 'track_id'>, classes: readonly string[], showScore: boolean): string {
  const clsName = classes[box.class] ?? `class ${box.class}`;
  const head = box.track_id !== undefined ? `#${box.track_id} ${clsName}` : clsName;
  return showScore && box.score !== undefined ? `${head} ${score(box.score)}` : head;
}

/**
 * Bounded per-track center history for rendering short motion trails on `LiveMediaCanvas`.
 * Trails prune automatically after a track disappears and clear immediately on tracker reset.
 */
export class TrackTrailHistory {
  #entries = new Map<number, { class: number; points: TrackTrailPoint[]; missed: number }>();

  constructor(
    readonly maxPoints = MAX_TRAIL_POINTS,
    readonly staleAfter = TRAIL_STALE_FRAMES,
  ) {}

  update(boxes: readonly Pick<FoloTrackedBox, 'track_id' | 'class' | 'x1' | 'y1' | 'x2' | 'y2'>[]): TrackTrail[] {
    const seen = new Set<number>();
    for (const b of boxes) {
      seen.add(b.track_id);
      const pt = { x: (b.x1 + b.x2) / 2, y: (b.y1 + b.y2) / 2 };
      const prev = this.#entries.get(b.track_id);
      if (prev) {
        prev.class = b.class;
        prev.missed = 0;
        prev.points.push(pt);
        if (prev.points.length > this.maxPoints) prev.points.shift();
      } else {
        this.#entries.set(b.track_id, { class: b.class, points: [pt], missed: 0 });
      }
    }
    for (const [id, entry] of this.#entries) {
      if (seen.has(id)) continue;
      entry.missed++;
      if (entry.missed > this.staleAfter) this.#entries.delete(id);
    }
    const out: TrackTrail[] = [];
    for (const [track_id, entry] of this.#entries) {
      if (entry.points.length >= 2) {
        out.push({ track_id, class: entry.class, points: [...entry.points] });
      }
    }
    return out;
  }

  clear(): void {
    this.#entries.clear();
  }
}

/**
 * Bounded frame-indexed observation recorder for Playground ByteTrack sessions.
 * Retains the most recent `maxObservations` observations (dropping oldest FIFO when full) and maintains
 * per-class unique track ID counts without unbounded memory growth during long camera/video sessions.
 */
export class TrackHistoryRecorder {
  #observations: TrackObservation[] = [];
  #seenTracks = new Map<number, { class_id: number; class_name: string; observations: number }>();
  #framesRecorded = 0;
  #activeTracks = 0;
  #droppedObservations = 0;
  #trackSummaryCapped = false;

  constructor(
    readonly maxObservations = MAX_TRACK_OBSERVATIONS,
    readonly maxTrackSummaries = MAX_TRACK_SUMMARIES,
  ) {}

  /** Records one tracked frame's observations and returns a fresh immutable summary snapshot. */
  record(
    frame: number,
    timestampMs: number,
    boxes: readonly Pick<FoloTrackedBox, 'track_id' | 'class' | 'name' | 'score' | 'x1' | 'y1' | 'x2' | 'y2'>[],
    classes: readonly string[],
  ): TrackSessionSummary {
    const safeFrame = Number.isFinite(frame) ? Math.max(0, Math.round(frame)) : this.#framesRecorded + 1;
    const safeTs = Number.isFinite(timestampMs) ? Math.max(0, Math.round(timestampMs)) : 0;
    this.#framesRecorded++;
    this.#activeTracks = boxes.length;

    for (const b of boxes) {
      const className = classes[b.class] ?? b.name ?? `class ${b.class}`;
      const obs: TrackObservation = {
        frame: safeFrame,
        timestamp_ms: safeTs,
        track_id: b.track_id,
        class_id: b.class,
        class_name: className,
        confidence: Number.isFinite(b.score) ? Number(b.score.toFixed(4)) : 0,
        x1: Number.isFinite(b.x1) ? Number(b.x1.toFixed(2)) : 0,
        y1: Number.isFinite(b.y1) ? Number(b.y1.toFixed(2)) : 0,
        x2: Number.isFinite(b.x2) ? Number(b.x2.toFixed(2)) : 0,
        y2: Number.isFinite(b.y2) ? Number(b.y2.toFixed(2)) : 0,
      };
      this.#observations.push(obs);

      const prev = this.#seenTracks.get(b.track_id);
      if (prev) {
        prev.class_id = b.class;
        prev.class_name = className;
        prev.observations++;
      } else if (this.#seenTracks.size < this.maxTrackSummaries) {
        this.#seenTracks.set(b.track_id, { class_id: b.class, class_name: className, observations: 1 });
      } else {
        this.#trackSummaryCapped = true;
      }
    }

    if (this.#observations.length > this.maxObservations) {
      const excess = this.#observations.length - this.maxObservations;
      this.#observations.splice(0, excess);
      this.#droppedObservations += excess;
    }

    return this.summary();
  }

  /** Returns a shallow copy of recorded observations in chronological order. */
  observations(): TrackObservation[] {
    return this.#observations.slice();
  }

  /** Computes the current session summary with per-class unique track totals sorted descending. */
  summary(): TrackSessionSummary {
    const byClassMap = new Map<number, TrackClassCount>();
    for (const entry of this.#seenTracks.values()) {
      const cur = byClassMap.get(entry.class_id);
      if (cur) {
        cur.unique_tracks++;
        cur.observations += entry.observations;
      } else {
        byClassMap.set(entry.class_id, {
          class_id: entry.class_id,
          class_name: entry.class_name,
          unique_tracks: 1,
          observations: entry.observations,
        });
      }
    }
    const by_class = [...byClassMap.values()].sort(
      (a, b) => b.unique_tracks - a.unique_tracks || b.observations - a.observations || a.class_id - b.class_id,
    );
    return {
      active_tracks: this.#activeTracks,
      unique_tracks: this.#seenTracks.size,
      frames_recorded: this.#framesRecorded,
      observations_recorded: this.#observations.length,
      observations_dropped: this.#droppedObservations,
      truncated: this.#droppedObservations > 0 || this.#trackSummaryCapped,
      by_class,
    };
  }

  clear(): void {
    this.#observations = [];
    this.#seenTracks.clear();
    this.#framesRecorded = 0;
    this.#activeTracks = 0;
    this.#droppedObservations = 0;
    this.#trackSummaryCapped = false;
  }
}

/** Builds a structured JSON trajectory payload for Playground track export. */
export function buildTrackExportJson(opts: {
  sourceName: string;
  mode: 'video' | 'webcam';
  modelId: string | null;
  modelName: string | null;
  summary: TrackSessionSummary;
  observations: readonly TrackObservation[];
}): TrackExportPayload {
  return {
    source: opts.sourceName || opts.mode,
    mode: opts.mode,
    model_id: opts.modelId,
    model_name: opts.modelName,
    summary: opts.summary,
    observations: [...opts.observations],
  };
}

/** Builds RFC-4180 CSV rows aligned with `TRACK_CSV_HEADERS`. */
export function buildTrackExportCsvRows(observations: readonly TrackObservation[]): CsvCell[][] {
  return observations.map((o) => [
    o.frame,
    o.timestamp_ms,
    o.track_id,
    o.class_id,
    o.class_name,
    o.confidence,
    o.x1,
    o.y1,
    o.x2,
    o.y2,
  ]);
}
