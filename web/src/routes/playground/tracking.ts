// Pure helpers for Playground live ByteTrack tracking: option mapping, label formatting, and bounded motion trails.
// Never duplicates ByteTrack math — all association and Kalman state live in `folo-wasm`'s `FoloTracker`.
import type { ViewBox } from '$lib/detect/DetectionView';
import type { FoloTrackedBox, FoloTrackOptions } from '$lib/inference/browser';
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

/** Maximum recent box-center points retained per active track for short motion trails. */
export const MAX_TRAIL_POINTS = 20;
/** Frames a missing track's trail fades out before pruning. */
const TRAIL_STALE_FRAMES = 3;

/** Builds valid `FoloTrackOptions` aligned with Playground's active confidence slider (`lowThresh <= highThresh <= newTrackThresh`). */
export function trackerOptionsForConf(conf: number): FoloTrackOptions {
  const high = Math.max(0.01, Math.min(1, Number.isFinite(conf) ? conf : 0.25));
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
