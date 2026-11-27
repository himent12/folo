// Navigation and formatting helpers for the labeling editor (`Editor.tsx`): next image to review,
// next unlabeled image (`U` shortcut), and compact box dimension formatting.
import type { Box, LabelImage } from '$lib/api/types';

/** Ordered ring of images after `fromId`, wrapping around the start (excluding `fromId`). */
function ringAfter(images: readonly LabelImage[], fromId: string): LabelImage[] {
  const i = images.findIndex((x) => x.id === fromId);
  const ordered = i < 0 ? [...images] : [...images.slice(i + 1), ...images.slice(0, i)];
  return ordered.filter((x) => x.id !== fromId);
}

/** The next image in `images` whose labels nobody has reviewed yet, wrapping around. */
export function nextToReview(images: readonly LabelImage[], fromId: string): string | null {
  return ringAfter(images, fromId).find((x) => x.status === 'labeled' && !x.reviewed)?.id ?? null;
}

/**
 * The next unlabeled image (`pending` or `failed`) in `images` after `fromId`, wrapping around.
 * Skips images currently being labeled by the agent (`labeling`) since they are locked for editing.
 */
export function nextUnlabeled(images: readonly LabelImage[], fromId: string): string | null {
  return ringAfter(images, fromId).find((x) => x.status === 'pending' || x.status === 'failed')?.id ?? null;
}

/** Formats a box's pixel dimensions (`W×H`). */
export const boxDimensions = (b: Box): string => `${Math.round(b.x2 - b.x1)}×${Math.round(b.y2 - b.y1)}`;

/** Contextual tooltip payload for the editor's Undo / Redo toolbar buttons. */
export function historyTooltip(kind: 'undo' | 'redo', count: number, locked: boolean): { text: string; keys?: string } {
  const label = kind === 'undo' ? 'Undo' : 'Redo';
  if (locked) return { text: `${label} unavailable while the agent is labeling` };
  const keys = kind === 'undo' ? 'Mod+Z' : 'Mod+Shift+Z';
  return count > 0
    ? { text: `${label} (${count} ${count === 1 ? 'step' : 'steps'})`, keys }
    : { text: `Nothing to ${kind}`, keys };
}
