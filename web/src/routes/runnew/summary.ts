import type { EstimatePhase } from '$lib/api/types';
import { bytes } from '$lib/format';
import { prefs } from '$lib/prefs.svelte';

// Estimates are ranges (lib/throughput `timeRange`); memory is one figure, in the app's one size format.
export const roughBytes = (b: number) => `~${bytes(b, prefs.units)}`;

/** What the Summary, its phone sheet and the dock show: the server's estimate when there is one, else the reference's. */
export interface Shown {
  lo: number;
  hi: number;
  vram: number;
  vramMeasured: boolean;
  basis: string;
  rough: boolean;
  how: string;
  phases: Pick<EstimatePhase, 'phase' | 'epochs' | 'train_imgs_per_s' | 'epoch_s'>[];
  val: { images: number; rate: number; reference: boolean };
  startup: number;
  updated: string | null;
}
