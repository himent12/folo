// The screens and states `npm run shoot` captures. Each scene is shot in every theme × viewport listed
// (default: dark + light, desktop + mobile). `setup(page)` drives the UI into the state before the shot.
// `search` adds query params to the page URL (e.g. `scenario=empty` for the mock), next to `mock=1`.

/** @typedef {{ name: string, path: string, themes?: ('dark' | 'light')[], viewports?: ('desktop' | 'laptop' | 'tablet' | 'mobile' | 'small')[], advance?: number, search?: string,
 *   fullPage?: boolean, keepScroll?: boolean, touch?: false, setup?: (page: import('playwright-core').Page, ctx: { vp: string, theme: string }) => Promise<void> }} Scene */
// `touch: false` keeps a scene out of --qa's coarse-pointer pass over phone states: a scene about a keyboard feature
// (shortcuts) or one that sets up touch itself (the touch hint).

import { runScenes, runPageScenes, runsQueueScenes, runsSelectionScenes } from './scenes/runs.mjs';
import { runNewScenes } from './scenes/runNew.mjs';
import { runCompareScenes } from './scenes/runCompare.mjs';
import { datasetScenes } from './scenes/datasets.mjs';
import { modelScenes } from './scenes/models.mjs';
import { playgroundScenes } from './scenes/playground.mjs';
import { stateScenes } from './scenes/states.mjs';
import { settingsScenes } from './scenes/settings.mjs';
import { cloudScenes } from './scenes/cloud.mjs';
import { labelScenes } from './scenes/label.mjs';
import { benchScenes } from './scenes/bench.mjs';

// One module per screen or area under ./scenes/ (the steps they share in ./scenes/drive.mjs). The order below is the
// order the scenes were written in; keep it: the shoot lists them in it, and --qa splits its states over CI's runners
// by position (FOLO_QA_SHARD).
/** @type {Scene[]} */
export const scenes = [
  ...runScenes,
  ...runNewScenes,
  ...runCompareScenes,
  ...runPageScenes,
  ...datasetScenes,
  ...runsQueueScenes,
  ...modelScenes,
  ...playgroundScenes,
  ...stateScenes,
  ...settingsScenes,
  ...runsSelectionScenes,
  ...cloudScenes,
  ...labelScenes,
  ...benchScenes,
];
