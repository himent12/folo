// The screens and states `npm run shoot` captures. Each scene is shot in every theme × viewport listed
// (default: dark + light, desktop + mobile). `setup(page)` drives the UI into the state before the shot.
// `search` adds query params to the page URL (e.g. `scenario=empty` for the mock), next to `mock=1`.

/** @typedef {{ name: string, path: string, themes?: ('dark' | 'light')[], viewports?: ('desktop' | 'laptop' | 'tablet' | 'mobile' | 'small')[], advance?: number, search?: string,
 *   fullPage?: boolean, keepScroll?: boolean, touch?: false, setup?: (page: import('playwright-core').Page, ctx: { vp: string, theme: string }) => Promise<void> }} Scene */
// `touch: false` keeps a scene out of --qa's coarse-pointer pass over phone states: a scene about a keyboard feature
// (shortcuts) or one that sets up touch itself (the touch hint).

import { overviewScenes } from './scenes/overview.mjs';
import { datasetScenes } from './scenes/datasets.mjs';
import { runScenes } from './scenes/runs.mjs';
import { runNewScenes } from './scenes/runNew.mjs';
import { runCompareScenes } from './scenes/runCompare.mjs';
import { modelScenes } from './scenes/models.mjs';
import { playgroundScenes } from './scenes/playground.mjs';
import { settingsScenes } from './scenes/settings.mjs';
import { labelScenes } from './scenes/label.mjs';
import { benchScenes } from './scenes/bench.mjs';
import { shellScenes } from './scenes/shell.mjs';
import { stateScenes } from './scenes/states.mjs';

// One module per screen under ./scenes/ (the steps they share in ./scenes/drive.mjs); the server's empty, loading,
// error and stale states of every screen are in ./scenes/states.mjs, the app's chrome in ./scenes/shell.mjs. Order
// only decides which CI shard checks which state (FOLO_QA_SHARD=i/n takes every n-th), and shots are named by scene.
/** @type {Scene[]} */
export const scenes = [
  ...overviewScenes,
  ...datasetScenes,
  ...runScenes,
  ...runNewScenes,
  ...runCompareScenes,
  ...modelScenes,
  ...playgroundScenes,
  ...settingsScenes,
  ...labelScenes,
  ...benchScenes,
  ...shellScenes,
  ...stateScenes,
];
