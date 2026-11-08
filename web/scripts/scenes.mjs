// The screens and states `npm run shoot` captures. Each scene is shot in every theme × viewport listed
// (default: dark + light, desktop + mobile). `setup(page)` drives the UI into the state before the shot.

/** @typedef {{ name: string, path: string, themes?: ('dark' | 'light')[], viewports?: ('desktop' | 'laptop' | 'mobile')[], advance?: number,
 *   fullPage?: boolean, setup?: (page: import('playwright-core').Page, ctx: { vp: string, theme: string }) => Promise<void> }} Scene */

/** @type {Scene[]} */
export const scenes = [
  { name: 'overview', path: '/', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'datasets', path: '/datasets' },
  { name: 'runs', path: '/runs', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'run-live', path: '/runs/run_7d1e', advance: 4000, viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'run-failed', path: '/runs/run_a2f0', viewports: ['desktop'] },
  { name: 'run-completed', path: '/runs/run_c19a', viewports: ['desktop', 'mobile'] },
  { name: 'run-queued', path: '/runs/run_q4n2', themes: ['dark'], viewports: ['desktop'] },
  { name: 'models', path: '/models' },
  { name: 'playground', path: '/playground' },
  { name: 'settings', path: '/settings' },
  { name: 'not-found', path: '/nope', themes: ['dark'], viewports: ['desktop'] },
];
