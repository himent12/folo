// The screens and states `npm run shoot` captures. Each scene is shot in every theme × viewport listed
// (default: dark + light, desktop + mobile). `setup(page)` drives the UI into the state before the shot.

/** @typedef {{ name: string, path: string, themes?: ('dark' | 'light')[], viewports?: ('desktop' | 'laptop' | 'mobile')[], advance?: number, search?: string,
 *   fullPage?: boolean, setup?: (page: import('playwright-core').Page, ctx: { vp: string, theme: string }) => Promise<void> }} Scene */

/** @type {Scene[]} */
export const scenes = [
  { name: 'overview', path: '/', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'datasets', path: '/datasets' },
  { name: 'runs', path: '/runs', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'run-live', path: '/runs/run_7d1e', advance: 4000, viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'run-failed', path: '/runs/run_a2f0', viewports: ['desktop'] },
  { name: 'run-completed', path: '/runs/run_c19a', viewports: ['desktop', 'mobile'] },
  { name: 'run-new', path: '/runs/new', viewports: ['desktop', 'laptop', 'mobile'] },
  {
    name: 'run-new-focus-mosaic',
    path: '/runs/new',
    themes: ['dark'],
    viewports: ['desktop'],
    setup: async (page) => {
      await page.getByText('Mosaic', { exact: true }).first().hover();
    },
  },
  { name: 'run-compare', path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0', viewports: ['desktop', 'mobile'] },
  { name: 'run-compare-empty', path: '/runs/compare', themes: ['dark'], viewports: ['desktop'] },
  { name: 'run-queued', path: '/runs/run_q4n2', themes: ['dark'], viewports: ['desktop'] },
  { name: 'models', path: '/models' },
  { name: 'playground', path: '/playground' },
  { name: 'settings', path: '/settings' },
  { name: 'overview-empty', path: '/', search: 'scenario=empty', viewports: ['desktop', 'mobile'] },
  { name: 'overview-loading', path: '/', search: 'scenario=loading', themes: ['dark'], viewports: ['desktop'] },
  { name: 'overview-error', path: '/', search: 'scenario=error', viewports: ['desktop'] },
  { name: 'runs-empty', path: '/runs', search: 'scenario=empty', themes: ['dark'], viewports: ['desktop'] },
  { name: 'run-loading', path: '/runs/run_7d1e', search: 'scenario=loading', themes: ['dark'], viewports: ['desktop'] },
  {
    name: 'palette-open',
    fullPage: false,
    path: '/',
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await page.keyboard.press('Control+k');
      await page.keyboard.type('fas');
    },
  },
  {
    name: 'shortcuts-open',
    fullPage: false,
    path: '/runs',
    themes: ['dark'],
    viewports: ['desktop'],
    setup: async (page) => {
      await page.keyboard.press('Shift+?');
    },
  },
  {
    name: 'run-stop-dialog',
    fullPage: false,
    path: '/runs/run_7d1e',
    themes: ['light'],
    viewports: ['desktop'],
    setup: async (page) => {
      await page.getByRole('button', { name: 'Stop' }).click();
    },
  },
  {
    name: 'chart-hover',
    path: '/runs/run_c19a',
    themes: ['dark'],
    viewports: ['desktop'],
    fullPage: false,
    setup: async (page) => {
      const svg = page.locator('#chart-map svg').first();
      const box = await svg.boundingBox();
      if (box) await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.4);
    },
  },
  { name: 'not-found', path: '/nope', themes: ['dark'], viewports: ['desktop'] },
];
