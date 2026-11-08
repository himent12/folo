// The screens and states `npm run shoot` captures. Each scene is shot in every theme × viewport listed
// (default: dark + light, desktop + mobile). `setup(page)` drives the UI into the state before the shot.
// `search` adds query params to the page URL (e.g. `scenario=empty` for the mock), next to `mock=1`.

/** @typedef {{ name: string, path: string, themes?: ('dark' | 'light')[], viewports?: ('desktop' | 'laptop' | 'mobile')[], advance?: number, search?: string,
 *   fullPage?: boolean, setup?: (page: import('playwright-core').Page, ctx: { vp: string, theme: string }) => Promise<void> }} Scene */

/**
 * Advances the page's fake clock until `selector` is visible (the mock's latency runs on that clock).
 * @param {import('playwright-core').Page} page
 * @param {string} selector
 */
async function until(page, selector, ms = 8000) {
  for (let t = 0; t < ms; t += 250) {
    if (await page.locator(selector).first().isVisible().catch(() => false)) return;
    await page.clock.runFor(250);
  }
  throw new Error(`timed out waiting for ${selector}`);
}

/** Runs the Playground on the first sample image and waits for the detections table. */
async function playgroundResult(/** @type {import('playwright-core').Page} */ page) {
  await until(page, 'button.sample');
  await page.locator('button.sample').first().click();
  await page.mouse.move(0, 0);
  await until(page, '#pg-panel tbody tr');
  await page.clock.runFor(500);
}

/** @type {Scene[]} */
export const scenes = [
  { name: 'overview', path: '/', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'datasets', path: '/datasets', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'runs', path: '/runs', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'run-live', path: '/runs/run_7d1e', advance: 4000, viewports: ['desktop', 'laptop', 'mobile'] },
  {
    // The phone layout below the fold: the metrics instrument, then one section at a time (here: per-class results).
    name: 'run-live-phone-classes',
    path: '/runs/run_7d1e',
    advance: 4000,
    viewports: ['mobile'],
    fullPage: true,
    setup: async (page) => {
      await page.getByRole('radio', { name: 'Classes' }).click();
    },
  },
  { name: 'run-failed', path: '/runs/run_a2f0', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'run-completed', path: '/runs/run_c19a', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'run-new', path: '/runs/new', viewports: ['desktop', 'laptop', 'mobile'] },
  {
    name: 'run-new-focus-mosaic',
    fullPage: false,
    path: '/runs/new',
    themes: ['dark'],
    viewports: ['desktop'],
    setup: async (page) => {
      await page.getByText('Mosaic', { exact: true }).first().hover();
    },
  },
  {
    // Up to 1199 px the dock's estimate opens the run summary as a dialog (a bottom sheet on phones).
    name: 'run-new-summary',
    fullPage: false,
    path: '/runs/new',
    viewports: ['laptop', 'mobile'],
    setup: async (page) => {
      await until(page, '.dock-est');
      await page.locator('.dock-est').click();
      await page.locator('dialog[open]').waitFor();
    },
  },
  {
    // The viewport, not the whole page (a full-page shot grows the viewport, so the dock steps aside for the summary's
    // own button): up to 1199 px Start training stays in the bottom dock while the form scrolls.
    name: 'run-new-dock',
    path: '/runs/new',
    fullPage: false,
    viewports: ['laptop'],
    setup: (page) => until(page, '.dock-est'),
  },
  {
    name: 'run-new-aug-open',
    path: '/runs/new',
    themes: ['dark'],
    viewports: ['laptop'],
    setup: async (page) => {
      await until(page, '.aug-toggle');
      await page.locator('.aug-toggle').click();
    },
  },
  { name: 'run-compare', path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0', viewports: ['desktop', 'mobile'] },
  { name: 'run-compare-empty', path: '/runs/compare', themes: ['dark'], viewports: ['desktop'] },
  { name: 'run-queued', path: '/runs/run_q4n2', themes: ['dark'], viewports: ['desktop'] },
  { name: 'datasets-empty', path: '/datasets', search: 'scenario=empty' },
  {
    name: 'datasets-add',
    path: '/datasets',
    fullPage: false,
    setup: async (page) => {
      await page.getByRole('button', { name: 'Add dataset' }).first().click();
      await page.locator('dialog[open]').waitFor();
    },
  },
  { name: 'dataset-stats', path: '/datasets/ds_shapes_hard?tab=stats', viewports: ['desktop', 'laptop', 'mobile'] },
  {
    name: 'dataset-images',
    path: '/datasets/ds_fasteners?class=2',
    fullPage: false,
    setup: async (page) => {
      await page.locator('.cell img.loaded').nth(6).waitFor();
      await page.waitForTimeout(400);
    },
  },
  {
    name: 'dataset-plain',
    path: '/datasets/ds_fasteners',
    fullPage: false,
    setup: async (page) => {
      await page.locator('.cell img.loaded').nth(6).waitFor();
      await page.waitForTimeout(400);
    },
  },
  {
    // Phones: boxes and thumbnail size live behind the toolbar's Display button.
    name: 'dataset-display',
    path: '/datasets/ds_fasteners?class=2',
    fullPage: false,
    viewports: ['mobile'],
    setup: async (page) => {
      await page.locator('.cell img.loaded').first().waitFor();
      await page.getByRole('button', { name: 'Display options' }).click();
      await page.waitForTimeout(200);
    },
  },
  {
    name: 'dataset-inspector',
    path: '/datasets/ds_fasteners',
    viewports: ['desktop', 'laptop', 'mobile'],
    fullPage: false,
    setup: async (page) => {
      await page.locator('.cell img.loaded').first().waitFor();
      await page.locator('.cell').nth(1).click();
      await page.locator('dialog[open] .stage img.loaded').waitFor();
      await page.locator('dialog[open] .labels button').nth(1).hover();
      await page.waitForTimeout(200);
    },
  },
  { name: 'runs', path: '/runs' },
  { name: 'models', path: '/models', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'models-empty', path: '/models', search: 'scenario=empty' },
  { name: 'models-error', path: '/models', search: 'scenario=error', viewports: ['desktop'] },
  { name: 'models-loading', path: '/models', search: 'scenario=loading', viewports: ['desktop', 'laptop', 'mobile'] },
  {
    // Phones open on the Curves section (the matrix is behind the section picker).
    name: 'model-detail',
    path: '/models/m_fast_n',
    viewports: ['desktop', 'laptop', 'mobile'],
    setup: (page, { vp }) => until(page, vp === 'mobile' ? '#pr svg' : '#cm .cell'),
  },
  {
    // Laptop: the details fold into one line above the matrix; opened here, they list in two columns. Phones: the
    // Details section.
    name: 'model-detail-details',
    path: '/models/m_fast_n',
    viewports: ['laptop', 'mobile'],
    fullPage: true,
    setup: async (page, { vp }) => {
      if (vp === 'mobile') {
        await until(page, '#pr svg');
        await page.getByRole('radio', { name: 'Details' }).click();
        return;
      }
      await until(page, '#cm .cell');
      await page.locator('.d-line').click();
    },
  },
  {
    name: 'model-coco',
    path: '/models/m_coco_s',
    viewports: ['desktop', 'laptop'],
    setup: (page) => until(page, '#cm .cell'),
  },
  {
    // Phones, whole page: the KPI hero row, the section picker, then the Curves section.
    name: 'model-phone',
    path: '/models/m_fast_n',
    viewports: ['mobile'],
    fullPage: true,
    setup: (page) => until(page, '#pr svg'),
  },
  {
    // Phones: the Classes section with one class opened (its details row).
    name: 'model-phone-classes',
    path: '/models/m_fast_n',
    viewports: ['mobile'],
    fullPage: true,
    setup: async (page) => {
      await until(page, '#pr svg');
      await page.getByRole('radio', { name: 'Classes' }).click();
      await page.locator('#pr button.cls').first().click();
    },
  },
  {
    // Phones: the Confusion section, with its column range and miss summary.
    name: 'model-phone-confusion',
    path: '/models/m_fast_n',
    viewports: ['mobile'],
    fullPage: true,
    setup: async (page) => {
      await until(page, '#pr svg');
      await page.getByRole('radio', { name: 'Confusion' }).click();
      await until(page, '#cm .cell');
    },
  },
  {
    // Phones: a reduced 13-column matrix (top 12 classes + Other) that scrolls sideways.
    name: 'model-coco-phone',
    path: '/models/m_coco_s',
    viewports: ['mobile'],
    fullPage: true,
    setup: async (page) => {
      await until(page, '#pr svg');
      await page.getByRole('radio', { name: 'Confusion' }).click();
      await until(page, '#cm .cell');
    },
  },
  {
    name: 'model-export',
    path: '/models/m_shapes_s',
    fullPage: false,
    setup: async (page) => {
      await until(page, 'button:has-text("Export")');
      await page.locator('button:has-text("Export")').first().click();
      await until(page, 'dialog[open] fieldset');
    },
  },
  { name: 'playground-empty', path: '/playground', setup: (page) => until(page, 'button.sample') },
  { name: 'playground-result', path: '/playground?model=m_hard_s', viewports: ['desktop', 'laptop', 'mobile'], setup: playgroundResult },
  {
    name: 'playground-selected',
    path: '/playground?model=m_hard_s',
    viewports: ['desktop', 'laptop', 'mobile'],
    setup: async (page) => {
      await playgroundResult(page);
      await page.locator('#pg-panel tbody tr').nth(2).click();
      await page.mouse.move(0, 0);
      await page.evaluate(() => window.scrollTo(0, 0));
    },
  },
  {
    // Phones: IoU, max boxes and the display toggles open in a non-modal bottom sheet over the lower screen; the image
    // stays visible above it. A first-screen shot, since the sheet is fixed to the viewport.
    name: 'playground-thresholds',
    path: '/playground?model=m_hard_s',
    viewports: ['mobile'],
    setup: async (page) => {
      await playgroundResult(page);
      await page.getByRole('button', { name: 'Thresholds' }).click();
      await page.locator('#pg-more').waitFor();
    },
  },
  {
    // Phones: classes and timing as two disclosures under the detections (both opened here).
    name: 'playground-discs',
    path: '/playground?model=m_hard_s',
    viewports: ['mobile'],
    fullPage: true,
    setup: async (page) => {
      await playgroundResult(page);
      for (const s of await page.locator('details.disc > summary').all()) await s.click();
    },
  },
  {
    name: 'playground-json',
    path: '/playground?model=m_hard_s',
    viewports: ['desktop'],
    setup: async (page) => {
      await playgroundResult(page);
      await page.locator('#pg-json').click();
      await until(page, '#pg-panel pre');
    },
  },
  {
    name: 'playground-picker',
    path: '/playground',
    fullPage: false,
    themes: ['dark'],
    setup: async (page) => {
      await until(page, 'button:has-text("Pick from a dataset")');
      await page.locator('button:has-text("Pick from a dataset")').click();
      await until(page, 'dialog[open] button.thumb');
      await page.clock.runFor(1000);
    },
  },
  { name: 'settings', path: '/settings', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'overview-empty', path: '/', search: 'scenario=empty', viewports: ['desktop', 'mobile'] },
  { name: 'overview-loading', path: '/', search: 'scenario=loading', viewports: ['desktop', 'laptop', 'mobile'] },
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
      // Scroll the chart into view first: the pointer cannot hover outside the viewport. Then require the tooltip.
      const svg = page.locator('#run-chart svg[role="img"]').first();
      await svg.scrollIntoViewIfNeeded();
      const box = await svg.boundingBox();
      if (!box) throw new Error('chart-hover: no chart');
      await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.45);
      // The legend is the readout: it switches to the hovered epoch.
      await page.locator('#run-chart .legend .note.at').waitFor({ state: 'visible', timeout: 3000 });
    },
  },
  { name: 'not-found', path: '/nope', themes: ['dark'], viewports: ['desktop'] },
];
