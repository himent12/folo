// Across the screens: empty, loading, error and stale states, hover, selection and keyboard on the lists, the app's
// overlays (command palette, shortcuts, Stop's confirm), chart hover and Not found.

import { until } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** @type {Scene[]} */
export const stateScenes = [
  { name: 'settings', path: '/settings', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  { name: 'overview-empty', path: '/', search: 'scenario=empty', viewports: ['desktop', 'mobile'] },
  { name: 'overview-loading', path: '/', search: 'scenario=loading', viewports: ['desktop', 'laptop', 'mobile'] },
  // Nothing loads: one recovery block on the page (Retry, Connection settings), not four panels each saying so.
  { name: 'overview-error', path: '/', search: 'scenario=error', viewports: ['desktop', 'laptop', 'mobile'] },
  // The server stops answering after the first loads, its live stream too: the page keeps what it has under one
  // stale-data notice, and the live card shows its last known state ("Last known: Running · 2m ago", "~7m left, as of
  // 2m ago", no heartbeat). Two minutes on, so the ages read as minutes.
  { name: 'overview-stale', path: '/', search: 'scenario=stale', advance: 125_000, viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  // Only the lists stop refreshing; the live run's stream still streams: the notice says the lists are behind and the
  // live card stays live.
  { name: 'overview-stale-lists', path: '/', search: 'scenario=stale-lists', advance: 125_000, viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  // A model scored on its training split (fasteners-mini has no validation images): Best models tags its score
  // ("Training split"), ranks it only within its own dataset and split, and counts imported and published weights apart.
  { name: 'overview-train-split', path: '/', search: 'scenario=train-split', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  { name: 'models-train-split', path: '/models', search: 'scenario=train-split', viewports: ['desktop', 'mobile'] },
  { name: 'runs-empty', path: '/runs', search: 'scenario=empty', viewports: ['desktop', 'mobile'] },
  // Empty and error states sit in the table frame (header row, the message inside, a few rows tall).
  { name: 'runs-error', path: '/runs', search: 'scenario=error', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'runs-loading', path: '/runs', search: 'scenario=loading', viewports: ['desktop', 'tablet', 'mobile'] },
  { name: 'datasets-loading', path: '/datasets', search: 'scenario=loading', viewports: ['desktop', 'tablet', 'mobile'] },
  { name: 'datasets-error', path: '/datasets', search: 'scenario=error', viewports: ['desktop', 'laptop', 'mobile'] },
  {
    // Offline: the create action is disabled and says why on hover (and to assistive tech).
    name: 'datasets-error-why',
    path: '/datasets',
    search: 'scenario=error',
    fullPage: false,
    viewports: ['desktop'],
    setup: async (page) => {
      await until(page, 'header .gate');
      const box = await page.locator('main header .gate').first().boundingBox();
      if (!box) throw new Error('datasets-error-why: no Add dataset');
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.clock.runFor(600);
    },
  },
  {
    // Row checkboxes appear on hover (the pointer is on the second row) and are hidden elsewhere.
    name: 'runs-hover',
    path: '/runs',
    fullPage: false,
    viewports: ['desktop'],
    setup: async (page) => {
      await until(page, 'tr[data-row]');
      await page.locator('tr[data-row]').nth(1).hover();
    },
  },
  {
    // Once one run is selected every row shows its checkbox, so picking the rest is one click each.
    name: 'runs-selected',
    path: '/runs',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet'],
    setup: async (page) => {
      await until(page, 'tr[data-row]');
      await page.locator('tr[data-row] .check label').first().click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // Invalid fields: the danger edge, the danger focus ring on the focused one, and the message with its icon.
    name: 'datasets-add-invalid',
    path: '/datasets',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await page.getByRole('button', { name: 'Add dataset' }).first().click();
      await page.locator('dialog[open]').waitFor();
      // Register stays disabled until there is a folder path; a name the server would refuse is checked on submit,
      // and Register is then gated on it ("Fix the name first") until the name is fixed.
      await page.locator('dialog[open] input[placeholder="e.g. D:/data/my-dataset"]').fill('D:/data/my dataset');
      await page.locator('dialog[open] input[id$="-name"]').fill('my dataset!');
      await page.locator('dialog[open] button[type="submit"]').click();
      await page.locator('dialog[open] input[aria-invalid="true"]').first().focus();
    },
  },
  {
    name: 'datasets-nomatch',
    path: '/datasets',
    themes: ['light'],
    viewports: ['desktop'],
    setup: async (page) => {
      await until(page, 'input[aria-label="Filter datasets"]');
      await page.fill('input[aria-label="Filter datasets"]', 'zebra');
    },
  },
  {
    // Keyboard rows: J moves a roving focus over the rows, X selects the focused run for Compare.
    name: 'runs-keyboard',
    path: '/runs',
    fullPage: false,
    viewports: ['desktop'],
    setup: async (page) => {
      await until(page, 'tr[data-row]');
      await page.keyboard.press('j');
      await page.keyboard.press('j');
      await page.keyboard.press('x');
      await page.keyboard.press('j');
      await page.keyboard.press('x');
    },
  },
  {
    // The engine benchmarks, opened from their one-line summary under the inventory.
    name: 'models-bench',
    path: '/models',
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      // The outer Engine benchmarks fold (the panel inside has a "How this was measured" fold of its own).
      await until(page, 'details.bench > summary');
      await page.locator('details.bench > summary').first().click();
      await page.mouse.move(0, 0);
    },
  },
  { name: 'run-loading', path: '/runs/run_7d1e', search: 'scenario=loading', themes: ['dark'], viewports: ['desktop', 'laptop'] },
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
    // Every group, General to Datasets: two columns where wide, one with an edge shadow on phones.
    // Touch-only devices (phones) get no shortcuts reference (DESIGN_DIRECTION §9): a laptop's narrow tier instead.
    name: 'shortcuts-open',
    fullPage: false,
    path: '/runs',
    viewports: ['desktop', 'laptop'],
    setup: async (page) => {
      await page.keyboard.press('Shift+?');
    },
  },
  {
    // Stop's confirm; on phones Stop is the icon button beside ⋯ (named "Stop <run>") and the confirm a bottom sheet.
    name: 'run-stop-dialog',
    fullPage: false,
    path: '/runs/run_7d1e',
    themes: ['light'],
    viewports: ['desktop', 'mobile'],
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
      const svg = page.locator('#run-chart svg[aria-roledescription="chart"]').first();
      await svg.scrollIntoViewIfNeeded();
      const box = await svg.boundingBox();
      if (!box) throw new Error('chart-hover: no chart');
      await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.45);
      // The legend is the readout: it switches to the hovered epoch.
      await page.locator('#run-chart .legend .note.at').waitFor({ state: 'visible', timeout: 3000 });
    },
  },
  { name: 'not-found', path: '/nope', viewports: ['desktop', 'mobile'] },
  {
    // Thumbnails still loading: a raised fill with faint viewfinder corners, not a flat dark photo.
    name: 'dataset-images-loading',
    path: '/datasets/ds_fasteners',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      // Hold every image at "loading" (a src that never arrives), then ask the grid for a fresh set of thumbnails.
      await page.locator('.cell img.loaded').first().waitFor();
      await page.evaluate(() => {
        Object.defineProperty(HTMLImageElement.prototype, 'src', { set() {}, get: () => '', configurable: true });
        const set = Element.prototype.setAttribute;
        Element.prototype.setAttribute = function (name, value) {
          if (!(this instanceof HTMLImageElement && name === 'src')) set.call(this, name, value);
        };
      });
      await page.goto(page.url().replace('ds_fasteners', 'ds_shapes_hard'));
      await until(page, '.dsimg[aria-busy="true"] .cover');
    },
  },
];
