// Runs: the first look at Overview, Datasets and Runs, a run's page in every state (live, failed, completed, queued,
// its charts, Config and logs), and the Runs list's queue menu and run selection.

import { until } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** @type {Scene[]} */
export const runScenes = [
  // The live run at the moment run-live shows it (the same `advance`), so the two screens' epoch and progress agree.
  { name: 'overview', path: '/', advance: 4000, viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  { name: 'datasets', path: '/datasets', viewports: ['desktop', 'laptop', 'mobile'] },
  {
    // A class-balance strip is one disclosure: opened, each class's name, box count and share, largest first.
    name: 'datasets-class-balance',
    path: '/datasets',
    viewports: ['desktop', 'mobile'],
    fullPage: false,
    setup: async (page) => {
      const strip = page.locator('.cs-btn:visible').first();
      await strip.waitFor();
      await strip.click();
      await page.mouse.move(0, 0);
    },
  },
  { name: 'runs', path: '/runs', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'run-live', path: '/runs/run_7d1e', advance: 4000, viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  {
    // The loss chart: total in ink, box / cls / dfl in their design/README.md hues (amber, violet dashed, teal dotted).
    name: 'run-loss',
    path: '/runs/run_c19a',
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      // Tabs from 768 px, a picker on phones; wait for either (the QA run's clock is paused, so data lands later).
      const tab = page.locator('[data-tab="loss"]');
      const pick = page.locator('select[aria-label="Metric to chart"]:visible');
      await page.locator('[data-tab="loss"]:visible, select[aria-label="Metric to chart"]:visible').first().waitFor();
      if (await tab.isVisible()) await tab.click();
      else await pick.selectOption('loss');
      await page.mouse.move(0, 0);
      await page.clock.runFor(500);
    },
  },
  {
    // The loss terms opened by a click (a tap on phones, Enter or Space from the keyboard): ui/InfoButton's popover.
    name: 'run-loss-terms',
    path: '/runs/run_c19a',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await page.locator('[data-tab="loss"]:visible, select[aria-label="Metric to chart"]:visible').first().waitFor();
      const tab = page.locator('[data-tab="loss"]');
      if (await tab.isVisible()) await tab.click();
      else await page.locator('select[aria-label="Metric to chart"]:visible').selectOption('loss');
      await page.clock.runFor(500);
      await page.getByRole('button', { name: 'What the loss terms mean' }).click();
      await page.mouse.move(0, 0);
      await page.clock.runFor(300);
    },
  },
  {
    // Phones, one section at a time (here: per-class results): the chart folds to its one-line readout above it.
    name: 'run-live-phone-classes',
    path: '/runs/run_7d1e',
    advance: 4000,
    viewports: ['mobile'],
    fullPage: true,
    setup: async (page) => {
      await page.getByRole('tab', { name: 'Classes', exact: true }).click();
      await page.mouse.move(0, 0);
    },
  },
  // run_a2f0 ran out of GPU memory: the plain cause, the smaller settings as actions, the driver's report folded.
  { name: 'run-failed', path: '/runs/run_a2f0', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  {
    name: 'run-failed-report',
    path: '/runs/run_a2f0',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'mobile'],
    setup: async (page) => {
      // "More fixes": the other smaller settings and the driver's report.
      await until(page, 'button.c-more');
      await page.locator('button.c-more').first().click();
      // The pointer leaves the label it clicked: the shot shows the opened disclosure, not its hover fill.
      await page.mouse.move(0, 0);
    },
  },
  { name: 'run-completed', path: '/runs/run_c19a', viewports: ['desktop', 'laptop', 'mobile'] },
  // A run on fasteners-mini, which has no validation images: every epoch was scored on the training split, so the page
  // says so once ("Evaluated on the training split") and its titles name that split, never "Validation".
  { name: 'run-train-split', path: '/runs/run_m1n1', search: 'scenario=train-split', viewports: ['desktop', 'laptop', 'mobile'] },
];

/** @type {Scene[]} */
export const runPageScenes = [
  // A queued run, first of two waiting: its place, what it waits for and when it should start lead; from 768 px up the
  // config beside the queue and one quiet list of what the run will have (no empty frames). Phones keep their sections.
  { name: 'run-queued', path: '/runs/run_q4n2', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  // Second in the queue (queue control): "2nd in queue", when it should start ("Starts in about …", each run in front with
  // its own time), and Start next (it never interrupts the live run, so not "now").
  { name: 'run-queued-second', path: '/runs/run_q7c5', viewports: ['desktop', 'laptop', 'tablet', 'mobile'], setup: (page) => until(page, '.callout button') },
  {
    // The queued run's ⋯: the queue's one action set, as on its Runs row: Start next, Move up (it is last, so no Move
    // down), then Remove from queue, last under a hairline.
    name: 'run-queued-menu',
    path: '/runs/run_q7c5',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, 'button[aria-label="More run actions"]:visible');
      await page.locator('button[aria-label="More run actions"]:visible').click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // What the run trains with (Run.resolved): "AdamW (auto)", "0.000833 (requested 0.01)", batch × accumulate, warmup.
    // Phones: the Config section.
    name: 'run-config-resolved',
    path: '/runs/run_c19a',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page, { vp }) => {
      if (vp === 'desktop' || vp === 'laptop') await until(page, '#config dd');
      else await page.getByRole('tab', { name: 'Config' }).click();
      await page.locator('#config').scrollIntoViewIfNeeded();
      await page.mouse.move(0, 0);
    },
  },
  {
    // Server PR #140 (`run_cli`): the run's `folo train` command (GET /api/runs/{id}/cli) closes its Config, under
    // "In PowerShell:" on one copyable line. Phones and tablets: the Config section. The whole page: the harness shoots
    // from the top, and Config sits below the charts.
    name: 'run-cli',
    path: '/runs/run_c19a',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page, { vp }) => {
      if (vp !== 'desktop' && vp !== 'laptop') await page.getByRole('tab', { name: 'Config' }).click();
      await until(page, '#config .cli pre.line');
      await page.mouse.move(0, 0);
    },
  },
  {
    // DOGFOOD-3 S2: a 15-epoch fine-tune asked for close mosaic 10, which the trainer caps at 7: the bar hatches the last
    // 7 epochs, the charts mark epoch 8, and Config says "last 7 epochs (requested 10)". Phones: the Config section.
    name: 'run-short-mosaic',
    path: '/runs/run_f15c',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page, { vp }) => {
      if (vp === 'desktop' || vp === 'laptop') await until(page, '#config dd');
      else await page.getByRole('tab', { name: 'Config' }).click();
      // The rail's Config folds Augmentation, where Close mosaic is.
      const fold = page.locator('#config details > summary');
      if (await fold.isVisible()) await fold.click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // DOGFOOD-3 S4: the header's meta names what differs from the defaults: Mixed precision and the end-to-end head.
    name: 'run-header-amp-head',
    path: '/runs/run_f15c',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, 'text=Completed in');
      await page.mouse.move(0, 0);
    },
  },
  {
    // DOGFOOD-3 S2, S4: Compare's settings: the capped close mosaic with the request, the head that differs.
    name: 'run-compare-short-mosaic',
    path: '/runs/compare?ids=run_f15c,run_c19a',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, 'td.cfg');
      await page.mouse.move(0, 0);
    },
  },
  {
    // A 5-epoch run from scratch: the trainer's warnings that it is too short to learn, as one callout.
    name: 'run-short-warning',
    path: '/runs/run_s5q1',
    search: 'scenario=short-run',
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
  },
  {
    // Phones: with Logs picked, the chart folds to its one-line readout, so the log starts in the first screen.
    name: 'run-phone-logs',
    path: '/runs/run_7d1e',
    advance: 4000,
    fullPage: false,
    viewports: ['mobile'],
    setup: async (page) => {
      await page.getByRole('tab', { name: 'Logs', exact: true }).click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // Keyboard focus on a chart: the readout row says how to read it.
    // From the tablet tier up: phones and touch-only devices get no key hint.
    name: 'chart-key-hint',
    path: '/runs/run_c19a',
    fullPage: false,
    viewports: ['desktop', 'tablet'],
    setup: async (page) => {
      const svg = page.locator('#run-chart svg[aria-roledescription="chart"]').first();
      await svg.scrollIntoViewIfNeeded();
      await page.keyboard.press('Tab');
      await svg.focus();
      await page.locator('#run-chart .legend .note.hint').waitFor({ state: 'visible', timeout: 3000 });
    },
  },
  {
    // A touch screen: the first chart of the session says "Tap to pin a value" (Compare: in its readout row).
    name: 'chart-touch-hint',
    path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0',
    fullPage: false,
    touch: false,
    viewports: ['mobile'],
    setup: async (page) => {
      const cdp = await page.context().newCDPSession(page);
      await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 5 });
      await page.reload({ waitUntil: 'networkidle' });
      await page.clock.runFor(2000);
      await until(page, '.chips-head .at.hint');
    },
  },
  {
    // The loss chart's one control open: smoothing and y scale behind a single button.
    name: 'run-loss-options',
    path: '/runs/run_c19a',
    viewports: ['desktop', 'mobile'],
    fullPage: false,
    setup: async (page) => {
      // Tabs from 768 px, a picker on phones; wait for either (the QA run's clock is paused, so data lands later).
      const tab = page.locator('[data-tab="loss"]');
      const pick = page.locator('select[aria-label="Metric to chart"]:visible');
      await page.locator('[data-tab="loss"]:visible, select[aria-label="Metric to chart"]:visible').first().waitFor();
      if (await tab.isVisible()) await tab.click();
      else await pick.selectOption('loss');
      await page.locator('button.trigger').click();
      await page.clock.runFor(500);
    },
  },
];

/** @type {Scene[]} */
export const runsQueueScenes = [
  { name: 'runs', path: '/runs' },
  {
    // The waiting runs say their place ("1st in queue", "2nd in queue"); the second's ⋯ opened: Start next, Move up,
    // Remove from queue (the queue's one action set).
    name: 'runs-queue-menu',
    path: '/runs',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      const menu = 'button[aria-label="Queue actions for fasteners-m"]:visible';
      await until(page, menu);
      await page.locator(menu).click();
      await page.mouse.move(0, 0);
    },
  },
];

/** @type {Scene[]} */
export const runsSelectionScenes = [
  {
    // Phones: one run picked. Compare (disabled, "Select 2–4 runs") stays secondary and New run keeps the one primary.
    name: 'runs-selected-phone',
    path: '/runs',
    fullPage: false,
    viewports: ['mobile'],
    setup: async (page) => {
      await until(page, '.rows li');
      await page.getByRole('button', { name: 'Select to compare' }).click();
      await page.locator('.rows li .r-check').nth(1).click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // Four runs picked, the most Compare takes: the line under the selection says so, and the other boxes are off.
    name: 'runs-selected-four',
    path: '/runs',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page, { vp }) => {
      if (vp === 'mobile') {
        await until(page, '.rows li');
        await page.getByRole('button', { name: 'Select to compare' }).click();
        for (let i = 0; i < 4; i++) await page.locator('.rows li .r-check').nth(i).click();
      } else {
        await until(page, 'tr[data-row]');
        for (let i = 0; i < 4; i++) await page.locator('tr[data-row] .check label').nth(i).click();
      }
      await page.mouse.move(0, 0);
    },
  },
  {
    // Phones: two runs picked; Compare sits in the list's own header row, by the checkboxes.
    name: 'runs-compare-phone',
    path: '/runs',
    fullPage: false,
    viewports: ['mobile'],
    setup: async (page) => {
      await until(page, '.rows li');
      await page.getByRole('button', { name: 'Select to compare' }).click();
      await page.locator('.rows li .r-check').nth(1).click();
      await page.locator('.rows li .r-check').nth(2).click();
    },
  },
];
