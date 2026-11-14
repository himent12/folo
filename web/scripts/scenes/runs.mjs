// Runs: the list (queue menu, hover, selection, keyboard rows) and a run's page in every state (live, failed, completed,
// queued, stopping, on a cloud instance), its charts, Config and logs.

import { until } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** @type {Scene[]} */
export const runScenes = [
  { name: 'runs', path: '/runs', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'run-live', path: '/runs/run_7d1e', advance: 4000, viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  {
    // The loss chart beside Validation mAP: total in ink, box / cls / dfl in their design/README.md hues (amber, violet
    // dashed, teal dotted), the last raw step and the smoothed total in its header.
    name: 'run-loss',
    path: '/runs/run_c19a',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      // The QA run's clock is paused, so data lands later: wait for the legend.
      await page.locator('#chart-loss .legend').waitFor();
      await page.locator('#chart-loss').scrollIntoViewIfNeeded();
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
      await page.locator('#chart-loss .legend').waitFor();
      await page.locator('#chart-loss').scrollIntoViewIfNeeded();
      await page.clock.runFor(500);
      await page.getByRole('button', { name: 'What the loss terms mean' }).click();
      await page.mouse.move(0, 0);
      await page.clock.runFor(300);
    },
  },
  {
    // Phones, one section at a time (here: per-class results), under the KPI tiles, Validation mAP and Training loss.
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
      // No completed banner: the page is in once its KPI tiles have the run's numbers.
      await until(page, '.kpis .tile .value');
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
    // Phones: with Logs picked, the section picker and the log under the tiles and the two charts.
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
    // Phones: Precision / Recall and Learning rate are sections of the picker (design/UNIFIED.md §4 Run, smaller tiers).
    name: 'run-phone-pr',
    path: '/runs/run_c19a',
    fullPage: false,
    viewports: ['mobile'],
    setup: async (page) => {
      await page.getByRole('tab', { name: 'P / R', exact: true }).click();
      await page.locator('#chart-pr .legend').waitFor();
      await page.locator('.phone-pick').scrollIntoViewIfNeeded();
      await page.mouse.move(0, 0);
    },
  },
  {
    // Keyboard focus on a chart (Validation mAP): the readout row says how to read it.
    // From the tablet tier up: phones and touch-only devices get no key hint.
    name: 'chart-key-hint',
    path: '/runs/run_c19a',
    fullPage: false,
    viewports: ['desktop', 'tablet'],
    setup: async (page) => {
      const svg = page.locator('#chart-map svg[aria-roledescription="chart"]').first();
      await svg.scrollIntoViewIfNeeded();
      await page.keyboard.press('Tab');
      await svg.focus();
      await page.locator('#chart-map .legend .note.hint').waitFor({ state: 'visible', timeout: 3000 });
    },
  },
  {
    // The loss chart's one control open: smoothing and y scale behind a single button.
    name: 'run-loss-options',
    path: '/runs/run_c19a',
    viewports: ['desktop', 'mobile'],
    fullPage: false,
    setup: async (page) => {
      // The QA run's clock is paused, so data lands later: wait for the legend.
      await page.locator('#chart-loss .legend').waitFor();
      await page.locator('#chart-loss').scrollIntoViewIfNeeded();
      await page.locator('#chart-loss button.trigger').click();
      await page.clock.runFor(500);
    },
  },
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
    // The one cursor of the run's four charts: hovering Validation mAP reads the same epoch in every chart's legend.
    name: 'chart-hover',
    path: '/runs/run_c19a',
    themes: ['dark'],
    viewports: ['desktop'],
    fullPage: false,
    setup: async (page) => {
      // Scroll the chart into view first: the pointer cannot hover outside the viewport. Then require the readout.
      const svg = page.locator('#chart-map svg[aria-roledescription="chart"]').first();
      await svg.scrollIntoViewIfNeeded();
      const box = await svg.boundingBox();
      if (!box) throw new Error('chart-hover: no chart');
      await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.45);
      // The legend is the readout: it switches to the hovered epoch.
      await page.locator('#chart-map .legend .note.at').waitFor({ state: 'visible', timeout: 3000 });
      await page.locator('#chart-lr .legend .note.at').waitFor({ state: 'visible', timeout: 3000 });
    },
  },
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
  {
    // A cloud run uploading its dataset: the waiting state is the phase ("Uploading data · 25%"), no queue controls,
    // and the Cloud panel's phases with the transfer.
    name: 'run-cloud-upload',
    path: '/runs/run_c9e4',
    search: 'scenario=cloud-upload',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile', 'small'],
    setup: async (page) => {
      await until(page, '#cloud .xfer:visible');
      await page.mouse.move(0, 0);
    },
  },
  {
    // The Runs list with that run: its status is the phase ("Uploading data", the waiting run's neutral mark), never
    // "Queued", and it has no place in the queue.
    name: 'runs-cloud-upload',
    path: '/runs',
    search: 'scenario=cloud-upload',
    fullPage: false,
    viewports: ['desktop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, 'a[href$="/runs/run_c9e4"]:visible');
      await page.mouse.move(0, 0);
    },
  },
  {
    // A cloud run training: where it runs in the header ("RunPod · RTX 4090") and the Cloud panel in the rail (phones:
    // its section).
    name: 'run-cloud-training',
    path: '/runs/run_c7b1',
    search: 'scenario=cloud-training',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page, { vp }) => {
      if (vp === 'desktop' || vp === 'laptop') await until(page, '#cloud dd');
      else {
        await until(page, '[role="tab"][data-value="cloud"], [role="tab"]:has-text("Cloud")');
        await page.getByRole('tab', { name: 'Cloud', exact: true }).click();
      }
      await page.mouse.move(0, 0);
    },
  },
  {
    // A finished cloud run: released, its final cost.
    name: 'run-cloud-done',
    path: '/runs/run_c3d5',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile', 'small'],
    setup: async (page, { vp }) => {
      if (vp === 'desktop' || vp === 'laptop') await until(page, '#cloud dd');
      else {
        await until(page, '[role="tab"]:has-text("Cloud")');
        await page.getByRole('tab', { name: 'Cloud', exact: true }).click();
      }
      await page.mouse.move(0, 0);
    },
  },
];
