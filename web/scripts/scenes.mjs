// The screens and states `npm run shoot` captures. Each scene is shot in every theme × viewport listed
// (default: dark + light, desktop + mobile). `setup(page)` drives the UI into the state before the shot.
// `search` adds query params to the page URL (e.g. `scenario=empty` for the mock), next to `mock=1`.

/** @typedef {{ name: string, path: string, themes?: ('dark' | 'light')[], viewports?: ('desktop' | 'laptop' | 'tablet' | 'mobile')[], advance?: number, search?: string,
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

/** Starts New run on a GPU the mock then reports gone (404 device_not_found) and waits for the Device error. */
async function unplugGpu(/** @type {import('playwright-core').Page} */ page) {
  // An <option> is never "visible": wait for it to exist, running the mock's clock meanwhile.
  for (let t = 0; t < 8000 && !(await page.locator('select#device option[value="gpu:1"]').count()); t += 250) await page.clock.runFor(250);
  await page.selectOption('select#device', 'gpu:1');
  await page.locator('form.layout').evaluate((f) => /** @type {HTMLFormElement} */ (f).requestSubmit());
  await until(page, '#device-error');
  await page.clock.runFor(1000);
}

/** @type {Scene[]} */
export const scenes = [
  { name: 'overview', path: '/', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  { name: 'datasets', path: '/datasets', viewports: ['desktop', 'laptop', 'mobile'] },
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
    // Phones, one section at a time (here: per-class results): the chart folds to its one-line readout above it.
    name: 'run-live-phone-classes',
    path: '/runs/run_7d1e',
    advance: 4000,
    viewports: ['mobile'],
    fullPage: true,
    setup: async (page) => {
      await page.getByRole('radio', { name: 'Classes' }).click();
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
    },
  },
  { name: 'run-completed', path: '/runs/run_c19a', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'run-new', path: '/runs/new', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  {
    // "Retry with batch 16" on the out-of-memory run: the note says what failed and what changed, the field is marked.
    name: 'run-new-oom-retry',
    path: '/runs/new?from=run_a2f0&batch=16',
    viewports: ['desktop', 'laptop', 'mobile'],
    setup: (page) => until(page, '.oom-note'),
  },
  {
    // Start on a GPU that went away (404 device_not_found): the cause and the adapters at the Device field.
    name: 'run-new-device-not-found',
    path: '/runs/new',
    search: 'scenario=gpu-unplugged',
    // The whole form: the harness shoots from the top, and the message sits at the Device field further down.
    fullPage: true,
    viewports: ['desktop', 'laptop', 'mobile'],
    setup: unplugGpu,
  },
  {
    // The server's own words fold under "Technical details" at the Device field; here opened.
    name: 'run-new-device-details',
    path: '/runs/new',
    search: 'scenario=gpu-unplugged',
    fullPage: false,
    viewports: ['laptop'],
    setup: async (page) => {
      await unplugGpu(page);
      await page.locator('.dev-detail summary').click();
      await page.locator('#device').scrollIntoViewIfNeeded();
    },
  },
  {
    // The short-run scenario's run (run_s5q1) set up again: shapes, folo-n from scratch, batch 32, 5 epochs. 25 batches an epoch
    // make 13 optimizer steps, 65 in all: the Summary warns in a one-line callout above Start, the reasons and the
    // trainer's fix (16 epochs, as the run page says) behind Why? (opened on desktop, closed on laptop); below 960 px
    // the dock says so too.
    name: 'run-new-short',
    path: '/runs/new?dataset=ds_shapes',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page, { vp }) => {
      await until(page, 'input#epochs');
      if (vp === 'mobile') await page.locator('#arch-pick [role="radio"]', { hasText: 'N' }).click();
      else await page.locator('label.arch', { hasText: 'folo-n' }).click();
      await page.locator('#batch [role="radio"]', { hasText: '32' }).click();
      await page.locator('input#epochs').fill('5');
      await page.mouse.move(0, 0);
      if (vp === 'mobile') {
        await page.locator('.dock-est').click();
        await page.locator('dialog[open]').waitFor();
      } else if (vp === 'laptop') {
        await until(page, '.summary .short');
        await page.locator('.summary .short').scrollIntoViewIfNeeded();
      } else if (vp === 'desktop') {
        await until(page, '.summary .short');
        await page.locator('.summary .short button.more').click();
      } else await until(page, '.dock-warn');
    },
  },
  {
    // Mixed precision switched on: the Training section's switch, the Summary's row, a faster estimate.
    name: 'run-new-amp',
    path: '/runs/new',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, 'input#amp');
      await page.locator('.amp .switch').click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // A server whose GPU has no tensor cores through DX12: the switch is off and disabled (its label a field label), one
    // sentence, "In PowerShell:" and the Vulkan command on one copyable line, the Driver report folded (as it starts).
    name: 'run-new-amp-off',
    path: '/runs/new',
    search: 'scenario=no-tensor-cores',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, '#amp-why');
      await page.mouse.move(0, 0);
    },
  },
  {
    // The end-to-end head (TrainConfig.head "o2o"): Advanced opened and the switch on, with its one line; the Summary's
    // Head row and the CLI command (opened) carry it.
    name: 'run-new-e2e',
    path: '/runs/new',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, '#sec-train details.advanced > summary');
      await page.locator('#sec-train details.advanced > summary').click();
      await page.locator('.switch:has(#head)').click();
      await page.locator('.summary details.cli > summary').click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // The Summary's schedule in plain words ("30 steps / epoch"); the batches behind it in a tooltip, here hovered.
    name: 'run-new-steps',
    path: '/runs/new',
    fullPage: false,
    viewports: ['desktop', 'laptop'],
    setup: async (page) => {
      await until(page, '.summary .steps');
      await page.locator('.summary .steps').hover();
      await page.clock.runFor(600);
    },
  },
  {
    // The same on tablet and phone, in the viewport: the dock's Start says why it is off.
    name: 'run-new-device-dock',
    path: '/runs/new',
    search: 'scenario=gpu-unplugged',
    fullPage: false,
    viewports: ['tablet', 'mobile'],
    setup: async (page) => {
      await unplugGpu(page);
      await page.evaluate(() => window.scrollTo(0, 0));
      await until(page, '#dock-why');
    },
  },
  {
    // A setting isolates itself in the preview on hover or focus: the augmentation section opened, Mosaic focused (focus
    // survives the harness's scroll to the top, a hover would not) and the preview showing mosaic alone. The whole page,
    // since the section sits below the first screen.
    name: 'run-new-focus-mosaic',
    fullPage: true,
    path: '/runs/new',
    themes: ['dark', 'light'],
    viewports: ['desktop', 'laptop'],
    setup: async (page) => {
      await until(page, '.aug-toggle');
      await page.locator('.aug-toggle').click();
      await page.locator('#sec-aug input[type="range"]').first().focus();
      await until(page, '.aug-row.focused');
      await page.clock.runFor(500);
    },
  },
  {
    // Phones: the size is a segmented pick with the chosen model in words; the five-row comparison opens on demand.
    name: 'run-new-compare-sizes',
    path: '/runs/new',
    fullPage: true,
    viewports: ['mobile'],
    setup: async (page) => {
      await until(page, 'details.a-compare > summary');
      await page.locator('details.a-compare > summary').click();
    },
  },
  {
    // Below 960 px the dock's estimate opens the run summary as a dialog (a bottom sheet on phones).
    name: 'run-new-summary',
    fullPage: false,
    path: '/runs/new',
    viewports: ['tablet', 'mobile'],
    setup: async (page) => {
      await until(page, '.dock-est');
      await page.locator('.dock-est').click();
      await page.locator('dialog[open]').waitFor();
    },
  },
  {
    // The viewport, not the whole page (a full-page shot grows the viewport, so the dock steps aside for the summary's
    // own button): below 960 px Start training stays in the bottom dock while the form scrolls.
    name: 'run-new-dock',
    path: '/runs/new',
    fullPage: false,
    viewports: ['tablet'],
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
  // Plans of 100 and 150 epochs: the x axis stays "By epoch"; each line ends in its run's status mark.
  { name: 'run-compare', path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  {
    // "By progress": every run over its planned epochs, so the failed and stopped runs end where they ended.
    name: 'run-compare-progress',
    path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, '.chip .key');
      await page.getByRole('radio', { name: 'By progress' }).click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // A run kept in front from the legend: its card selected, its line (here the dotted fourth run) painted over the rest.
    name: 'run-compare-pinned',
    path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, '.chip .key');
      await page.locator('.chip .key').nth(3).click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // Phones: Edit turns the compact two-column legend into one run per row with open and remove.
    name: 'run-compare-edit',
    path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0',
    fullPage: false,
    viewports: ['mobile'],
    setup: async (page) => {
      await until(page, 'button.manage');
      await page.locator('button.manage').click();
    },
  },
  // Nothing picked: the inline empty state with the standard secondary "Add run…" button.
  {
    // Phones: the side-by-side table two runs at a time, the baseline and the next run (paged once), differing values
    // marked.
    name: 'run-compare-pair-phone',
    path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0',
    viewports: ['mobile'],
    fullPage: true,
    setup: async (page) => {
      await until(page, '#cmp-table .pair');
      await page.getByRole('button', { name: 'Next run' }).click();
      await page.mouse.move(0, 0);
    },
  },
  { name: 'run-compare-empty', path: '/runs/compare', viewports: ['desktop', 'mobile'] },
  // A queued run in the loaded frame: KPIs at "—", the chart's axes with what it waits for, the config in its rail.
  // A queued run: the dependency and start estimate lead, the empty frames stay compact until there is data.
  { name: 'run-queued', path: '/runs/run_q4n2', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
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
      await page.getByRole('radio', { name: 'Logs' }).click();
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
  { name: 'dataset-stats', path: '/datasets/ds_shapes_hard?tab=stats', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  {
    // The health line opened: every finding, its evidence and next step.
    name: 'dataset-health-open',
    path: '/datasets/ds_shapes_hard?tab=stats',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, '#ds-health summary');
      for (const s of await page.locator('#ds-health summary').all()) await s.click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // The warning finding's "Show the warnings": the check's warnings open under that finding (the one issue list,
    // filtered to its severity), so the caption and rows are the warnings.
    name: 'dataset-issues',
    path: '/datasets/ds_shapes_hard?tab=stats',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, '#ds-health summary');
      await page.locator('#ds-health summary').click();
      await page.getByRole('button', { name: 'Show the warnings' }).click();
      await until(page, '#ds-health .evidence .fix');
      await page.clock.runFor(500);
    },
  },
  {
    // The box-size histogram with a bin pinned by a tap (phones) or under the mouse: the readout row states its count.
    // Both are sent as pointer events on the plot: the shot scrolls to the top first, which would take a real mouse off
    // the chart and end its hover.
    name: 'dataset-histogram',
    path: '/datasets/ds_shapes_hard?tab=stats',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'mobile'],
    setup: async (page, { vp }) => {
      await until(page, '#ds-size svg');
      const svg = page.locator('#ds-size svg[aria-roledescription=chart]');
      await svg.evaluate((el, touch) => {
        const r = el.getBoundingClientRect();
        const p = { clientX: r.left + r.width * 0.4, clientY: r.top + r.height * 0.6, bubbles: true };
        const types = touch ? ['pointerdown', 'pointerup'] : ['pointermove'];
        for (const type of types) el.dispatchEvent(new PointerEvent(type, { ...p, pointerType: touch ? 'touch' : 'mouse' }));
      }, vp === 'mobile');
      // The readout row switches from the total to the bin.
      await page.locator('#ds-size .readout.at').waitFor({ state: 'visible', timeout: 3000 });
      await page.clock.runFor(300);
    },
  },
  {
    // The watched run finished while another page was open: a toast says so, with "Open run" (no timer: it has an
    // action). Mock scenario `finishing`: the live run's last epoch lands 3 s after load.
    name: 'toast-run-finished',
    path: '/models',
    search: 'scenario=finishing',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, 'text=finished training', 12000);
      await page.mouse.move(0, 0);
    },
  },
  {
    // Registering a folder that is already in the list opens it and says so.
    name: 'datasets-add-existing',
    path: '/datasets',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await page.getByRole('button', { name: 'Add dataset' }).first().click();
      await page.locator('dialog[open]').waitFor();
      await page.locator('dialog[open] input[placeholder="e.g. D:/data/my-dataset"]').fill('D:/data/shapes');
      await page.locator('dialog[open] form').evaluate((f) => /** @type {HTMLFormElement} */ (f).requestSubmit());
      await until(page, 'text=Already registered as shapes');
      // Hold the page clock (the shoot's runs in real time) so the toast is still up when the thumbnails have loaded:
      // they decode in real time but wait on the mock's latency, so the clock steps 250 ms at a time, at most 2.5 s of
      // the toast's 4.
      await page.clock.pauseAt((await page.evaluate(() => Date.now())) + 100);
      for (let t = 0; t < 2500; t += 250) {
        if (await page.locator('.cell img.loaded').nth(17).isVisible().catch(() => false)) break;
        await page.clock.runFor(250);
        await page.waitForTimeout(600);
      }
    },
  },
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
    // Thumbnails mid-load and failed: a skeleton holds each tile until its image is ready (no boxes on an empty frame),
    // and an image that could not be read says so, with Retry. Two images are made to fail after the first load.
    name: 'dataset-images-unavailable',
    path: '/datasets/ds_fasteners',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await page.locator('.cell img.loaded').nth(6).waitFor();
      await page.locator('.cell img').evaluateAll((imgs) => [1, 4].forEach((i) => imgs[i]?.dispatchEvent(new Event('error'))));
      await until(page, '.cell .failed');
      await page.waitForTimeout(200);
    },
  },
  {
    // Small thumbnails of pastel photos: every class stroke (amber on beige, sky on blue-grey) holds in its 2 px halo.
    name: 'dataset-images-pastel',
    path: '/datasets/ds_shapes_hard',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await page.evaluate(() => localStorage.setItem('folo.datasets.density', 's'));
      await page.reload({ waitUntil: 'networkidle' });
      await page.clock.runFor(2000);
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
  {
    // A label clicked (selected: bg.active + accent bar, aria-current, the image zoomed to its box: the box takes the
    // 3 px accent stroke and the others recede) while another is hovered (a lighter neutral step; its box takes its
    // class fill and chip), so the two states show side by side.
    name: 'dataset-inspector-selected',
    path: '/datasets/ds_fasteners',
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    fullPage: false,
    setup: async (page) => {
      await page.locator('.cell img.loaded').first().waitFor();
      await page.locator('.cell').nth(1).click();
      await page.locator('dialog[open] .stage img.loaded').waitFor();
      await page.locator('dialog[open] .labels button').nth(0).click();
      await page.locator('dialog[open] .labels button').nth(1).hover();
      await page.waitForTimeout(300);
    },
  },
  { name: 'runs', path: '/runs' },
  { name: 'models', path: '/models', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  { name: 'models-empty', path: '/models', search: 'scenario=empty' },
  { name: 'models-error', path: '/models', search: 'scenario=error', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'models-loading', path: '/models', search: 'scenario=loading', viewports: ['desktop', 'laptop', 'mobile'] },
  {
    // Phones open on the Curves section (the matrix is behind the section picker).
    name: 'model-detail',
    path: '/models/m_fast_n',
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: (page, { vp }) => until(page, vp === 'mobile' ? '#pr svg' : '#cm .cell'),
  },
  {
    // Laptop: the details fold into one line above the matrix; opened here, they list in two columns. Phones: the
    // Details section.
    name: 'model-detail-details',
    path: '/models/m_fast_n',
    viewports: ['tablet', 'mobile'],
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
    // An o2o model (shapes-s-best, as the server mock's m_31ab): Details says Head "End-to-end (no NMS)" and how it was
    // trained. Tablets fold the details (laptops and desktops show the rail): opened. Phones: the Details section.
    name: 'model-o2o',
    path: '/models/m_shapes_s',
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    fullPage: true,
    setup: async (page, { vp }) => {
      await until(page, '#pr svg');
      if (vp === 'mobile') await page.getByRole('radio', { name: 'Details' }).click();
      else if (vp === 'tablet') await page.locator('.d-line').click();
    },
  },
  {
    // fasteners-n's f16 export (as the server mock's m_53d2): Details says "Stored as f16 · fused, inference only" and,
    // with no training record, nothing about training. Phones: the Details section.
    name: 'model-f16',
    path: '/models/m_fast_n_f16',
    viewports: ['desktop', 'mobile'],
    fullPage: true,
    setup: async (page, { vp }) => {
      await until(page, '#pr svg');
      if (vp === 'mobile') await page.getByRole('radio', { name: 'Details' }).click();
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
    // Phones: the Classes section (the metrics folded to one line) with one class traced and its details row opened:
    // two controls per row, the name traces, the chevron opens.
    name: 'model-phone-classes',
    path: '/models/m_fast_n',
    viewports: ['mobile'],
    fullPage: true,
    setup: async (page) => {
      await until(page, '#pr svg');
      await page.getByRole('radio', { name: 'Classes' }).click();
      await page.locator('#per-class button.cls').first().click();
      await page.locator('#per-class button.disc').first().click();
    },
  },
  {
    // Phones: a class's "View curve" switched to Curves with the class traced, and the way back to Classes.
    name: 'model-phone-curve',
    path: '/models/m_fast_n',
    viewports: ['mobile'],
    fullPage: false,
    setup: async (page) => {
      await until(page, '#pr svg');
      await page.getByRole('radio', { name: 'Classes' }).click();
      await page.locator('#per-class button.disc').nth(2).click();
      await page.locator('#per-class tr.detail button', { hasText: 'View curve' }).click();
      await until(page, '#pr .legend .back');
      await page.clock.runFor(500);
    },
  },
  {
    // Phones: the Confusion section opens on the ranked confusions and misses, the matrix one step away.
    name: 'model-phone-confusion',
    path: '/models/m_fast_n',
    viewports: ['mobile'],
    fullPage: true,
    setup: async (page) => {
      await until(page, '#pr svg');
      await page.getByRole('radio', { name: 'Confusion' }).click();
      await until(page, '#cm .rk');
    },
  },
  {
    // Phones: "Explore the matrix": the full matrix with its column range, the overview map and the miss summary.
    name: 'model-phone-matrix',
    path: '/models/m_fast_n',
    viewports: ['mobile'],
    fullPage: true,
    setup: async (page) => {
      await until(page, '#pr svg');
      await page.getByRole('radio', { name: 'Confusion' }).click();
      await page.getByRole('button', { name: 'Explore the matrix' }).click();
      await until(page, '#cm .cell');
    },
  },
  {
    // Phones: a reduced 13-column matrix (top 12 classes + Other) that scrolls sideways, behind "Explore the matrix".
    name: 'model-coco-phone',
    path: '/models/m_coco_s',
    viewports: ['mobile'],
    fullPage: true,
    setup: async (page) => {
      await until(page, '#pr svg');
      await page.getByRole('radio', { name: 'Confusion' }).click();
      await page.getByRole('button', { name: 'Explore the matrix' }).click();
      await until(page, '#cm .cell');
    },
  },
  {
    name: 'model-export',
    path: '/models/m_shapes_s',
    fullPage: false,
    setup: async (page, { vp }) => {
      // Phones keep Export in the header's ⋯ menu.
      if (vp === 'mobile') {
        await until(page, 'button[aria-label="More model actions"]');
        await page.locator('button[aria-label="More model actions"]').click();
        await page.getByRole('menuitem', { name: 'Export' }).click();
      } else {
        await until(page, 'button:has-text("Export")');
        await page.locator('button:has-text("Export")').first().click();
      }
      await until(page, 'dialog[open] fieldset');
    },
  },
  {
    // The f16 export finished: Download (the primary) for the file, by the name it has on the server.
    name: 'model-export-done',
    path: '/models/m_shapes_s',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page, { vp }) => {
      if (vp === 'mobile') {
        await until(page, 'button[aria-label="More model actions"]');
        await page.locator('button[aria-label="More model actions"]').click();
        await page.getByRole('menuitem', { name: 'Export' }).click();
      } else {
        await until(page, 'button:has-text("Export")');
        await page.locator('button:has-text("Export")').first().click();
      }
      await until(page, 'dialog[open] fieldset');
      await page.locator('dialog[open] footer button:has-text("Export")').click();
      await until(page, 'dialog[open] .where');
    },
  },
  {
    // ONNX picked on a server with the export switches: decode, dynamic and half as checkboxes under it.
    name: 'model-export-onnx-options',
    path: '/models/m_shapes_s',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page, { vp }) => {
      if (vp === 'mobile') {
        await until(page, 'button[aria-label="More model actions"]');
        await page.locator('button[aria-label="More model actions"]').click();
        await page.getByRole('menuitem', { name: 'Export' }).click();
      } else {
        await until(page, 'button:has-text("Export")');
        await page.locator('button:has-text("Export")').first().click();
      }
      await until(page, 'dialog[open] fieldset');
      await page.locator('dialog[open] label:has-text("ONNX")').first().click();
      await until(page, 'dialog[open] .switches');
      await page.locator('dialog[open] .switches label').first().click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // ONNX has no download endpoint: the dialog says so next to the server path.
    name: 'model-export-onnx',
    path: '/models/m_shapes_s',
    fullPage: false,
    viewports: ['desktop'],
    setup: async (page) => {
      await until(page, 'button:has-text("Export")');
      await page.locator('button:has-text("Export")').first().click();
      await until(page, 'dialog[open] fieldset');
      await page.locator('dialog[open] label:has-text("ONNX")').first().click();
      await page.locator('dialog[open] footer button:has-text("Export")').click();
      await until(page, 'dialog[open] .where');
    },
  },
  {
    // The metric help under the KPI strip (AP vs mAP, IoU, the precision / recall operating point), and the header's
    // lineage: fasteners-n was fine-tuned from shapes-n. Phones: the summary line opened too.
    name: 'model-metric-help',
    path: '/models/m_fast_n',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page, { vp }) => {
      await until(page, '#pr svg');
      if (vp === 'mobile') await page.locator('button.meta-line').click();
      await page.getByRole('button', { name: 'What these metrics mean' }).click();
      await page.clock.runFor(1500);
    },
  },
  {
    // Per-class rows trace curves: the hint above the table, two classes selected (bg.active + accent bar) and Clear.
    name: 'model-classes-selected',
    path: '/models/m_fast_n',
    fullPage: true,
    viewports: ['desktop', 'laptop'],
    setup: async (page) => {
      await until(page, '#cm .cell');
      await page.locator('#per-class button.cls').nth(1).click();
      await page.locator('#per-class button.cls').nth(4).click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // The engine benchmark with its "How this was measured" fold open.
    name: 'models-engine-how',
    path: '/models',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      // The panel lives in the collapsed Engine benchmarks fold: open that, then its own "How this was measured".
      await until(page, 'details.bench > summary');
      await page.locator('details.bench > summary').first().click();
      await until(page, '#engine-speed');
      await page.locator('#engine-speed summary', { hasText: 'How this was measured' }).click();
    },
  },
  { name: 'playground-empty', path: '/playground', viewports: ['desktop', 'laptop', 'tablet', 'mobile'], setup: (page) => until(page, 'button.sample') },
  {
    // Server inference ran out of GPU memory (507): the cause, the hint, and the ways out (this browser, a smaller model),
    // in a strip above the image (where the result would be), which stays whole.
    name: 'playground-oom',
    path: '/playground?model=m_shapes_s',
    search: 'scenario=oom',
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, 'button.sample');
      await page.locator('button.sample').first().click();
      await page.mouse.move(0, 0);
      await until(page, '.notice .err');
      // The toast repeats the cause; let it go so the shot shows the state in place.
      await page.clock.runFor(6000);
    },
  },
  // Tablet: the model and Run on in a toolbar row under the title, the rail a two-column block under the image.
  { name: 'playground-result', path: '/playground?model=m_hard_s', viewports: ['desktop', 'laptop', 'tablet', 'mobile'], setup: playgroundResult },
  {
    // A browser without WebGPU: the option is off and the header says why, and what is left.
    name: 'playground-no-webgpu',
    path: '/playground?model=m_hard_s',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await page.addInitScript(() => {
        delete (/** @type {any} */ (Navigator.prototype)).gpu;
      });
      await page.reload({ waitUntil: 'networkidle' });
      await page.clock.runFor(2000);
      await until(page, '.runon-note');
    },
  },
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
    viewports: ['desktop', 'laptop', 'mobile'],
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
  { name: 'settings', path: '/settings', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  { name: 'overview-empty', path: '/', search: 'scenario=empty', viewports: ['desktop', 'mobile'] },
  { name: 'overview-loading', path: '/', search: 'scenario=loading', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'overview-error', path: '/', search: 'scenario=error', viewports: ['desktop', 'mobile'] },
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
      // Register stays disabled until there is a folder path; a name the server would refuse is checked on submit.
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
  {
    // Connection says what it means ("Using demo data", "Leave blank to use this server"); the mechanics are folded.
    name: 'settings-connection-details',
    path: '/settings',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await page.locator('details.tech summary').click();
      await page.mouse.move(0, 0);
    },
  },
  // Settings' System panel (GET /api/system capabilities): tensor cores on (default), available but off, and absent.
  ...[
    { name: 'settings-system', search: '' },
    { name: 'settings-system-tc-off', search: 'scenario=tensor-cores-off' },
    { name: 'settings-system-no-tc', search: 'scenario=no-tensor-cores' },
  ].map(
    ({ name, search }) =>
      /** @type {Scene} */ ({
        name,
        path: '/settings',
        search,
        // The whole page: the harness shoots from the top, and the panel sits near the bottom.
        fullPage: true,
        viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
        // The Driver report stays folded, as it starts in every state; settings-system-report opens it.
        setup: async (page) => {
          await until(page, '#system');
          await page.mouse.move(0, 0);
        },
      }),
  ),
  {
    // The server's own words opened under Driver report: wrapped prose in a well, copyable whole.
    name: 'settings-system-report',
    path: '/settings',
    search: 'scenario=no-tensor-cores',
    fullPage: true,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, '#system');
      await page.locator('#system details.report summary').click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // Single-key shortcuts turned off in Settings; the dialog (from the Settings button) says so.
    name: 'shortcuts-off',
    path: '/settings',
    fullPage: false,
    themes: ['light'],
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await page.getByRole('switch', { name: /Single-key shortcuts/ }).click({ force: true });
      await page.getByRole('button', { name: 'Show shortcuts' }).click();
      await page.locator('dialog[open]').waitFor();
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
