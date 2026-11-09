// The screens and states `npm run shoot` captures. Each scene is shot in every theme × viewport listed
// (default: dark + light, desktop + mobile). `setup(page)` drives the UI into the state before the shot.
// `search` adds query params to the page URL (e.g. `scenario=empty` for the mock), next to `mock=1`.

/** @typedef {{ name: string, path: string, themes?: ('dark' | 'light')[], viewports?: ('desktop' | 'laptop' | 'tablet' | 'mobile' | 'small')[], advance?: number, search?: string,
 *   fullPage?: boolean, keepScroll?: boolean, touch?: false, setup?: (page: import('playwright-core').Page, ctx: { vp: string, theme: string }) => Promise<void> }} Scene */
// `touch: false` keeps a scene out of --qa's coarse-pointer pass over phone states: a scene about a keyboard feature
// (shortcuts) or one that sets up touch itself (the touch hint).

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

/**
 * New run's estimate with what it stands on opened: in the Summary rail on wide screens, in the summary sheet (from the
 * dock) below 960 px. `served`: wait for the server's estimate to replace the reference.
 * @param {import('playwright-core').Page} page
 * @param {string} vp
 * @param {boolean} served
 */
async function openEstimate(page, vp, served) {
  await until(page, '.summary .est-how > summary');
  if (served) await until(page, '.summary .est-how > summary:has-text("Measured")');
  if (vp === 'mobile' || vp === 'tablet') {
    await page.locator('.dock-est').click();
    await page.locator('dialog[open]').waitFor();
    await page.locator('dialog[open] details.est-how > summary').click();
  } else {
    await page.locator('.summary details.est-how > summary').click();
  }
  await page.mouse.move(0, 0);
}

/**
 * Presses Tab until `selector` holds focus, as a keyboard user reaches it (so `:focus-visible` is the keyboard's).
 * @param {import('playwright-core').Page} page
 * @param {string} selector
 */
async function tabTo(page, selector, max = 60) {
  await until(page, selector);
  for (let i = 0; i < max; i++) {
    if (await page.evaluate((sel) => !!document.activeElement?.matches(sel), selector)) return;
    await page.keyboard.press('Tab');
  }
  throw new Error(`Tab never reached ${selector}`);
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
  { name: 'run-new', path: '/runs/new', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  {
    // The run page's retry, followed as a person would: shapes-hard-m ran out of GPU memory, its callout offers the
    // largest batch the forecast fits ("Retry with batch 8 · ~13.6 GiB"), and that link opens New run with it. The
    // note says what failed and what changed, with the figure ("Now batch 8 · ~13.6 GiB"), and Start is the primary.
    name: 'run-new-oom-retry',
    path: '/runs/run_a2f0',
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      const retry = page.locator('a.btn.primary', { hasText: 'Retry with' }).first();
      await until(page, 'a.btn.primary:has-text("Retry with")');
      await retry.click();
      await until(page, '.oom-note');
      await until(page, '.summary .est-how > summary');
      await page.clock.runFor(500);
      await page.mouse.move(0, 0);
    },
  },
  {
    // A preset that is still too big (batch 16, ~26.9 / 16 GiB): the note says the change is not enough rather than
    // "Now batch 16", the figure, meter and line are danger, "Use batch 8" is the one primary and Start a secondary;
    // the phone dock says "~26.9 GiB / 16.0 GiB · Likely won't fit" beside "Use batch 8".
    name: 'run-new-oom-still-over',
    path: '/runs/new?from=run_a2f0&batch=16',
    viewports: ['desktop', 'mobile'],
    setup: (page) => until(page, '.start .btn.primary:has-text("Use batch")'),
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
    // DOGFOOD-3 S1: a fine-tune's CLI command: no --model (it conflicts with --weights) and the weights file by path;
    // this server does not report it, so a placeholder and one line under the command say what to put there.
    name: 'run-new-cli-finetune',
    path: '/runs/new',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, 'input#name');
      // Phones pick the size from a segmented control; wider screens from the table's radios.
      if (await page.locator('#arch-pick').isVisible()) await page.locator('#arch-pick button', { hasText: 'N' }).click();
      else await page.locator('input[name="arch"][value="folo-n"]').check({ force: true });
      for (let t = 0; t < 8000 && !(await page.locator('select#weights option[value="m_fast_n"]').count()); t += 250) await page.clock.runFor(250);
      await page.selectOption('select#weights', 'm_fast_n');
      await page.locator('.amp .switch').click();
      await page.locator('.summary details.cli > summary').click();
      await page.locator('.summary details.cli').scrollIntoViewIfNeeded();
      // The estimate for these settings lands (300 ms after the last change, then the server's answer), so the Summary
      // is current; run-new-estimate-updating shows the moment before.
      await page.clock.runFor(1000);
      await page.mouse.move(0, 0);
    },
  },
  {
    // The settings just changed and their estimate is still out: the last answer stays, dimmed to tertiary in the
    // Summary and the dock, and the basis line reads "Updating…" with a spinner (a figure is current or marked as not).
    name: 'run-new-estimate-updating',
    path: '/runs/new',
    viewports: ['desktop', 'laptop', 'mobile'],
    setup: async (page) => {
      await until(page, 'input#name');
      await page.clock.runFor(1000);
      await until(page, '.summary .est-how summary');
      if (await page.locator('#arch-pick').isVisible()) await page.locator('#arch-pick button', { hasText: 'N' }).click();
      else await page.locator('input[name="arch"][value="folo-n"]').check({ force: true });
      await until(page, '.est-how.stale');
      await page.mouse.move(0, 0);
    },
  },
  {
    // Server PR #140: published weights (folo-n-coco, `folo weights pull`) in Start from, a group of their own. Picked,
    // they set the size ("folo-n, from folo-n-coco" above the picker), their `about` is the field's hint, and the
    // command names them by their published name.
    name: 'run-new-published',
    path: '/runs/new',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, 'input#name');
      for (let t = 0; t < 8000 && !(await page.locator('select#weights option[value="folo-n-coco"]').count()); t += 250) await page.clock.runFor(250);
      await page.selectOption('select#weights', 'folo-n-coco');
      await until(page, '.a-from');
      if (await page.locator('.summary details.cli > summary').isVisible()) await page.locator('.summary details.cli > summary').click();
      await page.clock.runFor(1000);
      await page.mouse.move(0, 0);
    },
  },
  {
    // Server PR #140: a POST refused with `501 not_supported` (the server restarted without tensor cores after the page
    // loaded): never a toast; from the capabilities read again, the switch off and held off, so the refusal is answered
    // and the field says quietly why and how to turn tensor cores back on (no danger line, no button repeating Start).
    name: 'run-new-amp-refused',
    path: '/runs/new',
    search: 'scenario=amp-refused',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, 'input#amp');
      await page.locator('.amp .switch').click();
      await page.locator('form.layout').evaluate((f) => /** @type {HTMLFormElement} */ (f).requestSubmit());
      await until(page, '#amp-why');
      await page.clock.runFor(1000);
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
    // The Summary's schedule in plain words ("30 steps / epoch"); the batches behind it in a popover, here clicked.
    name: 'run-new-steps',
    path: '/runs/new',
    fullPage: false,
    viewports: ['desktop', 'laptop'],
    setup: async (page) => {
      await until(page, '.summary button.steps');
      await page.locator('.summary button.steps').click();
      await page.mouse.move(0, 0);
      await page.clock.runFor(300);
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
    // The server's estimate (POST /api/runs/estimate): folo-s at 640, batch 16, f32 was measured on the mock's GPU, so
    // the time is a narrow range on "Measured on this GPU"; opened, the phases, validation and startup behind it.
    name: 'run-new-estimate',
    path: '/runs/new',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: (page, { vp }) => openEstimate(page, vp, true),
  },
  {
    // A server from before measured throughput (no `run_estimate`): the reference figures, as a wide range said to be
    // the reference, in secondary ink.
    name: 'run-new-reference',
    path: '/runs/new',
    search: 'scenario=no-throughput',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: (page, { vp }) => openEstimate(page, vp, false),
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
    // The Summary sheet's "30 steps / epoch" opened by a tap: the batches behind it in a popover over the sheet (top
    // layer), never behind the modal.
    name: 'run-new-summary-steps',
    fullPage: false,
    path: '/runs/new',
    viewports: ['tablet', 'mobile'],
    setup: async (page) => {
      await until(page, '.dock-est');
      await page.locator('.dock-est').click();
      await page.locator('dialog[open] button.steps').click();
      await page.clock.runFor(300);
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
  // One run trained 150 epochs, the others 23 to 61: the x axis defaults to "By progress" (the header says why, "one run
  // trained far longer"), so the three shorter runs are not squeezed into the first 40 % of an epoch axis; each line ends
  // in its run's status mark.
  // `small` (320 px): the phone control row must hold at the narrowest phone too, its values whole.
  { name: 'run-compare', path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0', viewports: ['desktop', 'laptop', 'tablet', 'mobile', 'small'] },
  {
    // Phones: the automatic axis's reason opened from the ⓘ beside the axis control, in place under the controls (the
    // legend and the plot move down, never covered).
    name: 'run-compare-axis-info',
    path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0',
    fullPage: false,
    viewports: ['mobile'],
    setup: async (page) => {
      await until(page, '.chip .key');
      await page.getByRole('button', { name: 'About the progress axis' }).click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // Phones: the y axis's choice (Fit / From 0) in How to read this, the control row having no room. By progress a
    // phone opens on From 0 (the whole climb); Fit chosen by hand, so the plot starts at its floor and says so.
    name: 'run-compare-yaxis-phone',
    path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0',
    viewports: ['mobile'],
    setup: async (page) => {
      await until(page, '.chip .key');
      await page.locator('details.about > summary').click();
      await page.locator('.about-axis').getByRole('radio', { name: 'Fit' }).click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // "By epoch", chosen by hand over the default "By progress": one epoch scale to the longest plan. On a tablet the
    // header keeps one layout whatever the axis (the qualifier wraps under the metric tabs, the control on their row);
    // a phone picks it in its compact X axis select.
    name: 'run-compare-epoch',
    path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0',
    fullPage: false,
    viewports: ['desktop', 'tablet', 'mobile'],
    setup: async (page, { vp }) => {
      await until(page, '.chip .key');
      if (vp === 'mobile') await page.getByRole('combobox', { name: 'X axis' }).selectOption('epoch');
      else await page.getByRole('radio', { name: 'By epoch' }).click();
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
    // Phones: Edit opens a sheet, one run per row with how it started, open and remove, then Add run; the legend stays
    // two columns of keys.
    name: 'run-compare-edit',
    path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0',
    fullPage: false,
    viewports: ['mobile'],
    setup: async (page) => {
      await until(page, 'button.manage');
      await page.locator('button.manage').click();
      await page.locator('dialog[open]').waitFor();
      await page.clock.runFor(500);
    },
  },
  {
    // Phones: the fourth run's line kept in front while the third is the table's baseline: emphasis and baseline are
    // separate choices, and the baseline holds while other lines are brought forward.
    name: 'run-compare-baseline-phone',
    path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0',
    viewports: ['mobile'],
    fullPage: true,
    setup: async (page) => {
      await until(page, '#cmp-table .pair');
      await page.getByRole('button', { name: 'Next run' }).click();
      await page.locator('#cmp-table .set-base').click();
      await page.locator('.chip .key').nth(3).click();
      await page.mouse.move(0, 0);
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
  {
    // Scrolled into the pair's settings: the pager and the head row (both names, Baseline, Swap) hold under the top bar
    // while the table is in view.
    name: 'run-compare-pair-sticky',
    path: '/runs/compare?ids=run_7d1e,run_c19a,run_5be3,run_a2f0',
    viewports: ['mobile'],
    fullPage: false,
    keepScroll: true,
    setup: async (page) => {
      await until(page, '#cmp-table .pair');
      await page.evaluate(() => {
        const rows = document.querySelectorAll('#cmp-table tbody tr');
        rows[Math.min(6, rows.length - 1)]?.scrollIntoView({ block: 'center' });
      });
      await page.mouse.move(0, 0);
    },
  },
  { name: 'run-compare-empty', path: '/runs/compare', viewports: ['desktop', 'laptop', 'mobile'] },
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
  {
    // Keyboard focus on fields: the one global 2 px ring on the field's frame (a text field), on a native select, and
    // on the class chips' frame, never a lime edge or halo (DESIGN_DIRECTION §2.9).
    name: 'datasets-add-focus',
    path: '/datasets',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await page.getByRole('button', { name: 'Add dataset' }).first().click();
      await page.locator('dialog[open]').waitFor();
      await page.getByRole('tab', { name: 'Generate sample set' }).click();
      await tabTo(page, 'dialog[open] .chips input');
    },
  },
  {
    name: 'run-new-focus-field',
    path: '/runs/new',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, 'input#name');
      await page.locator('input#name').focus();
      await page.keyboard.press('Shift+Tab');
      await tabTo(page, 'input#name');
    },
  },
  {
    name: 'run-new-focus-select',
    path: '/runs/new',
    fullPage: false,
    viewports: ['desktop'],
    setup: async (page) => {
      await until(page, 'select#weights');
      await page.locator('select#weights').focus();
      await page.keyboard.press('Shift+Tab');
      await tabTo(page, 'select#weights');
      await page.evaluate(() => window.scrollTo(0, 0));
    },
  },
  { name: 'dataset-stats', path: '/datasets/ds_shapes_hard?tab=stats', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  {
    // Box size's ⓘ opened by a click or tap: the definition in a popover under it (the whole page, so the panel shows).
    name: 'dataset-stats-box-size',
    path: '/datasets/ds_shapes_hard?tab=stats',
    fullPage: true,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, 'button[aria-label="How box size is measured"]');
      await page.locator('button[aria-label="How box size is measured"]').click();
      await page.mouse.move(0, 0);
      await page.clock.runFor(300);
    },
  },
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
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    fullPage: false,
    setup: async (page) => {
      await page.locator('.cell img.loaded').first().waitFor();
      await page.locator('.cell').nth(1).click();
      await page.locator('dialog[open] .stage img.loaded').waitFor();
      // Nothing selected and nothing hovered: the pointer that clicked the cell leaves the dialog, so no label row
      // carries a highlight of a state the scene does not show (round 22's stray "rivet" row was this hover).
      await page.mouse.move(0, 0);
      await page.waitForTimeout(200);
    },
  },
  {
    // A tooltip inside the inspector (a modal dialog): Zoom in hovered, its key hint drawn over the dialog.
    name: 'dataset-inspector-tooltip',
    path: '/datasets/ds_fasteners',
    viewports: ['desktop', 'mobile'],
    fullPage: false,
    setup: async (page) => {
      await page.locator('.cell img.loaded').first().waitFor();
      await page.locator('.cell').nth(1).click();
      await page.locator('dialog[open] .stage img.loaded').waitFor();
      await page.locator('dialog[open] button[aria-label="Zoom in"]:visible').first().hover();
      await page.clock.runFor(600);
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
      // The selection alone, as the Playground's selected scene: the pointer leaves the list, so no row keeps a hover
      // (and on touch there is none to keep).
      await page.mouse.move(0, 0);
      await page.waitForTimeout(300);
    },
  },
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
    // The Details rail is sticky (DESIGN_DIRECTION §2.2): scrolled to the per-class table, it holds under the top bar
    // beside it, so the space under it in the full-page shot is the room it travels in.
    name: 'model-detail-scrolled',
    path: '/models/m_fast_n',
    viewports: ['desktop', 'laptop'],
    fullPage: false,
    keepScroll: true,
    setup: async (page) => {
      await until(page, '#cm .cell');
      await page.locator('.cls-part').first().evaluate((el) => el.scrollIntoView({ block: 'end' }));
      await page.mouse.move(0, 0);
    },
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
        await page.getByRole('tab', { name: 'Details', exact: true }).click();
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
      if (vp === 'mobile') await page.getByRole('tab', { name: 'Details', exact: true }).click();
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
      if (vp === 'mobile') await page.getByRole('tab', { name: 'Details', exact: true }).click();
    },
  },
  {
    // fasteners-mini-n, imported, evaluated on fasteners-mini, which has no validation images: /eval fell back to the
    // training split (docs/API.md, Split fallback). The page says so once, under the metrics, and names the training
    // split wherever it names what was evaluated (the curves' meta, the per-class footer, Details, the matrix).
    name: 'model-eval-train-fallback',
    path: '/models/m_mini_n',
    viewports: ['desktop', 'mobile'],
    fullPage: true,
    setup: (page, { vp }) => until(page, vp === 'mobile' ? '#pr svg' : '#cm .cell'),
  },
  {
    name: 'model-coco',
    path: '/models/m_coco_s',
    viewports: ['desktop', 'laptop'],
    setup: (page) => until(page, '#cm .cell'),
  },
  {
    // Phones, whole page: the section picker under the header, then the Curves section headed by the 2 × 2 KPI grid.
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
      await page.getByRole('tab', { name: 'Classes', exact: true }).click();
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
      await page.getByRole('tab', { name: 'Classes', exact: true }).click();
      // Trace the third class, then the selection bar's View curve (the one place it lives).
      await page.locator('#per-class button.cls').nth(2).click();
      await page.locator('#per-class .to-curve').click();
      await until(page, '#pr .tools .back');
      await page.clock.runFor(500);
    },
  },
  {
    // Phones: tracing classes from Curves itself: the picker open under the bar, two classes traced (each a tap to
    // remove), the graph and its legend still in view.
    name: 'model-phone-trace',
    path: '/models/m_fast_n',
    viewports: ['mobile'],
    fullPage: false,
    setup: async (page) => {
      await until(page, '#pr svg');
      await page.locator('#pr .trace').click();
      await page.locator('#pr .t-chip').nth(1).click();
      await page.locator('#pr .t-chip').nth(3).click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // A quiet grey PR curve identified where it is drawn: the pointer near it picks out its class, which takes its hue
    // and its name and precision in the readout ("Click to trace"). Phones: a tap, and the readout offers Trace.
    name: 'model-pr-pick',
    path: '/models/m_fast_n',
    keepScroll: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    fullPage: false,
    setup: async (page, { vp }) => {
      await until(page, '#pr svg');
      await page.locator('#pr').scrollIntoViewIfNeeded();
      // A point three-quarters along the third grey curve, in page coordinates.
      const at = await page.locator('#pr path.line.context').nth(2).evaluate((p) => {
        const path = /** @type {SVGPathElement} */ (p);
        const pt = path.getPointAtLength(path.getTotalLength() * 0.75);
        const r = /** @type {SVGSVGElement} */ (path.ownerSVGElement).getBoundingClientRect();
        return { x: r.left + pt.x, y: r.top + pt.y };
      });
      if (vp !== 'mobile') {
        await page.mouse.move(at.x, at.y);
        return;
      }
      await page.locator('#pr svg[aria-roledescription=chart]').evaluate((svg, a) => {
        for (const type of ['pointerdown', 'pointerup'])
          svg.dispatchEvent(new PointerEvent(type, { pointerType: 'touch', clientX: a.x, clientY: a.y, bubbles: true }));
      }, at);
      await page.clock.runFor(100);
    },
  },
  {
    // Wider screens trace classes as phones do: Trace a class opens the classes as chips under the readout row, the
    // traced first; two traced here.
    name: 'model-trace-wide',
    path: '/models/m_fast_n',
    keepScroll: true,
    viewports: ['desktop', 'laptop', 'tablet'],
    fullPage: false,
    setup: async (page) => {
      await until(page, '#pr svg');
      await page.locator('#pr button.trace').click();
      await page.locator('#pr .t-chip').nth(1).click();
      await page.locator('#pr .t-chip').nth(3).click();
      await page.mouse.move(0, 0);
      await page.locator('#pr').scrollIntoViewIfNeeded();
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
      await page.getByRole('tab', { name: 'Confusion', exact: true }).click();
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
      await page.getByRole('tab', { name: 'Confusion', exact: true }).click();
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
      await page.getByRole('tab', { name: 'Confusion', exact: true }).click();
      await page.getByRole('button', { name: 'Explore the matrix' }).click();
      await until(page, '#cm .cell');
    },
  },
  {
    // Before an export, the demo says it creates no file ("Demo: no file is created"), under the formats.
    name: 'model-export',
    path: '/models/m_shapes_s',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
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
    // The demo's f16 export: a neutral preview (no success check, no saved file, no Download) whose one primary is
    // "Connect to a server". A real server's finished export (Download, the path) is ExportDialog.svelte.test.ts's.
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
      await until(page, 'dialog[open] .done');
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
    // ONNX in the demo: the same neutral preview as f16, by the file a server would write (fasteners-n-best.onnx).
    name: 'model-export-onnx',
    path: '/models/m_fast_n',
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
      await page.locator('dialog[open] label:has-text("ONNX")').first().click();
      await until(page, 'dialog[open] .switches');
      await page.locator('dialog[open] footer button:has-text("Export")').click();
      await until(page, 'dialog[open] .done');
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
      if (vp === 'mobile') {
        await page.locator('button.meta-line').click();
        // Phones open every section on the folded headline: unfold the grid its ⓘ sits in.
        await page.locator('.kpi-line').click();
      }
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
  {
    // The empty Playground. Phones are shot as the touch-only devices they are: "Choose an image to detect objects", no
    // drag-and-drop or Ctrl V hint (DropZone, the shared touch detection).
    name: 'playground-empty',
    path: '/playground',
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page, { vp }) => {
      if (vp === 'mobile') {
        const cdp = await page.context().newCDPSession(page);
        await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 5 });
        await page.reload({ waitUntil: 'networkidle' });
        await page.clock.runFor(2000);
      }
      await until(page, 'button.sample');
    },
  },
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
      await until(page, '.vbar-note');
    },
  },
  {
    // DOGFOOD-3 S5: a Studio build without the in-browser runtime: WebGPU and CPU are off, the line under them says
    // why, and the hovered CPU choice shows the reason as its tip.
    name: 'playground-no-wasm',
    path: '/playground?model=m_hard_s',
    search: 'scenario=no-wasm',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page, { vp }) => {
      await until(page, '.vbar-note');
      if (vp === 'desktop' || vp === 'laptop') await page.locator('.runon .tip').last().hover();
      else await page.mouse.move(0, 0);
      await page.clock.runFor(800);
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
    // Tablets: Confidence and Overlap stay in the strip under the half-height image; More settings opens Max detections
    // and the display toggles in the same non-modal sheet, over the detections.
    name: 'playground-more-tablet',
    path: '/playground?model=m_hard_s',
    viewports: ['tablet'],
    fullPage: false,
    setup: async (page) => {
      await playgroundResult(page);
      await page.getByRole('button', { name: 'More settings' }).click();
      await page.locator('#pg-more').waitFor();
      await page.clock.runFor(500);
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
        // The System panel is the subject: the page opens on it, so the section strip marks System.
        path: '/settings?section=system',
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
    // Server PR #140 (`gpu_pool`): FOLO's GPU memory in Settings › System: in use, cached and the peak, in the one size
    // format, the device beside it (the whole page: the panel sits near the bottom).
    name: 'settings-gpu-pool',
    path: '/settings?section=system',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, '#system .pool');
      await page.mouse.move(0, 0);
    },
  },
  {
    // Server PR #140: published weights listed as models, their own group ("Published weights") after the datasets.
    name: 'models-published',
    path: '/models',
    viewports: ['desktop', 'laptop', 'mobile'],
  },
  {
    // The published weights' page: "published weights" in its meta line, its Source what they are.
    name: 'model-published',
    path: '/models/folo-n-coco',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
  },
  {
    // The server's own words opened under Driver report: wrapped prose in a well, copyable whole.
    name: 'settings-system-report',
    path: '/settings?section=system',
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
    // Settings › System › Throughput: the rows this server measured (training, predictions; one from another adapter),
    // their spread, samples and age, and Measure now.
    name: 'settings-throughput',
    path: '/settings?section=system',
    // The whole page: the harness shoots from the top, and the block sits near the bottom.
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, '.tp-row');
      await page.mouse.move(0, 0);
    },
  },
  {
    // Measure now running: the button busy, the elapsed seconds counted against "up to 10 s".
    name: 'settings-measuring',
    path: '/settings?section=system',
    // The whole page: the harness shoots from the top, and the block sits near the bottom.
    fullPage: true,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, '.tp-row');
      await page.getByRole('button', { name: 'Measure now' }).click();
      await page.clock.runFor(1100);
      await page.mouse.move(0, 0);
    },
  },
  {
    // Single-key shortcuts turned off in Settings; the dialog (from the Settings button) says so.
    name: 'shortcuts-off',
    touch: false,
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
  // ---- Cloud compute (docs/CLOUD.md): Settings › Cloud, New run › Run on, and a cloud run's page ----
  {
    // One connection and a cloud run training on its rented RTX 4090: the instance running, its cost so far and when it
    // stops; the shapes-m run's instance ended, quiet.
    name: 'settings-cloud',
    path: '/settings?section=cloud',
    search: 'scenario=cloud-training',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile', 'small'],
    setup: async (page) => {
      await until(page, '#cloud-instances .st:visible');
      await page.mouse.move(0, 0);
    },
  },
  {
    // No connection yet: "No cloud connected" and the one way to connect.
    name: 'settings-cloud-empty',
    path: '/settings?section=cloud',
    search: 'scenario=cloud-none',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, '#cloud .empty');
      await page.mouse.move(0, 0);
    },
  },
  {
    // The key was revoked: the check's reason in the danger ink, Check again, and an instance whose termination is not
    // confirmed said as a problem above the list, with the pod to look for in the provider's console.
    name: 'settings-cloud-trouble',
    path: '/settings?section=cloud',
    search: 'scenario=cloud-trouble',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, '#cloud-instances .callout');
      await page.mouse.move(0, 0);
    },
  },
  {
    // Connect: RunPod chosen, the form empty, the key masked, "Where to find it" beside its label.
    name: 'settings-cloud-connect',
    path: '/settings?section=cloud',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, '#cloud header .btn');
      await page.locator('#cloud header .btn').click();
      await page.locator('dialog[open] input[type="password"]').waitFor();
      await page.clock.runFor(300);
      await page.mouse.move(0, 0);
    },
  },
  {
    // A key RunPod rejects: said at the key's field, nowhere else.
    name: 'settings-cloud-connect-rejected',
    path: '/settings?section=cloud',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, '#cloud header .btn');
      await page.locator('#cloud header .btn').click();
      await page.locator('dialog[open] input[type="password"]').fill('rpa_bad_0f3c9a71');
      await page.locator('dialog[open] button[type="submit"]').click();
      await until(page, 'dialog[open] .msg.error');
      await page.mouse.move(0, 0);
    },
  },
  {
    // New run on RunPod: the GPU preselected (the cheapest with the memory), Stop after with what it caps, the device a
    // fact; the Summary says where it runs, the cost range and the limit.
    name: 'run-new-cloud',
    path: '/runs/new',
    fullPage: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      for (let t = 0; t < 8000 && !(await page.locator('select#runon option[value="cc_4d2a"]').count()); t += 250) await page.clock.runFor(250);
      await page.selectOption('select#runon', 'cc_4d2a');
      await until(page, '#offer-hint');
      await until(page, '.summary .cost-note');
      await page.clock.runFor(1000);
      await page.mouse.move(0, 0);
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
