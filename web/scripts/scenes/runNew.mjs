// New run: the form, its estimate and Summary, the dock below 960 px, focus rings, Run on a cloud GPU, and the errors
// Start can meet.

import { until, tabTo } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

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
export const runNewScenes = [
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
];
