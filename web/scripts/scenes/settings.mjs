// Settings: Connection, System (tensor cores, GPU pool, throughput, the driver report) and shortcuts; with them the
// published weights as models.

import { until } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** @type {Scene[]} */
export const settingsScenes = [
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
];
