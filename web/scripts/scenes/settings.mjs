// Settings: Connection, System (tensor cores, GPU pool, throughput, the driver report) and Cloud.

import { until } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** @type {Scene[]} */
export const settingsScenes = [
  { name: 'settings', path: '/settings', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
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
  // ---- Cloud compute (docs/CLOUD.md): Settings › Cloud ----
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
    viewports: ['desktop', 'tablet', 'mobile', 'small'],
    setup: async (page) => {
      await until(page, '#cloud header .btn');
      await page.locator('#cloud header .btn').click();
      await page.locator('dialog[open] input[type="password"]').waitFor();
      await page.clock.runFor(300);
      await page.mouse.move(0, 0);
    },
  },
  {
    // Connect to a remote FOLO server: its sentence names the command to run there, in mono and never broken inside a
    // flag; the token is optional.
    name: 'settings-cloud-connect-remote',
    path: '/settings?section=cloud',
    fullPage: false,
    viewports: ['desktop', 'tablet', 'mobile', 'small'],
    setup: async (page) => {
      await until(page, '#cloud header .btn');
      await page.locator('#cloud header .btn').click();
      await page.locator('dialog[open]').getByRole('radio', { name: 'Remote FOLO server' }).click();
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
];
