// Cloud compute (docs/CLOUD.md): Settings › Cloud, New run › Run on, and a cloud run on its pages.

import { until } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** @type {Scene[]} */
export const cloudScenes = [
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
    // Overview's recent runs with that run: the phase in the Status column, as in Runs.
    name: 'overview-cloud-upload',
    path: '/',
    search: 'scenario=cloud-upload',
    fullPage: false,
    advance: 4000,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
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
