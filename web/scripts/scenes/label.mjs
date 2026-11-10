// Agent labeling: the jobs, New labeling job, a job's page in each state, the editor and Create dataset.

import { until } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** A labeling job's page with its first tiles loaded (on phones the agent panel comes first and the grid is out of view). */
async function tiles(/** @type {import('playwright-core').Page} */ page, /** @type {{ vp: string }} */ { vp }) {
  await until(page, '.agent');
  if (vp !== 'mobile' && (await page.locator('.cell').count())) await page.locator('.cell img.loaded').nth(3).waitFor();
  await page.mouse.move(0, 0);
}

/** @type {Scene[]} */
export const labelScenes = [
  // ---------- Agent labeling (Datasets › Label images) ----------
  { name: 'label-jobs', path: '/datasets/label', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  { name: 'label-jobs-empty', path: '/datasets/label', search: 'scenario=empty', viewports: ['desktop', 'mobile'] },
  {
    // The New labeling job dialog, filled in: two classes as chips, instructions for the agent.
    name: 'label-new-dialog',
    path: '/datasets/label?new=1',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, 'dialog[open] input[id$="-name"]');
      await page.locator('dialog[open] input[id$="-name"]').fill('street-scenes');
      await page.locator('dialog[open] input[id$="-classes"]').fill('car, person, bicycle');
      await page.keyboard.press('Enter');
      await page.mouse.move(0, 0);
    },
  },
  {
    // A job just made: no images yet, the drop target in the grid's place.
    name: 'label-job-fresh',
    path: '/datasets/label?new=1',
    viewports: ['desktop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, 'dialog[open] input[id$="-name"]');
      await page.locator('dialog[open] input[id$="-name"]').fill('street-scenes');
      await page.locator('dialog[open] input[id$="-classes"]').fill('car, person');
      await page.keyboard.press('Enter');
      await page.locator('dialog[open] button[type="submit"]').click();
      await until(page, '.zone');
      await page.clock.runFor(500);
      await page.mouse.move(0, 0);
    },
  },
  // A job's page: its images (a virtualised grid, so shot in the viewport, as a dataset's), the agent rail and its log.
  { name: 'label-job-pending', path: '/datasets/label/lb_7c21', fullPage: false, viewports: ['desktop', 'laptop', 'tablet', 'mobile'], setup: tiles },
  // An agent labeling now: tiles with the agent, live progress, Stop.
  { name: 'label-job-running', path: '/datasets/label/lb_3f9a?show=pending', fullPage: false, viewports: ['desktop', 'laptop', 'tablet', 'mobile'], setup: tiles },
  // Everything labeled and partly corrected by hand: Create dataset is the one action.
  { name: 'label-job-review', path: '/datasets/label/lb_51d0', fullPage: false, viewports: ['desktop', 'laptop', 'tablet', 'mobile'], setup: tiles },
  // The last run failed (Claude Code not signed in): its cause, the fix and Try again.
  { name: 'label-job-failed', path: '/datasets/label/lb_9e44', fullPage: false, viewports: ['desktop', 'tablet', 'mobile'], setup: tiles },
  {
    // The editor on an image corrected by hand, one box selected: its handles, its class, Delete box.
    name: 'label-editor',
    path: '/datasets/label/lb_51d0?image=5',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, 'dialog[open] .boxes button');
      await page.clock.runFor(800);
      await page.locator('dialog[open] .boxes button').first().click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // Create dataset: the name, the validation share and what goes in.
    name: 'label-dataset-dialog',
    path: '/datasets/label/lb_51d0',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, '.agent');
      const header = page.locator('header').getByRole('button', { name: 'Create dataset' });
      if (await header.isVisible()) await header.click();
      await until(page, 'dialog[open] input[id$="-name"]');
      await page.mouse.move(0, 0);
    },
  },
  // `folo serve --mock` (or an older server): one calm state in the page frame.
  { name: 'label-unavailable', path: '/datasets/label', search: 'scenario=no-labeling', viewports: ['desktop', 'mobile'] },
];
