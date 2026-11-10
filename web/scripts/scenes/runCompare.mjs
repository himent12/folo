// Compare runs: the chart's axes, the legend, the side-by-side table on phones, and the empty state.

import { until } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** @type {Scene[]} */
export const runCompareScenes = [
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
];
