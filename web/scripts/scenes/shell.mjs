// The app's chrome on any screen: a toast, the command palette, the shortcuts reference, Not found.

import { until } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** @type {Scene[]} */
export const shellScenes = [
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
  { name: 'not-found', path: '/nope', viewports: ['desktop', 'mobile'] },
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
