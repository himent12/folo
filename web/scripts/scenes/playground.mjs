// Playground: empty, a result, the browser runtimes it lacks, and its settings on tablets and phones.

import { until } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** Runs the Playground on the first sample image and waits for the detections table. */
async function playgroundResult(/** @type {import('playwright-core').Page} */ page) {
  await until(page, 'button.sample');
  await page.locator('button.sample').first().click();
  await page.mouse.move(0, 0);
  await until(page, '#pg-panel tbody tr');
  await page.clock.runFor(500);
}

/** @type {Scene[]} */
export const playgroundScenes = [
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
  { name: 'playground-result', path: '/playground?model=m_hard_s', viewports: ['desktop', 'wide', 'laptop', 'tablet', 'mobile'], setup: playgroundResult },
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
    // A Studio build without the in-browser runtime (DOGFOOD-3 S5, models F14): it runs on the server, with no Run on
    // choice it could never use and no line about it; the model picker stands alone.
    name: 'playground-no-wasm',
    path: '/playground?model=m_hard_s',
    search: 'scenario=no-wasm',
    fullPage: false,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, '.model-pick');
      await page.mouse.move(0, 0);
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
];
