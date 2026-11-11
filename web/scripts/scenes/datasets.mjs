// Datasets: the list (class balance, Add dataset and its focus rings, errors and filtering) and a dataset's page (stats,
// health, issues, the histogram, images, display options and the inspector).

import { until, tabTo } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** @type {Scene[]} */
export const datasetScenes = [
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
];
