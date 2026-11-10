// Models: the list, a model's page (curves, matrix, classes, trace, COCO) on every tier, Export, and the engine benchmark.

import { until } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** @type {Scene[]} */
export const modelScenes = [
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
];
