// Vision benchmarks: the list, a leaderboard, New entry, conditions and the plan, loops, an entry's results, Compare.

import { until, optionIn, untilDrawn, imagesDecoded } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** @type {Scene[]} */
export const benchScenes = [
  // ---------- vision benchmarks ----------
  // The list: four benchmarks, one fresh, one running an entry.
  { name: 'bench-list', path: '/benchmarks', viewports: ['desktop', 'laptop', 'tablet', 'mobile'], setup: async (page, { vp }) => until(page, vp === 'mobile' ? '.rows li a' : 'table.benches tbody tr[data-row]') },
  {
    // New benchmark, filled in: a name, what the camera sees, a dataset's validation split.
    name: 'bench-new-dialog',
    path: '/benchmarks?new=1',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, 'dialog[open] input[id$="-name"]');
      await page.locator('dialog[open] input[id$="-name"]').fill('Warehouse bins');
      await page.locator('dialog[open] textarea').fill('A camera over the parts bins at the end of the line: fluorescent light, parts in piles that hide each other.');
      await page.clock.runFor(500);
      await page.mouse.move(0, 0);
    },
  },
  // A leaderboard with an entry of every kind and state: ranked scores, partial agents, a failure, one running.
  { name: 'bench-board', path: '/benchmarks/bn_3c90', viewports: ['desktop', 'laptop', 'tablet', 'mobile'], setup: async (page, { vp }) => until(page, vp === 'mobile' ? 'ul.rows li a' : 'table.board tbody tr') },
  {
    // Two entries picked: Compare is the one primary.
    name: 'bench-board-compare',
    path: '/benchmarks/bn_3c90',
    fullPage: false,
    viewports: ['desktop'],
    setup: async (page) => {
      await until(page, 'table.board tbody tr .check input');
      const boxes = page.locator('table.board tbody tr .check input');
      await boxes.nth(1).check({ force: true });
      await boxes.nth(2).check({ force: true });
      await page.mouse.move(0, 0);
    },
  },
  // A fresh benchmark: no entries yet, its plan still open.
  { name: 'bench-board-empty', path: '/benchmarks/bn_9f03', viewports: ['desktop', 'mobile'] },
  {
    // Add entry › Model: a model that knows two of the eight classes.
    name: 'bench-add-model',
    path: '/benchmarks/bn_3c90?add=model',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, 'dialog[open] select[id$="-model"] option[value="m_mini_n"]').catch(() => {});
      for (let t = 0; t < 8000 && !(await page.locator('dialog[open] select[id$="-model"] option[value="m_mini_n"]').count()); t += 250) await page.clock.runFor(250);
      await page.selectOption('dialog[open] select[id$="-model"]', 'm_mini_n');
      await page.clock.runFor(300);
      await page.mouse.move(0, 0);
    },
  },
  {
    // Add entry › Agent: Claude Code with the defaults (48 images, unchanged only) and what they mean.
    name: 'bench-add-agent',
    path: '/benchmarks/bn_3c90?add=agent',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, 'dialog[open] [role=radiogroup][aria-label="Effort"]');
      await page.mouse.move(0, 0);
    },
  },
  {
    // Add entry › External: the file, the confidences switch and how to make the file, open.
    name: 'bench-add-external',
    path: '/benchmarks/bn_3c90?add=external',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, 'dialog[open] details.how > summary');
      await page.locator('dialog[open] details.how > summary').click();
      await page.mouse.move(0, 0);
    },
  },
  {
    // The conditions of a fresh benchmark: the default plan, editable, and the preview of haze on its images.
    name: 'bench-conditions',
    path: '/benchmarks/bn_9f03?tab=conditions',
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, '.stage img');
      await page.clock.runFor(800);
      await page.mouse.move(0, 0);
    },
  },
  {
    // The preview under distance: the scene and its boxes shrink towards the centre.
    name: 'bench-preview-distance',
    path: '/benchmarks/bn_3c90?tab=conditions',
    fullPage: false,
    viewports: ['desktop'],
    setup: async (page) => {
      await until(page, 'select[id$="-cond"]');
      await page.selectOption('select[id$="-cond"]', 'distance');
      await page.locator('[role=radiogroup][aria-label="Severity"] [role=radio]').nth(4).click();
      await until(page, '.stage img');
      await page.clock.runFor(1200);
      await page.mouse.move(0, 0);
    },
  },
  // A plan Claude Code designed, frozen by its entries: read-only, with the one-line reason.
  { name: 'bench-conditions-frozen', path: '/benchmarks/bn_7a41?tab=conditions', viewports: ['desktop', 'mobile'], setup: async (page) => until(page, '.stage img') },
  {
    // Design with an agent: the request, then the proposal against the default plan.
    name: 'bench-design',
    path: '/benchmarks/bn_9f03?tab=conditions',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, 'button:has-text("Design with an agent")');
      await page.locator('button:has-text("Design with an agent")').click();
      await until(page, 'dialog[open] button:has-text("Ask Claude Code")');
      await page.locator('dialog[open] button:has-text("Ask Claude Code")').click();
      await until(page, 'dialog[open] button:has-text("Use this plan")', 12000);
      await page.mouse.move(0, 0);
    },
  },
  {
    // The proposal used: the editor holds it, unsaved.
    name: 'bench-design-used',
    path: '/benchmarks/bn_9f03?tab=conditions',
    fullPage: false,
    viewports: ['desktop'],
    setup: async (page) => {
      await until(page, 'button:has-text("Design with an agent")');
      await page.locator('button:has-text("Design with an agent")').click();
      await until(page, 'dialog[open] button:has-text("Ask Claude Code")');
      await page.locator('dialog[open] button:has-text("Ask Claude Code")').click();
      await until(page, 'dialog[open] button:has-text("Use this plan")', 12000);
      await page.locator('dialog[open] button:has-text("Use this plan")').click();
      await page.clock.runFor(400);
      await page.mouse.move(0, 0);
    },
  },
  {
    // The agent's answer is not a plan: the server's message and its fix.
    name: 'bench-design-failed',
    path: '/benchmarks/bn_9f03?tab=conditions',
    search: 'scenario=bench-agent-fails',
    fullPage: false,
    viewports: ['desktop'],
    setup: async (page) => {
      await until(page, 'button:has-text("Design with an agent")');
      await page.locator('button:has-text("Design with an agent")').click();
      await until(page, 'dialog[open] button:has-text("Ask Claude Code")');
      await page.locator('dialog[open] button:has-text("Ask Claude Code")').click();
      await until(page, 'dialog[open] button:has-text("Try again")', 12000);
      await page.mouse.move(0, 0);
    },
  },
  // ---------- the result screens ----------
  // Loops: the tab (one improved over three generations, one held), New loop from it.
  { name: 'bench-loops', path: '/benchmarks/bn_5e12?tab=loops', viewports: ['desktop', 'mobile'], setup: async (page, { vp }) => until(page, vp === 'mobile' ? 'ul.rows a.m-row' : 'table.loops tbody tr') },
  {
    // New loop on Fasteners val: the eligible teachers first, the student, the summary of what it will cost.
    name: 'bench-loop-new',
    path: '/benchmarks/bn_3c90?tab=loops',
    fullPage: false,
    viewports: ['desktop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, 'button:has-text("New loop")');
      await page.locator('button:has-text("New loop")').first().click();
      await until(page, 'dialog[open] .s-text');
      await page.locator('dialog[open] [role=radiogroup][aria-label="Generations"] [role=radio]').nth(2).click();
      await page.clock.runFor(600);
      await page.mouse.move(0, 0);
    },
  },
  // A loop's page: improved over three generations, held after one, and one running now (its student training).
  { name: 'bench-loop-improved', path: '/benchmarks/bn_5e12/loops/bl_07d1', viewports: ['desktop', 'laptop', 'tablet', 'mobile'], setup: async (page) => until(page, '.top svg[aria-roledescription=chart]') },
  { name: 'bench-loop-held', path: '/benchmarks/bn_5e12/loops/bl_5h02', viewports: ['desktop', 'mobile'], setup: async (page) => until(page, '.top svg[aria-roledescription=chart]') },
  { name: 'bench-loop-running', path: '/benchmarks/bn_3c90/loops/bl_f3a0', viewports: ['desktop', 'mobile'], setup: async (page) => until(page, '.steps li') },
  { name: 'bench-loop-mobile-full', path: '/benchmarks/bn_5e12/loops/bl_07d1', fullPage: true, viewports: ['mobile'], setup: async (page) => until(page, '.top svg[aria-roledescription=chart]') },
  // A loop that has just started: generation 0, one step (the benchmark grades the first teacher), its entry waiting
  // for the GPU, nothing graded on the chart yet.
  { name: 'bench-loop-gen0', path: '/benchmarks/bn_3c90/loops/bl_g0a1', search: 'scenario=bench-loop-gen0', viewports: ['desktop', 'tablet', 'mobile'], setup: async (page) => until(page, '.steps li') },
  // A loop that failed before labeling: its teacher's entry found no threshold and none was set (a loop never guesses).
  { name: 'bench-loop-starved', path: '/benchmarks/bn_5e12/loops/bl_9e40', viewports: ['desktop', 'mobile'], setup: async (page) => until(page, 'table.gens tbody tr') },
  {
    // New loop with a teacher whose entry found no threshold: the confidence is required, said before anything else.
    name: 'bench-loop-new-weak',
    path: '/benchmarks/bn_5e12?tab=loops',
    fullPage: false,
    viewports: ['desktop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, 'button:has-text("New loop")');
      await page.locator('button:has-text("New loop")').first().click();
      await optionIn(page, 'dialog[open] select[id$="-teacher"]', 'm_shapes_e5');
      await page.selectOption('dialog[open] select[id$="-teacher"]', 'm_shapes_e5');
      await until(page, 'dialog[open] .fc.warn');
      await page.evaluate(() => document.querySelector('dialog[open] .fc')?.scrollIntoView({ block: 'center' }));
      await page.clock.runFor(300);
      await page.mouse.move(0, 0);
    },
  },
  ...[
    // A confidence where it keeps a handful: the loop would stop at labeling; a lower one, and what it costs.
    ['stops', '0.9'],
    ['lower', '0.30'],
  ].map(([which, value]) => ({
    name: `bench-loop-new-${which}`,
    path: '/benchmarks/bn_5e12?tab=loops',
    fullPage: false,
    viewports: /** @type {('desktop' | 'mobile')[]} */ (which === 'stops' ? ['desktop', 'mobile'] : ['desktop']),
    setup: async (/** @type {import('playwright-core').Page} */ page) => {
      await until(page, 'button:has-text("New loop")');
      await page.locator('button:has-text("New loop")').first().click();
      await optionIn(page, 'dialog[open] select[id$="-teacher"]', 'm_shapes_e5');
      await page.selectOption('dialog[open] select[id$="-teacher"]', 'm_shapes_e5');
      await until(page, 'dialog[open] input[id$="-conf"]');
      await page.locator('dialog[open] input[id$="-conf"]').fill(value);
      await page.clock.runFor(300);
      await page.evaluate(() => document.querySelector('dialog[open] .fc')?.scrollIntoView({ block: 'center' }));
      await page.mouse.move(0, 0);
    },
  })),
  {
    name: 'bench-entry-error-explorer',
    path: '/benchmarks/bn_3c90/entries/be_f101',
    fullPage: false,
    keepScroll: true,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, '#images .tile img');
      await page.evaluate(() => document.getElementById('images')?.scrollIntoView({ block: 'start' }));
      await page.locator('[role="radiogroup"][aria-label="Filter images by error"] [role="radio"]').nth(1).click();
      await page.clock.runFor(500);
      await page.mouse.move(0, 0);
    },
  },
  // Photos of every shape (Pond fish was made from people's phone photos): the gallery's tiles in one shape, each photo
  // whole on its well with its boxes; the large view of a portrait one and of a panorama.
  {
    name: 'bench-entry-mixed',
    path: '/benchmarks/bn_7a41/entries/be_p201',
    fullPage: false,
    keepScroll: true,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
    setup: async (page) => {
      await until(page, '#images .tile img');
      await page.evaluate(() => document.getElementById('images')?.scrollIntoView({ block: 'start' }));
      await page.clock.runFor(800);
      await page.mouse.move(0, 0);
    },
  },
  .../** @type {[string, number, number][]} */ ([
    ['portrait', 0, 0.9],
    ['wide', 2.5, 99],
  ]).map(([which, lo, hi]) => /** @type {Scene} */ ({
    name: `bench-entry-viewer-${which}`,
    path: '/benchmarks/bn_7a41/entries/be_p201',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (/** @type {import('playwright-core').Page} */ page) => {
      await until(page, '#images .tile img');
      await page.clock.runFor(800);
      // The first tile whose photo has that shape (its own pixels say it), once the thumbnails have drawn.
      let k = -1;
      for (let i = 0; i < 60 && k < 0; i++) {
        await imagesDecoded(page);
        k = await page.evaluate(
          ([a, b]) => [...document.querySelectorAll('#images .tile img')].findIndex((im) => im instanceof HTMLImageElement && im.naturalHeight > 0 && im.naturalWidth / im.naturalHeight > a && im.naturalWidth / im.naturalHeight < b),
          [lo, hi],
        );
        if (k < 0) {
          await page.clock.runFor(250);
          await page.waitForTimeout(100);
        }
      }
      if (k < 0) throw new Error(`no ${which} photo in the first page`);
      await page.locator('#images .tile').nth(k).click();
      await untilDrawn(page, 'dialog[open] .v-img img');
      await page.clock.runFor(600);
      await page.mouse.move(0, 0);
    },
  })),
  {
    // The preview on photos of every shape: a portrait one in the benchmark's 4:3 stage, under distance (its boxes
    // shrink about its own centre).
    name: 'bench-preview-mixed',
    path: '/benchmarks/bn_7a41?tab=conditions',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, 'select[id$="-cond"]');
      await page.selectOption('select[id$="-cond"]', 'distance');
      await page.locator('.stage').scrollIntoViewIfNeeded();
      await page.locator('[role=radiogroup][aria-label="Severity"] [role=radio]').nth(3).click({ force: true });
      await untilDrawn(page, '.stage img');
      // Step to the first portrait photo.
      for (let i = 0; i < 24; i++) {
        await page.clock.runFor(400);
        await imagesDecoded(page);
        const tall = await page.evaluate(() => [...document.querySelectorAll('.stage img')].some((im) => im instanceof HTMLImageElement && im.naturalHeight > im.naturalWidth));
        if (tall) break;
        await page.locator('button[aria-label="Next image"]').click({ force: true });
        await untilDrawn(page, '.stage img');
      }
      await page.clock.runFor(800);
      await page.evaluate(() => document.querySelector('.stage')?.scrollIntoView({ block: 'center' }));
      await page.mouse.move(0, 0);
    },
  },
  {
    // A large predictions file on its way: an open entry filled a pass at a time, the records sent as a bar.
    name: 'bench-add-external-sending',
    path: '/benchmarks/bn_9f03?add=external',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, 'dialog[open] .file button');
      // Every image of the benchmark under every pass of its default plan, 12 boxes each: over 8 MiB as JSON.
      const passes = [[null, null], ...['low_light', 'overexposure', 'low_contrast', 'defocus', 'motion_blur', 'sensor_noise', 'haze', 'color_cast', 'jpeg', 'low_resolution', 'distance', 'occlusion'].flatMap((c) => [1, 3, 5].map((s) => [c, s]))];
      const box = { class: 0, score: 0.62, x1: 120.5, y1: 88.25, x2: 210.75, y2: 160.5 };
      const records = passes.flatMap(([c, s]) => Array.from({ length: 200 }, (_, i) => ({ image: String(i + 1), ...(c ? { condition: c, severity: s } : {}), boxes: Array.from({ length: 12 }, () => box) })));
      await page.locator('dialog[open] input[type="file"]').setInputFiles({ name: 'rtdetr-l-predictions.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({ predictions: records })) });
      await until(page, 'dialog[open] .f-state.ok', 20000);
      await page.locator('dialog[open] button[type="submit"]').click();
      await until(page, 'dialog[open] .f-send');
      for (let i = 0; i < 12; i++) await page.clock.runFor(300);
      await page.mouse.move(0, 0);
    },
  },
  // Calibration read honestly: the pond's mixed-sign reliability (mostly under, a thin over top), and a scored entry
  // whose confidences never reach 0.1 (no calibration, partial for that alone).
  ...[
    ['mixed', 'be_p201', '#calibration'],
    ['nocal', 'be_p205', '#score'],
  ].map(([which, entryId, anchor]) => ({
    name: `bench-entry-calibration-${which}`,
    path: `/benchmarks/bn_7a41/entries/${entryId}`,
    fullPage: false,
    keepScroll: true,
    viewports: /** @type {('desktop' | 'mobile')[]} */ (['desktop', 'mobile']),
    setup: async (/** @type {import('playwright-core').Page} */ page) => {
      await until(page, anchor);
      await page.evaluate((a) => document.querySelector(a)?.scrollIntoView({ block: 'start' }), anchor);
      await page.clock.runFor(600);
      await page.mouse.move(0, 0);
    },
  })),
  // Self-labeling with no threshold: a precise handful said to be a handful, the label count where the threshold was.
  {
    name: 'bench-entry-selflabel-none',
    path: '/benchmarks/bn_5e12/entries/be_5a30',
    fullPage: false,
    keepScroll: true,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, '#self-labeling');
      await page.evaluate(() => document.getElementById('self-labeling')?.scrollIntoView({ block: 'start' }));
      await page.clock.runFor(600);
      await page.mouse.move(0, 0);
    },
  },
  // The leaderboard with partial entries after every complete one; the pond's, where a partial entry scores higher.
  { name: 'bench-board-pond', path: '/benchmarks/bn_7a41', viewports: ['desktop', 'mobile'], setup: async (page, { vp }) => until(page, vp === 'mobile' ? 'ul.rows li a' : 'table.board tbody tr') },
  // An entry graded on an earlier version of the benchmark (another fingerprint): its page says so; Compare calls it out.
  { name: 'bench-entry-stale', path: '/benchmarks/bn_3c90/entries/be_f100', fullPage: false, viewports: ['desktop', 'mobile'], setup: async (page) => until(page, '#score') },
  { name: 'bench-compare-stale', path: '/benchmarks/bn_3c90/compare?entries=be_f101,be_f100', fullPage: false, viewports: ['desktop', 'mobile'], setup: async (page) => until(page, '.chips .chip') },
  // A fresh benchmark whose only entry failed: its plan still open (a failed entry graded nothing).
  { name: 'bench-conditions-failed-only', path: '/benchmarks/bn_9f03?tab=conditions', search: 'scenario=bench-failed-only', viewports: ['desktop', 'mobile'], setup: async (page) => until(page, '.stage img') },
  {
    // An entry's next step: a loop with this model as the first teacher.
    name: 'bench-entry-loop',
    path: '/benchmarks/bn_3c90/entries/be_f103',
    fullPage: false,
    viewports: ['desktop', 'mobile'],
    setup: async (page) => {
      await until(page, 'button:has-text("Test it: start a loop")');
      await page.locator('button:has-text("Test it: start a loop")').click();
      await until(page, 'dialog[open] .s-text');
      await page.clock.runFor(600);
      await page.mouse.move(0, 0);
    },
  },
  // Compare: two entries, four (an agent and a model that knows two classes among them: not like for like), and a
  // link with one entry left.
  { name: 'bench-compare', path: '/benchmarks/bn_3c90/compare?entries=be_f101,be_f103', viewports: ['desktop', 'laptop', 'mobile'], setup: async (page) => until(page, '.pair svg[aria-roledescription=chart]') },
  { name: 'bench-compare-4', path: '/benchmarks/bn_3c90/compare?entries=be_f101,be_f102,be_f104,be_f106', viewports: ['desktop', 'tablet', 'mobile'], setup: async (page) => until(page, '.pair svg[aria-roledescription=chart]') },
  { name: 'bench-compare-4-full', path: '/benchmarks/bn_3c90/compare?entries=be_f101,be_f102,be_f104,be_f106', fullPage: true, viewports: ['mobile'], setup: async (page) => until(page, '.pair svg[aria-roledescription=chart]') },
  { name: 'bench-compare-one', path: '/benchmarks/bn_3c90/compare?entries=be_f101,be_gone', viewports: ['desktop', 'mobile'] },
  {
    name: 'bench-board-pick',
    path: '/benchmarks/bn_3c90',
    viewports: ['mobile'],
    setup: async (page) => {
      await until(page, 'button:has-text("Select")');
      await page.locator('button:has-text("Select")').click();
      const boxes = page.locator('ul.rows .m-check input');
      await boxes.nth(0).check();
      await boxes.nth(2).check();
      await page.mouse.move(0, 0);
    },
  },
  // `folo serve --mock` (or an older server): the section says what it needs.
  { name: 'bench-unavailable', path: '/benchmarks', search: 'scenario=no-bench', viewports: ['desktop', 'mobile'] },
];
