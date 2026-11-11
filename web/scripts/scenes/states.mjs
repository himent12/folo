// The server states every screen answers, by scenario: nothing registered (`empty`), still loading, unreachable
// (`error`), and stopped answering after the first loads (`stale`, `stale-lists`).

import { until } from './drive.mjs';

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** @type {Scene[]} */
export const stateScenes = [
  { name: 'datasets-empty', path: '/datasets', search: 'scenario=empty' },
  { name: 'models-empty', path: '/models', search: 'scenario=empty' },
  { name: 'models-error', path: '/models', search: 'scenario=error', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'models-loading', path: '/models', search: 'scenario=loading', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'overview-empty', path: '/', search: 'scenario=empty', viewports: ['desktop', 'mobile'] },
  { name: 'overview-loading', path: '/', search: 'scenario=loading', viewports: ['desktop', 'laptop', 'mobile'] },
  // Nothing loads: one recovery block on the page (Retry, Connection settings), not four panels each saying so.
  { name: 'overview-error', path: '/', search: 'scenario=error', viewports: ['desktop', 'laptop', 'mobile'] },
  // The server stops answering after the first loads, its live stream too: the page keeps what it has under one
  // stale-data notice, and the live card shows its last known state ("Last known: Running · 2m ago", "~7m left, as of
  // 2m ago", no heartbeat). Two minutes on, so the ages read as minutes.
  { name: 'overview-stale', path: '/', search: 'scenario=stale', advance: 125_000, viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  // Only the lists stop refreshing; the live run's stream still streams: the notice says the lists are behind and the
  // live card stays live.
  { name: 'overview-stale-lists', path: '/', search: 'scenario=stale-lists', advance: 125_000, viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  { name: 'runs-empty', path: '/runs', search: 'scenario=empty', viewports: ['desktop', 'mobile'] },
  // Empty and error states sit in the table frame (header row, the message inside, a few rows tall).
  { name: 'runs-error', path: '/runs', search: 'scenario=error', viewports: ['desktop', 'laptop', 'mobile'] },
  { name: 'runs-loading', path: '/runs', search: 'scenario=loading', viewports: ['desktop', 'tablet', 'mobile'] },
  { name: 'datasets-loading', path: '/datasets', search: 'scenario=loading', viewports: ['desktop', 'tablet', 'mobile'] },
  { name: 'datasets-error', path: '/datasets', search: 'scenario=error', viewports: ['desktop', 'laptop', 'mobile'] },
  {
    // Offline: the create action is disabled and says why on hover (and to assistive tech).
    name: 'datasets-error-why',
    path: '/datasets',
    search: 'scenario=error',
    fullPage: false,
    viewports: ['desktop'],
    setup: async (page) => {
      await until(page, 'header .gate');
      const box = await page.locator('main header .gate').first().boundingBox();
      if (!box) throw new Error('datasets-error-why: no Add dataset');
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.clock.runFor(600);
    },
  },
  { name: 'run-loading', path: '/runs/run_7d1e', search: 'scenario=loading', themes: ['dark'], viewports: ['desktop', 'laptop'] },
  { name: 'label-jobs-empty', path: '/datasets/label', search: 'scenario=empty', viewports: ['desktop', 'mobile'] },
  { name: 'bench-list-empty', path: '/benchmarks', search: 'scenario=empty', viewports: ['desktop', 'mobile'] },
  // Loading (the skeletons have the loaded page's geometry) and a server that stopped answering.
  { name: 'bench-list-loading', path: '/benchmarks', search: 'scenario=loading', viewports: ['desktop', 'mobile'] },
  { name: 'bench-list-error', path: '/benchmarks', search: 'scenario=error', viewports: ['desktop', 'mobile'] },
  { name: 'bench-board-loading', path: '/benchmarks/bn_3c90', search: 'scenario=loading', viewports: ['desktop', 'mobile'] },
  { name: 'bench-board-error', path: '/benchmarks/bn_3c90', search: 'scenario=error', viewports: ['desktop', 'mobile'] },
];
