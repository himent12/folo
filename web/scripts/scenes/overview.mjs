// Overview: the page, a model scored on its training split, and a cloud run uploading its dataset.

/** @typedef {import('../scenes.mjs').Scene} Scene */

/** @type {Scene[]} */
export const overviewScenes = [
  // The live run at the moment run-live shows it (the same `advance`), so the two screens' epoch and progress agree.
  { name: 'overview', path: '/', advance: 4000, viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  // A model scored on its training split (fasteners-mini has no validation images): Best models tags its score
  // ("Training split"), ranks it only within its own dataset and split, and counts imported and published weights apart.
  { name: 'overview-train-split', path: '/', search: 'scenario=train-split', viewports: ['desktop', 'laptop', 'tablet', 'mobile'] },
  {
    // Overview's recent runs with that run: the phase in the Status column, as in Runs.
    name: 'overview-cloud-upload',
    path: '/',
    search: 'scenario=cloud-upload',
    fullPage: false,
    advance: 4000,
    viewports: ['desktop', 'laptop', 'tablet', 'mobile'],
  },
];
