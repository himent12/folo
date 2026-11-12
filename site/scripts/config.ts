// What the site is built from and how it is laid out. The Markdown stays the source of truth: this file only says
// which files become pages, in what order, and under which sidebar label.
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const SITE_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const REPO_DIR = resolve(SITE_DIR, '..');
/** Generated pages and public files; Vite's root. Never committed. */
export const GEN_DIR = resolve(SITE_DIR, '.gen');
export const DIST_DIR = resolve(SITE_DIR, 'dist');

export const REPO_URL = 'https://github.com/himent12/folo';
/** Links to repository files that are not pages point here. */
export const BLOB_URL = `${REPO_URL}/blob/main`;
export const TREE_URL = `${REPO_URL}/tree/main`;

export interface NavPage {
  /** Repository-relative Markdown path. */
  src: string;
  /** Sidebar label (the page title is the file's H1). */
  label: string;
}

export interface NavGroup {
  title: string;
  pages: NavPage[];
}

// Every docs/*.md and examples/*/README.md must appear here; the build fails otherwise, so a new doc is never left
// out of the site by accident. The engineering notes under docs/notes are not pages: links to them go to GitHub.
export const NAV: NavGroup[] = [
  {
    title: 'Get started',
    pages: [
      { src: 'docs/QUICKSTART.md', label: 'Quickstart' },
      { src: 'docs/CLI.md', label: 'CLI' },
      { src: 'docs/AGENT_COOKBOOK.md', label: 'Agent cookbook' },
    ],
  },
  {
    title: 'Guides',
    pages: [
      { src: 'docs/LABELING.md', label: 'Labeling with agents' },
      { src: 'docs/CLOUD.md', label: 'Cloud compute' },
      { src: 'docs/BENCH.md', label: 'Vision benchmarks' },
      { src: 'examples/shapes/README.md', label: 'Shapes: the convergence gate' },
      { src: 'examples/coco/README.md', label: 'COCO: pretraining' },
    ],
  },
  {
    title: 'Reference',
    pages: [
      { src: 'docs/API.md', label: 'HTTP API' },
      { src: 'docs/MCP.md', label: 'MCP server' },
      { src: 'docs/FORMAT.md', label: '.folo file format' },
      { src: 'docs/MODEL.md', label: 'Detector spec' },
      { src: 'docs/NN.md', label: 'NN: layers and optimizers' },
      { src: 'docs/DATA.md', label: 'Data' },
      { src: 'docs/METRICS.md', label: 'Metrics' },
      { src: 'docs/GPU.md', label: 'GPU' },
      { src: 'docs/WASM.md', label: 'In the browser' },
    ],
  },
  {
    title: 'Project',
    pages: [
      { src: 'docs/ARCHITECTURE.md', label: 'Architecture' },
      { src: 'docs/BENCHMARKS.md', label: 'Speed benchmarks' },
      { src: 'docs/ROADMAP.md', label: 'Roadmap' },
      { src: 'docs/RELEASE.md', label: 'Releasing' },
      { src: 'docs/CI.md', label: 'CI' },
    ],
  },
];

/** The landing page is built from the README. */
export const LANDING_SRC = 'README.md';
/** The agents' index; the site's llms.txt is this file with its links pointed at the site. */
export const LLMS_SRC = 'llms.txt';

/** URL path of a page, relative to the site root, without a leading slash ('' is the landing page). */
export function slugFor(src: string): string {
  if (src === LANDING_SRC) return '';
  const ex = /^examples\/([^/]+)\/README\.md$/.exec(src);
  if (ex) return `examples/${ex[1]}/`;
  const doc = /^docs\/([^/]+)\.md$/.exec(src);
  if (doc) return `${doc[1].toLowerCase().replace(/_/g, '-')}/`;
  throw new Error(`no URL scheme for ${src}`);
}
