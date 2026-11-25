// Pure item providers assembling command palette entries from page commands, navigation, actions, and server entities.
import Activity from '$icons/activity';
import Box from '$icons/box';
import GitCompare from '$icons/git-compare-arrows';
import Images from '$icons/images';
import Keyboard from '$icons/keyboard';
import PanelLeft from '$icons/panel-left';
import Plus from '$icons/plus';
import Repeat from '$icons/repeat';
import ScanSearch from '$icons/scan-search';
import SettingsIcon from '$icons/settings';
import SunMoon from '$icons/sun-moon';
import Tags from '$icons/tags';
import Target from '$icons/target';
import type { Bench, BenchLoop, Dataset, LabelJob, Model, Run, RunStatus } from '$lib/api/types';
import { count } from '$lib/format';
import { failureCause } from '$lib/oom';
import { prefs } from '$lib/prefs';
import { router } from '$lib/router';
import { scoreContext, scoreText } from '$lib/scoreSummary';
import { statusWord } from '$lib/ui/StatusBadge';
import { mainNav, settingsNav } from '../nav';
import type { PageCommand } from '../pageCommands';
import { sidebar } from '../sidebarState';
import { themeCommands } from '../themeCommands';
import { SETTINGS_SECTIONS, type Item } from './types';

const ACTIVE: RunStatus[] = ['running', 'stopping', 'queued'];

export interface PaletteItemSources {
  pageCommands: readonly PageCommand[];
  runs: readonly Run[];
  datasets: readonly Dataset[];
  models: readonly Model[];
  benches: readonly Bench[];
  benchesOk: boolean;
  labelJobs: readonly LabelJob[];
  loops: readonly BenchLoop[];
  cloud: boolean;
  lacksLabel: boolean;
  touchOnly: boolean;
  onshortcuts: () => void;
}

/** Builds the complete list of rankable palette items from current shell & entity state. */
export function buildPaletteItems(src: PaletteItemSources): Item[] {
  const go = (path: string) => () => router.go(path);

  // What the page on screen offers (shell/pageCommands): first, under This page.
  const page: Item[] = src.pageCommands.map((c) => ({
    id: `page-${c.id}`,
    group: 'This page',
    label: c.label,
    match: c.match,
    keys: c.keys,
    icon: c.icon,
    run: c.run,
  }));

  const nav: Item[] = [...mainNav, settingsNav].map((n) => ({
    id: `nav-${n.id}`,
    group: 'Go to',
    label: n.label,
    keys: n.keys,
    icon: n.icon,
    run: go(n.path),
  }));

  const actions: Item[] = [
    { id: 'new-run', group: 'Actions', label: 'New training run', keys: 'N', icon: Plus, run: go('/runs/new') },
    { id: 'compare', group: 'Actions', label: 'Compare runs', icon: GitCompare, run: go('/runs/compare') },
    // Agent labeling, under Datasets (no nav item of its own).
    { id: 'label', group: 'Actions', label: 'Label images', match: ['Labeling jobs', 'Claude Code', 'Codex', 'Antigravity'], icon: Tags, run: go('/datasets/label') },
    // Not on a server whose health said it cannot label (`folo serve --mock`): its form could only fail (#324).
    ...(src.lacksLabel ? [] : [{ id: 'label-new', group: 'Actions', label: 'New labeling job', icon: Plus, run: go('/datasets/label?new=1') }]),
    // Vision benchmarks: offered where the server has them (its list answered).
    ...(src.benchesOk
      ? [{ id: 'bench-new', group: 'Actions', label: 'New benchmark', match: ['Benchmark a model', 'Vision benchmark'], icon: Plus, run: go('/benchmarks?new=1') }]
      : []),
    {
      id: 'theme',
      group: 'Actions',
      label: `Switch theme (now ${prefs.theme})`,
      match: ['Toggle theme', 'Dark theme', 'Light theme'],
      icon: SunMoon,
      run: () => prefs.cycleTheme(),
    },
    // One per theme, custom ones too (design/THEMES.md, Choosing a theme): for a query only ("theme", "midnight"),
    // so seven or more never crowd the open palette. Picking one switches the mode under Dark or Light, not System.
    ...themeCommands(),
    // Touch-only devices have no keys to look up, and a phone or tablet held in the hand no sidebar to fold.
    ...(src.touchOnly
      ? []
      : [
          {
            id: 'sidebar',
            group: 'Actions',
            label: sidebar.collapsed ? 'Expand sidebar' : 'Collapse sidebar',
            keys: '[',
            icon: PanelLeft,
            run: () => sidebar.toggle(),
          },
          { id: 'shortcuts', group: 'Actions', label: 'Keyboard shortcuts', keys: '?', icon: Keyboard, run: () => src.onshortcuts() },
        ]),
  ];

  // Settings' sections, by name and by what they hold ("cloud", "theme", "server URL"): for a query only.
  // Cloud only where Settings has it (a server with the cloud feature).
  const settings: Item[] = SETTINGS_SECTIONS.filter((x) => !(src.touchOnly && x.id === 'shortcuts') && (x.id !== 'cloud' || src.cloud)).map((x) => ({
    id: `settings-${x.id}`,
    group: 'Settings',
    label: x.label,
    match: x.match,
    scoped: true,
    icon: SettingsIcon,
    run: () => {
      const path = `/settings?section=${x.id}`;
      if (router.route.name === 'settings' && router.location.name === 'settings' && router.route.query.get('section') === x.id) {
        if (location.hash !== `#${path}`) router.go(path, { replace: true });
        window.dispatchEvent(new CustomEvent('folo:settings-section', { detail: x.id }));
        return;
      }
      router.go(path);
    },
  }));

  const runs: Item[] = src.runs.map((x) => ({
    id: `run-${x.id}`,
    entityId: x.id,
    group: 'Runs',
    label: x.name,
    hint: [failureCause(x) ? 'Failed, out of GPU memory' : statusWord(x), { ident: x.config.arch }, `epoch ${x.epoch}/${x.epochs}`],
    icon: Activity,
    status: x.status,
    // With no query: navigation, the actions and the runs in flight. Read the status enum, never the display word.
    scoped: !ACTIVE.includes(x.status),
    run: go(`/runs/${x.id}`),
  }));

  const datasets: Item[] = src.datasets.map((x) => ({
    id: `ds-${x.id}`,
    entityId: x.id,
    group: 'Datasets',
    label: x.name,
    hint: [count(x.classes.length, 'class'), count(x.splits.train + x.splits.val + x.splits.test, 'image')],
    icon: Images,
    scoped: true,
    run: go(`/datasets/${x.id}`),
  }));

  // A model's score with what it is on (lib/scoreSummary): "mAP50-95 0.604 on fasteners", "mAP50-95 0.701 · split unknown".
  const runById = new Map(src.runs.map((x) => [x.id, x]));
  const dsById = new Map(src.datasets.map((x) => [x.id, x]));
  const benches: Item[] = src.benches.map((x) => ({
    id: `bench-${x.id}`,
    entityId: x.id,
    group: 'Benchmarks',
    label: x.name,
    hint: [count(x.images, 'image'), count(x.entries, 'entry', 'entries'), ...(x.best ? [`best ${x.best.score.toFixed(1)}`] : [])],
    icon: Target,
    scoped: true,
    run: go(`/benchmarks/${x.id}`),
  }));

  const benchName = new Map(src.benches.map((x) => [x.id, x.name]));
  const jobs: Item[] = src.labelJobs.map((x) => ({
    id: `label-${x.id}`,
    entityId: x.id,
    group: 'Labeling jobs',
    label: x.name,
    match: x.classes,
    hint: [x.status === 'idle' ? `${x.counts.labeled.toLocaleString('en-US')} of ${count(x.counts.images, 'image')} labeled` : 'Labeling', count(x.classes.length, 'class')],
    icon: Tags,
    scoped: true,
    run: go(`/datasets/label/${x.id}`),
  }));

  const loops: Item[] = src.loops.map((x) => ({
    id: `loop-${x.id}`,
    entityId: x.id,
    group: 'Loops',
    label: x.name,
    hint: [`${x.status.charAt(0).toUpperCase()}${x.status.slice(1)}`, `on ${benchName.get(x.bench_id) ?? 'a benchmark'}`],
    icon: Repeat,
    scoped: true,
    run: go(`/benchmarks/${x.bench_id}/loops/${x.id}`),
  }));

  const models: Item[] = src.models.map((x) => ({
    id: `model-${x.id}`,
    entityId: x.id,
    group: 'Models',
    label: x.name,
    hint: [{ ident: x.arch }, x.metrics ? scoreText(x.metrics.map50_95, scoreContext(x, runById, dsById)) : 'not scored'],
    icon: Box,
    scoped: true,
    run: go(`/models/${x.id}`),
  }));

  // Actions that take what the query names: train on a dataset, try a model. Ranked with the entities themselves.
  const onEntity: Item[] = [
    ...src.datasets.map((x) => ({
      id: `act-train-${x.id}`,
      group: 'Actions',
      label: `New run on ${x.name}`,
      hint: 'Train a detector on this dataset',
      describes: true,
      match: [x.name, `Train ${x.name}`],
      scoped: true,
      icon: Plus,
      run: go(`/runs/new?dataset=${encodeURIComponent(x.id)}`),
    })),
    ...src.models.map((x) => ({
      id: `act-try-${x.id}`,
      group: 'Actions',
      label: `Open ${x.name} in Playground`,
      hint: 'Run it on an image',
      describes: true,
      match: [x.name, `Playground ${x.name}`, `Try ${x.name}`],
      scoped: true,
      icon: ScanSearch,
      run: go(`/playground?model=${encodeURIComponent(x.id)}`),
    })),
  ];

  return [...page, ...nav, ...actions, ...settings, ...onEntity, ...runs, ...datasets, ...models, ...benches, ...jobs, ...loops];
}
