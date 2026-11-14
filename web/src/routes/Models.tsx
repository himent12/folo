// The Models list (`#/models`): every model, grouped by the dataset and split its score is on (published weights and
// imported files in groups of their own), each group ranked on its own. The parts are in ./models/.
import Box from '$icons/box';
import Plus from '$icons/plus';
import Search from '$icons/search';
import { computed, css, For, signal } from 'fw';
import { hub } from '$lib/api/index';
import { bestMark, bestWord } from '$lib/bestMark';
import { modelsResource, refreshOnModelsChange } from '$lib/api/models';
import { Resource } from '$lib/api/resource';
import type { Dataset, Model, Run } from '$lib/api/types';
import { ago, bytes, compact, date, metric, sizeParts } from '$lib/format';
import { prefs } from '$lib/prefs';
import { href, router } from '$lib/router';
import Page from '$lib/shell/Page';
import PhoneSections from '$lib/shell/PhoneSections';
import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Input from '$lib/ui/Input';
import Panel from '$lib/ui/Panel';
import Versus from './models/Versus';
import { isLast } from './model/weightsEpoch';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import Skeleton from '$lib/ui/Skeleton';
import SortHeader from '$lib/ui/SortHeader';
import TableState from '$lib/ui/TableState';
import Ident from '$lib/ui/Ident';
import Meta from '$lib/ui/Meta';
import { rowNav } from '$lib/ui/rowNav';
import { unlike, versusYolo26n } from '$lib/benchmarks';
import { filterLabel, groupKey, groupLabel, rankKey, scoreContext, splitLabel, type ScoreContext } from '$lib/scoreSummary';
import { loadThroughput, measuredLatency, measuredTraining } from '$lib/throughput';

type Key = 'name' | 'arch' | 'classes' | 'bytes' | 'map50_95' | 'map50' | 'source' | 'created';
interface Col {
  key: Key;
  label: string;
  num?: boolean;
  get: (m: Model) => number | string;
  hide?: 'md' | 'lg';
  /** Fixed width (header label, sort mark and padding fit in it); the name column takes what is left. */
  w?: string;
}
interface EvalSet {
  key: string;
  label: string;
  filter: string;
  ctx: ScoreContext;
  ranked: boolean;
}
type Group = EvalSet & { rows: Model[] };

/** Loading: placeholder dataset groups (models per group), so the list keeps its grouped rhythm when it arrives. */
const SK_GROUPS = [2, 3, 2];
// A score compares only with scores on the same dataset and split (lib/scoreSummary): a trained model is scored on
// its run's dataset, on its validation split or, when the dataset has none, its training split. The list is grouped
// by that, each group headed by what its scores are on ("Validation split of fasteners") and ranked on its own.
// Published weights and imported files have no known split here: a group each, after the datasets, never ranked.
// The filter picks a group of the list: an evaluation dataset, or a source with none ("Published weights",
// "Imported"), so it is "Group", never "Dataset". Its accessible name says what it filters; the phone picker's keeps
// its visible word first (WCAG 2.5.3, label in name).
const GROUP_NAME = 'Filter by evaluation dataset or source';
const GROUP_LABEL = `Group: ${GROUP_NAME.charAt(0).toLowerCase()}${GROUP_NAME.slice(1)}`;
/** Groups that are not ranked (published, imported, unknown split): no "best", after the ranked ones, in this order. */
const otherRank = (k: string) => (k === '~published' ? 1 : k === '~imported' ? 2 : k.startsWith('~') ? 3 : 0);
const splitRank = (s: string | null) => (s === 'val' ? 0 : s === 'test' ? 1 : 2);
const groupOrder = (a: { key: string; ctx: ScoreContext }, b: { key: string; ctx: ScoreContext }) =>
  otherRank(a.key) - otherRank(b.key) || (a.ctx.datasetName ?? '').localeCompare(b.ctx.datasetName ?? '') || splitRank(a.ctx.split) - splitRank(b.ctx.split);
const hideCls = (c: Col) => (c.hide ? `hide-${c.hide}` : '');
/** One line for the collapsed engine benchmarks: what is measured and how many rows are not like for like (a win
 *  count would add up unlike conditions; each row's own qualifier says which way it goes). */
const benchUnlike = versusYolo26n.filter((v) => unlike[v.metric]).length;
const range = (n: number) => Array.from({ length: n }, (_, i) => i);

export default function Models() {
  setCrumbs({ label: 'Models' });

  // Kept current: a run that ends or is deleted, or the tab shown again, fetches the list again (lib/api/models).
  const models = modelsResource();
  // No models yet: the empty state's New run is the page's create action, so the top bar's steps aside.
  ownCreateAction(() => models.data?.length === 0);
  // Only for source-run names and evaluation datasets; the list still renders if these fail.
  const runs = new Resource((s) => hub.api.runs(s));
  refreshOnModelsChange(runs);
  const datasets = new Resource((s) => hub.api.datasets(s));
  // Where imported .folo files go, for the empty state (the server's data dir, when it says; the mock has none).
  const system = new Resource((s) => hub.api.system(s));
  const modelsDir = computed(() => (system.data?.data_dir ? `\`${system.data.data_dir}/models\`` : "the models folder of the server's data directory"));
  // This server's measured speed, for the engine benchmarks' "Measured on this server" lines (null on older servers).
  const throughput = new Resource((s) => loadThroughput(hub.api, s));
  const benchHere = computed(() => [measuredLatency(throughput.data, 'folo-n'), measuredTraining(throughput.data, 'folo-n')].filter(Boolean).length);

  const runById = computed(() => new Map<string, Run>((runs.data ?? []).map((r) => [r.id, r])));
  /** A run's `-last` model when the run's best epoch was its last: the same weights as its `-best` (model/weightsEpoch). */
  const sameAsBest = (m: Model) => {
    const r = isLast(m) && m.run_id ? runById.get().get(m.run_id) : undefined;
    return !!r && r.status === 'completed' && r.best?.epoch === r.epochs;
  };
  const sourceName = (m: Model) => (m.run_id ? (runById.get().get(m.run_id)?.name ?? m.run_id) : '');

  const dsById = computed(() => new Map<string, Dataset>((datasets.data ?? []).map((d) => [d.id, d])));
  // Each model's evaluation set, worked out once per change of the lists (legacy worked it out again for the groups,
  // the search and the filter's options, two or three times per model per keystroke).
  const evals = computed(() => {
    const runMap = runById.get();
    const dsMap = dsById.get();
    const out = new Map<string, EvalSet>();
    for (const m of models.data ?? []) {
      const ctx = scoreContext(m, runMap, dsMap);
      out.set(m.id, { key: groupKey(ctx), label: groupLabel(ctx), filter: filterLabel(ctx), ctx, ranked: rankKey(ctx) !== null });
    }
    return out;
  });
  const evalSet = (m: Model) => evals.get().get(m.id)!;

  // The columns of design/UNIFIED.md §4 (Models): Params, GFLOPs and where a model started from are on its own page.
  // Source run: a run's name ends in what tells it from its siblings ("shapes-hard-s" beside "shapes-hard-m"), so it is
  // never cut (DESIGN_DIRECTION §2.3); a longer one wraps at its hyphens. A longer model name is cut with an ellipsis
  // (.name-cell), whole in its tooltip. Tablets drop Classes, Size and Source run (the phone rows keep their facts).
  // Size as Desktop's: each cell its number with its own unit ("37.0 MiB", "1.2 GiB"), the unit one step smaller in
  // tertiary, the heading plain "Size"; sorted by bytes (prefs.units: MiB or MB).
  const cols = computed<Col[]>(() => [
    { key: 'name', label: 'Model', get: (m) => m.name.toLowerCase() },
    { key: 'arch', label: 'Architecture', get: (m) => m.arch, w: '112px' },
    { key: 'classes', label: 'Classes', num: true, get: (m) => m.classes.length, hide: 'md', w: '88px' },
    { key: 'bytes', label: 'Size', num: true, get: (m) => m.bytes, hide: 'md', w: '96px' },
    { key: 'map50_95', label: 'mAP50-95', num: true, get: (m) => m.metrics?.map50_95 ?? -1, w: '136px' },
    { key: 'map50', label: 'mAP50', num: true, get: (m) => m.metrics?.map50 ?? -1, w: '88px' },
    { key: 'source', label: 'Source run', get: (m) => sourceName(m) || '~', hide: 'md', w: '144px' },
    { key: 'created', label: 'Created', num: true, get: (m) => Date.parse(m.created), w: '112px' },
  ]);

  // Default: best first within each dataset.
  const sortKey = signal<Key>('map50_95');
  const sortDir = signal<'asc' | 'desc'>('desc');
  const query = signal('');
  const dsFilter = signal('all');

  function sortBy(c: Col) {
    if (sortKey.peek() === c.key) sortDir.set(sortDir.peek() === 'asc' ? 'desc' : 'asc');
    else {
      sortKey.set(c.key);
      sortDir.set(c.num ? 'desc' : 'asc');
    }
  }

  const matches = (m: Model, q: string) => !q || `${m.name} ${m.arch} ${m.id} ${sourceName(m)} ${evalSet(m).ctx.datasetName ?? ''}`.toLowerCase().includes(q);

  /** Groups in display order (datasets A–Z, validation first, then published, imported, unknown), rows sorted. */
  const groups = computed(() => {
    const q = query.get().trim().toLowerCase();
    const all = cols.get();
    const col = all.find((c) => c.key === sortKey.get()) ?? all[0]!;
    const sign = sortDir.get() === 'asc' ? 1 : -1;
    const filter = dsFilter.get();
    const by = new Map<string, Group>();
    for (const m of models.data ?? []) {
      const e = evalSet(m);
      if ((filter !== 'all' && e.key !== filter) || !matches(m, q)) continue;
      if (!by.has(e.key)) by.set(e.key, { ...e, rows: [] });
      by.get(e.key)!.rows.push(m);
    }
    const out = [...by.values()].sort(groupOrder);
    for (const g of out)
      g.rows.sort((a, b) => {
        const x = col.get(a);
        const y = col.get(b);
        const d = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y));
        return d * sign || a.name.localeCompare(b.name);
      });
    return out;
  });
  const shownCount = computed(() => groups.get().reduce((a, g) => a + g.rows.length, 0));
  /** Rows to sort, or their skeleton: the sort mark shows only then. */
  const hasRows = computed(() => (models.data ? shownCount.get() > 0 : !models.error));
  /** Models to filter, or on their way: the toolbar is live (else it stays in place, disabled). */
  const hasData = computed(() => (models.data ? models.data.length > 0 : !models.error));

  /**
   * Each comparable group's best model (highest mAP50-95), marked with tertiary "best" text whatever the sort, "joint
   * best" on every model of an exact tie (lib/bestMark): within a group the scores compare, across groups they do not,
   * so no mark spans groups (and none on an unranked one). A group whose best differs from another score only past
   * three decimals shows its scores with as many as tell them apart.
   */
  const best = computed(() => {
    const marks = new Map<string, string>();
    const decimals = new Map<string, number>();
    for (const g of groups.get()) {
      if (!g.ranked || g.rows.length < 2) continue;
      // A run's last checkpoint that is its best epoch is the same weights: it is not a second contender.
      const m = bestMark(g.rows.map((r) => (sameAsBest(r) ? null : r.metrics?.map50_95)));
      for (const i of m.best) marks.set(g.rows[i]!.id, bestWord(m));
      for (const r of g.rows) if (sameAsBest(r)) marks.set(r.id, 'same as best');
      if (m.decimals > 3) for (const r of g.rows) decimals.set(r.id, m.decimals);
    }
    return { marks, decimals };
  });
  /** A model's mAP50-95 as the list shows it: three decimals, more where its group needs them to tell the best apart. */
  const score = (m: Model) => {
    const d = best.get().decimals;
    return m.metrics && d.has(m.id) ? m.metrics.map50_95.toFixed(d.get(m.id)) : metric(m.metrics?.map50_95);
  };

  const dsOptions = computed(() => {
    const keys = new Map<string, EvalSet & { n: number }>();
    for (const m of models.data ?? []) {
      const e = evalSet(m);
      const k = keys.get(e.key) ?? { ...e, n: 0 };
      k.n++;
      keys.set(e.key, k);
    }
    return [
      { value: 'all', label: 'All', meta: String(models.data?.length ?? 0) },
      ...[...keys.values()].sort(groupOrder).map((k) => ({ value: k.key, label: k.filter, meta: String(k.n) })),
    ];
  });
  const pickOptions = computed(() => dsOptions.get().map((o) => ({ value: o.value, label: `${o.label} · ${o.meta}` })));
  // The header's facts line (design/UNIFIED.md §2, Page header): what the page holds, "7 models · 2 datasets ·
  // 52.4 MiB", whatever the filter; a narrowed list says what it shows under it.
  const facts = computed(() => {
    const all = models.data;
    if (!all?.length) return null;
    const sets = new Set(all.map((m) => evalSet(m).ctx.datasetId).filter(Boolean)).size;
    const size = all.reduce((a, m) => a + m.bytes, 0);
    return [`${all.length} ${all.length === 1 ? 'model' : 'models'}`, sets ? `${sets} ${sets === 1 ? 'dataset' : 'datasets'}` : '', bytes(size, prefs.units)]
      .filter(Boolean)
      .join(' · ');
  });
  /** Medium weight goes to the name and the score the list is ranked by; specs stay regular. */
  const strongScore = computed<Key>(() => (sortKey.get() === 'map50' ? 'map50' : 'map50_95'));

  // What a group's scores are on: "Validation split of fasteners" (the dataset linked), or an unranked group's name.
  const groupHead = (g: () => Group) => (
    <>
      {g().ranked && g().ctx.split && g().ctx.datasetId ? (
        <span>
          {splitLabel(g().ctx.split!)} of{' '}
          <a class="g-name" href={href(`/datasets/${g().ctx.datasetId}`)}>
            {g().ctx.datasetName}
          </a>
        </span>
      ) : (
        <span>{g().label}</span>
      )}{' '}
      <span class="faint num">{g().rows.length}</span>{' '}
      {!g().ranked ? <span class="faint">not ranked</span> : null}
    </>
  );

  const bestText = (m: () => Model) =>
    best.get().marks.has(m().id) ? (
      <span class="best" title={sameAsBest(m()) ? 'Best and last are the same epoch: the same weights' : undefined}>
        {best.get().marks.get(m().id)}
      </span>
    ) : null;

  return (
    <Page
      title="Models"
      // What the page holds (how to import a .folo file is the empty state's; ranking is the group heads').
      sub={facts.get() ? () => <span class="num">{facts.get()}</span> : undefined}
      // Phones: one line, so the list starts 16 px under it.
      subPhone={facts.get() ? () => <span class="num">{facts.get()}</span> : undefined}
    >
      {/* The list toolbar (base.css): one row on laptops and up; tablets and phones put the filter field on a row of its
          own. It is always here, so the table never moves between states (DESIGN_DIRECTION §2.8): over an empty or
          failed list it is disabled, "All" alone and no counts. */}
      <PhoneSections
        label="Models and benchmarks"
        items={[
          { label: 'Models', path: '/models', current: true },
          { label: 'Benchmarks', path: '/benchmarks', current: false },
        ]}
      />{' '}
      <div class="list-toolbar">
        <div class="lt-filter">
          {models.data?.length ? (
            <>
              <div class="seg" data-scrolls>
                <Segmented label={GROUP_NAME} bind:value={dsFilter} options={dsOptions.get()} />
              </div>{' '}
              {/* Phones: every dataset in a native picker; a strip that scrolls sideways hides the last ones. */}
              <div class="ds-pick">
                <label for="models-ds">Group</label>{' '}
                <Select id="models-ds" label={GROUP_LABEL} bind:value={dsFilter} options={pickOptions.get()} />
              </div>
            </>
          ) : models.data || models.error ? (
            <>
              <div class="seg">
                <Segmented label={GROUP_NAME} value="all" disabled options={[{ value: 'all', label: 'All' }]} />
              </div>{' '}
              <div class="ds-pick">
                <label for="models-ds">Group</label>{' '}
                <Select id="models-ds" label={GROUP_LABEL} value="all" disabled options={[{ value: 'all', label: 'All' }]} />
              </div>
            </>
          ) : (
            // The same two controls at their loaded size: the dataset strip (a picker on phones).
            <>
              <div class="seg ctl-sk" style:width="320px">
                <Skeleton h="100%" r="var(--radius-md)" />
              </div>{' '}
              <div class="ds-pick" aria-hidden="true">
                <span class="ds-label">Group</span>{' '}
                <span class="ctl-sk">
                  <Skeleton h="100%" r="var(--radius-md)" />
                </span>
              </div>
            </>
          )}
        </div>{' '}
        <div class="lt-search">
          <Input bind:value={query} icon={Search} placeholder="Filter by name, architecture or run" aria-label="Filter models" type="search" disabled={!hasData.get()} />
        </div>
      </div>{' '}
      <Panel pad="none">
        {/* With no rows (empty, no match, an error) the frame drops its column heads and holds the state alone. */}
        <div class="scroll" class:state={!hasRows.get()}>
          <table class="table models" aria-busy={!models.data} aria-label="Models by evaluation dataset" use:rowNav>
            {hasRows.get() ? (
              <thead>
                <tr>
                  <For each={cols.get()} key={(c) => c.key}>
                    {(c) => (
                      <SortHeader
                        label={c().label}
                        class={hideCls(c())}
                        align={c().num ? 'right' : 'left'}
                        width={c().w}
                        dir={sortKey.get() === c().key ? sortDir.get() : null}
                        disabled={!models.data}
                        onsort={() => sortBy(c())}
                      />
                    )}
                  </For>
                </tr>
              </thead>
            ) : null}
            {models.error && !models.data ? (
              <tbody>
                <TableState cols={cols.get().length}>
                  <ErrorState error={models.error} retry={() => models.refresh()} title="Could not load models" />
                </TableState>
              </tbody>
            ) : models.data?.length === 0 ? (
              <tbody>
                <TableState cols={cols.get().length}>
                  <EmptyState
                    icon={Box}
                    title="No models yet"
                    description={`A training run saves its best weights here when it finishes. To import a .folo file, copy it into ${modelsDir.get()} on the server's machine.`}
                    // The page's only action (its header has none), so primary.
                    actions={() => (
                      <Button variant="primary" icon={Plus} href={href('/runs/new')}>
                        New run
                      </Button>
                    )}
                  />
                </TableState>
              </tbody>
            ) : models.data && shownCount.get() === 0 ? (
              <tbody>
                <TableState cols={cols.get().length}>
                  <EmptyState
                    icon={Search}
                    title="No models match"
                    description="Try another dataset or search term."
                    actions={() => (
                      <Button
                        size="sm"
                        onclick={() => {
                          dsFilter.set('all');
                          query.set('');
                        }}
                      >
                        Clear filters
                      </Button>
                    )}
                  />
                </TableState>
              </tbody>
            ) : !models.data ? (
              // The loaded rhythm: dataset groups (a header row, then their models), with placeholder content.
              <For each={SK_GROUPS.map((n, gi) => ({ n, gi }))} key={(g) => g.gi}>
                {(g) => (
                  <tbody aria-hidden="true">
                    <tr class="group">
                      <th colspan={cols.get().length}>
                        <Skeleton h="10px" w="96px" />
                      </th>
                    </tr>
                    <For each={range(g().n)} key={(i) => i}>
                      {() => (
                        <tr>
                          <For each={cols.get()} key={(c) => c.key}>
                            {(c) => (
                              <td class={hideCls(c())} class:r={!!c().num}>
                                <Skeleton h="12px" w={c().key === 'name' ? '112px' : '40px'} />
                              </td>
                            )}
                          </For>
                        </tr>
                      )}
                    </For>
                  </tbody>
                )}
              </For>
            ) : (
              <For each={groups.get()} key={(g) => g.key}>
                {(g) => (
                  <tbody>
                    <tr class="group">
                      <th colspan={cols.get().length} scope="colgroup">
                        {groupHead(g)}
                      </th>
                    </tr>
                    <For each={g().rows} key={(m) => m.id}>
                      {(m) => {
                        return (
                          <tr class="link" data-row data-id={m().id} aria-label={m().name} onclick={() => router.go(`/models/${m().id}`)}>
                            <td class="name-cell">
                              <a class="name truncate" href={href(`/models/${m().id}`)} title={m().about ? `${m().name} · ${m().about}` : m().name} onclick={(e) => e.stopPropagation()}>
                                {m().name}
                              </a>
                            </td>
                            <td class="spec">
                              <Ident value={m().arch} />
                            </td>
                            <td class="r spec hide-md" title={m().classes.slice(0, 12).join(', ') + (m().classes.length > 12 ? ', …' : '')}>
                              {m().classes.length}
                            </td>
                            <td class="r spec hide-md">
                              {sizeParts(m().bytes, prefs.units)[0]}{' '}
                              <span class="size-unit">{sizeParts(m().bytes, prefs.units)[1]}</span>
                            </td>
                            <td class="r score" class:strong={strongScore.get() === 'map50_95'}>
                              {bestText(m)}
                              {score(m())}
                            </td>
                            <td class="r score" class:strong={strongScore.get() === 'map50'}>
                              {metric(m().metrics?.map50)}
                            </td>
                            <td class="hide-md">
                              {m().run_id ? (
                                <a class="src" href={href(`/runs/${m().run_id}`)} title={sourceName(m())} onclick={(e) => e.stopPropagation()}>
                                  {sourceName(m())}
                                </a>
                              ) : (
                                <span class="faint">—</span>
                              )}
                            </td>
                            <td class="r spec" title={date(m().created)}>
                              {ago(m().created)}
                            </td>
                          </tr>
                        );
                      }}
                    </For>
                  </tbody>
                )}
              </For>
            )}
          </table>
        </div>{' '}
        {/* Phones: the same groups as two-line rows (name and one meta line, mAP on the right). */}
        <div class="rows">
          {!models.data && !models.error ? (
            <For each={SK_GROUPS.map((n, gi) => ({ n, gi }))} key={(g) => g.gi}>
              {(g) => (
                <section aria-hidden="true">
                  <h2 class="m-group">
                    <Skeleton h="10px" w="96px" />
                  </h2>{' '}
                  <ul>
                    <For each={range(g().n)} key={(i) => i}>
                      {() => (
                        <li>
                          <span class="row-sk">
                            <span class="m-main">
                              <span class="l-name">
                                <Skeleton w="55%" h="14px" />
                              </span>
                              <span class="l-meta">
                                <Skeleton w="75%" h="10px" />
                              </span>
                            </span>{' '}
                            <span class="m-side">
                              <span class="l-name">
                                <Skeleton w="40px" h="14px" />
                              </span>
                            </span>
                          </span>
                        </li>
                      )}
                    </For>
                  </ul>
                </section>
              )}
            </For>
          ) : null}
          <For each={groups.get()} key={(g) => g.key}>
            {(g, gi) => (
              <section aria-label={g().label}>
                {/* The value column is named once, on the first group's header, not under every value. */}
                <h2 class="m-group">
                  <span class="g-head">{groupHead(g)}</span>{' '}
                  {gi() === 0 ? <span class="m-col">mAP50-95</span> : null}
                </h2>{' '}
                <ul>
                  <For each={g().rows} key={(m) => m.id}>
                    {(m) => (
                      <li>
                        <a href={href(`/models/${m().id}`)} title={m().about}>
                          <span class="m-main">
                            <span class="name truncate">{m().name}</span>{' '}
                            <span class="m-meta num">
                              <Meta line>
                                <Ident value={m().arch} />
                                <span>{m().imgsz} px</span>
                                <span>{compact(m().params)}</span>
                                <span>{bytes(m().bytes, prefs.units)}</span>
                              </Meta>
                            </span>
                          </span>{' '}
                          <span class="m-side">
                            <span class="num map">
                              {bestText(m)}
                              <span class="sr-only">mAP50-95 </span>
                              {score(m())}
                            </span>
                          </span>
                        </a>
                      </li>
                    )}
                  </For>
                </ul>
              </section>
            )}
          </For>
        </div>
      </Panel>{' '}
      {/* No totals line under the table (they are the header's facts): only a narrowed list says what it shows. */}
      {models.data && shownCount.get() < models.data.length ? (
        <p class="list-foot num">
          {shownCount.get()} of {models.data.length} {models.data.length === 1 ? 'model' : 'models'}
        </p>
      ) : null}{' '}
      {/* Only with models to look at. The engine comparison is reference: one line until it is asked for. */}
      {models.data?.length ? (
        <details class="bench">
          <summary>
            <span class="chev" aria-hidden="true"></span>{' '}
            <span class="b-title">Engine benchmarks</span>{' '}
            <span class="b-sum">
              {versusYolo26n.length} reference measurements · {benchUnlike} under different conditions
              {benchHere.get() ? <>{' '}· {benchHere.get()} measured on this server</> : null}
            </span>
          </summary>{' '}
          <div class="b-body">
            <Versus table={throughput.data} />
          </div>
        </details>
      ) : null}
    </Page>
  );
}

css`
  .seg {
    max-width: 100%;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .ds-pick {
    display: none;
    align-items: center;
    gap: var(--space-3);
    width: 100%;
  }
  .ds-pick label,
  .ds-label {
    flex-shrink: 0;
    color: var(--text-tertiary);
  }
  .ds-pick :global(.select),
  .ds-pick .ctl-sk {
    flex: 1;
  }
  .ctl-sk {
    display: block;
    max-width: 100%;
    height: var(--control-md);
  }
  @media (pointer: coarse) {
    .ctl-sk {
      height: 44px;
    }
  }
  .scroll {
    overflow-x: auto;
  }
  /* A 32 px section header per evaluation dataset: the groups are ranked separately. */
  /* Placeholders in number columns sit where the numbers will: at the right. */
  td.r :global(.sk) {
    margin-left: auto;
  }
  .group th {
    height: 32px;
    padding: 0 var(--space-4);
    background: var(--bg-raised);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    text-align: left;
    border-bottom: 1px solid var(--line-subtle);
  }
  .group th .faint {
    margin-left: var(--space-2);
    font-weight: var(--weight-regular);
  }
  .g-name:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  /* The name takes what the fixed columns leave, cut with an ellipsis (whole in its tooltip): a 68-character name made
     the table 1,315 px in its 1,158 px frame, Started from and Created out of view. Never under 160 px. */
  .name-cell {
    max-width: 0;
    min-width: 160px;
  }
  .name-cell .name {
    display: block;
  }
  /* The base table drops the last row's border; inside a group it still separates the next header. */
  .models tbody:not(:last-child) tr:last-child td {
    border-bottom: 1px solid var(--line-subtle);
  }
  /* Specs (and the score the list is not ranked by) are reference: regular and secondary, so the name and the ranked
     score lead the row. */
  .spec,
  .score {
    color: var(--text-secondary);
  }
  .score.strong {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  /* A size's own unit, one step under the cell's figure in tertiary (design/README.md, Type), as Desktop's "37.0 MiB". */
  .size-unit {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* The group's top model: a word in tertiary, never a colour (scores compare only within a dataset). */
  .best {
    margin-right: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .src,
  td :global(.ident.src) {
    color: var(--text-secondary);
  }
  /* A run's name is never cut: its suffix is what tells it from its siblings ("shapes-hard-s" beside "shapes-hard-m",
     DESIGN_DIRECTION §2.3). The column holds the usual names on one line (the widths above); a longer one wraps after
     a hyphen and the row grows a line, never an ellipsis and never wider than its column. */
  a.src {
    display: inline-block;
    max-width: 100%;
    white-space: normal;
    overflow-wrap: break-word;
  }
  /* The engine benchmarks: a collapsed row with a one-line summary; the comparison opens beneath it. */
  .bench {
    min-width: 0;
  }
  .bench summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: 40px;
    padding: var(--space-2) var(--space-4);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
    cursor: pointer;
    list-style: none;
    transition: border-color var(--dur-fast) var(--ease-standard);
  }
  .bench summary:hover {
    border-color: var(--line-strong);
  }
  .bench summary::-webkit-details-marker {
    display: none;
  }
  .bench summary:focus-visible {
    border-radius: var(--radius-lg);
  }
  .chev {
    flex-shrink: 0;
    width: 6px;
    height: 6px;
    margin: 0 var(--space-1) 0 2px;
    border-right: 1.5px solid var(--text-tertiary);
    border-bottom: 1.5px solid var(--text-tertiary);
    transform: rotate(-45deg);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .bench[open] .chev {
    transform: rotate(45deg);
  }
  .b-title {
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    font-weight: var(--weight-medium);
    white-space: nowrap;
  }
  .b-sum {
    min-width: 0;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* Open: the summary row becomes the head of one frame, the comparison panel its body. */
  .bench[open] summary {
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
  }
  /* Measurement names and their notes are sentences: they wrap (table cells default to one line). */
  .b-body :global(td:first-child) {
    white-space: normal;
  }
  .b-body > :global(.panel) {
    border-top: 0;
    border-top-left-radius: 0;
    border-top-right-radius: 0;
  }
  @media (pointer: coarse) {
    .bench summary {
      min-height: 44px;
    }
  }
  .map {
    font-weight: var(--weight-medium);
  }
  .rows {
    display: none;
  }
  @media (max-width: 767px) {
    /* Phones show the grouped rows below; the table frame stays only to hold an empty or error state (no column heads at any width). */
    .scroll:not(.state) {
      display: none;
    }
    /* The summary wraps under the title, on the title's edge (past the chevron). */
    .b-sum {
      flex-basis: 100%;
      padding-left: 20px;
    }
    .bench summary {
      flex-wrap: wrap;
      row-gap: 0;
    }
    .rows {
      display: block;
    }
    .seg {
      display: none;
    }
    .ds-pick {
      display: flex;
    }
    .m-group {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      height: 32px;
      padding: 0 var(--space-4);
      background: var(--bg-raised);
      border-bottom: 1px solid var(--line-subtle);
      font-size: var(--type-sm-size);
      font-weight: var(--weight-medium);
      color: var(--text-secondary);
    }
    section + section .m-group {
      border-top: 1px solid var(--line-subtle);
    }
    section:first-child .m-group {
      border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    }
    .m-group .faint {
      font-weight: var(--weight-regular);
    }
    .g-head {
      min-width: 0;
    }
    .g-head .faint {
      margin-left: var(--space-2);
    }
    li + li {
      border-top: 1px solid var(--line-subtle);
    }
    li a,
    .row-sk {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      min-height: 56px;
      padding: var(--space-2) var(--space-4);
    }
    li a:active {
      background: var(--bg-hover);
    }
    .m-main {
      flex: 1;
      min-width: 0;
      display: grid;
    }
    /* Skeleton lines in the line boxes of the name (20 px) and meta line (16 px). */
    .l-name,
    .l-meta {
      display: flex;
      align-items: center;
      height: 20px;
    }
    .l-meta {
      height: 16px;
    }
    .m-side .l-name {
      justify-content: flex-end;
    }
    .m-meta {
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-tertiary);
    }
    .m-side {
      display: grid;
      justify-items: end;
      flex-shrink: 0;
    }
    /* "best" over the score, not beside it: the side stays narrow, so the name's meta line never wraps (and rows with
       and without the word are the same height). */
    .m-side .map {
      display: grid;
      justify-items: end;
    }
    .m-side .best {
      margin-right: 0;
      line-height: var(--type-sm-line);
    }
    .m-col {
      margin-left: auto;
      font-weight: var(--weight-regular);
      color: var(--text-tertiary);
    }
  }
`;
