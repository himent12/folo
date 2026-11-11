// fw port of BenchCompare.svelte.
/**
 * Two to four finished entries of one benchmark side by side and overlaid (`#/benchmarks/{id}/compare?entries=a,b`):
 * what makes them not like for like, said first (fewer images, conditions left out, no confidences, unknown classes);
 * their scores and five axes against the first (the baseline, chosen in the table's head), the best of each row
 * marked; mAP50-95 with each 95% interval on one scale and a plain sentence on whether they are told apart; the
 * accuracy each keeps per condition as an aligned dot plot (who wins where); their reliability curves and their labels'
 * precision against coverage overlaid; and AP per class as aligned dots. Each entry keeps one hue and one name on
 * every chart (RunCompare's slots: sky, amber, magenta dashed, teal dotted; a dot plot's shapes), and its legend chip
 * keeps its line in front. Entries are added and removed here, the URL following, on phones too.
 */
import ArrowLeft from '$icons/arrow-left';
import ArrowUpRight from '$icons/arrow-up-right';
import GitCompare from '$icons/git-compare-arrows';
import Scale from '$icons/scale';
import Plus from '$icons/plus';
import X from '$icons/x';
import { computed, css, For, Index, MediaQuery } from 'fw';
import { ApiError } from '$lib/api/index';
import type { BenchEntry } from '$lib/api/types';
import { TARGET_PRECISION } from '$lib/bench/rules';
import XYChart from '$lib/charts/XYChart';
import { media } from '$lib/design/tokens';
import { conf, metric, num, pct } from '$lib/format';
import { href } from '$lib/router';
import Page from '$lib/shell/Page';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import NoValue from '$lib/ui/NoValue';
import Panel from '$lib/ui/Panel';
import Skeleton from '$lib/ui/Skeleton';
import Callout from './run/Callout';
import { diffAsPrinted, RELIABILITY_X, SLOT_PATTERN } from './bench/charts';
import DotRows from './bench/compare/DotRows';
import { CLS_SHOWN, compareState, type Row } from './bench/compare/state';
import IntervalPlot from './bench/IntervalPlot';
import { along, COMPARE_MAX, stale } from './bench/leaderboard';
import { capTicks } from '$lib/charts/domain';
import Unsupported from './bench/Unsupported';
import { intervalText, keptText, scoreText } from './bench/words.fw';

const BIN_MIDS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
const diff = diffAsPrinted;
const unitTick = (v: number) => (v === 0 || v === 1 ? String(v) : v.toFixed(2));

export default function BenchCompare() {
  const phone = new MediaQuery(media.phone, false);
  const {
    id,
    ids,
    live,
    bench,
    unsupported,
    picked,
    shown,
    results,
    setIds,
    candidates,
    slotOf,
    color,
    fill,
    keys,
    hovered,
    pinned,
    highlight,
    res,
    loaded,
    failedResult,
    caveats,
    nameOf,
    rows,
    old,
    best,
    makeBaseline,
    ivDomain,
    verdict,
    condRows,
    condDomain,
    noRobust,
    calSeries,
    noCal,
    calCursor,
    slSeries,
    slMarks,
    clsRows,
    allClasses,
    clsShown,
    clsDomain,
  } = compareState();

  const key = (eid: string) => <i class="key" style={{ '--c': fill(eid) }} data-pattern={SLOT_PATTERN[slotOf(eid)]} aria-hidden="true"></i>;

  const adder = () => (
    <label class="add">
      <Plus size={14} strokeWidth={1.75} aria-hidden="true" />{' '}
      <span aria-hidden="true">Add entry…</span>{' '}
      <select
        aria-label="Add an entry to the comparison"
        onchange={(ev) => {
          const v = (ev.currentTarget as HTMLSelectElement).value;
          if (v) setIds([...ids.peek(), v]);
        }}
      >
        <option value="">Add entry…</option>
        <For each={candidates.get()} key={(c) => c.id}>
          {(c) => (
            <option value={c().id}>
              {c().name} · score {scoreText(c().score!.total)}
            </option>
          )}
        </For>
      </select>
    </label>
  );

  const legendRead = (_c: () => number | null, hint: () => string | null) => (
    <>{hint() ? <span class="hint">{hint()}</span> : <span class="lr">The entries’ hues and names are the chips above.</span>}</>
  );

  const calRead = (c: () => number | null, hint: () => string | null) => (
    <>
      {hint() ? (
        <span class="hint">{hint()}</span>
      ) : c() === null ? (
        <span class="lr">Each line one entry’s bins: the share right against what it said; the dashed line is said = right.</span>
      ) : (
        <>
          <span class="t">
            Bin {num(c()! - 0.05, 1)}–{num(c()! + 0.05, 1)}
          </span>{' '}
          <For each={shown.get()} key={(e) => e.id}>
            {(e, i) => {
              const b = () => res.get()[i()]?.calibration?.bins.find((x) => Math.abs(x.lo + 0.05 - c()!) < 1e-6);
              return (
                <span class="rv">
                  {key(e().id)}
                  {b() && b()!.count ? `says ${num(b()!.confidence!, 2)}, right ${pct(b()!.precision!, 0)}` : 'none'}
                </span>
              );
            }}
          </For>
        </>
      )}
    </>
  );

  /** A value's difference from the baseline's, unless either is graded on an earlier version. */
  const diffOf = (row: () => Row, v: () => number, i: number) => {
    const o = old.get();
    return o[i] || o[0] ? null : diff(v(), row().values[0] ?? null, row().d);
  };

  const cmpCell = (row: () => Row, b: () => Set<number>, v: () => number | null, i: number) => (
    <td class="r num" class:top={b().has(i)}>
      {v() === null ? (
        <NoValue label="not measured" />
      ) : (
        <>
          {b().has(i) ? <span class="best">best</span> : null}
          {row().fmt(v()!)}{' '}
          {i > 0 ? <>{diffOf(row, v as () => number, i) ? <span class="dd">{diffOf(row, v as () => number, i)}</span> : null}</> : null}
        </>
      )}
    </td>
  );

  const sbsValue = (row: () => Row, b: () => Set<number>, e: () => BenchEntry, i: () => number) => {
    const v = () => row().values[i()] ?? null;
    return (
      <span class="sbs-v" class:top={b().has(i())}>
        <span class="sbs-n">
          {key(e().id)}
          <span class="truncate">{e().name}</span>
        </span>{' '}
        <span class="num">
          {v() === null ? (
            <NoValue label="not measured" />
          ) : (
            <>
              {row().fmt(v()!)}
              {i() > 0 ? (
                <>{diffOf(row, v as () => number, i()) ? <span class="dd">{diffOf(row, v as () => number, i())}</span> : null}</>
              ) : (
                <span class="dd">baseline</span>
              )}
              {b().has(i()) ? <span class="dd">best</span> : null}
            </>
          )}
        </span>
      </span>
    );
  };

  const compared = () => (
    <>
      {caveats.get().length ? (
        <Callout tone="neutral" icon={Scale} title="Not like for like" role="note">
          <ul class="cav">
            <Index each={caveats.get()}>{(c) => <li>{c().text}</li>}</Index>
          </ul>
        </Callout>
      ) : null}{' '}
      <Panel title="Side by side" meta="against the first, the baseline" pad="none">
        <div class="scroll">
          <table class="table cmp" aria-label="Scores and axes side by side">
            <thead>
              <tr>
                <th class="lead m-col">
                  <span class="sr-only">Measure</span>
                </th>
                <For each={shown.get()} key={(e) => e.id}>
                  {(e, i) => (
                    <th class="r e-col">
                      <span class="eh">
                        {key(e().id)}
                        <span class="eh-n">{e().name}</span>
                      </span>{' '}
                      <span class="role">
                        {i() === 0 ? (
                          'Baseline'
                        ) : (
                          <button type="button" class="swap" onclick={() => makeBaseline(e().id)}>
                            Make baseline
                          </button>
                        )}
                      </span>
                    </th>
                  )}
                </For>
              </tr>
            </thead>
            <tbody>
              <For each={rows.get()} key={(r) => r.key}>
                {(row) => {
                  const b = computed(() => best(row().key, row().values, row().d));
                  return (
                    <tr class:total={row().key === 'score'}>
                      <td class="lead">{row().label}</td>
                      <Index each={row().values}>{(v, i) => cmpCell(row, () => b.get(), v, i)}</Index>
                    </tr>
                  );
                }}
              </For>
            </tbody>
          </table>
        </div>
        {/* Phones: one block per measure, each entry's value (and its difference from the baseline) in a 2 × 2 grid. */}
        <ul class="sbs" aria-label="Scores and axes, entry by entry">
          <For each={rows.get()} key={(r) => r.key}>
            {(row) => {
              const b = computed(() => best(row().key, row().values, row().d));
              return (
                <li>
                  <span class="sbs-h">{row().label}</span>{' '}
                  <span class="sbs-g">
                    <For each={shown.get()} key={(e) => e.id}>
                      {(e, i) => sbsValue(row, () => b.get(), e, i)}
                    </For>
                  </span>
                </li>
              );
            }}
          </For>
        </ul>
      </Panel>{' '}
      <Panel title="mAP50-95 with 95% intervals" meta="unchanged images">
        {ivDomain.get() ? (
          <>
            <ul class="iv" aria-label="mAP50-95 and its 95% interval per entry, on one scale">
              <For each={shown.get()} key={(e) => e.id}>
                {(e) => (
                  <li class:dim={highlight.get() !== null && highlight.get() !== e().id}>
                    <span class="iv-n">
                      {key(e().id)}
                      <span class="truncate">{e().name}</span>
                    </span>{' '}
                    <span class="iv-v num">{metric(e().map50_95)}</span>{' '}
                    <span class="iv-plot">
                      {e().ci95 ? <IntervalPlot value={e().map50_95 ?? 0} ci={e().ci95!} domain={ivDomain.get()!} height={28} color={color(e().id)} /> : null}
                    </span>{' '}
                    <span class="iv-ci num">{e().ci95 ? intervalText(e().ci95!) : ''}</span>
                  </li>
                )}
              </For>{' '}
              <li class="iv-axis" aria-hidden="true">
                <span></span>
                <span></span>{' '}
                <span class="iv-ticks">
                  <For each={capTicks(ivDomain.get()![0], ivDomain.get()![1], 5)} key={(t) => t}>
                    {(t) => (
                      <span class="num" style:left={`${(along(ivDomain.get()!, t()) * 100).toFixed(2)}%`}>
                        {t().toFixed(2)}
                      </span>
                    )}
                  </For>
                </span>{' '}
                <span></span>
              </li>
            </ul>{' '}
            <p class="verdict">{verdict.get()}</p>
          </>
        ) : null}
      </Panel>{' '}
      {!loaded.get() && failedResult.get() ? (
        <ErrorState error={failedResult.get()!.err} title={`Could not load the result of ${failedResult.get()!.e.name}`} retry={() => results.retry()} />
      ) : !loaded.get() ? (
        <>
          <Skeleton h="280px" r="var(--radius-lg)" />
          <Skeleton h="320px" r="var(--radius-lg)" />
          <Skeleton h="320px" r="var(--radius-lg)" />
        </>
      ) : (
        results_()
      )}
    </>
  );

  const results_ = () => (
    <>
      <Panel title="Robustness by condition" meta="accuracy kept">
        {condRows.get().length ? (
          <DotRows
            rows={condRows.get().map((r) => ({ ...r, name: nameOf(r.key) }))}
            keys={keys.get()}
            domain={condDomain.get()}
            format={(v) => keptText(v)}
            label="Accuracy kept per condition, one dot per entry"
            highlight={highlight.get()}
          />
        ) : null}{' '}
        {noRobust.get().length ? (
          <p class="note">
            {noRobust
              .get()
              .map((e) => e.name)
              .join(', ')}{' '}
            {noRobust.get().length === 1 ? 'ran' : 'ran'} no condition: {noRobust.get().length === 1 ? 'it has' : 'they have'} no dots here.
          </p>
        ) : null}
      </Panel>{' '}
      <div class="pair">
        <Panel title="Calibration" meta="share right by confidence">
          {calSeries.get().length ? (
            <XYChart
              label="Reliability of each entry's confidences"
              series={calSeries.get()}
              highlight={highlight.get()}
              xDomain={RELIABILITY_X}
              yDomain={[0, 1]}
              xTicks={[0.1, 0.4, 0.7, 1]}
              yTicks={[0, 0.25, 0.5, 0.75, 1]}
              xFormat={(v) => v.toFixed(1)}
              yFormat={unitTick}
              xLabel="Confidence it gave"
              yLabel="Share right"
              height={phone.current ? 240 : 280}
              refs={[{ axis: 'diagonal', label: 'said = right' }]}
              stops={BIN_MIDS}
              bind:cursor={calCursor}
              readout={calRead}
              table={{
                caption: 'ECE per entry',
                head: ['Entry', 'ECE', 'Detections'],
                rows: shown.get().flatMap((e, i) => {
                  const c = res.get()[i]?.calibration;
                  return c ? [[e.name, num(c.ece, 3), num(c.detections)]] : [];
                }),
              }}
            />
          ) : null}{' '}
          <p class="note">
            <For each={shown.get()} key={(e) => e.id}>
              {(e, i) => (
                <>
                  {res.get()[i()]?.calibration ? (
                    <span class="ece">
                      {key(e().id)}
                      {/* One text node, as Svelte makes of "ECE {num(…)}": Chrome shapes each text node on its own. */}
                      {`ECE ${num(res.get()[i()]!.calibration!.ece, 2)}`}
                    </span>
                  ) : null}
                </>
              )}
            </For>{' '}
            {/* Not measured without confidences, or with confidences that never reach 0.1 (docs/BENCH.md, Calibration). */}
            <For each={noCal.get()} key={(e) => e.id}>
              {(e) => (
                <span>
                  {e().name}: {e().scored ? 'no detection reached confidence 0.1' : 'no confidences'}, not measured.
                </span>
              )}
            </For>
          </p>
        </Panel>{' '}
        <Panel title="Self-labeling" meta="labels right by objects found">
          <XYChart
            label="Precision of each entry's labels against the share of objects they find, as its threshold moves"
            series={slSeries.get()}
            highlight={highlight.get()}
            xDomain={[0, 1]}
            yDomain={[0, 1]}
            xTicks={[0, 0.25, 0.5, 0.75, 1]}
            yTicks={[0, 0.25, 0.5, 0.75, 1]}
            xFormat={unitTick}
            yFormat={unitTick}
            xLabel="Objects found (recall)"
            yLabel="Labels right"
            height={phone.current ? 240 : 280}
            refs={[{ axis: 'y', value: TARGET_PRECISION, label: '90% right' }]}
            marks={slMarks.get()}
            stops={[]}
            readout={legendRead}
            table={{
              caption: 'Self-labeling per entry',
              head: ['Entry', 'Threshold t*', 'Objects at 90%', 'Labels right'],
              rows: shown.get().flatMap((e, i) => {
                const s = res.get()[i]?.self_labeling;
                return s ? [[e.name, s.threshold !== null ? conf(s.threshold) : '—', pct(s.recall, 0), pct(s.precision, 0)]] : [];
              }),
            }}
          />{' '}
          <p class="note">
            The furthest right a line gets above 90% is its self-labeling axis (ringed).{' '}
            <For each={shown.get()} key={(e) => e.id}>
              {(e, i) => {
                const s = () => res.get()[i()]?.self_labeling;
                return (
                  <>
                    {s() ? (
                      <span class="ece">
                        {key(e().id)}
                        {pct(s()!.recall, 0)}
                        {s()!.threshold !== null ? <>{' '}from {conf(s()!.threshold!)}</> : null}
                      </span>
                    ) : null}
                  </>
                );
              }}
            </For>
          </p>
        </Panel>
      </div>{' '}
      <Panel
        title="AP50-95 per class"
        meta={clsRows.get().length > CLS_SHOWN && !allClasses.get() ? `the ${CLS_SHOWN} hardest of ${clsRows.get().length}` : `${clsRows.get().length} classes, hardest first`}
      >
        <DotRows
          rows={clsShown.get().map((r) => ({ key: r.key, name: r.name, values: r.values }))}
          keys={keys.get()}
          domain={clsDomain.get()}
          format={(v) => metric(v)}
          tickFormat={(v) => v.toFixed(2)}
          label="AP50-95 per class, one dot per entry"
          highlight={highlight.get()}
          sub={(k) => {
            // The benchmark's count; an entry graded on a part of the images says on how many of them.
            const row = clsRows.get().find((r) => r.key === k);
            const n = row?.truths ?? 0;
            const b = bench();
            const part = shown.get().flatMap((e, i) => (b && e.images < b.images && row?.graded[i] != null ? [`${num(row.graded[i]!)} for ${e.name}`] : []));
            return `${num(n)} true ${n === 1 ? 'box' : 'boxes'}${part.length ? ` (graded on ${part.join(', ')})` : ''}`;
          }}
        />{' '}
        {clsRows.get().length > CLS_SHOWN ? (
          <Button size="sm" variant="ghost" onclick={() => allClasses.set((a) => !a)}>
            {allClasses.get() ? `Show the ${CLS_SHOWN} hardest` : `Show all ${clsRows.get().length} classes`}
          </Button>
        ) : null}
      </Panel>
    </>
  );

  const page = () => (
    <Page
      title="Compare entries"
      inlineActions
      sub={() => (
        <span>
          {bench()!.name} · {shown.get().length} {shown.get().length === 1 ? 'entry' : 'entries'}
          {old.get().some((x) => x) ? ', not all on its current fingerprint' : ' on one fingerprint'}
        </span>
      )}
      subPhone={() => (
        <span>
          {bench()!.name} · {shown.get().length} {shown.get().length === 1 ? 'entry' : 'entries'}
        </span>
      )}
      actions={() => (
        <>
          {!phone.current ? (
            <Button icon={ArrowLeft} href={href(`/benchmarks/${bench()?.id}`)}>
              Leaderboard
            </Button>
          ) : null}
        </>
      )}
    >
      {/* The legend: one chip per entry, its key, its score, open and remove; then Add. */}
      <div class="chips" role="group" aria-label="Entries being compared">
        <For each={shown.get()} key={(e) => e.id}>
          {(e) => (
            <span
              class="chip"
              class:pinned={pinned.get() === e().id}
              style={{ '--c': fill(e().id) }}
              onpointerenter={() => hovered.set(e().id)}
              onpointerleave={() => hovered.set(null)}
              onfocusin={() => hovered.set(e().id)}
              onfocusout={() => hovered.set(null)}
            >
              <button
                type="button"
                class="ck"
                aria-pressed={pinned.get() === e().id}
                title={pinned.get() === e().id ? 'Let it fall back' : 'Keep it in front'}
                onclick={() => pinned.set((p) => (p === e().id ? null : e().id))}
              >
                {key(e().id)}
                <span class="nm">{e().name}</span>
              </button>{' '}
              <span class="v num">
                {scoreText(e().score!.total)}
                {e().score!.partial ? <span class="st">{' · partial'}</span> : null}
                {bench() && stale(e(), bench()!.fingerprint) ? <span class="st">{' · earlier version'}</span> : null}
              </span>{' '}
              <a class="act" href={href(`/benchmarks/${bench()?.id}/entries/${e().id}`)} aria-label={`Open ${e().name}`} title="Open its result">
                <ArrowUpRight size={14} />
              </a>{' '}
              <button
                type="button"
                class="act rm"
                aria-label={`Remove ${e().name}`}
                title="Remove from the comparison"
                onclick={() => setIds(ids.peek().filter((x) => x !== e().id))}
              >
                <X size={14} />
              </button>
            </span>
          )}
        </For>{' '}
        {ids.get().length < COMPARE_MAX && candidates.get().length ? <>{adder()}</> : null}
      </div>{' '}
      {shown.get().length < 2 ? (
        <section class="state">
          <EmptyState
            icon={GitCompare}
            title={shown.get().length ? 'Add one more entry to compare' : 'Pick two to four finished entries'}
            description={
              picked.get().missing.length
                ? `${picked.get().missing.length === 1 ? 'One entry in this link is' : `${picked.get().missing.length} entries in this link are`} not on ${bench()!.name} or not finished. Add finished entries here, or pick them on the leaderboard.`
                : `Add finished entries of ${bench()!.name} here, or pick them on the leaderboard.`
            }
            actions={() => (
              <Button icon={ArrowLeft} href={href(`/benchmarks/${bench()?.id}`)}>
                Back to the leaderboard
              </Button>
            )}
          />
        </section>
      ) : (
        compared()
      )}
    </Page>
  );

  return (
    <>
      {unsupported.get() ? (
        <Page title="Benchmarks">
          <Unsupported />
        </Page>
      ) : live.error && !bench() ? (
        <Page
          title="Compare entries"
          actions={() => (
            <Button icon={ArrowLeft} href={href('/benchmarks')}>
              All benchmarks
            </Button>
          )}
        >
          {live.error instanceof ApiError && live.error.status === 404 ? (
            <ErrorState error={live.error} title="Benchmark not found" />
          ) : (
            <ErrorState error={live.error} title="Could not load the entries" retry={() => live.retry()} />
          )}
        </Page>
      ) : !bench() || live.loading ? (
        <div class="sk-page" aria-busy="true" aria-label="Loading comparison">
          <div class="sk-head">
            <Skeleton w="min(240px, 60%)" h="24px" />
            <Skeleton w="min(420px, 90%)" h="14px" />
          </div>
          <div class="sk-chips">
            <Skeleton h="44px" r="var(--radius-md)" />
            <Skeleton h="44px" r="var(--radius-md)" />
          </div>
          <Skeleton h="320px" r="var(--radius-lg)" />
          <Skeleton h="240px" r="var(--radius-lg)" />
        </div>
      ) : (
        page()
      )}
    </>
  );
}

css`
  /* The legend: one card per entry (RunCompare's chips), side by side, a 2 × 2 grid from 768 to 1279 px. */
  /* Chips as wide as what they hold (at least 240 px), so a name like "Claude Code (haiku)" stays on one line beside
     its score instead of breaking inside its parentheses in a cell a quarter of the row wide. */
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    margin-top: calc(-1 * var(--space-2));
  }
  .chips > :global(*) {
    flex: 0 1 auto;
    min-width: 240px;
    max-width: 100%;
  }
  .chip {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto auto;
    align-items: center;
    column-gap: var(--space-1);
    min-height: 44px;
    padding: var(--space-1) var(--space-1) var(--space-1) var(--space-2);
    border-radius: var(--radius-md);
    background: var(--bg-panel);
    border: 1px solid var(--line-default);
  }
  .chip:hover {
    border-color: var(--line-strong);
  }
  .chip.pinned {
    background: var(--bg-active);
    box-shadow: var(--select-bar);
  }
  .ck {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    min-height: 32px;
    padding: 0 var(--space-1);
    border-radius: var(--radius-sm);
    text-align: left;
  }
  .nm {
    font-weight: var(--weight-medium);
    overflow-wrap: anywhere;
  }
  .v {
    padding: 0 var(--space-1);
    font-weight: var(--weight-medium);
    white-space: nowrap;
  }
  .st {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .act {
    position: relative;
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  @media (hover: hover) {
    .ck:hover,
    .act:hover {
      background: var(--bg-hover);
      color: var(--text-primary);
    }
  }
  @media (pointer: coarse) {
    .act {
      width: 40px;
      height: 40px;
    }
  }
  /* A key is a fill: the palette hue with the swatch hairline, its slot's pattern cut along it (RunCompare). */
  .key {
    flex-shrink: 0;
    display: inline-block;
    width: 16px;
    height: 4px;
    border-radius: 2px;
    background: var(--c);
    box-shadow: inset 0 0 0 1px var(--chart-swatch-edge);
  }
  .key[data-pattern='dash'] {
    mask-image: repeating-linear-gradient(90deg, #000 0 5px, transparent 5px 7px);
    border-radius: 0;
  }
  .key[data-pattern='dot'] {
    mask-image: repeating-linear-gradient(90deg, #000 0 3px, transparent 3px 5px);
    border-radius: 0;
  }
  /* Add: a secondary button's face, the native picker over all of it. */
  .add {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    min-height: 44px;
    padding: 0 var(--space-3);
    border-radius: var(--radius-md);
    border: 1px dashed var(--line-strong);
    color: var(--text-secondary);
  }
  .add select {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }
  @media (pointer: coarse) {
    .add select {
      font-size: var(--type-lg-size);
    }
  }
  .add:has(select:focus-visible) {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: var(--size-focus-offset);
  }
  .state {
    display: grid;
    place-items: center;
    min-height: 320px;
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
  }
  .cav {
    display: grid;
    gap: var(--space-1);
    margin-top: var(--space-1);
  }
  .scroll {
    overflow-x: auto;
  }
  /* The measure a fixed column, the entries sharing the rest equally (DESIGN_DIRECTION §2.7: a comparison). */
  .cmp {
    table-layout: fixed;
  }
  .m-col {
    width: 200px;
  }
  .cmp thead th {
    height: auto;
    padding-top: var(--space-2);
    padding-bottom: var(--space-2);
    vertical-align: bottom;
  }
  .eh {
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    gap: var(--space-2);
    max-width: 100%;
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    white-space: normal;
  }
  .eh-n {
    overflow-wrap: anywhere;
    text-align: right;
  }
  .role {
    display: block;
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .swap {
    color: var(--text-secondary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .cmp tr.total td {
    font-weight: var(--weight-semibold);
  }
  .top {
    font-weight: var(--weight-semibold);
  }
  .best {
    margin-right: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .dd {
    display: inline-block;
    min-width: 4.5em;
    margin-left: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    text-align: right;
  }
  .iv {
    display: grid;
  }
  .iv li {
    display: grid;
    grid-template-columns: minmax(120px, 220px) 56px minmax(0, 1fr) 112px;
    align-items: center;
    gap: var(--space-3);
    min-height: 36px;
    border-top: 1px solid var(--line-subtle);
  }
  .iv li.dim {
    opacity: 0.45;
  }
  .iv li.iv-axis {
    min-height: 20px;
    border-top: 0;
  }
  .iv-n {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .iv-v {
    text-align: right;
    font-weight: var(--weight-medium);
  }
  .iv-ci {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .iv-ticks {
    position: relative;
    display: block;
    height: 16px;
    color: var(--chart-label);
    /* small: chart tick */
    font-size: var(--type-xs-size);
  }
  .iv-ticks span {
    position: absolute;
    top: 0;
    transform: translateX(-50%);
  }
  .verdict {
    margin-top: var(--space-3);
    max-inline-size: var(--measure-dialog);
  }
  .sbs {
    display: none;
  }
  .pair {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-4);
    align-items: start;
  }
  .note {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-4);
    margin-top: var(--space-3);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .ece,
  .rv {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }
  .rv {
    margin-left: var(--space-3);
    color: var(--text-primary);
  }
  .t,
  .lr {
    color: var(--text-tertiary);
  }
  .hint {
    color: var(--text-secondary);
  }
  .sk-page {
    max-width: calc(var(--size-content-max) + 2 * var(--space-6));
    margin: 0 auto;
    padding: var(--space-6);
    display: grid;
    gap: var(--space-5);
  }
  .sk-head {
    display: grid;
    gap: var(--space-2);
  }
  .sk-chips {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-2);
  }
  /* Narrower columns: the difference under its value. */
  @media (max-width: 1279px) {
    .cmp .dd {
      display: block;
      min-width: 0;
      margin-left: 0;
    }
  }
  @media (max-width: 1023px) {
    .pair {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  @media (max-width: 767px) {
    .sk-page {
      padding: var(--space-4);
    }
    .chips > :global(*) {
      flex-basis: 100%;
    }
    .scroll:has(.cmp) {
      display: none;
    }
    .sbs {
      display: grid;
    }
    .sbs li {
      display: grid;
      gap: var(--space-2);
      padding: var(--space-3) var(--space-4);
      border-top: 1px solid var(--line-subtle);
    }
    .sbs li:first-child {
      border-top: 0;
    }
    .sbs-h {
      font-weight: var(--weight-medium);
    }
    .sbs-g {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-2) var(--space-4);
    }
    .sbs-v {
      display: grid;
      min-width: 0;
    }
    .sbs-n {
      display: flex;
      align-items: center;
      gap: var(--space-2);
      min-width: 0;
      font-size: var(--type-sm-size);
      line-height: var(--type-sm-line);
      color: var(--text-secondary);
    }
    .sbs-v .dd {
      min-width: 0;
    }
    .iv li {
      grid-template-columns: minmax(0, 1fr) 48px;
      grid-template-areas: 'n v' 'p p';
      row-gap: 0;
      padding: var(--space-1) 0;
    }
    .iv-n {
      grid-area: n;
    }
    .iv-v {
      grid-area: v;
    }
    .iv-plot {
      grid-area: p;
    }
    .iv-ci {
      display: none;
    }
    .iv li.iv-axis {
      grid-template-areas: 'p p';
    }
    .iv li.iv-axis > span:not(.iv-ticks) {
      display: none;
    }
    .iv-ticks {
      grid-area: p;
    }
  }
`;
