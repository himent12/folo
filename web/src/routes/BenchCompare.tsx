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
import Download from '$icons/download';
import GitCompare from '$icons/git-compare-arrows';
import Scale from '$icons/scale';
import Plus from '$icons/plus';
import X from '$icons/x';
import { css, effect, For, Index, MediaQuery, onCleanup, onDocument, signal, untrack } from 'fw';
import { ApiError } from '$lib/api/index';
import { TARGET_PRECISION } from '$lib/bench/rules';
import XYChart from '$lib/charts/XYChart';
import { media } from '$lib/design/tokens';
import { conf, metric, num, pct } from '$lib/format';
import { isTyping } from '$lib/keys';
import { href } from '$lib/router';
import Page from '$lib/shell/Page';
import { pageKeys } from '$lib/shell/pageCommands';
import { gChord } from '$lib/shell/shortcuts';
import { exportTableCsv, exportTableJson } from '$lib/tableExport';
import Button from '$lib/ui/Button';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Ident from '$lib/ui/Ident';
import { keyPrefs } from '$lib/ui/keyPrefs';
import Menu, { type MenuItem } from '$lib/ui/Menu';
import Panel from '$lib/ui/Panel';
import Skeleton from '$lib/ui/Skeleton';
import Callout from './run/Callout';
import { RELIABILITY_X, SLOT_PATTERN } from './bench/charts';
import { compareExport } from './bench/export';
import DotRows from './bench/compare/DotRows';
import SideBySide from './bench/compare/SideBySide';
import { CLS_SHOWN, compareState } from './bench/compare/state';
import IntervalPlot from './bench/IntervalPlot';
import { along, COMPARE_MAX, stale } from './bench/leaderboard';
import { capTicks } from '$lib/charts/domain';
import Unsupported from './bench/Unsupported';
import { calibrationNone, intervalText, keptText, scoreText } from './bench/words';

const BIN_MIDS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
const unitTick = (v: number) => (v === 0 || v === 1 ? String(v) : v.toFixed(2));

export default function BenchCompare() {
  const phone = new MediaQuery(media.phone, false);
  const {
    id,
    live,
    bench,
    unsupported,
    picked,
    shown,
    pending,
    dropped,
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
    labelOf,
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
  const liveNote = signal('');
  let alive = true;
  let noteSeq = 0;
  onCleanup(() => {
    alive = false;
    noteSeq++;
  });
  function announce(message: string) {
    const seq = ++noteSeq;
    if (liveNote.peek() === message) {
      liveNote.set('');
      queueMicrotask(() => {
        if (alive && seq === noteSeq) liveNote.set(message);
      });
    } else {
      liveNote.set(message);
    }
  }
  let chipsEl: HTMLDivElement | undefined;

  let lastAnnouncedDropped = '';
  effect(() => {
    const omitted = dropped.get();
    const keyText = omitted.join(',');
    if (!omitted.length || keyText === lastAnnouncedDropped) {
      if (!omitted.length) lastAnnouncedDropped = '';
      return;
    }
    lastAnnouncedDropped = keyText;
    const bName = bench()?.name;
    untrack(() =>
      announce(
        `${omitted.length === 1 ? 'Omitted unavailable entry' : `Omitted ${omitted.length} unavailable entries`}: ${omitted.join(', ')}${bName ? ` (not on ${bName} or not finished)` : ''}.`,
      ),
    );
  });

  function dismissMissing() {
    dropped.set([]);
    queueMicrotask(() => {
      if (!alive || !chipsEl?.isConnected) return;
      if (document.activeElement && document.activeElement !== document.body) return;
      chipsEl.querySelector<HTMLElement>('button.ck, .add select')?.focus();
    });
  }

  function setBaseline(eid: string) {
    const entry = shown.peek().find((x) => x.id === eid);
    const label = entry ? labelOf(entry) : '';
    makeBaseline(eid);
    if (label) announce(`Baseline set to ${label}.`);
  }

  function togglePin(eid: string) {
    const entry = shown.peek().find((x) => x.id === eid);
    const next = pinned.peek() === eid ? null : eid;
    pinned.set(next);
    if (entry) announce(next ? `Pinned ${labelOf(entry)} to front.` : `Unpinned ${labelOf(entry)}.`);
  }

  function addEntry(eid: string) {
    const entry = candidates.peek().find((x) => x.id === eid);
    const current = shown.peek().map((x) => x.id);
    if (current.length >= COMPARE_MAX || current.includes(eid)) return;
    setIds([...current, eid]);
    if (entry) announce(`Added ${entry.name} to comparison.`);
    // Adding the 4th entry (or last candidate) unmounts the adder <select>: move keyboard focus to the new chip.
    queueMicrotask(() => {
      if (!alive || !chipsEl?.isConnected) return;
      if (document.activeElement && document.activeElement !== document.body && chipsEl.contains(document.activeElement)) return;
      const buttons = [...chipsEl.querySelectorAll<HTMLButtonElement>('button.ck')];
      const target = buttons.find((b) => b.dataset.entryId === eid) ?? buttons.at(-1);
      target?.focus();
    });
  }

  function removeEntry(eid: string, index: number) {
    const entry = shown.peek().find((x) => x.id === eid);
    const label = entry ? labelOf(entry) : '';
    setIds(shown.peek().map((x) => x.id).filter((x) => x !== eid));
    if (label) announce(`Removed ${label} from comparison.`);
    // The Remove button unmounts with its chip: move keyboard focus to a neighbouring chip, the adder, or the heading.
    queueMicrotask(() => {
      if (!alive || !chipsEl?.isConnected) return;
      const buttons = [...chipsEl.querySelectorAll<HTMLButtonElement>('button.ck')];
      const next = buttons[Math.min(index, buttons.length - 1)] ?? chipsEl.querySelector<HTMLSelectElement>('select') ?? document.querySelector<HTMLElement>('h1[data-route-focus]');
      next?.focus();
    });
  }

  function exportComparison(format: 'csv' | 'json') {
    const b = bench();
    const list = shown.peek();
    if (!b || list.length < 2) return;
    const payload = compareExport(b, list, rows.peek(), old.peek(), best, caveats.peek(), res.peek());
    const ok =
      format === 'csv'
        ? exportTableCsv(payload.filenameBase, payload.headers, payload.rows)
        : exportTableJson(payload.filenameBase, payload.json);
    if (ok) announce(`Exported comparison of ${list.length} entries as ${format.toUpperCase()}.`);
  }

  const exportItems: MenuItem[] = [
    { label: 'Export CSV', icon: Download, onclick: () => exportComparison('csv') },
    { label: 'Export JSON', icon: Download, onclick: () => exportComparison('json') },
  ];

  // Single-key shortcuts via pageKeys (window bubble, after shell/shortcuts so G B navigates to Benchmarks cleanly):
  // 1–4 pin/unpin an entry across every chart; B rotates the baseline through all shown entries.
  pageKeys((k) => {
    if (document.querySelector('[aria-expanded="true"][aria-haspopup], [role="menu"]')) return false;
    const list = shown.peek();
    if (list.length < 2) return false;
    if (k >= '1' && k <= '4') {
      const target = list[Number(k) - 1];
      if (!target) return false;
      togglePin(target.id);
      return true;
    }
    if (k.toLowerCase() === 'b') {
      const shownIds = list.map((x) => x.id);
      setIds([...shownIds.slice(1), shownIds[0]!]);
      const nextBase = list[1];
      if (nextBase) announce(`Baseline set to ${labelOf(nextBase)}.`);
      return true;
    }
    return false;
  });

  // Escape unpins an active chip before global Escape returns to the leaderboard (never intercepting a G-chord cancel or popover).
  const hasOpenPopover = () => {
    try {
      return !!document.querySelector('[popover]:popover-open');
    } catch {
      return false;
    }
  };
  onDocument('keydown', (e) => {
    if (e.key !== 'Escape' || e.repeat || e.defaultPrevented || e.isComposing || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey || isTyping(e)) return;
    if (!keyPrefs.singleKeys || gChord.peek() || hasOpenPopover() || document.querySelector('dialog[open], [aria-expanded="true"][aria-haspopup], [role="menu"], #folo-tooltip[data-show], .info-pop.open')) return;
    const p = pinned.peek();
    if (p !== null && shown.peek().some((x) => x.id === p)) {
      e.preventDefault();
      togglePin(p);
    }
  });

  const adder = () => (
    <label class="add">
      <Plus size={14} strokeWidth={1.75} aria-hidden="true" />{' '}
      <span aria-hidden="true">Add entry…</span>{' '}
      <select
        aria-label="Add an entry to the comparison"
        onchange={(ev) => {
          const el = ev.currentTarget as HTMLSelectElement;
          const v = el.value;
          if (v) addEntry(v);
          el.value = '';
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

  const compared = () => (
    <>
      {caveats.get().length ? (
        <Callout tone="neutral" icon={Scale} title="Not like for like" role="note">
          <ul class="cav">
            <Index each={caveats.get()}>{(c) => <li>{c().text}</li>}</Index>
          </ul>
        </Callout>
      ) : null}{' '}
      <SideBySide shown={shown.get()} rows={rows.get()} old={old.get()} labelOf={labelOf} best={best} makeBaseline={setBaseline} keyMark={key} />{' '}
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
            ran no conditions: {noRobust.get().length === 1 ? 'it has' : 'they have'} no dots here.
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
                  return c ? [[labelOf(e), num(c.ece, 3), num(c.detections)]] : [];
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
                  {e().name}: {calibrationNone(e())}, not measured.
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
                return s ? [[labelOf(e), s.threshold !== null ? conf(s.threshold) : '—', pct(s.recall, 0), pct(s.precision, 0)]] : [];
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
          {shown.get().length >= 2 ? (
            <Menu icon={Download} text={phone.current ? undefined : 'Export'} label="Export comparison" items={exportItems} />
          ) : null}
          {!phone.current ? (
            <Button icon={ArrowLeft} href={href(`/benchmarks/${bench()?.id}`)}>
              Leaderboard
            </Button>
          ) : null}
        </>
      )}
    >
      {/* The legend: one chip per entry, its key, its score, open and remove; then Add. */}
      <div ref={(el) => (chipsEl = el)} class="chips" role="group" aria-label="Entries being compared">
        <For each={shown.get()} key={(e) => e.id}>
          {(e, i) => (
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
                data-entry-id={e().id}
                aria-label={labelOf(e()) !== e().name ? labelOf(e()) : undefined}
                aria-pressed={pinned.get() === e().id}
                aria-keyshortcuts={keyPrefs.singleKeys && shown.get().length >= 2 ? String(i() + 1) : undefined}
                title={pinned.get() === e().id ? 'Let it fall back' : 'Keep it in front'}
                onclick={() => togglePin(e().id)}
              >
                {key(e().id)}
                <span class="nm">{e().name}</span>
              </button>{' '}
              <span class="v num">
                {scoreText(e().score!.total)}
                {e().score!.partial ? <span class="st">{' · partial'}</span> : null}
                {bench() && stale(e(), bench()!.fingerprint) ? <span class="st">{' · earlier version'}</span> : null}
              </span>{' '}
              <a class="act" href={href(`/benchmarks/${bench()?.id}/entries/${e().id}`)} aria-label={`Open ${labelOf(e())}`} title="Open its result">
                <ArrowUpRight size={14} />
              </a>{' '}
              <button
                type="button"
                class="act rm"
                aria-label={`Remove ${labelOf(e())}`}
                title="Remove from the comparison"
                onclick={() => removeEntry(e().id, i())}
              >
                <X size={14} />
              </button>
            </span>
          )}
        </For>{' '}
        {shown.get().length < COMPARE_MAX && candidates.get().length ? adder() : null}
      </div>{' '}
      <p class="sr-only" role="status" aria-live="polite">
        {liveNote.get()}
      </p>{' '}
      {shown.get().length < 2 ? (
        <section class="state">
          <EmptyState
            icon={GitCompare}
            title={shown.get().length ? 'Add one more entry to compare' : 'Pick two to four finished entries'}
            description={
              dropped.get().length || picked.get().missing.length
                ? `${(dropped.get().length || picked.get().missing.length) === 1 ? 'One entry in this link is' : `${dropped.get().length || picked.get().missing.length} entries in this link are`} not on ${bench()!.name} or not finished. Add finished entries here, or pick them on the leaderboard.`
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
        <>
          {dropped.get().length ? (
            <div class="missing-note" data-missing-entries>
              <span>
                {dropped.get().length === 1 ? 'Omitted unavailable entry from this comparison: ' : `Omitted ${dropped.get().length} unavailable entries from this comparison: `}
                <For each={dropped.get()} key={(eid) => eid}>
                  {(eid, i) => (
                    <>
                      {i() > 0 ? ', ' : null}
                      <Ident value={eid()} />
                    </>
                  )}
                </For>
                {` (not on ${bench()!.name} or not finished).`}
              </span>{' '}
              <Button size="sm" variant="ghost" onclick={dismissMissing}>
                Dismiss
              </Button>
            </div>
          ) : null}
          {pending.get().length ? (
            <p class="missing-note" data-pending-entries>
              {pending.get().length === 1
                ? `Waiting for ${pending.get()[0]!.name} to finish — it will join this comparison automatically.`
                : `Waiting for ${pending.get().length} entries (${pending.get().map((e) => e.name).join(', ')}) to finish — they will join automatically.`}
            </p>
          ) : null}
          {compared()}
        </>
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
  .missing-note {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    margin: 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
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
