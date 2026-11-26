/**
 * Streaming training log: level filter, follow mode (auto-scroll unless you scroll up). The view always starts on a
 * whole entry: following, it pins the last entry to the bottom and, when that would cut the top entry (the wrapped
 * tail of a message, without its time and level), pads the end by the cut so the top entry starts on the edge; read
 * by hand, it snaps to entries. The box is at least `--log-h` tall (300 px) and grows with its panel when the page
 * stretches it (Run's columns end together).
 *
 * **Silent by default** (DESIGN_DIRECTION §2.9, accessibility): a log streams several lines a second, and a live
 * region on it would read every one with no way to stop it (follow mode only scrolls). The log is `role="log"` with
 * `aria-live="off"`; "Announce new lines" (a pressed toggle in its header, while the run is live) turns on a polite
 * status line that reads them in batches, at most one announcement every `ANNOUNCE_MS`: the one new line, or how
 * many arrived and the latest. The run's milestones and failures are not the log's to say: its start is the shell's
 * live region (shell/SidebarLive) and its end, failure included, a toast (shell/RunWatch), announced whether or not
 * this is on.
 *
 * Streaming cost (the Svelte original redid all of it on every new line): the lines are read from the run's append
 * log (`live.logsLog`), so a batch of new lines costs its own size: the warn / error lists and counts are kept up to
 * date with the lines added and the lines the cap dropped, the 500 lines shown are one slice, and each row is keyed by
 * its line (a stable object), so an appended line inserts one row and a dropped one removes one (keyed by index, every
 * row re-rendered). Pinning finds the top entry by bisection instead of reading every line's position.
 */
import ArrowDownToLine from '$icons/arrow-down-to-line';
import Copy from '$icons/copy';
import Search from '$icons/search';
import Volume2 from '$icons/volume-2';
import VolumeOff from '$icons/volume-off';
import { batch, computed, css, effect, For, signal, untrack } from 'fw';
import type { AppendLog, LogCursor } from '$lib/api/appendLog';
import type { LogLevel, LogLine } from '$lib/api/types';
import { clock } from '$lib/format';
import Button from '$lib/ui/Button';
import { feedback } from '$lib/ui/feedback';
import Input from '$lib/ui/Input';
import Panel from '$lib/ui/Panel';
import Segmented from '$lib/ui/Segmented';
import { tooltip } from '$lib/ui/tooltip';
import { logPieces } from './logTokens';

export interface LogsProps {
  /** The run's log lines (`live.logsLog`): read after `version` bumps, only what was added (or dropped) since. */
  log: AppendLog<LogLine>;
  version: number;
  live: boolean;
  /** Lines of the run's log older than `log` (`live.logsEarlier`): counted in the header, said above the first line. */
  earlier?: number;
}

const order: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 };
/** The most lines the box holds: the latest 500 the level filter shows, until "Show earlier" asks for all it has. */
const SHOWN = 500;

/** A line at warn level or above, with its absolute index in the log (the cap trims the front: older ones leave). */
interface Kept {
  line: LogLine;
  at: number;
}

/**
 * The log's warn-and-up lines and error lines, kept up to date from an append log's reads: each read costs the lines
 * added and dropped since the last, never the whole log. Exported for tests.
 */
export class LevelIndex {
  #log: AppendLog<LogLine> | null = null;
  #cursor: LogCursor | null = null;
  /** Lines at warn level or above, oldest first. */
  warnUp: Kept[] = [];
  /** Error lines, oldest first. */
  errors: Kept[] = [];

  /** Where the lists stand in the log (null before the first sync). */
  get cursor(): LogCursor | null {
    return this.#cursor;
  }

  /** Brings the lists up to `log` (what changed since the last call). */
  sync(log: AppendLog<LogLine>): void {
    if (log !== this.#log) {
      this.#log = log;
      this.#cursor = null;
    }
    const r = log.read(this.#cursor);
    this.#cursor = r.cursor;
    if (r.reset) {
      this.warnUp = [];
      this.errors = [];
    } else if (r.dropped) {
      const start = r.cursor.start;
      drop(this.warnUp, start);
      drop(this.errors, start);
    }
    let at = r.cursor.end - r.added.length;
    for (const line of r.added) {
      const o = order[line.level];
      if (o >= 2) this.warnUp.push({ line, at });
      if (o >= 3) this.errors.push({ line, at });
      at++;
    }
  }
  get warn(): number {
    return this.warnUp.length - this.errors.length;
  }
  get error(): number {
    return this.errors.length;
  }
}

/** Removes the entries before absolute index `start` (the front of a sorted list). */
function drop(list: Kept[], start: number) {
  let n = 0;
  while (n < list.length && list[n]!.at < start) n++;
  if (n) list.splice(0, n);
}

/** The last `n` lines of a kept list. */
const lastLines = (list: Kept[], n: number) => {
  const from = Math.max(0, list.length - n);
  const out = new Array<LogLine>(list.length - from);
  for (let i = from; i < list.length; i++) out[i - from] = list[i]!.line;
  return out;
};

/** "1 line", "204 lines". */
const lineCount = (n: number) => `${n.toLocaleString('en-US')} ${n === 1 ? 'line' : 'lines'}`;

export default function Logs(props: LogsProps) {
  const notify = feedback();
  const level = signal<'all' | 'warn' | 'error'>('all');
  const query = signal('');
  const needle = computed(() => query.get().trim().toLowerCase());
  /** How many of the latest lines the box shows: `SHOWN`, or every line held once asked ("Show earlier"). */
  const cap = signal(SHOWN);
  const follow = signal(true);
  const box = signal<HTMLDivElement>();
  // Preserve keyboard focus onto the log container if the empty-state "Clear filters" button unmounts while focused
  // (either on click or when a matching live log line arrives).
  let emptyFocused = false;
  const preserveEmptyFocus = (el: HTMLElement) => ({
    destroy() {
      const hadFocus = emptyFocused || el.contains(document.activeElement);
      emptyFocused = false;
      const logEl = box.peek();
      if (hadFocus && logEl?.isConnected && (!document.activeElement || document.activeElement === document.body || el.contains(document.activeElement))) {
        logEl.focus({ preventScroll: true });
      }
    },
  });

  // Scoped span templates cloned per immutable log token so rendering 500+ lines creates zero per-token effects.
  const wordTpl = (<span class="w" />) as HTMLSpanElement;
  const longTpl = (<span class="w long" />) as HTMLSpanElement;
  function fillMessage(el: HTMLSpanElement, message: string) {
    el.textContent = '';
    const frag = document.createDocumentFragment();
    for (const p of logPieces(message)) {
      if (p.space) {
        frag.appendChild(document.createTextNode(p.text));
      } else {
        const span = (p.long ? longTpl : wordTpl).cloneNode(false) as HTMLSpanElement;
        span.textContent = p.text;
        frag.appendChild(span);
      }
    }
    el.appendChild(frag);
  }

  const index = new LevelIndex();
  // `log` changes in place; `version` is the signal that it did.
  const counts = computed(
    () => {
      void props.version;
      index.sync(props.log);
      return { held: props.log.length, earlier: props.earlier ?? 0, warn: index.warn, error: index.error };
    },
    { equals: (a, b) => a.held === b.held && a.earlier === b.earlier && a.warn === b.warn && a.error === b.error },
  );
  /** Every line of the run's log, the ones older than those held included: what the header counts. */
  const total = () => counts.get().held + counts.get().earlier;
  // The lines shown: a new list only when the log or the filter changed (steps bump `version` too, several times a
  // second, and each new list would re-pin the box: a layout read).
  let shownKey = '';
  let shownLog: AppendLog<LogLine> | null = null;
  let shownState: { lines: LogLine[]; matched: number } = { lines: [], matched: 0 };
  const filtered = computed(() => {
    void props.version;
    const ct = counts.get();
    const lv = level.get();
    const q = needle.get();
    const n = cap.get();
    const c = index.cursor;
    const key = c ? `${c.gen}|${c.start}|${c.end}|${lv}|${n}|${q}` : `${lv}|${n}|${q}`;
    if (key !== shownKey || shownLog !== props.log) {
      shownKey = key;
      shownLog = props.log;
      if (!q) {
        const lines = lv === 'all' ? props.log.items.slice(-n) : lastLines(lv === 'warn' ? index.warnUp : index.errors, n);
        const matched = lv === 'all' ? ct.held : lv === 'warn' ? ct.warn + ct.error : ct.error;
        shownState = { lines, matched };
      } else {
        const matchedLines: LogLine[] = [];
        if (lv === 'all') {
          const items = props.log.items;
          for (let i = 0; i < items.length; i++) {
            const line = items[i]!;
            if (line.message.toLowerCase().includes(q) || line.level.includes(q)) matchedLines.push(line);
          }
        } else {
          const list = lv === 'warn' ? index.warnUp : index.errors;
          for (let i = 0; i < list.length; i++) {
            const line = list[i]!.line;
            if (line.message.toLowerCase().includes(q) || line.level.includes(q)) matchedLines.push(line);
          }
        }
        shownState = {
          lines: matchedLines.length > n ? matchedLines.slice(-n) : matchedLines,
          matched: matchedLines.length,
        };
      }
    }
    return shownState;
  });
  const shown = computed(() => filtered.get().lines);
  /**
   * The lines above the first one shown, at this level: the ones the box leaves out (it can show them), and for All
   * without a search filter the ones older than this view holds (the run folder's log.jsonl has them).
   */
  const before = computed(() => {
    const lv = level.get();
    const f = filtered.get();
    return {
      hidden: Math.max(0, f.matched - f.lines.length),
      older: lv === 'all' || needle.get() ? counts.get().earlier : 0,
    };
  });

  // Polite live status for search results: updates when the user edits the query or level filter, or when search state
  // flips between empty/non-empty matches or truncated/untruncated (never on every streaming count tick).
  const hasMatches = computed(() => filtered.get().matched > 0);
  const hasEarlier = computed(() => counts.get().earlier > 0);
  const searchStatus = signal('');
  effect(() => {
    const q = needle.get();
    const lv = level.get();
    if (!q) {
      searchStatus.set('');
      return;
    }
    void hasMatches.get();
    void hasEarlier.get();
    untrack(() => {
      const m = filtered.get().matched;
      const ct = counts.get();
      const raw = query.peek().trim();
      const scope = lv !== 'all' ? ` at ${lv} level` : '';
      const older = ct.earlier ? ` (${lineCount(ct.earlier)} earlier are in log.jsonl)` : '';
      searchStatus.set(
        m
          ? `${m === 1 ? '1 line matches' : `${lineCount(m)} match`} "${raw}"${scope}${older}.`
          : `No log lines match "${raw}"${scope}${older}.`,
      );
    });
  });

  async function copyVisible() {
    const lines = shown.peek();
    if (!lines.length) return;
    const text = lines.map((l) => `${clock(l.t)} ${l.level} ${l.message}`).join('\n');
    try {
      await navigator.clipboard.writeText(text);
      notify.success('Logs copied', lineCount(lines.length));
    } catch (e) {
      notify.error(e, 'Could not copy the log lines');
    }
  }

  /** Space after the last entry, so that, pinned to the bottom, the top entry starts whole on the box's top edge. */
  let pad = 0;
  let pinEl: HTMLDivElement | undefined;
  /** The box's top padding: an entry the view starts on sits this far under the edge, as the first one does. */
  const INSET = 8;

  /** Scrolls to the end, landing on a whole entry at the top (padding the end by what would otherwise be cut). */
  function pin() {
    const el = box.peek();
    if (!el) return;
    if (el !== pinEl) {
      pinEl = el;
      pad = 0;
    }
    const end = el.scrollHeight - pad - el.clientHeight;
    let next = 0;
    if (end > 0) {
      // The first entry starting at or below `end`: entries are stacked top to bottom inside `inner`, so their offsets
      // only grow and a direct bisection over `inner.children` avoids both scanning every line and querySelectorAll.
      const kids = inner.peek()?.children;
      if (kids && kids.length) {
        let lo = kids[0]!.classList.contains('earlier') ? 1 : 0;
        if (lo < kids.length && kids[lo]!.classList.contains('line')) {
          let hi = kids.length;
          while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if ((kids[mid] as HTMLElement).offsetTop - INSET >= end - 0.5) hi = mid;
            else lo = mid + 1;
          }
          const top = lo < kids.length ? (kids[lo] as HTMLElement) : undefined;
          next = top ? Math.max(0, top.offsetTop - INSET - end) : 0;
        }
      }
    }
    const targetScroll = Math.max(0, end + next);
    // Avoid forcing a second layout pass: when tail padding is unchanged or shrinking, assign `scrollTop` while layout
    // is still clean before mutating `--tail`. Only when `--tail` grows must it precede `scrollTop` so the browser does
    // not clamp to the smaller prior scroll height.
    if (next <= pad) {
      el.scrollTop = targetScroll;
      if (next !== pad) {
        pad = next;
        el.style.setProperty('--tail', `${next}px`);
      }
    } else {
      pad = next;
      el.style.setProperty('--tail', `${next}px`);
      el.scrollTop = targetScroll;
    }
  }

  effect(() => {
    shown.get();
    if (follow.get()) pin();
    else {
      const el = box.peek();
      if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 24) follow.set(true);
    }
  });
  // Re-pin when the box's height follows the page (Run stretches it to the rail) or the entries re-wrap (the mono
  // font arriving, a narrower column).
  const inner = signal<HTMLDivElement>();
  effect(() => {
    const el = box.get();
    const content = inner.get();
    if (!el || !content) return;
    // Next frame: pinning changes the tail padding, which would resize what is observed inside this callback.
    let frame = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => follow.peek() && pin());
    });
    ro.observe(el);
    ro.observe(content);
    return () => (cancelAnimationFrame(frame), ro.disconnect());
  });

  function onscroll() {
    const el = box.peek();
    if (!el) return;
    follow.set(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
  }

  /** At most one announcement this often: a batch of the lines that arrived since the last. */
  const ANNOUNCE_MS = 10_000;
  const announce = signal(false);
  const said = signal('');
  /** The last line announced (or the last one there when announcing was turned on): later ones are new. */
  let heard: LogLine | undefined;
  const lineText = (l: LogLine) => (l.level === 'info' ? l.message : `${l.level}: ${l.message}`);
  /** The lines since `heard` that the level filter shows, said as one batch. */
  function flushSaid() {
    const logs = props.log.items;
    const at = heard ? logs.lastIndexOf(heard) : -1;
    const min = level.peek() === 'all' ? 0 : order[level.peek() as 'warn' | 'error'];
    // `heard` may have left the capped list (or a history load replaced it): then what came after it in time.
    const since = at >= 0 || !heard ? logs.slice(at + 1) : logs.filter((l) => l.t > heard!.t);
    const fresh = since.filter((l) => order[l.level] >= min);
    heard = logs[logs.length - 1] ?? heard;
    if (!fresh.length) return;
    const last = fresh[fresh.length - 1]!;
    said.set(fresh.length === 1 ? lineText(last) : `${fresh.length} new log lines. Latest: ${lineText(last)}`);
  }
  effect(() => {
    if (!announce.get() || !props.live) return;
    // From now on: what was there when it was turned on is not news.
    untrack(() => {
      const logs = props.log.items;
      heard = logs[logs.length - 1];
      said.set('');
    });
    const t = setInterval(flushSaid, ANNOUNCE_MS);
    return () => clearInterval(t);
  });

  const levels = () => (
    <>
      {props.live ? (
        // Off by default: the log never speaks unless asked (see the header comment).
        <span class="announce" use:tooltip={announce.get() ? 'Stop announcing new lines' : 'Announce new lines'}>
          <Button
            size="sm"
            variant="ghost"
            icon={announce.get() ? Volume2 : VolumeOff}
            aria-pressed={announce.get()}
            aria-label="Announce new lines"
            onclick={() => announce.set(!announce.peek())}
          />
        </span>
      ) : null}
      <Segmented
        size="sm"
        label="Log level"
        bind:value={level}
        options={[
          { value: 'all', label: 'All' },
          { value: 'warn', label: 'Warn', meta: counts.get().warn ? String(counts.get().warn) : undefined },
          { value: 'error', label: 'Error', meta: counts.get().error ? String(counts.get().error) : undefined },
        ]}
      />
    </>
  );

  // No lines at all (a queued run): the state-height rule, the header and one compact line; nothing to filter yet.
  return (
    <Panel title="Logs" meta={total() ? String(total()) : undefined} pad="none" id="logs" actions={total() ? levels : undefined}>
      {total() === 0 ? (
        <p class="none">No log lines yet.</p>
      ) : (
        <div class="wrap">
          <div class="toolbar">
            <Input
              class="search"
              size="sm"
              type="search"
              icon={Search}
              placeholder="Search logs"
              aria-label="Search logs"
              bind:value={query}
            />
            {needle.get() ? <span class="matches">{lineCount(filtered.get().matched)}</span> : null}
            <Button
              class="copy"
              size="sm"
              variant="ghost"
              icon={Copy}
              disabled={shown.get().length === 0}
              onclick={copyVisible}
            >
              Copy visible
            </Button>
          </div>
          {/* Silent: new lines are read only through the batched status line below, when asked for. */}
          <div class="log" ref={box} onscroll={onscroll} role="log" aria-live="off" aria-label="Training log" tabindex={0}>
            <div ref={inner}>
              {before.get().hidden || (before.get().older && (!needle.get() || shown.get().length > 0)) ? (
                <p class="earlier">
                  {before.get().hidden ? (
                    <>
                      {lineCount(before.get().hidden + (needle.get() ? 0 : before.get().older))} earlier{' '}
                      <Button size="sm" variant="ghost" onclick={() => cap.set(Infinity)}>
                        Show {before.get().older ? lineCount(before.get().hidden) : 'them'}
                      </Button>
                      {needle.get() && before.get().older
                        ? ` (${lineCount(before.get().older)} older are in the run folder's log.jsonl).`
                        : null}
                    </>
                  ) : needle.get() ? (
                    <>
                      Search covers the latest {lineCount(counts.get().held)}; {lineCount(before.get().older)} earlier are in the run folder's log.jsonl.
                    </>
                  ) : (
                    <>{lineCount(before.get().older)} earlier are in the run folder's log.jsonl.</>
                  )}
                </p>
              ) : null}
              <For
                each={shown.get()}
                key={(l) => l}
                fallback={
                  <div
                    class="empty"
                    use:preserveEmptyFocus
                    onfocusin={() => (emptyFocused = true)}
                    onfocusout={(e) => {
                      if (!e.currentTarget.contains(e.relatedTarget as Node | null)) emptyFocused = false;
                    }}
                  >
                    <p>
                      {needle.get()
                        ? `No log lines match "${query.get().trim()}"${level.get() !== 'all' ? ` at ${level.get()} level` : ''}${
                            counts.get().earlier
                              ? ` in the latest ${lineCount(counts.get().held)} (${lineCount(counts.get().earlier)} earlier are in the run folder's log.jsonl)`
                              : ''
                          }.`
                        : `No log lines${level.get() !== 'all' ? ` at ${level.get()} level` : ''} yet.`}
                    </p>
                    {needle.get() || level.get() !== 'all' ? (
                      <Button
                        size="sm"
                        onclick={() => {
                          batch(() => {
                            query.set('');
                            level.set('all');
                          });
                          box.peek()?.focus({ preventScroll: true });
                        }}
                      >
                        Clear filters
                      </Button>
                    ) : null}
                  </div>
                }
              >
                {(l) => {
                  // Each row is keyed by its immutable LogLine reference; tokens are cloned statically with zero effects.
                  const item = untrack(l);
                  return (
                    <p class={`line ${item.level}`}>
                      <span class="t">{clock(item.t)}</span>
                      <span class="lv">{item.level}</span>
                      <span class="m" ref={(el) => fillMessage(el, item.message)}></span>
                    </p>
                  );
                }}
              </For>
            </div>
          </div>
          <p class="sr-only" role="status">
            {announce.get() && props.live ? said.get() : ''}
          </p>
          <p class="sr-only" aria-live="polite" aria-atomic="true">
            {searchStatus.get()}
          </p>
          {!follow.get() ? (
            <div class="jump">
              <Button
                size="sm"
                icon={ArrowDownToLine}
                onclick={() => {
                  follow.set(true);
                  pin();
                  box.peek()?.focus({ preventScroll: true });
                }}
              >
                Follow
              </Button>
            </div>
          ) : null}
        </div>
      )}
    </Panel>
  );
}

css`
  .earlier {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin: 0;
    padding: 0 var(--space-4) var(--space-2);
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    /* A snap point as the lines are, so the box can come to rest on it (scrolled to the top). */
    scroll-snap-align: start;
  }
  .announce {
    display: inline-flex;
  }
  .wrap {
    position: relative;
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .toolbar {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    background: var(--bg-panel);
  }
  .toolbar :global(.search) {
    flex: 1 1 160px;
    max-width: 320px;
  }
  .matches {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    font-variant-numeric: tabular-nums;
  }
  .toolbar :global(.copy) {
    margin-left: auto;
  }
  .log {
    position: relative;
    flex: 1 1 auto;
    height: var(--log-h, 300px);
    overflow: auto;
    /* The end's padding grows by --tail when pinned, so the top entry is whole (pin()). */
    padding: var(--space-2) 0 calc(var(--space-2) + var(--tail, 0px));
    /* Read by hand, the view comes to rest on an entry's start. */
    scroll-snap-type: y proximity;
    scroll-padding-top: var(--space-2);
    background: var(--bg-inset);
    border-top: 1px solid var(--line-subtle);
    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    line-height: 18px;
  }
  /* The top padding stays clear while scrolled: a band in the log's own fill over the end of the entry above the
     one the view starts on (pin()), so the first line shown is always a whole entry. */
  .log::before {
    content: '';
    position: sticky;
    top: calc(-1 * var(--space-2));
    z-index: 1;
    display: block;
    height: var(--space-2);
    margin: calc(-1 * var(--space-2)) 0 0;
    background: var(--bg-inset);
  }
  .log:focus-visible {
    outline-offset: -2px;
  }
  .line {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    padding: 0 var(--space-4);
    scroll-snap-align: start;
  }
  /* Wrap at spaces only: a piece (a word, a \`key value\` pair, a flag and its value) never breaks inside, except one
     too long for any line, as a last resort. */
  .m {
    white-space: pre-wrap;
  }
  .w {
    white-space: nowrap;
  }
  .w.long {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .line:hover {
    background: var(--bg-hover);
  }
  .t {
    color: var(--text-tertiary);
    flex-shrink: 0;
  }
  .lv {
    width: 40px;
    flex-shrink: 0;
    color: var(--text-tertiary);
  }
  .m {
    color: var(--text-secondary);
    min-width: 0;
  }
  .debug .m {
    color: var(--text-tertiary);
  }
  .warn .lv,
  .warn .m {
    color: var(--warning-text);
  }
  .error .lv,
  .error .m {
    color: var(--danger-text);
  }
  .none {
    padding: 0 var(--space-4) var(--space-4);
    color: var(--text-tertiary);
  }
  .empty {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: var(--space-2);
    padding: var(--space-4);
    color: var(--text-tertiary);
    font-family: var(--font-sans);
  }
  .jump {
    position: absolute;
    right: var(--space-4);
    bottom: var(--space-3);
  }
  @media (max-width: 767px) {
    /* Phones: time and level on one 12 px tertiary line above the message, which gets the full width. */
    .line {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      align-items: start;
      gap: 2px var(--space-2);
      padding: 6px var(--space-3);
    }
    /* A hairline between entries, so each meta line reads as the head of the message under it. */
    .line + .line {
      border-top: 1px solid var(--line-subtle);
    }
    .t,
    .lv {
      font-family: var(--font-sans);
      font-size: var(--type-sm-size);
      line-height: 16px;
      font-variant-numeric: tabular-nums;
    }
    .lv {
      width: auto;
    }
    .lv::before {
      content: '·';
      margin-right: var(--space-2);
      color: var(--text-tertiary);
    }
    .m {
      grid-column: 1 / -1;
    }
  }
`;
