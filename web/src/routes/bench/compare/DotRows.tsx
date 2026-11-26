/**
 * An aligned dot plot: one row per thing (a condition, a class), one dot per entry on one shared 0–1 scale, so who
 * wins where reads down the rows. Each entry keeps its hue and a shape of its own; a row pointed at (or focused)
 * reads out every entry's value in words under the plot; an entry kept in front (the legend) draws its dots over the
 * others' and the rest step back. The scale's ticks sit over the first row and under the last.
 */
import { computed, css, For, signal } from 'fw';
import { capTicks } from '$lib/charts/domain';
import { along } from '../leaderboard';

/** A compared entry as the dot plots draw it: its hue and its shape (the shape says whose dot it is without the hue). */
export interface DotKey {
  id: string;
  name: string;
  /** `var(--chart-N)`: the mark's ink. */
  color: string;
  shape: 'circle' | 'square' | 'diamond' | 'triangle';
}

export interface DotRow {
  key: string;
  name: string;
  values: (number | null)[];
  note?: string;
}

export interface DotRowsProps {
  rows: DotRow[];
  keys: DotKey[];
  domain: [number, number];
  format: (v: number) => string;
  /** The scale's tick labels (default `format`). */
  tickFormat?: (v: number) => string;
  label: string;
  highlight?: string | null;
  /** A second line under a row's name (a group, a count), small. */
  sub?: (key: string) => string;
}

/** The entry's shape, filled with its hue. */
const shape = (k: () => DotKey) => (
  <>
    {k().shape === 'circle' ? (
      <circle r="4.5" style:fill={k().color} />
    ) : k().shape === 'square' ? (
      <rect x="-4" y="-4" width="8" height="8" style:fill={k().color} />
    ) : k().shape === 'diamond' ? (
      <path d="M0 -5.5 L5.5 0 L0 5.5 L-5.5 0 Z" style:fill={k().color} />
    ) : (
      <path d="M0 -5 L5 4.5 L-5 4.5 Z" style:fill={k().color} />
    )}
  </>
);

export default function DotRows(props: DotRowsProps) {
  const highlight = () => props.highlight ?? null;
  const pointed = signal<string | null>(null);
  // 3–5 nice ticks inside the scale (./lib/charts/domain), never a crowded last one at its edge.
  const ticks = computed(() => capTicks(props.domain[0], props.domain[1], 5));
  const pos = (v: number) => `${(along(props.domain, v) * 100).toFixed(2)}%`;
  const read = computed(() => props.rows.find((r) => r.key === pointed.get()));
  const say = (r: DotRow) => `${r.name}: ${props.keys.map((k, i) => `${k.name} ${r.values[i] == null ? 'not run' : props.format(r.values[i]!)}`).join(', ')}`;
  const spoken = signal('');
  let listEl: HTMLUListElement | undefined;
  function onkeydown(e: KeyboardEvent) {
    const rows = props.rows;
    if (!rows.length) return;
    const i = rows.findIndex((r) => r.key === pointed.peek());
    let j = i;
    if (e.key === 'ArrowDown') j = Math.min(rows.length - 1, i + 1);
    else if (e.key === 'ArrowUp') j = i < 0 ? 0 : Math.max(0, i - 1);
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = rows.length - 1;
    else if (e.key === 'Escape') {
      if (i >= 0) e.preventDefault();
      pointed.set(null);
      spoken.set('');
      return;
    } else return;
    e.preventDefault();
    pointed.set(rows[j]!.key);
    spoken.set(say(rows[j]!));
    listEl?.children[j]?.scrollIntoView?.({ block: 'nearest' });
  }
  /** Painted last: the entry kept in front. */
  const order = computed(() => {
    const h = highlight();
    return props.keys.map((k, i) => ({ k, i })).sort((a, b) => Number(a.k.id === h) - Number(b.k.id === h));
  });
  /** The pointed row's value of the entry at `i`, in words. */
  const readText = (i: number) => {
    const r = read.get();
    if (!r) return '';
    return r.values[i] === null ? 'not run' : props.format(r.values[i]!);
  };
  const tickText = (t: number) => (props.tickFormat ?? props.format)(t);

  const mark = (k: () => DotKey, v: () => number, dim: () => boolean) => (
    <svg class="m" class:dim={dim()} style:left={pos(v())} width="12" height="12" viewBox="-6 -6 12 12" aria-hidden="true">
      {shape(k)}
    </svg>
  );
  const glyph = (k: () => DotKey) => (
    <svg class="g" width="12" height="12" viewBox="-6 -6 12 12" aria-hidden="true">
      {shape(k)}
    </svg>
  );

  return (
    <div
      class="dots"
      role="group"
      aria-roledescription="chart"
      aria-label={`${props.label}. Focus and use the up and down arrow keys to read each row.`}
      tabindex={0}
      onkeydown={onkeydown}
      onblur={() => pointed.set(null)}
    >
      <div class="row axis" aria-hidden="true">
        <span></span>
        <span class="scale">
          <For each={ticks.get()} key={(t) => t}>
            {(t) => (
              <span class="tk num" style:left={pos(t())}>
                {tickText(t())}
              </span>
            )}
          </For>
        </span>
      </div>
      <ul ref={(el) => (listEl = el)}>
        <For each={props.rows} key={(r) => r.key}>
          {(r) => (
            <li class="row" class:on={pointed.get() === r().key} onpointerenter={() => pointed.set(r().key)} onpointerleave={() => pointed.set(null)}>
              <span class="name">
                <span class="nm">{r().name}</span>
                {props.sub ? <span class="sub">{props.sub(r().key)}</span> : null}
              </span>
              <span class="track">
                <For each={ticks.get()} key={(t) => t}>
                  {(t) => <span class="grid" style:left={pos(t())}></span>}
                </For>
                <For each={order.get()} key={(o) => o.k.id}>
                  {(o) => {
                    const v = () => r().values[o().i] ?? null;
                    return <>{v() !== null ? mark(() => o().k, () => v()!, () => highlight() !== null && highlight() !== o().k.id) : null}</>;
                  }}
                </For>
              </span>
            </li>
          )}
        </For>
      </ul>
      <p class="sr-only" aria-live="polite">
        {spoken.get()}
      </p>
      <p class="readout num" aria-hidden="true">
        {read.get() ? (
          <>
            <span class="t">{read.get()?.name}</span>
            <For each={props.keys} key={(k) => k.id}>
              {(k, i) => (
                <span class="rv">
                  {glyph(k)}
                  {readText(i())}
                </span>
              )}
            </For>
          </>
        ) : (
          <span class="t">Point at a row to read it.</span>
        )}
      </p>
    </div>
  );
}

css`
  .dots {
    display: grid;
    min-width: 0;
  }
  .row {
    display: grid;
    grid-template-columns: minmax(120px, 200px) minmax(0, 1fr);
    align-items: center;
    gap: var(--space-4);
    min-height: 32px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
  }
  li.row {
    border-top: 1px solid var(--line-subtle);
    border-radius: 0;
  }
  li.row.on {
    background: var(--bg-hover);
  }

  .row.axis {
    min-height: 20px;
  }
  .name {
    display: grid;
    min-width: 0;
  }
  .nm {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .sub {
    font-size: var(--type-sm-size);
    line-height: 14px;
    color: var(--text-tertiary);
  }
  .scale,
  .track {
    position: relative;
    display: block;
    margin: 0 8px;
  }
  .scale {
    height: 16px;
  }
  .track {
    height: 28px;
  }
  .tk {
    position: absolute;
    top: 0;
    transform: translateX(-50%);
    color: var(--chart-label);
    /* small: chart tick */
    font-size: var(--type-xs-size);
  }
  .grid {
    position: absolute;
    top: 2px;
    bottom: 2px;
    width: 1px;
    background: var(--chart-grid);
  }
  .m {
    position: absolute;
    top: 50%;
    margin: -6px 0 0 -6px;
    overflow: visible;
  }
  .m :global(:is(circle, rect, path)) {
    stroke: var(--bg-panel);
    stroke-width: 1.5;
  }
  .m.dim {
    opacity: 0.35;
  }
  .readout {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-4);
    min-height: 32px;
    padding: var(--space-2) var(--space-2) 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    border-top: 1px solid var(--line-subtle);
  }
  .t {
    color: var(--text-secondary);
  }
  .rv {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--text-primary);
  }
  .g {
    flex-shrink: 0;
  }
  @media (max-width: 767px) {
    .row {
      grid-template-columns: minmax(84px, 112px) minmax(0, 1fr);
      gap: var(--space-2);
    }
  }
`;
