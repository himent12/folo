/**
 * Per-class validation metrics of the latest epoch, with the change since the previous epoch in a column of its own.
 * A class's own metric is AP (AP50-95, AP50); mAP is only their mean.
 */
import { computed, css, For, Index } from 'fw';
import type { Epoch } from '$lib/api/types';
import { classInk } from '$lib/design/classInk';
import { unitTop } from '$lib/charts/domain';
import Panel from '$lib/ui/Panel';
import Skeleton from '$lib/ui/Skeleton';
import { tooltip } from '$lib/ui/tooltip';

export interface PerClassProps {
  classes: string[];
  epochs: Epoch[];
  version: number;
  loading: boolean;
  /** Rail width: class, AP50-95, change, bar. */
  compact?: boolean;
  /** The run's history failed to load: one quiet line in place of "appear after the first validation pass"; the
   * metrics panel has the error and its Retry. */
  unavailable?: boolean;
}

/** The last two epochs, compared by identity: a step (which bumps `version` too) leaves them as they were. */
const sameTwo = (a: [Epoch | undefined, Epoch | undefined], b: [Epoch | undefined, Epoch | undefined]) => a[0] === b[0] && a[1] === b[1];

export default function PerClass(props: PerClassProps) {
  // `epochs` changes in place; `version` is the signal that it did. Steps bump it several times a second while epochs
  // change once an epoch: the table is rebuilt only when the last two epochs are other objects (the Svelte original
  // rebuilt and re-sorted it on every step).
  const lastTwo = computed(
    (): [Epoch | undefined, Epoch | undefined] => {
      void props.version;
      const e = props.epochs;
      return [e[e.length - 1], e[e.length - 2]];
    },
    { equals: sameTwo },
  );
  const rows = computed(() => {
    const [last, prev] = lastTwo.get();
    const classes = props.classes;
    if (!last) return [];
    return last.val.per_class
      .map((c) => ({
        ...c,
        name: classes[c.class] ?? `class ${c.class}`,
        delta: prev ? c.map50_95 - (prev.val.per_class.find((p) => p.class === c.class)?.map50_95 ?? c.map50_95) : 0,
      }))
      .sort((a, b) => b.map50_95 - a.map50_95);
  });
  // The bars run 0–1 while any class reaches 0.5; below that a 0–1 bar is a dot, so they run from 0 to just over the
  // column's best (../../lib/charts/domain unitTop, as the charts' y axes do), and the panel's meta line says so.
  const barTop = computed(() => {
    const max = Math.max(0, ...rows.get().map((r) => r.map50_95));
    const top = unitTop(max);
    return max > 0 && top < 0.5 ? top : 1;
  });
  const epochNo = computed(() => lastTwo.get()[0]?.epoch ?? null);
  const compact = () => props.compact ?? false;

  return (
    <Panel
      title="Per class"
      meta={epochNo.get() ? `epoch ${epochNo.get()}${barTop.get() < 1 ? ` · bars 0–${+barTop.get().toFixed(3)}` : ''}` : undefined}
      pad="none"
      id="per-class"
    >
      {props.unavailable ? (
        <p class="none">Per-class results show once the run's metrics load.</p>
      ) : props.loading ? (
        <div class="sk">
          <Index each={[0, 1, 2, 3]}>{() => <Skeleton h="18px" />}</Index>
        </div>
      ) : rows.get().length === 0 ? (
        <p class="none">Per-class metrics appear after the first validation pass.</p>
      ) : (
        <div class="scroll">
          <table class="table dense">
            <thead>
              <tr>
                <th>Class</th>
                <th class="r">AP50-95</th>
                {/* "Δ prev" keeps the header short and on one line at every width; the full name is its tooltip and
                    accessible name. */}
                <th class="r delta-h" use:tooltip={'Change since the previous epoch'}>
                  <span aria-hidden="true">Δ prev</span>
                  <span class="sr-only">Change since the previous epoch</span>
                </th>
                <th class="bar-col">
                  <span class="sr-only">{barTop.get() < 1 ? `AP50-95 bar, scaled to 0–${+barTop.get().toFixed(3)}` : 'AP50-95 bar'}</span>
                </th>
                {!compact() ? (
                  <>
                    <th class="r">AP50</th>
                    <th class="r">Precision</th>
                    <th class="r">Recall</th>
                  </>
                ) : null}
              </tr>
            </thead>
            <tbody>
              <For each={rows.get()} key={(c) => c.class}>
                {(c) => (
                  <tr>
                    <td>
                      <span class="cls">
                        <span class="sw" style:background={classInk(c().class)}></span>
                        {c().name}
                      </span>
                    </td>
                    <td class="r num">{c().map50_95.toFixed(3)}</td>
                    <td class="r num delta" class:up={c().delta > 0.0005} class:down={c().delta < -0.0005}>
                      {Math.abs(c().delta) >= 0.0005 ? (
                        <span aria-label={`${c().delta > 0 ? 'up' : 'down'} ${Math.abs(c().delta).toFixed(3)}`}>
                          {c().delta > 0 ? '▲' : '▼'}&#8201;{Math.abs(c().delta).toFixed(3)}
                        </span>
                      ) : (
                        <span aria-label="no change">·</span>
                      )}
                    </td>
                    <td class="bar-col">
                      <span class="bar">
                        <span style:width={`${Math.min(1, c().map50_95 / barTop.get()) * 100}%`}></span>
                      </span>
                    </td>
                    {!compact() ? (
                      <>
                        <td class="r num">{c().map50.toFixed(3)}</td>
                        <td class="r num faint">{c().precision.toFixed(3)}</td>
                        <td class="r num faint">{c().recall.toFixed(3)}</td>
                      </>
                    ) : null}
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

css`
  .scroll {
    overflow-x: auto;
  }
  .sk {
    display: grid;
    gap: var(--space-3);
    padding: var(--space-4);
  }
  /* The state-height rule (DESIGN_DIRECTION §2.8): the header, then one compact line with the panel's padding. */
  .none {
    padding: 0 var(--space-4) var(--space-4);
    color: var(--text-tertiary);
  }
  .cls {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    box-shadow: 0 0 0 1px var(--class-halo);
  }
  .delta {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  /* Direction by mark (▲ / ▼) in ink, not by status hue: a metric moving is not a status. */
  .delta.up {
    color: var(--text-secondary);
  }
  .delta.down {
    color: var(--text-tertiary);
  }
  .bar-col {
    width: 28%;
    /* Low enough that the table fits the rail at 960 px (about 278 px wide). */
    min-width: 40px;
  }
  .bar {
    display: block;
    height: 4px;
    border-radius: 2px;
    background: var(--bg-active);
    overflow: hidden;
  }
  th.delta-h {
    white-space: nowrap;
  }
  /* The table rule: one neutral inline bar. The class hue stays on the swatch; the bar is ink, primary on the
     hovered row (the rows select nothing). */
  .bar span {
    display: block;
    height: 100%;
    border-radius: 2px;
    background: var(--text-tertiary);
    transition:
      width var(--dur-chart) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard);
  }
  @media (hover: hover) {
    tr:hover .bar span {
      background: var(--text-primary);
    }
  }
`;
