// fw port of RunningPanel.svelte.
/**
 * An entry that has not finished: where it is, live (the benchmark's stream). Queued: its place behind the model
 * entries that run first. Running: the share done, the pass it is on (the unchanged images, or a condition at a
 * severity), and the plan's passes as a checklist (done, now, to come), so "haze 3 of 12 conditions" is something
 * seen. Stopping: that it is ending what is in flight. An open external entry is receiving its predictions in parts
 * from its sender (any order, so no checklist): the records received of all the plan could take, and that it is
 * scored when its sender finishes it. The score lands here when it is done.
 */
import Check from '$icons/check';
import { computed, css, For } from 'fw';
import type { Bench, BenchCatalog, BenchEntry } from '$lib/api/types';
import { num } from '$lib/format';
import Panel from '$lib/ui/Panel';
import Progress from '$lib/ui/Progress';
import { progressFacts, progressShare, receiving } from '../words.fw';

export interface RunningPanelProps {
  entry: BenchEntry;
  bench: Bench;
  catalog: BenchCatalog | undefined;
  lost: boolean;
  ahead: BenchEntry[];
}

const stateWord = (st: string) => (st === 'done' ? 'done' : st === 'now' ? 'running now' : 'to come');

export default function RunningPanel(props: RunningPanelProps) {
  const nameOf = (id: string) => props.catalog?.conditions.find((c) => c.id === id)?.name ?? id;
  /** The passes it runs, in order: the unchanged images, then each planned condition at each severity. */
  const passes = computed(() => [
    { key: 'clean', condition: null as string | null, severities: [] as number[] },
    ...(props.entry.conditions
      ? props.bench.plan.conditions.map((c) => ({ key: c.condition, condition: c.condition as string | null, severities: c.severities }))
      : []),
  ]);
  /** How many image passes are done: whole passes of the entry's images. */
  const passesDone = computed(() => (props.entry.images ? Math.floor(props.entry.progress.done / props.entry.images) : 0));
  /** Each (condition, severity) pass's index in the order run. */
  const indexOf = (cond: string | null, sev: number | null) => {
    let k = 0;
    for (const p of passes.get()) {
      if (p.condition === null) {
        if (cond === null) return k;
        k++;
        continue;
      }
      for (const s of p.severities) {
        if (p.condition === cond && s === sev) return k;
        k++;
      }
    }
    return -1;
  };
  const passState = (cond: string | null, sev: number | null) => {
    const i = indexOf(cond, sev);
    if (props.entry.status === 'queued') return 'todo';
    return i < passesDone.get() ? 'done' : i === passesDone.get() ? 'now' : 'todo';
  };
  const p = computed(() => progressFacts(props.entry, props.catalog));
  const recv = computed(() => receiving(props.entry));

  const pill = (st: () => string, label: () => string, text: () => string | number) => (
    <span class={`pill ${st()}`} aria-label={label()}>
      {st() === 'done' ? <Check size={12} strokeWidth={2} aria-hidden="true" /> : null}
      {text()}
    </span>
  );

  return (
    <Panel
      title={props.entry.status === 'queued' ? 'Waiting to run' : props.entry.status === 'stopping' ? 'Stopping' : recv.get() ? 'Receiving predictions' : 'Running'}
      meta={recv.get() ? undefined : (p.get().pct ?? undefined)}
      id="progress"
    >
      <div class="run">
        {recv.get() ? (
          <>
            <Progress value={progressShare(props.entry)} size="md" tone={props.lost ? 'neutral' : 'accent'} label={`${props.entry.name}: records received`} />
            <p class="lead num">{p.get().pct}, sent in parts by whatever runs the model (a script, or Add entry here).</p>
            <p class="cap">
              It is scored when its sender finishes it: every unchanged image answered, each condition whole or left out. Stopping it discards what was
              received.
            </p>
          </>
        ) : props.entry.status === 'queued' ? (
          <p class="lead">
            Model entries run one at a time on the server’s GPU.{' '}
            {props.ahead.length ? (
              <>
                {props.ahead.length === 1 ? 'One entry runs' : `${props.ahead.length} entries run`} before it: {props.ahead.map((e) => e.name).join(', ')}.
              </>
            ) : null}
          </p>
        ) : (
          <>
            <Progress
              value={progressShare(props.entry)}
              size="md"
              tone={props.lost || props.entry.status === 'stopping' ? 'neutral' : 'accent'}
              label={`${props.entry.name} progress`}
            />
            <p class="lead num">
              {p.get().where}
              {props.entry.status === 'running' ? (
                <>
                  {' '}· {num(props.entry.progress.done)} of {num(props.entry.progress.total)} image passes
                </>
              ) : null}
            </p>
          </>
        )}
        {!recv.get() ? (
          <>
            <ol class="passes" aria-label="Passes it runs">
              <For each={passes.get()} key={(ps) => ps.key}>
                {(ps) => (
                  <li>
                    <span class="p-name">{ps().condition ? nameOf(ps().condition!) : 'Unchanged images'}</span>
                    <span class="p-sev">
                      {ps().condition === null ? (
                        pill(
                          () => passState(null, null),
                          () => stateWord(passState(null, null)),
                          () => `${num(props.entry.images)} images`,
                        )
                      ) : (
                        <For each={ps().severities} key={(s) => s}>
                          {(s) =>
                            pill(
                              () => passState(ps().condition, s()),
                              () => `Severity ${s()}: ${stateWord(passState(ps().condition, s()))}`,
                              () => s(),
                            )
                          }
                        </For>
                      )}
                    </span>
                  </li>
                )}
              </For>
            </ol>
            <p class="cap">
              The score and the full result appear here when it is done
              {props.entry.conditions || !props.bench.plan.conditions.length ? '' : '; it runs the unchanged images only, so it will have no robustness'}.
            </p>
          </>
        ) : null}
      </div>
    </Panel>
  );
}

css`
  .run {
    display: grid;
    gap: var(--space-3);
  }
  .lead {
    color: var(--text-secondary);
  }
  .passes {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 0 var(--space-6);
    margin-top: var(--space-1);
  }
  .passes li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    min-height: 36px;
    border-top: 1px solid var(--line-subtle);
  }
  .p-name {
    color: var(--text-secondary);
  }
  .p-sev {
    display: flex;
    gap: var(--space-1);
  }
  .pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 2px;
    min-width: 24px;
    height: 24px;
    padding: 0 6px;
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    font-variant-numeric: tabular-nums;
  }
  .pill.done {
    background: var(--bg-active);
    color: var(--text-primary);
  }
  /* Running now: the live mark, lime meaning "now" (DESIGN_DIRECTION §2.6). */
  .pill.now {
    box-shadow: inset 0 0 0 1.5px var(--accent-line);
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .pill.todo {
    color: var(--text-tertiary);
    box-shadow: inset 0 0 0 1px var(--line-default);
  }
  .cap {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
`;
