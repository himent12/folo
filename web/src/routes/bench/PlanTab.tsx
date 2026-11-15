/**
 * A benchmark's conditions (its plan): the catalog's conditions by group, each on or off, at which severities and how
 * much it counts, with its reason when the plan has one; who designed the plan and why; and a preview of any
 * condition on the benchmark's own images. The plan can change only while the benchmark has no entries: then it is
 * edited here (saved with Save plan; the page holds the edits and asks before they are lost) or designed by an agent;
 * after, it reads as a fixed test, with the one-line reason why.
 */
import Eye from '$icons/eye';
import Lock from '$icons/lock';
import Sparkles from '$icons/sparkles';
import { bindable, computed, css, effect, For, signal, untrack, type Signal } from 'fw';
import type { Bench, BenchCatalog, BenchCondition, BenchPlan } from '$lib/api/types';
import { planCondition, planError, planPasses, setWeight, toggleCondition, toggleSeverity, weightError } from '$lib/bench/plan';
import { GROUP_NAME, GROUPS, SEVERITIES } from '$lib/bench/rules';
import { AGENT_NAME } from '$lib/label';
import Button from '$lib/ui/Button';
import Gate from '$lib/ui/Gate';
import Panel from '$lib/ui/Panel';
import Switch from '$lib/ui/Switch';
import { tooltip } from '$lib/ui/tooltip';
import Callout from '../run/Callout';
import Preview from './Preview';
import { FROZEN } from './words';

export interface PlanTabProps {
  bench: Bench;
  /** The plan can still change: no entry finished or is under way ($lib/bench/rules planEditable). */
  editable: boolean;
  catalog: BenchCatalog;
  /** Bindable: the plan as edited (the page holds it, saved or not). */
  draft?: BenchPlan;
  dirty: boolean;
  saving: boolean;
  onsave: () => void;
  ondiscard: () => void;
  /** Opens "Design with an agent". */
  ondesign: () => void;
  /** Why designing cannot start now, or null. */
  designWhy: string | null;
}

export default function PlanTab(props: PlanTabProps) {
  const draft = bindable(props, 'draft') as Signal<BenchPlan>;

  /** The plan shown: the edits while it can change, else the benchmark's own. */
  const plan = computed(() => (props.editable ? draft.get() : props.bench.plan));
  /** The condition the preview shows: the plan's first at the start, then the one last looked at. */
  const first = () => props.bench.plan.conditions[0]?.condition ?? props.catalog.conditions[0]?.id ?? 'haze';
  const preview = signal(untrack(first));
  const severity = signal(3);
  /** Weights as typed, per condition, so a half-typed number is not thrown away. */
  const typed = signal<Record<string, string>>({});
  // A weight field emptied (or holding no number) leaves the draft at the last number typed: until it holds one again
  // it is wrong, so the field does not read as valid and Save plan does not store a weight the field does not show.
  const error = computed(() => {
    if (!props.editable) return null;
    const d = draft.get();
    const t = typed.get();
    const empty = d.conditions.find((c) => unreadable(t[c.condition]));
    if (!empty) return planError(d, props.catalog.conditions);
    const name = props.catalog.conditions.find((c) => c.id === empty.condition)?.name ?? empty.condition;
    return `${name}: the weight is empty. ${weightError(Number.NaN)}.`;
  });
  effect(() => {
    void props.bench.id;
    typed.set({});
  });
  // A draft changed from outside the field (Discard, a condition off and on, a save) drops what was typed for a weight
  // it no longer has: the field shows the plan's.
  effect(() => {
    const d = draft.get();
    untrack(() => {
      const t = typed.get();
      const kept = Object.entries(t).filter(([id, v]) => Number(v) === planCondition(d, id)?.weight);
      if (kept.length !== Object.keys(t).length) typed.set(Object.fromEntries(kept));
    });
  });

  const groups = computed(() =>
    GROUPS.map((g) => ({ g, conditions: props.catalog.conditions.filter((c) => c.group === g) })).filter((x) => x.conditions.length),
  );
  const designer = computed(() => {
    const by = plan.get().designed_by;
    return by ? `${AGENT_NAME[by.agent]}${by.model ? ` (${by.model[0]!.toUpperCase()}${by.model.slice(1)})` : ''}` : null;
  });
  const isDefault = computed(() => {
    const p = plan.get();
    return (
      !p.designed_by &&
      p.conditions.length === props.catalog.default_plan.conditions.length &&
      p.conditions.every((c) => c.weight === 1 && c.severities.join() === '1,3,5')
    );
  });

  function setTyped(id: string, v: string) {
    typed.set({ ...typed.peek(), [id]: v });
    const w = Number(v);
    if (v.trim() && Number.isFinite(w)) draft.set(setWeight(draft.peek(), id, w));
  }
  function show(id: string) {
    preview.set(id);
    const c = planCondition(plan.peek(), id);
    if (c && !c.severities.includes(severity.peek())) severity.set(c.severities.at(-1) ?? 3);
  }

  const row = (c: () => BenchCondition) => {
    const pc = computed(() => planCondition(plan.get(), c().id));
    const on = () => !!pc.get();
    const wErr = () => {
      const p = pc.get();
      if (!p || !props.editable) return null;
      return unreadable(typed.get()[c().id]) ? weightError(Number.NaN) : weightError(p.weight);
    };
    return (
      <li class="cond" class:off={!on()} class:shown={preview.get() === c().id}>
        <div class="c-main">
          {props.editable ? (
            <Switch
              label={c().name}
              hint={c().description}
              checked={on()}
              onchange={(v) => draft.set(toggleCondition(draft.peek(), c().id, v, props.catalog, props.bench.plan))}
            />
          ) : (
            <>
              <span class="c-name">{c().name}</span>
              <span class="c-desc">{c().description}</span>
            </>
          )}
        </div>
        {pc.get() ? (
          <>
            <div class="c-sev" role="group" aria-label={`Severities of ${c().name}`}>
              <For each={SEVERITIES} key={(s) => s}>
                {(s) => {
                  const has = () => pc.get()!.severities.includes(s());
                  const only = () => has() && pc.get()!.severities.length === 1;
                  return (
                    <>
                      {props.editable ? (
                        <button
                          type="button"
                          class="sev"
                          class:has={has()}
                          aria-pressed={has()}
                          aria-label={`Severity ${s()}`}
                          disabled={only()}
                          use:tooltip={only() ? 'A condition runs at one severity at least: turn it off instead' : null}
                          onclick={() => draft.set(toggleSeverity(draft.peek(), c().id, s()))}
                        >
                          {s()}
                        </button>
                      ) : (
                        <span class="sev static" class:has={has()} aria-label={has() ? `Severity ${s()}: runs` : `Severity ${s()}: not run`}>
                          {s()}
                        </span>
                      )}
                    </>
                  );
                }}
              </For>
            </div>
            <div class="c-weight">
              {props.editable ? (
                <label class="w">
                  <span class="w-x" aria-hidden="true">
                    ×
                  </span>
                  <input
                    type="number"
                    min="0.5"
                    max="10"
                    step="0.5"
                    inputmode="decimal"
                    aria-label={`Weight of ${c().name}`}
                    aria-invalid={wErr() ? 'true' : undefined}
                    value={typed.get()[c().id] ?? String(pc.get()!.weight)}
                    oninput={(e) => setTyped(c().id, e.currentTarget.value)}
                  />
                </label>
              ) : (
                <span class="w-static num" aria-label={`Weight ${pc.get()!.weight}`}>
                  × {pc.get()!.weight}
                </span>
              )}
            </div>
          </>
        ) : (
          <div class="c-not">Not in the plan</div>
        )}
        <div class="c-eye">
          <Button variant="ghost" size="sm" icon={Eye} aria-label={`Preview ${c().name}`} aria-pressed={preview.get() === c().id} onclick={() => show(c().id)} />
        </div>
        {pc.get()?.why ? (
          <span class="c-why">
            <Sparkles size={12} strokeWidth={1.75} aria-hidden="true" />
            <span>{pc.get()!.why}</span>
          </span>
        ) : null}
        {wErr() ? (
          <p class="c-err" role="alert">
            {c().name}: {wErr()}.
          </p>
        ) : null}
      </li>
    );
  };

  return (
    <div class="tab">
      {!props.editable ? (
        <div class="frozen">
          <Callout tone="neutral" icon={Lock} title={FROZEN} role="note">
            To change it, delete those entries (failed and stopped ones do not count) or make a new benchmark with the plan you want.
          </Callout>
        </div>
      ) : null}
      <div class="grid">
        <div class="plan">
          <Panel
            title="Plan"
            meta={`${plan.get().conditions.length} of ${props.catalog.conditions.length} conditions · ${planPasses(plan.get())} severities`}
            class={props.editable && props.dirty ? 'unsaved-plan' : ''}
            pad="none"
            actions={() => (
              <>
                {props.editable ? (
                  <>
                    {props.dirty ? (
                      <>
                        <span class="unsaved" data-gate-why>
                          Unsaved changes
                        </span>
                        <Button variant="ghost" size="sm" onclick={props.ondiscard}>
                          Discard
                        </Button>
                        <Gate reason={error.get() ? 'Fix the plan first' : null}>
                          {(why) => (
                            <Button variant="primary" size="sm" loading={props.saving} disabled={!!why()} aria-describedby={why()} onclick={props.onsave}>
                              Save plan
                            </Button>
                          )}
                        </Gate>
                      </>
                    ) : (
                      <Gate reason={props.designWhy}>
                        {(why) => (
                          <Button size="sm" icon={Sparkles} disabled={!!why()} aria-describedby={why()} onclick={props.ondesign}>
                            Design with an agent
                          </Button>
                        )}
                      </Gate>
                    )}
                  </>
                ) : null}
              </>
            )}
          >
            <div class="about">
              {designer.get() ? (
                <>
                  <p class="by">
                    <Sparkles size={14} strokeWidth={1.75} aria-hidden="true" />
                    <span>Designed by {designer.get()}{props.dirty ? ' · not saved yet' : ''}</span>
                  </p>
                  {plan.get().rationale ? <p class="rationale">{plan.get().rationale}</p> : null}
                </>
              ) : isDefault.get() ? (
                <p class="rationale">
                  The default plan: every condition at severities 1, 3 and 5, each counting the same. Weight what your camera meets, or let an agent design it from the description.
                </p>
              ) : !plan.get().conditions.length ? (
                <p class="rationale">No conditions: entries run the unchanged images only, and the score has no robustness axis.</p>
              ) : (
                <p class="rationale">Written by hand.</p>
              )}
              {error.get() ? (
                <p class="plan-err" role="alert">
                  {error.get()}
                </p>
              ) : null}
            </div>
            <For each={groups.get()} key={(x) => x.g}>
              {(x) => (
                <section class="group" aria-labelledby={`grp-${x().g}`}>
                  <h3 class="group-label" id={`grp-${x().g}`}>
                    {GROUP_NAME[x().g]}
                  </h3>
                  <ul>
                    <For each={x().conditions} key={(c) => c.id}>
                      {(c) => row(c)}
                    </For>
                  </ul>
                </section>
              )}
            </For>
          </Panel>
        </div>
        <aside class="rail" aria-label="Preview">
          <Preview bench={props.bench} catalog={props.catalog} bind:condition={preview} bind:severity={severity} />
        </aside>
      </div>
    </div>
  );
}

css`
  .tab {
    display: grid;
    gap: var(--space-4);
  }
  /* The plan on 8 columns, the preview a sticky rail on 4 (DESIGN_DIRECTION §2.2). */
  .grid {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: start;
  }
  /* With unsaved edits, the header holding Save plan stays in view under the top bar while lower rows are edited. */
  .plan :global(.panel.unsaved-plan > header) {
    position: sticky;
    top: var(--size-topbar);
    z-index: var(--z-sticky);
    background: var(--bg-panel);
    border-bottom: 1px solid var(--line-subtle);
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
  }
  .rail {
    position: sticky;
    top: calc(var(--size-topbar) + var(--space-4));
    min-width: 0;
  }
  .about {
    display: grid;
    gap: var(--space-2);
    padding: 0 var(--space-4) var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
  }
  .by {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-weight: var(--weight-medium);
  }
  .by :global(svg) {
    color: var(--text-tertiary);
  }
  .rationale {
    color: var(--text-secondary);
    max-inline-size: none;
  }
  .plan-err,
  .c-err {
    font-size: var(--type-sm-size);
    color: var(--danger-text);
  }
  .unsaved {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
    margin-right: var(--space-1);
  }
  .group + .group {
    border-top: 1px solid var(--line-subtle);
  }
  .group .group-label {
    padding: var(--space-3) var(--space-4) var(--space-1);
  }
  /* A row: the condition (its switch at the end of the text while the plan can change), its severities, its weight,
     the preview button. */
  .cond {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 136px 72px auto;
    align-items: center;
    column-gap: var(--space-4);
    padding: var(--space-3) var(--space-3) var(--space-3) var(--space-4);
  }
  .cond.shown {
    background: var(--bg-active);
    box-shadow: var(--select-bar);
  }
  .c-main {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .c-main :global(.switch .label) {
    font-weight: var(--weight-medium);
  }
  .cond.off .c-main :global(.switch .label) {
    color: var(--text-secondary);
  }
  .c-main :global(.switch .hint) {
    max-inline-size: none;
  }
  .c-name {
    font-weight: var(--weight-medium);
  }
  .cond.off .c-name {
    color: var(--text-secondary);
  }
  .c-desc,
  .c-why {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  /* The agent's reason is a line of its own under the condition, so the switch, the severities and the weight share
     the row's first line. */
  .c-why {
    grid-column: 1;
    display: grid;
    grid-template-columns: 12px minmax(0, 1fr);
    gap: 6px;
    margin-top: var(--space-2);
    color: var(--text-secondary);
  }
  .c-why :global(svg) {
    margin-top: 2px;
    color: var(--text-tertiary);
  }
  .c-sev {
    display: flex;
    gap: var(--space-1);
  }
  .c-not {
    grid-column: span 2;
    justify-self: start;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .sev {
    position: relative;
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--line-control);
    background: none;
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    font-variant-numeric: tabular-nums;
  }
  /* On: an ink edge on the active fill, the number in ink at 500 (a selection is ink, never lime); off, a faint
     outline and a tertiary number, so the five read as one set with the run ones standing out. */
  .sev.has {
    border-color: var(--text-primary);
    background: var(--bg-active);
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  @media (hover: hover) {
    button.sev:hover:not(:disabled):not(.has) {
      border-color: var(--line-control-hover);
      color: var(--text-primary);
    }
  }
  button.sev:disabled {
    cursor: not-allowed;
  }
  /* Touch: 40 px squares, each a 44 px target (the gap between them is its margin). */
  @media (pointer: coarse) {
    .sev {
      width: 40px;
      height: 40px;
    }
    button.sev::after {
      content: '';
      position: absolute;
      inset: -2px;
    }
  }
  /* Read-only: a run severity is a filled mark, one that does not run its bare number, faint. */
  .sev.static {
    border-color: transparent;
    color: var(--text-tertiary);
  }
  .sev.static.has {
    border-color: transparent;
    background: var(--bg-active);
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .c-weight {
    justify-self: end;
  }
  .w {
    display: flex;
    align-items: center;
    gap: 2px;
    height: var(--control-md);
    padding: 0 var(--space-2);
    border-radius: var(--radius-md);
    border: 1px solid var(--line-control);
    background: var(--bg-panel);
  }
  .w:has(input:focus-visible) {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: var(--size-focus-offset);
  }
  .w:has(input[aria-invalid='true']) {
    border-color: var(--danger-solid);
  }
  .w-x {
    color: var(--text-tertiary);
  }
  .w input {
    width: 40px;
    border: 0;
    background: none;
    color: var(--text-primary);
    font: inherit;
    font-variant-numeric: tabular-nums;
    text-align: right;
    outline: none;
  }
  @media (pointer: coarse) {
    .w input {
      font-size: var(--type-lg-size);
    }
  }
  .w-static {
    color: var(--text-secondary);
  }
  .c-err {
    grid-column: 1 / -1;
    margin-top: var(--space-1);
  }
  @media (max-width: 1023px) {
    .grid {
      grid-template-columns: minmax(0, 1fr);
    }
    .rail {
      position: static;
      order: -1;
    }
    /* Twelve rows under a preview: denser, each condition's sentence on demand (its eye shows it in the preview, where
       the condition is seen), the row being previewed keeping it. */
    .cond {
      padding-top: var(--space-2);
      padding-bottom: var(--space-2);
    }
    .cond:not(.shown) .c-main :global(.switch .hint),
    .cond:not(.shown) .c-desc {
      display: none;
    }
  }
  /* Phones: the condition (and its switch) over its severities and weight, the preview button beside it. */
  @media (max-width: 767px) {
    .cond {
      grid-template-columns: minmax(0, 1fr) auto auto;
      row-gap: var(--space-2);
      padding-right: var(--space-2);
    }
    .c-main {
      grid-row: 1;
      grid-column: 1 / span 2;
    }
    .c-eye {
      grid-row: 1;
      grid-column: 3;
      align-self: start;
    }
    .c-sev {
      grid-row: 2;
      grid-column: 1;
    }
    .c-weight {
      grid-row: 2;
      grid-column: 2;
    }
    .c-not {
      grid-row: 2;
      grid-column: 1 / span 2;
      justify-self: start;
    }
    .c-why {
      grid-row: 3;
      grid-column: 1 / -1;
      margin-top: 0;
    }
  }
`;

/** A weight field's text that is not a number (empty, or what the number input could not read). */
const unreadable = (v: string | undefined) => v !== undefined && !(v.trim() && Number.isFinite(Number(v)));
