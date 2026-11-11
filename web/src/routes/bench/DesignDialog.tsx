// fw port of DesignDialog.svelte.
/**
 * "Design with an agent" (docs/BENCH.md, Designing a plan with an agent): asks Claude Code or Codex once (no tools)
 * for a plan drawn from the catalog, from the benchmark's description, its classes and four of its images. While the
 * request runs, the dialog says that one request is in flight and what it carries. The answer is a proposal shown
 * against the current plan: what it adds, removes and changes, each with the agent's reason, and why the whole; "Use
 * this plan" loads it into the editor, still unsaved, "Discard" drops it. A failure shows the server's message and
 * its fix, with Try again.
 */
import RotateCw from '$icons/rotate-cw';
import Sparkles from '$icons/sparkles';
import { bindable, computed, createUniqueId, css, effect, For, signal } from 'fw';
import { hub } from '$lib/api/index';
import type { Bench, BenchCatalog, BenchDesign, BenchPlan, BenchPlanCondition, LabelAgent } from '$lib/api/types';
import { DESIGN_NOTE_MAX } from '$lib/bench/rules';
import { diffPlans, planPasses, type PlanChange } from '$lib/bench/plan';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import Field from '$lib/ui/Field.tsx';
import Gate from '$lib/ui/Gate';
import Meta from '$lib/ui/Meta';
import Spinner from '$lib/ui/Spinner';
import TextArea from '$lib/ui/TextArea';
import { modelLabel, usageFacts } from '../label/words.fw';
import AgentFields from '../label/AgentFields';
import type { AgentChoice } from '../label/agentChoice';
import { conditionsText } from './words.fw';

export interface DesignDialogProps {
  open?: boolean;
  bench: Bench;
  catalog: BenchCatalog;
  /** The plan the proposal is set against: the editor's, saved or not. */
  current: BenchPlan;
  agents: readonly LabelAgent[] | undefined;
  agentsError: unknown;
  retryAgents: () => void;
  choice: AgentChoice;
  /** Loads the proposal into the editor (unsaved). */
  onuse: (plan: BenchPlan) => void;
}

export default function DesignDialog(props: DesignDialogProps) {
  const open = bindable(props, 'open', false);
  const uid = createUniqueId();
  const note = signal('');
  const step = signal<'ask' | 'asking' | 'proposal'>('ask');
  const design = signal<BenchDesign | null>(null);
  const error = signal<unknown>(undefined);
  let ctrl: AbortController | null = null;

  effect(() => {
    if (open.get()) return;
    ctrl?.abort();
    ctrl = null;
    step.set('ask');
    design.set(null);
    error.set(undefined);
  });

  const agent = computed(() => props.choice.agent);
  const asked = computed(() => {
    const a = agent.get();
    return a ? `${a.name} (${modelLabel(props.choice.model || null, a)})` : 'The agent';
  });
  const why = computed(() =>
    !agent.get()
      ? props.agents
        ? 'Install Claude Code or Codex on the server’s machine'
        : 'Checking which agents are installed'
      : note.get().length > DESIGN_NOTE_MAX
        ? 'Shorten the note first'
        : null,
  );

  async function ask() {
    const a = props.choice.request();
    if (!a) return;
    ctrl?.abort();
    ctrl = new AbortController();
    const signal = ctrl.signal;
    step.set('asking');
    error.set(undefined);
    try {
      const d = await hub.api.designBench(props.bench.id, { agent: a.agent, model: a.model, effort: a.effort, note: note.peek().trim() }, signal);
      if (signal.aborted) return;
      design.set(d);
      step.set('proposal');
    } catch (e) {
      if (signal.aborted) return;
      error.set(e);
      step.set('ask');
    }
  }

  const diff = computed(() => {
    const d = design.get();
    return d ? diffPlans(props.current, d.plan, props.catalog.conditions) : null;
  });
  const name = (id: string) => props.catalog.conditions.find((c) => c.id === id)?.name ?? id;
  const sev = (c: BenchPlanCondition | null) => (c ? c.severities.join(', ') : '');
  const proposedBy = computed(() => {
    const by = design.get()?.plan.designed_by;
    return by
      ? `${by.agent === 'claude' ? 'Claude Code' : 'Codex'}${by.model ? ` (${modelLabel(by.model, props.agents?.find((x) => x.id === by.agent))})` : ''}`
      : asked.get();
  });

  function use() {
    const d = design.peek();
    if (!d) return;
    props.onuse(d.plan);
    open.set(false);
  }

  const change = (c: () => PlanChange, kind: 'added' | 'removed' | 'changed') => (
    <li class="ch">
      <span class="ch-head">
        <span class="ch-name">{name(c().condition)}</span>
        <span class="ch-what num">
          {kind === 'added' && c().to ? (
            <Meta>
              <span>severities {sev(c().to)}</span>
              <span>weight {c().to!.weight}</span>
            </Meta>
          ) : kind === 'removed' ? (
            <span>left out</span>
          ) : c().from && c().to ? (
            <Meta>
              {c().parts.includes('severities') ? (
                <span>
                  severities {sev(c().from)} → {sev(c().to)}
                </span>
              ) : null}
              {c().parts.includes('weight') ? (
                <span>
                  weight {c().from!.weight} → {c().to!.weight}
                </span>
              ) : null}
            </Meta>
          ) : null}
        </span>
      </span>
      {c().to?.why ? <span class="ch-why">{c().to!.why}</span> : null}
    </li>
  );

  return (
    <Dialog
      bind:open={open}
      size={step.get() === 'proposal' ? 'lg' : 'md'}
      title={step.get() === 'proposal' ? 'Proposed plan' : 'Design the plan with an agent'}
      description={
        step.get() === 'proposal'
          ? `${proposedBy.get()} proposes ${conditionsText(design.get()?.plan.conditions.length ?? 0)} for ${props.bench.name}, set against the current plan. Use this plan puts it in the editor, unsaved.`
          : 'One request, no tools: the agent reads the description, the classes and four of the images, and answers with a plan from the catalog. You see it next to the current plan first.'
      }
      footer={() => (
        <>
          {step.get() === 'proposal' ? (
            <>
              <Button variant="ghost" onclick={() => open.set(false)}>
                Discard
              </Button>
              <Button icon={RotateCw} onclick={() => step.set('ask')}>
                Ask again
              </Button>
              <Button variant="primary" onclick={use}>
                Use this plan
              </Button>
            </>
          ) : step.get() === 'asking' ? (
            <Button onclick={() => open.set(false)}>Cancel</Button>
          ) : (
            <>
              {why.get() ? (
                <p class="why" data-gate-why>
                  {why.get()}
                </p>
              ) : null}
              <Button variant="ghost" onclick={() => open.set(false)}>
                Cancel
              </Button>
              <Gate reason={why.get()}>
                {(reason) => (
                  <Button variant="primary" icon={error.get() ? RotateCw : Sparkles} disabled={!!reason()} aria-describedby={reason()} onclick={ask}>
                    {error.get() ? 'Try again' : `Ask ${agent.get()?.name ?? 'the agent'}`}
                  </Button>
                )}
              </Gate>
            </>
          )}
        </>
      )}
    >
      {step.get() === 'ask' ? (
        <div class="form">
          {/* A failed request first, where the eye lands after the wait: the server's message and its fix. */}
          {error.get() ? <ErrorState error={error.get()} title="The agent did not give a plan" compact /> : null}
          <div class="reads">
            <span class="group-label">What it reads</span>
            {props.bench.description.trim() ? (
              <p class="desc">{props.bench.description}</p>
            ) : (
              <p class="desc none">
                No description yet: the agent has only the classes and the images to go on. Describe the camera on the benchmark's page first for a plan that fits it.
              </p>
            )}
            <p class="facts num">
              <Meta>
                <span>
                  {props.bench.classes.length} classes: {props.bench.classes.join(', ')}
                </span>
                <span>4 of its images</span>
              </Meta>
            </p>
          </div>
          <AgentFields
            agents={props.agents}
            agentsError={props.agentsError}
            retryAgents={props.retryAgents}
            choice={props.choice}
            effortHint={(a) => `More effort means more reasoning before it answers. The default, ${a.default_effort}, is enough for one plan.`}
          />
          <Field label="Note" for={`${uid}-note`} hint="Optional. Anything to add for this request: what matters most, what to leave out.">
            <TextArea id={`${uid}-note`} bind:value={note} rows={2} placeholder="Most footage is at dusk. Ignore compression: the camera records to a card." />
          </Field>
        </div>
      ) : step.get() === 'asking' ? (
        <div class="asking" role="status">
          <Spinner size={16} />
          <div class="a-text">
            <p class="a-title">{asked.get()} is designing the plan</p>
            <p class="a-sub">
              One request in flight, with the description, {props.bench.classes.length} classes and 4 images. It can take up to two minutes.
            </p>
          </div>
        </div>
      ) : design.get() && diff.get() ? (
        <div class="proposal">
          {design.get()!.plan.rationale ? (
            <div class="rationale">
              <span class="group-label">Why</span>
              <p>{design.get()!.plan.rationale}</p>
            </div>
          ) : null}
          <p class="sum num">
            <Meta>
              <span>{conditionsText(design.get()!.plan.conditions.length)}</span>
              <span>{planPasses(design.get()!.plan)} severities</span>
              <For each={usageFacts(design.get()!.usage)} key={(f) => f}>
                {(f) => <span>{f()}</span>}
              </For>
            </Meta>
          </p>
          {diff.get()!.added.length ? (
            <section aria-label="Added">
              <h3 class="group-label">Added · {diff.get()!.added.length}</h3>
              <ul>
                <For each={diff.get()!.added} key={(c) => c.condition}>
                  {(c) => change(c, 'added')}
                </For>
              </ul>
            </section>
          ) : null}
          {diff.get()!.changed.length ? (
            <section aria-label="Changed">
              <h3 class="group-label">Changed · {diff.get()!.changed.length}</h3>
              <ul>
                <For each={diff.get()!.changed} key={(c) => c.condition}>
                  {(c) => change(c, 'changed')}
                </For>
              </ul>
            </section>
          ) : null}
          {diff.get()!.removed.length ? (
            <section aria-label="Left out">
              <h3 class="group-label">Left out · {diff.get()!.removed.length}</h3>
              <p class="removed">{diff.get()!.removed.map((c) => name(c.condition)).join(', ')}</p>
            </section>
          ) : null}
          {diff.get()!.kept.length ? (
            <section aria-label="Kept">
              <h3 class="group-label">Kept as they are · {diff.get()!.kept.length}</h3>
              <ul>
                <For each={diff.get()!.kept} key={(c) => c.condition}>
                  {(c) => (
                    <>
                      {c().to?.why ? (
                        change(c, 'changed')
                      ) : (
                        <li class="ch">
                          <span class="ch-name">{name(c().condition)}</span>
                        </li>
                      )}
                    </>
                  )}
                </For>
              </ul>
            </section>
          ) : null}
        </div>
      ) : null}
    </Dialog>
  );
}

css`
  .form {
    display: grid;
    gap: var(--space-4);
  }
  .form :global(.seg) {
    justify-self: start;
  }
  .reads {
    display: grid;
    gap: var(--space-1);
    padding: var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-raised);
  }
  .desc {
    color: var(--text-primary);
  }
  .desc.none,
  .facts {
    color: var(--text-secondary);
  }
  .facts {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .asking {
    display: grid;
    grid-template-columns: 16px minmax(0, 1fr);
    gap: var(--space-3);
    align-items: start;
    padding: var(--space-3) 0 var(--space-2);
  }
  .a-title {
    font-weight: var(--weight-medium);
  }
  .a-sub {
    color: var(--text-secondary);
  }
  .proposal {
    display: grid;
    gap: var(--space-4);
  }
  .rationale {
    display: grid;
    gap: var(--space-1);
  }
  .rationale p {
    color: var(--text-primary);
    font-weight: var(--weight-regular);
  }
  .sum {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  section {
    display: grid;
    gap: var(--space-2);
  }
  ul {
    display: grid;
    border-top: 1px solid var(--line-subtle);
  }
  .ch {
    display: grid;
    gap: 2px;
    padding: var(--space-2) 0;
    border-bottom: 1px solid var(--line-subtle);
  }
  .ch-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .ch-name {
    font-weight: var(--weight-medium);
  }
  .ch-what {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
    text-align: right;
  }
  .ch-why {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .removed {
    color: var(--text-secondary);
  }
  .why {
    margin-right: auto;
    align-self: center;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  @media (max-width: 767px) {
    .why {
      flex-basis: 100%;
    }
    .ch-head {
      flex-wrap: wrap;
    }
    .ch-what {
      text-align: left;
    }
  }
`;
