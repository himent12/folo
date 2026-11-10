<script lang="ts">
  /**
   * "Design with an agent" (docs/BENCH.md, Designing a plan with an agent): asks Claude Code or Codex once (no tools)
   * for a plan drawn from the catalog, from the benchmark's description, its classes and four of its images. While the
   * request runs, the dialog says that one request is in flight and what it carries. The answer is a proposal shown
   * against the current plan: what it adds, removes and changes, each with the agent's reason, and why the whole; "Use
   * this plan" loads it into the editor, still unsaved, "Discard" drops it. A failure shows the server's message and
   * its fix, with Try again.
   */
  import RotateCw from '@lucide/svelte/icons/rotate-cw';
  import Sparkles from '@lucide/svelte/icons/sparkles';
  import { hub } from '$lib/api/index.svelte';
  import type { Bench, BenchCatalog, BenchDesign, BenchPlan, BenchPlanCondition, LabelAgent } from '$lib/api/types';
  import { DESIGN_NOTE_MAX } from '$lib/bench/rules';
  import { diffPlans, planPasses, type PlanChange } from '$lib/bench/plan';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';
  import TextArea from '$lib/ui/TextArea.svelte';
  import { modelLabel, usageFacts } from '../label/words';
  import AgentFields from '../label/AgentFields.svelte';
  import type { AgentChoice } from '../label/agentChoice.svelte';
  import { conditionsText } from './words';

  let {
    open = $bindable(false),
    bench,
    catalog,
    current,
    agents,
    agentsError,
    retryAgents,
    choice,
    onuse,
  }: {
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
  } = $props();

  const uid = $props.id();
  let note = $state('');
  let step = $state<'ask' | 'asking' | 'proposal'>('ask');
  let design = $state.raw<BenchDesign | null>(null);
  let error = $state.raw<unknown>(undefined);
  let ctrl: AbortController | null = null;

  $effect(() => {
    if (open) return;
    ctrl?.abort();
    ctrl = null;
    step = 'ask';
    design = null;
    error = undefined;
  });

  const agent = $derived(choice.agent);
  const asked = $derived(agent ? `${agent.name} (${modelLabel(choice.model || null, agent)})` : 'The agent');
  const why = $derived(!agent ? (agents ? 'Install Claude Code or Codex on the server’s machine' : 'Checking which agents are installed') : note.length > DESIGN_NOTE_MAX ? 'Shorten the note first' : null);

  async function ask() {
    const a = choice.request();
    if (!a) return;
    ctrl?.abort();
    ctrl = new AbortController();
    const signal = ctrl.signal;
    step = 'asking';
    error = undefined;
    try {
      const d = await hub.api.designBench(bench.id, { agent: a.agent, model: a.model, effort: a.effort, note: note.trim() }, signal);
      if (signal.aborted) return;
      design = d;
      step = 'proposal';
    } catch (e) {
      if (signal.aborted) return;
      error = e;
      step = 'ask';
    }
  }

  const diff = $derived(design ? diffPlans(current, design.plan, catalog.conditions) : null);
  const name = (id: string) => catalog.conditions.find((c) => c.id === id)?.name ?? id;
  const sev = (c: BenchPlanCondition | null) => (c ? c.severities.join(', ') : '');
  const proposedBy = $derived(design?.plan.designed_by ? `${design.plan.designed_by.agent === 'claude' ? 'Claude Code' : 'Codex'}${design.plan.designed_by.model ? ` (${modelLabel(design.plan.designed_by.model, agents?.find((x) => x.id === design?.plan.designed_by?.agent))})` : ''}` : asked);

  function use() {
    if (!design) return;
    onuse(design.plan);
    open = false;
  }
</script>

{#snippet change(c: PlanChange, kind: 'added' | 'removed' | 'changed')}
  <li class="ch">
    <span class="ch-head">
      <span class="ch-name">{name(c.condition)}</span>
      <span class="ch-what num">
        {#if kind === 'added' && c.to}
          <Meta><span>severities {sev(c.to)}</span><span>weight {c.to.weight}</span></Meta>
        {:else if kind === 'removed'}
          <span>left out</span>
        {:else if c.from && c.to}
          <Meta>
            {#if c.parts.includes('severities')}<span>severities {sev(c.from)} → {sev(c.to)}</span>{/if}
            {#if c.parts.includes('weight')}<span>weight {c.from.weight} → {c.to.weight}</span>{/if}
          </Meta>
        {/if}
      </span>
    </span>
    {#if c.to?.why}<span class="ch-why">{c.to.why}</span>{/if}
  </li>
{/snippet}

<Dialog
  bind:open
  size={step === 'proposal' ? 'lg' : 'md'}
  title={step === 'proposal' ? 'Proposed plan' : 'Design the plan with an agent'}
  description={step === 'proposal'
    ? `${proposedBy} proposes ${conditionsText(design?.plan.conditions.length ?? 0)} for ${bench.name}, set against the current plan. Use this plan puts it in the editor, unsaved.`
    : 'One request, no tools: the agent reads the description, the classes and four of the images, and answers with a plan from the catalog. You see it next to the current plan first.'}
>
  {#if step === 'ask'}
    <div class="form">
      <!-- A failed request first, where the eye lands after the wait: the server's message and its fix. -->
      {#if error}<ErrorState {error} title="The agent did not give a plan" compact />{/if}
      <div class="reads">
        <span class="group-label">What it reads</span>
        {#if bench.description.trim()}
          <p class="desc">{bench.description}</p>
        {:else}
          <p class="desc none">No description yet: the agent has only the classes and the images to go on. Describe the camera on the benchmark's page first for a plan that fits it.</p>
        {/if}
        <p class="facts num"><Meta><span>{bench.classes.length} classes: {bench.classes.join(', ')}</span><span>4 of its images</span></Meta></p>
      </div>
      <AgentFields {agents} {agentsError} {retryAgents} {choice} effortHint={(a) => `More effort means more reasoning before it answers. The default, ${a.default_effort}, is enough for one plan.`} />
      <Field label="Note" for="{uid}-note" hint="Optional. Anything to add for this request: what matters most, what to leave out.">
        <TextArea id="{uid}-note" bind:value={note} rows={2} placeholder="Most footage is at dusk. Ignore compression: the camera records to a card." />
      </Field>
    </div>
  {:else if step === 'asking'}
    <div class="asking" role="status">
      <Spinner size={16} />
      <div class="a-text">
        <p class="a-title">{asked} is designing the plan</p>
        <p class="a-sub">One request in flight, with the description, {bench.classes.length} classes and 4 images. It can take up to two minutes.</p>
      </div>
    </div>
  {:else if design && diff}
    <div class="proposal">
      {#if design.plan.rationale}
        <div class="rationale">
          <span class="group-label">Why</span>
          <p>{design.plan.rationale}</p>
        </div>
      {/if}
      <p class="sum num">
        <Meta>
          <span>{conditionsText(design.plan.conditions.length)}</span>
          <span>{planPasses(design.plan)} severities</span>
          {#each usageFacts(design.usage) as f (f)}<span>{f}</span>{/each}
        </Meta>
      </p>
      {#if diff.added.length}
        <section aria-label="Added">
          <h3 class="group-label">Added · {diff.added.length}</h3>
          <ul>{#each diff.added as c (c.condition)}{@render change(c, 'added')}{/each}</ul>
        </section>
      {/if}
      {#if diff.changed.length}
        <section aria-label="Changed">
          <h3 class="group-label">Changed · {diff.changed.length}</h3>
          <ul>{#each diff.changed as c (c.condition)}{@render change(c, 'changed')}{/each}</ul>
        </section>
      {/if}
      {#if diff.removed.length}
        <section aria-label="Left out">
          <h3 class="group-label">Left out · {diff.removed.length}</h3>
          <p class="removed">{diff.removed.map((c) => name(c.condition)).join(', ')}</p>
        </section>
      {/if}
      {#if diff.kept.length}
        <section aria-label="Kept">
          <h3 class="group-label">Kept as they are · {diff.kept.length}</h3>
          <ul>{#each diff.kept as c (c.condition)}{#if c.to?.why}{@render change(c, 'changed')}{:else}<li class="ch"><span class="ch-name">{name(c.condition)}</span></li>{/if}{/each}</ul>
        </section>
      {/if}
    </div>
  {/if}
  {#snippet footer()}
    {#if step === 'proposal'}
      <Button variant="ghost" onclick={() => (open = false)}>Discard</Button>
      <Button icon={RotateCw} onclick={() => (step = 'ask')}>Ask again</Button>
      <Button variant="primary" onclick={use}>Use this plan</Button>
    {:else if step === 'asking'}
      <Button onclick={() => (open = false)}>Cancel</Button>
    {:else}
      {#if why}<p class="why" data-gate-why>{why}</p>{/if}
      <Button variant="ghost" onclick={() => (open = false)}>Cancel</Button>
      <Gate reason={why}>
        {#snippet children(reason)}
          <Button variant="primary" icon={error ? RotateCw : Sparkles} disabled={!!reason} aria-describedby={reason} onclick={ask}>{error ? 'Try again' : `Ask ${agent?.name ?? 'the agent'}`}</Button>
        {/snippet}
      </Gate>
    {/if}
  {/snippet}
</Dialog>

<style>
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
</style>
