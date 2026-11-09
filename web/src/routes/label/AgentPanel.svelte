<script lang="ts">
  /**
   * The job page's rail: first the run, "Label N images" while some have no labels (naming the agent, model and effort
   * it will use), live progress with Stop while an agent works, and how the last run ended (a failed one with its cause,
   * the fix the server gives, and Try again); then who labels (agent, model, effort) and what it is told (the
   * instructions, the classes).
   */
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import Pencil from '@lucide/svelte/icons/pencil';
  import RotateCw from '@lucide/svelte/icons/rotate-cw';
  import Sparkles from '@lucide/svelte/icons/sparkles';
  import Square from '@lucide/svelte/icons/square';
  import type { LabelAgent, LabelJob } from '$lib/api/types';
  import { classColor } from '$lib/design/tokens';
  import { num } from '$lib/format';
  import { AGENT_NAME, INSTRUCTIONS_MAX } from '$lib/label';
  import Button from '$lib/ui/Button.svelte';
  import CodeText from '$lib/ui/CodeText.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Progress from '$lib/ui/Progress.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import TextArea from '$lib/ui/TextArea.svelte';
  import Callout from '../run/Callout.svelte';
  import type { AgentChoice } from './agentChoice.svelte';
  import LastRun from './LastRun.svelte';
  import { images, modelLabel, runAgent, toLabel } from './words';

  let {
    job,
    agents,
    agentsError,
    retryAgents,
    choice,
    instructions = $bindable(),
    instructionsError,
    starting,
    startError,
    lost,
    onsaveinstructions,
    onstart,
    onagain,
    onstop,
    onedit,
  }: {
    job: LabelJob;
    agents: readonly LabelAgent[] | undefined;
    agentsError: unknown;
    retryAgents: () => void;
    choice: AgentChoice;
    instructions: string;
    instructionsError: string | null;
    starting: boolean;
    startError: unknown;
    /** The stream is down: progress is as last heard. */
    lost: boolean;
    onsaveinstructions: () => void;
    /** Labels the images without labels. */
    onstart: () => void;
    /** Labels every image again (asks first). */
    onagain: () => void;
    onstop: () => void;
    onedit: () => void;
  } = $props();

  const uid = $props.id();
  const running = $derived(job.status !== 'idle');
  const run = $derived(job.run);
  const failed = $derived(!running && run?.outcome === 'failed');
  const n = $derived(toLabel(job));
  const agent = $derived(choice.agent);
  /** The agents that are not set up, said in words under the choice (a disabled option's tooltip alone reaches no one on touch). */
  const notReady = $derived((agents ?? []).filter((a) => !a.available && a.hint));
  const modelOptions = $derived([
    ...(agent && agent.default_model === null ? [{ value: '', label: 'Default' }] : []),
    ...(agent?.models.map((m) => ({ value: m.id, label: m.label })) ?? []),
  ]);
  /** Why the run cannot start, in words beside the button; null when it can. */
  const why = $derived(
    job.counts.images === 0
      ? 'Add images to label first'
      : !agent
        ? agents
          ? 'Install Claude Code or Codex on the server’s machine'
          : 'Checking which agents are installed'
        : instructions.length > INSTRUCTIONS_MAX
          ? 'Shorten the instructions first'
          : null,
  );
  const done = $derived(run ? run.done + run.failed : 0);
</script>

<Panel title="Agent" pad="none">
  <div class="sec run">
    {#if running && run}
      <div class="progress">
        <p class="p-head num"><b>{num(run.done)} of {images(run.total)} labeled</b>{#if run.failed}<span class="faint">{num(run.failed)} failed</span>{/if}</p>
        <Progress value={run.total ? done / run.total : 0} size="md" tone={lost ? 'neutral' : 'accent'} label="{done} of {run.total} images done" />
        <p class="p-meta"><Meta>{#each runAgent(run, agents) as f (f)}<span>{f}</span>{/each}<span>{run.effort} effort</span></Meta></p>
      </div>
      {#if job.status === 'stopping'}
        <p class="note">Stopping: ending the calls in flight.</p>
      {:else}
        <Button variant="danger" icon={Square} block onclick={onstop}>Stop</Button>
      {/if}
    {:else}
      {#if failed && run}
        <Callout tone="danger" icon={CircleAlert} title="{AGENT_NAME[run.agent]} stopped" role="alert" label="Last run failed">
          <CodeText text={run.error ?? 'The agent kept failing.'} />
        </Callout>
        {#if run.hint}<p class="note"><CodeText text={run.hint} /></p>{/if}
      {/if}
      {#if n > 0 || job.counts.images === 0}
        <Gate reason={why}>
          {#snippet children(reason)}
            <Button variant={reason ? 'secondary' : 'primary'} icon={failed ? RotateCw : Sparkles} block loading={starting} disabled={!!reason} aria-describedby={reason} onclick={onstart}>
              {failed ? 'Try again' : n > 0 ? `Label ${images(n)}` : 'Label images'}
            </Button>
          {/snippet}
        </Gate>
        {#if why}<p class="note" data-gate-why>{why}</p>{:else if agent}<p class="note">{agent.name} · {modelLabel(choice.model || null, agent)} · {choice.effort} effort. The boxes are a draft for you to review.</p>{/if}
      {:else}
        <p class="note">Every image has labels. Review them, then create a dataset.</p>
        <Button icon={RotateCw} block disabled={!agent} onclick={onagain}>Label all again</Button>
      {/if}
      {#if startError}<ErrorState error={startError} title="Could not start the agent" compact />{/if}
      {#if run && !failed}<LastRun {job} {run} {agents} />{/if}
    {/if}
  </div>
  <!-- The settings are for the next run: while one runs, its own line above says what it uses. -->
  {#if !running}
    <div class="sec">
      {#if agentsError && !agents}
        <ErrorState error={agentsError} title="Could not ask which agents are installed" retry={retryAgents} compact />
      {:else if !agents}
        <div class="sk" aria-hidden="true"><Skeleton w="64px" h="12px" /><Skeleton h="32px" r="var(--radius-md)" /><Skeleton w="64px" h="12px" /><Skeleton h="32px" r="var(--radius-md)" /></div>
      {:else}
        <Field label="Agent" hint={notReady.map((a) => `${a.name}: ${a.hint}`).join(' ') || undefined}>
          <Segmented
            label="Agent"
            value={agent?.id ?? 'claude'}
            options={agents.map((a) => ({ value: a.id, label: a.name, disabled: !a.available, tip: a.hint ?? undefined }))}
            onchange={(id) => choice.pick(id)}
          />
        </Field>
        {#if agent}
          <Field label="Model" for="{uid}-model" hint={agent.id === 'claude' && agent.default_model === 'haiku' ? 'Haiku is the default: about as accurate as the larger models here, at a fraction of the cost.' : undefined}>
            <Select id="{uid}-model" bind:value={choice.model} options={modelOptions} />
          </Field>
          <Field label="Effort" hint="More effort means more reasoning tokens per call and a slower run. The default, {agent.default_effort}, is the best value for {agent.name}.">
            <Segmented
              label="Effort"
              bind:value={choice.effort}
              options={[
                { value: 'low', label: 'Low' },
                { value: 'medium', label: 'Medium' },
                { value: 'high', label: 'High' },
              ]}
            />
          </Field>
        {/if}
      {/if}
    </div>
  {/if}

  <div class="sec">
    <Field
      label="Instructions"
      for="{uid}-instructions"
      error={instructionsError}
      hint={running ? 'The agent has these for this run; change them once it is done.' : 'What counts as each class and what to leave out, in your own words. Sent with every image.'}
    >
      <TextArea
        id="{uid}-instructions"
        bind:value={instructions}
        rows={5}
        readonly={running}
        placeholder="Label every car and pedestrian, also partly hidden ones. Ignore reflections and posters."
        onblur={onsaveinstructions}
      />
    </Field>
    <Field label="Classes">
      {#snippet aside()}
        <Gate reason={running ? 'Classes can be changed once the agent is done' : null}>
          {#snippet children(reason)}
            <Button variant="ghost" size="sm" icon={Pencil} disabled={!!reason} aria-describedby={reason} onclick={onedit}>Edit</Button>
          {/snippet}
        </Gate>
      {/snippet}
      <ul class="chips" aria-label="Classes">
        {#each job.classes as c, i (i)}<li class="chip"><span class="sw" style:background={classColor(i)}></span>{c}</li>{/each}
      </ul>
    </Field>
  </div>

</Panel>

<style>
  .sec {
    display: grid;
    gap: var(--space-4);
    padding: var(--space-4);
    border-top: 1px solid var(--line-subtle);
  }
  .sec:first-child {
    border-top: 0;
    padding-top: 0;
  }
  .sec.run {
    gap: var(--space-3);
  }
  .sec :global(.seg) {
    justify-self: start;
  }
  .sk {
    display: grid;
    gap: var(--space-2);
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1);
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 24px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    background: var(--bg-active);
    font-size: var(--type-sm-size);
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
  }
  .progress {
    display: grid;
    gap: var(--space-2);
  }
  .p-head {
    display: flex;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .p-head b {
    font-weight: var(--weight-medium);
  }
  .p-meta,
  .note {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    max-inline-size: none;
  }
</style>
