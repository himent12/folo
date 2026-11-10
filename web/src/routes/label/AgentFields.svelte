<script lang="ts">
  /**
   * Who answers: the agent CLI (Claude Code or Codex, the ones not installed off with how to install them in words),
   * its model and its effort, for the `AgentChoice` they set. Agent labeling's rail and the vision benchmarks' agent
   * entry and plan design ask through it, so an agent is picked the same way everywhere. The caller frames it (a rail
   * section, a dialog's fields) and says what the effort costs in its own terms when it has more to say.
   */
  import type { LabelAgent } from '$lib/api/types';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import type { AgentChoice } from './agentChoice.svelte';

  let {
    agents,
    agentsError,
    retryAgents,
    choice,
    effortHint,
  }: {
    agents: readonly LabelAgent[] | undefined;
    agentsError: unknown;
    retryAgents: () => void;
    choice: AgentChoice;
    /** What more effort means where it is asked (labeling's: more reasoning tokens per call and a slower run). */
    effortHint?: (agent: LabelAgent) => string;
  } = $props();

  const uid = $props.id();
  const agent = $derived(choice.agent);
  /** The agents that are not set up, said in words under the choice (a disabled option's tooltip alone reaches no one on touch). */
  const notReady = $derived((agents ?? []).filter((a) => !a.available && a.hint));
  const modelOptions = $derived([
    ...(agent && agent.default_model === null ? [{ value: '', label: 'Default' }] : []),
    ...(agent?.models.map((m) => ({ value: m.id, label: m.label })) ?? []),
  ]);
  const hint = (a: LabelAgent) => effortHint?.(a) ?? `More effort means more reasoning tokens per call and a slower run. The default, ${a.default_effort}, is the best value for ${a.name}.`;
</script>

{#if agentsError && !agents}
  <ErrorState error={agentsError} title="Could not ask which agents are installed" retry={retryAgents} compact />
{:else if !agents}
  <div class="sk" aria-hidden="true"><Skeleton w="64px" h="12px" /><Skeleton h="32px" r="var(--radius-md)" /><Skeleton w="64px" h="12px" /><Skeleton h="32px" r="var(--radius-md)" /></div>
{:else}
  <Field label="CLI" hint={notReady.map((a) => `${a.name}: ${a.hint}`).join(' ') || undefined}>
    <Segmented
      label="Agent CLI"
      value={agent?.id ?? 'claude'}
      options={agents.map((a) => ({ value: a.id, label: a.name, disabled: !a.available, tip: a.hint ?? undefined }))}
      onchange={(id) => choice.pick(id)}
    />
  </Field>
  {#if agent}
    <Field label="Model" for="{uid}-model" hint={agent.id === 'claude' && agent.default_model === 'haiku' ? 'Haiku is the default: about as accurate as the larger models here, at a fraction of the cost.' : undefined}>
      <Select id="{uid}-model" bind:value={choice.model} options={modelOptions} />
    </Field>
    <Field label="Effort" hint={hint(agent)}>
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

<style>
  .sk {
    display: grid;
    gap: var(--space-2);
  }
</style>
