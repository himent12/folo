// fw port of AgentFields.svelte.
/**
 * Who answers: the agent CLI (Claude Code or Codex, the ones not installed off with how to install them in words),
 * its model and its effort, for the `AgentChoice` they set. Agent labeling's rail and the vision benchmarks' agent
 * entry and plan design ask through it, so an agent is picked the same way everywhere. The caller frames it (a rail
 * section, a dialog's fields) and says what the effort costs in its own terms when it has more to say.
 */
import { computed, createUniqueId, css, field } from 'fw';
import type { LabelAgent } from '$lib/api/types';
import ErrorState from '$lib/ui/ErrorState';
import Field from '$lib/ui/Field.tsx';
import Segmented from '$lib/ui/Segmented';
import Select from '$lib/ui/Select';
import Skeleton from '$lib/ui/Skeleton';
import type { AgentChoice } from './agentChoice';

export interface AgentFieldsProps {
  agents: readonly LabelAgent[] | undefined;
  agentsError: unknown;
  retryAgents: () => void;
  choice: AgentChoice;
  /** What more effort means where it is asked (labeling's: more reasoning tokens per call and a slower run). */
  effortHint?: (agent: LabelAgent) => string;
}

export default function AgentFields(props: AgentFieldsProps) {
  const uid = createUniqueId();
  const agent = computed(() => props.choice.agent);
  /** The agents that are not set up, said in words under the choice (a disabled option's tooltip alone reaches no one on touch). */
  const notReady = computed(() => (props.agents ?? []).filter((a) => !a.available && a.hint));
  const modelOptions = computed(() => {
    const a = agent.get();
    return [...(a && a.default_model === null ? [{ value: '', label: 'Default' }] : []), ...(a?.models.map((m) => ({ value: m.id, label: m.label })) ?? [])];
  });
  const hint = (a: LabelAgent) =>
    props.effortHint?.(a) ?? `More effort means more reasoning tokens per call and a slower run. The default, ${a.default_effort}, is the best value for ${a.name}.`;

  return (
    <>
      {props.agentsError && !props.agents ? (
        <ErrorState error={props.agentsError} title="Could not ask which agents are installed" retry={props.retryAgents} compact />
      ) : !props.agents ? (
        <div class="sk" aria-hidden="true">
          <Skeleton w="64px" h="12px" />
          <Skeleton h="32px" r="var(--radius-md)" />
          <Skeleton w="64px" h="12px" />
          <Skeleton h="32px" r="var(--radius-md)" />
        </div>
      ) : (
        <>
          <Field label="CLI" hint={notReady.get().map((a) => `${a.name}: ${a.hint}`).join(' ') || undefined}>
            <Segmented
              label="Agent CLI"
              value={agent.get()?.id ?? 'claude'}
              options={(props.agents ?? []).map((a) => ({ value: a.id, label: a.name, disabled: !a.available, tip: a.hint ?? undefined }))}
              onchange={(id) => props.choice.pick(id)}
            />
          </Field>
          {agent.get() ? (
            <>
              <Field
                label="Model"
                for={`${uid}-model`}
                hint={
                  agent.get()!.id === 'claude' && agent.get()!.default_model === 'haiku'
                    ? 'Haiku is the default: about as accurate as the larger models here, at a fraction of the cost.'
                    : undefined
                }
              >
                <Select id={`${uid}-model`} bind:value={field(props.choice, 'model')} options={modelOptions.get()} />
              </Field>
              <Field label="Effort" hint={hint(agent.get()!)}>
                <Segmented
                  label="Effort"
                  bind:value={field(props.choice, 'effort')}
                  options={[
                    { value: 'low', label: 'Low' },
                    { value: 'medium', label: 'Medium' },
                    { value: 'high', label: 'High' },
                  ]}
                />
              </Field>
            </>
          ) : null}
        </>
      )}
    </>
  );
}

css`
  .sk {
    display: grid;
    gap: var(--space-2);
  }
`;
