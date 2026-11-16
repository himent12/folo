/**
 * The job page's rail: first the run, "Label N images" while some have no labels (naming the agent, model and effort
 * it will use), live progress with Stop while an agent works, and how the last run ended (a failed one with its cause,
 * the fix the server gives, and Try again); then who labels (agent, model, effort) and what it is told (the
 * instructions, the classes). The instructions cannot be typed in while a run starts: what is saved then is what
 * the agent gets.
 */
import CircleAlert from '$icons/circle-alert';
import Pencil from '$icons/pencil';
import RotateCw from '$icons/rotate-cw';
import Sparkles from '$icons/sparkles';
import Square from '$icons/square';
import { bindable, computed, createUniqueId, css, effect, For, Index, signal } from 'fw';
import type { LabelAgent, LabelJob } from '$lib/api/types';
import { classColor } from '$lib/design/tokens';
import { count, duration, num, sentence } from '$lib/format';
import { AGENT_NAME, INSTRUCTIONS_MAX } from '$lib/label';
import Button from '$lib/ui/Button';
import CodeText from '$lib/ui/CodeText';
import ErrorState from '$lib/ui/ErrorState';
import Field from '$lib/ui/Field.tsx';
import Gate from '$lib/ui/Gate';
import Meta from '$lib/ui/Meta';
import Panel from '$lib/ui/Panel';
import Progress from '$lib/ui/Progress';
import TextArea from '$lib/ui/TextArea';
import Callout from '../run/Callout';
import AgentFields from './AgentFields';
import type { AgentChoice } from './agentChoice';
import LastRun from './LastRun';
import { images, modelLabel, runAgent, toLabel } from './words';

export interface AgentPanelProps {
  job: LabelJob;
  agents: readonly LabelAgent[] | undefined;
  agentsError: unknown;
  retryAgents: () => void;
  choice: AgentChoice;
  instructions?: string;
  instructionsError: string | null;
  /** The instructions were just saved and not edited since: the field says so. */
  instructionsSaved?: boolean;
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
}

export default function AgentPanel(props: AgentPanelProps) {
  const instructions = bindable(props, 'instructions', '');
  const uid = createUniqueId();
  const running = computed(() => props.job.status !== 'idle');
  const run = computed(() => props.job.run);
  const failed = computed(() => !running.get() && run.get()?.outcome === 'failed');
  const n = computed(() => toLabel(props.job));
  const agent = computed(() => props.choice.agent);
  /** Why the run cannot start, in words beside the button; null when it can. */
  const why = computed(() =>
    props.job.counts.images === 0
      ? 'Add images to label first'
      : !agent.get()
        ? props.agents
          ? 'Install Claude Code or Codex on the server’s machine'
          : 'Checking which agents are installed'
        : instructions.get().length > INSTRUCTIONS_MAX
          ? 'Shorten the instructions first'
          : null,
  );
  const done = computed(() => {
    const r = run.get();
    return r ? r.done + r.failed : 0;
  });
  // Until its first answer a run shows an empty bar: a line says it is working, and for how long.
  const waiting = computed(() => props.job.status === 'running' && !!run.get() && done.get() === 0);
  const elapsed = signal(0);
  effect(() => {
    const r = run.get();
    if (!waiting.get() || !r) return;
    const from = Date.parse(r.started);
    const tick = () => elapsed.set(Math.max(0, (Date.now() - from) / 1000));
    tick();
    const t = setInterval(tick, 1000);
    return () => clearInterval(t);
  });

  return (
    <Panel title="Agent" pad="none">
      <div class="sec run">
        {running.get() && run.get() ? (
          <>
            <div class="progress">
              <p class="p-head num">
                <b>
                  {num(run.get()!.done)} of {images(run.get()!.total)} labeled
                </b>
                {run.get()!.failed ? <span class="faint">{num(run.get()!.failed)} failed</span> : null}
              </p>
              <Progress
                value={run.get()!.total ? done.get() / run.get()!.total : 0}
                size="md"
                tone={props.lost ? 'neutral' : 'accent'}
                label={`${done.get()} of ${count(run.get()!.total, 'image')} done`}
              />
              {waiting.get() ? (
                <p class="p-wait num">
                  {AGENT_NAME[run.get()!.agent]} is working on the first images · {duration(elapsed.get())}
                </p>
              ) : null}
              <p class="p-meta">
                <Meta>
                  <For each={runAgent(run.get()!, props.agents)}>{(f) => <span>{f()}</span>}</For>
                  <span>{run.get()!.effort} effort</span>
                </Meta>
              </p>
            </div>
            {props.job.status === 'stopping' ? (
              <p class="note">Stopping: ending the calls in flight.</p>
            ) : (
              <Button variant="danger" icon={Square} block onclick={props.onstop}>
                Stop
              </Button>
            )}
          </>
        ) : (
          <>
            {failed.get() && run.get() ? (
              <>
                <Callout tone="danger" icon={CircleAlert} title={`${AGENT_NAME[run.get()!.agent]} stopped`} role="alert" label="Last run failed">
                  <CodeText text={sentence(run.get()!.error ?? 'The agent kept failing.')} />
                </Callout>
                {run.get()!.hint ? (
                  <p class="note">
                    <CodeText text={sentence(run.get()!.hint!)} />
                  </p>
                ) : null}
              </>
            ) : null}
            {n.get() > 0 || props.job.counts.images === 0 ? (
              <>
                <Gate reason={why.get()}>
                  {(reason) => (
                    <Button
                      variant={reason() ? 'secondary' : 'primary'}
                      icon={failed.get() ? RotateCw : Sparkles}
                      block
                      loading={props.starting}
                      disabled={!!reason()}
                      aria-describedby={reason()}
                      onclick={props.onstart}
                    >
                      {failed.get() ? 'Try again' : n.get() > 0 ? `Label ${images(n.get())}` : 'Label images'}
                    </Button>
                  )}
                </Gate>
                {why.get() ? (
                  <p class="note" data-gate-why>
                    {why.get()}
                  </p>
                ) : agent.get() ? (
                  <p class="note">
                    {agent.get()!.name} · {modelLabel(props.choice.model || null, agent.get())} · {props.choice.effort} effort. The boxes are a draft for you to
                    review.
                  </p>
                ) : null}
              </>
            ) : (
              <>
                <p class="note">Every image has labels. Review them, then create a dataset.</p>
                <Button icon={RotateCw} block disabled={!agent.get()} onclick={props.onagain}>
                  Label all again
                </Button>
              </>
            )}
            {props.startError ? <ErrorState error={props.startError} title="Could not start the agent" compact /> : null}
            {run.get() && !failed.get() ? <LastRun job={props.job} run={run.get()!} agents={props.agents} /> : null}
          </>
        )}
      </div>
      {/* The settings are for the next run: while one runs, its own line above says what it uses. */}
      {!running.get() ? (
        <div class="sec">
          <AgentFields agents={props.agents} agentsError={props.agentsError} retryAgents={props.retryAgents} choice={props.choice} />
        </div>
      ) : null}

      <div class="sec">
        <Field
          label="Instructions"
          for={`${uid}-instructions`}
          error={props.instructionsError}
          hint={
            running.get()
              ? 'The agent has these for this run; change them once it is done.'
              : props.instructionsSaved
                ? 'Saved. Sent with every image.'
                : 'What counts as each class and what to leave out, in your own words. Sent with every image.'
          }
        >
          <TextArea
            id={`${uid}-instructions`}
            bind:value={instructions}
            rows={5}
            readonly={running.get() || props.starting}
            placeholder="None: the agent goes by the class names alone."
            onblur={props.onsaveinstructions}
          />
        </Field>
        <Field
          label="Classes"
          aside={() => (
            <Gate reason={running.get() ? 'Classes can be changed once the agent is done' : null}>
              {(reason) => (
                <Button variant="ghost" size="sm" icon={Pencil} disabled={!!reason()} aria-describedby={reason()} onclick={props.onedit}>
                  Edit
                </Button>
              )}
            </Gate>
          )}
        >
          <ul class="chips" aria-label="Classes">
            <Index each={props.job.classes}>
              {(c, i) => (
                <li class="chip">
                  <span class="sw" style:background={classColor(i)}></span>
                  {c()}
                </li>
              )}
            </Index>
          </ul>
        </Field>
      </div>
    </Panel>
  );
}

css`
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
  .p-wait,
  .note {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    max-inline-size: none;
  }
`;
