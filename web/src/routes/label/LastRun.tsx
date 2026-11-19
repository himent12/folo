/**
 * How the job's last run ended, as facts: its outcome and counts, the agent and model, and what the agent reported it
 * used; the job's totals too once it has had more than one run.
 */
import { computed, css, For } from 'fw';
import type { LabelAgent, LabelJob, LabelRun } from '$lib/api/types';
import { ago, num } from '$lib/format';
import Meta from '$lib/ui/Meta';
import { OUTCOME, runAgent, usageFacts } from './words';

export interface LastRunProps {
  job: LabelJob;
  run: LabelRun;
  agents: readonly LabelAgent[] | undefined;
}

export default function LastRun(props: LastRunProps) {
  const used = computed(() => usageFacts(props.run.usage));
  const total = computed(() => (props.job.usage.calls > props.run.usage.calls ? usageFacts(props.job.usage) : []));
  const who = computed(() => runAgent(props.run, props.agents));

  return (
    <div class="last">
      <p class="group-label">Last run</p>
      <p class="line num">
        <Meta>
          <span>{props.run.outcome ? OUTCOME[props.run.outcome] : 'Running'}</span>
          <span>{num(props.run.done)} labeled</span>
          {props.run.failed ? <span>{num(props.run.failed)} failed</span> : null}
          <span>{who.get()[0]}</span>
          {/* A model id the agent does not list (a Bedrock profile's ARN) can be wider than the rail: the one item that
              may cut itself (ui/Meta), whole in its tooltip. */}
          <span class="truncate" title={who.get()[1]}>
            {who.get()[1]}
          </span>
          {props.run.finished ? <span>{ago(props.run.finished)}</span> : null}
        </Meta>
      </p>
      {used.get().length ? (
        <p class="line num">
          <Meta>
            <For each={used.get()}>{(f) => <span>{f()}</span>}</For>
          </Meta>
        </p>
      ) : null}
      {total.get().length ? (
        <p class="line num">
          <Meta>
            <span>All runs</span>
            <For each={total.get()}>{(f) => <span>{f()}</span>}</For>
          </Meta>
        </p>
      ) : null}
      {used.get().length ? <p class="note">Tokens and cost as reported by the agent.</p> : null}
    </div>
  );
}

css`
  /* One column no wider than the rail: a long model id cuts itself instead of widening the panel and the page. */
  .last {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: var(--space-1);
  }
  .line {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .note {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
`;
