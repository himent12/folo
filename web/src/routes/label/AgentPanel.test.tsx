// @vitest-environment happy-dom
// A run that fails after some paid calls (a usage limit on the third) still shows what it used, beside its cause and
// Try again: the cost counts in whether to try again (sweep 4, #320: the Last run summary was dropped on failure).
import { flush } from 'fw';
import { describe, expect, it } from 'vitest';
import { render } from '../../../framework/testing';
import { MockApi } from '$lib/api/mock';
import { mockAgents } from '$lib/api/mock/label';
import type { LabelJob, LabelUsage } from '$lib/api/types';
import AgentPanel from './AgentPanel';
import { AgentChoice } from './agentChoice';

const noop = () => {};

async function panel(edit: (job: LabelJob) => LabelJob) {
  const job = edit(await new MockApi('normal', 0).labelJob('lb_9e44'));
  const agents = mockAgents(true);
  const target = document.createElement('div');
  const view = render(
    () => (
      <AgentPanel
        job={job}
        agents={agents}
        agentsError={null}
        retryAgents={noop}
        choice={new AgentChoice(() => agents)}
        instructionsError={null}
        starting={false}
        startError={null}
        lost={false}
        onsaveinstructions={noop}
        onstart={noop}
        onagain={noop}
        onstop={noop}
        onedit={noop}
      />
    ),
    target,
  );
  flush();
  return { target, view };
}

describe('AgentPanel after a failed run', () => {
  it('shows the calls, tokens and cost the agent reported, with the cause and Try again', async () => {
    const usage: LabelUsage = { calls: 2, input_tokens: 1_200, cached_input_tokens: 0, output_tokens: 200, cost_usd: 0.024 };
    const { target, view } = await panel((job) => ({
      ...job,
      run: { ...job.run!, error: 'You have hit your usage limit', hint: null, usage },
      usage,
    }));
    expect(target.querySelector('[role="alert"]')?.textContent).toMatch(/Claude Code stopped.*usage limit/);
    expect([...target.querySelectorAll('button')].some((b) => b.textContent?.trim() === 'Try again')).toBe(true);
    const text = target.textContent ?? '';
    expect(text).toMatch(/Last run/);
    expect(text).toMatch(/2 calls.*1\.2k tokens in.*200 out.*\$0\.02/);
    expect(text).toMatch(/Tokens and cost as reported by the agent/);
    view.unmount();
  });

  it('shows no cost when the agent reported none', async () => {
    const usage: LabelUsage = { calls: 1, input_tokens: 900, cached_input_tokens: 0, output_tokens: 40, cost_usd: null };
    const { target, view } = await panel((job) => ({ ...job, run: { ...job.run!, usage }, usage }));
    const text = target.textContent ?? '';
    expect(text).toMatch(/1 call.*900 tokens in.*40 out/);
    expect(text).not.toMatch(/\$/);
    view.unmount();
  });
});
