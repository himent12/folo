// @vitest-environment happy-dom
// A run that fails after some paid calls (a usage limit on the third) still shows what it used, beside its cause and
// Try again: the cost counts in whether to try again (sweep 4, #320: the Last run summary was dropped on failure).
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
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

// A model id the agent does not list, as `start` takes it (up to 100 characters): one unbreakable item that widened
// the rail, and the page to 626 px at 390 (#322 review). The job page is where a person reads which model ran, so the id
// stays whole there (no ellipsis, no tooltip that Tab and touch never show) and wraps in a column no wider than the rail.
const ARN = 'arn:aws:bedrock:us-west-2:123456789012:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0';
const runs: [string, (job: LabelJob) => LabelJob][] = [
  ['failed', (job) => ({ ...job, run: { ...job.run!, model: ARN } })],
  ['completed', (job) => ({ ...job, run: { ...job.run!, model: ARN, outcome: 'completed', error: null, hint: null } })],
  ['live', (job) => ({ ...job, status: 'running', run: { ...job.run!, model: ARN, outcome: null, finished: null, error: null, hint: null } })],
];

describe('AgentPanel with a long model id', () => {
  for (const [state, edit] of runs) {
    it(`shows it whole in the ${state} run's line, an item that wraps`, async () => {
      const { target, view } = await panel(edit);
      const item = [...target.querySelectorAll<HTMLElement>('.ui-meta > *')].find((e) => e.textContent?.trim() === ARN);
      expect(item?.classList.contains('model')).toBe(true);
      // Not cut, and not left to a tooltip: no ellipsis or title on it, inside it or around it.
      expect(item?.closest('.truncate, [title]')).toBe(null);
      expect(item?.querySelector('.truncate, [title]')).toBe(null);
      view.unmount();
    });
  }

  it("wraps the id in each line's column, no wider than the rail; the jobs list's Agent cell stays capped", () => {
    // The rules themselves (happy-dom lays nothing out): the id wraps anywhere, each line is one `minmax(0, 1fr)` grid
    // column, and the jobs list (a summary row, the whole id a click away on the job page) keeps `max-width: 0`.
    const rule = (file: string, selector: string) => {
      const src = readFileSync(resolve(process.cwd(), file), 'utf8');
      const at = src.indexOf(`\n  ${selector} {`);
      return at < 0 ? '' : src.slice(at, src.indexOf('}', at));
    };
    expect(rule('src/routes/label/ModelId.tsx', ':global(.ui-meta) > .model')).toContain('white-space: normal');
    expect(rule('src/routes/label/ModelId.tsx', '.model > span')).toMatch(/min-width: 0;\s+overflow-wrap: anywhere/);
    expect(rule('src/routes/label/LastRun.tsx', '.last')).toContain('grid-template-columns: minmax(0, 1fr)');
    expect(rule('src/routes/label/AgentPanel.tsx', '.progress')).toContain('grid-template-columns: minmax(0, 1fr)');
    expect(rule('src/routes/LabelJobs.tsx', '.name-cell,\n  .agent-cell')).toContain('max-width: 0');
  });
});
