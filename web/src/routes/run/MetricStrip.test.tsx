// @vitest-environment happy-dom
// The live run's progress row in its first minute (runs F14): the epoch in progress said as such, the start's phase,
// and no time left until it has settled.
import { flush as flushSync } from 'fw';
import { render } from '../../../framework/testing';
import { describe, expect, it } from 'vitest';
import { MockApi } from '$lib/api/mock';
import type { Run } from '$lib/api/types';
import MetricStrip from './MetricStrip';

async function row(o: Partial<Run>, etaReady: boolean) {
  const run = { ...(await new MockApi('normal', 0).run('run_7d1e')), ...o };
  const target = document.createElement('div');
  const c = render(
    () => (
      <MetricStrip
        run={run}
        live
        progress={0.06}
        mosaicOff={false}
        closeFrom={1}
        closeId="c"
        closeText={{ cap: '', full: '' }}
        elapsed={null}
        etaReady={etaReady}
        tabs={[]}
        unitOf={() => null}
      />
    ),
    target,
  );
  flushSync();
  const text = [target.querySelector('.big')?.textContent?.replace(/\s+/g, ' ').trim(), target.querySelector('.epoch > .cap')?.textContent?.trim()];
  c.unmount();
  return text;
}

describe('the live progress row', () => {
  it('says "Epoch 1 of 5", and the time left only once an epoch has finished', async () => {
    const base = { status: 'running', epoch: 1, epochs: 5, step: 3, steps_per_epoch: 50, eta_s: 240 } as const;
    expect(await row(base, false)).toEqual(['Epoch1of 5', '1%estimating time left…']);
    expect((await row(base, true))[1]).toBe('1%4m left');
  });
  it('names the start while no step has come yet', async () => {
    expect((await row({ status: 'running', epoch: 0, epochs: 5, step: 0, eta_s: null }, false))[1]).toBe('Starting the first epoch…');
  });
});
