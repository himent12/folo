// @vitest-environment happy-dom
// The sidebar's live card says the epoch and the time left as the run page does: "Epoch 1 of 5", and no time left
// until the first epoch is done (lib/api/progress etaSettled).
import { flush } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, describe, expect, it } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Run } from '$lib/api/types';
import SidebarLive from './SidebarLive';
import { watched } from './runList';

let target: HTMLElement;
let c: { unmount(): void } | undefined;
afterEach(() => {
  c?.unmount();
  target?.remove();
  watched.runs = undefined;
});

describe('SidebarLive', () => {
  it('reads "Epoch 1 of 5" and holds the time left back until the first epoch is done', async () => {
    const [any] = await new MockApi('normal', 0).runs();
    const run = { ...any!, status: 'running', epoch: 1, epochs: 5, steps_per_epoch: 20, step: 7, eta_s: 240, best: null, cloud: undefined } as Run;
    watched.runs = [run];
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <SidebarLive collapsed={false} />, target);
    flush();
    const card = () => target.querySelector('.card')!.textContent ?? '';
    expect(card()).toContain('Epoch 1 of 5');
    expect(card()).toContain('estimating time left…');
    expect(card()).not.toContain('4m left');
    watched.runs = [{ ...run, epoch: 2, step: 21, eta_s: 180 }];
    flush();
    expect(card()).toContain('Epoch 2 of 5');
    expect(card()).toContain('3m left');
  });

  it('says nothing of the run training on a server just switched to: it did not start then (review R4)', async () => {
    const [any] = await new MockApi('normal', 0).runs();
    const run = (id: string) => ({ ...any!, id, name: id, status: 'running', cloud: undefined }) as Run;
    watched.runs = [run('on-a')];
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <SidebarLive collapsed={false} />, target);
    flush();
    const said = () => target.querySelector('[aria-live]')!.textContent;
    expect(target.textContent).toContain('on-a');

    hub.epoch++;
    flush();
    expect(target.querySelector('.card')).toBeNull();
    watched.runs = [run('on-b')];
    flush();
    expect(target.textContent).toContain('on-b');
    expect(said()).toBe('');
    // A run that starts on it afterwards is news.
    watched.runs = [run('next-on-b')];
    flush();
    expect(said()).toBe('next-on-b started training');
  });
});
