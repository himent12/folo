// @vitest-environment happy-dom
// A queued run's wait: the estimate of each run in front, asked of the server with that run's settings.
import { flush } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { LiveRun } from '$lib/api/live';
import { MockApi } from '$lib/api/mock';
import { Resource } from '$lib/api/resource';
import type { Run, SystemInfo, TrainConfig } from '$lib/api/types';
import { render, type Rendered } from '../../../framework/testing';
import { queueState } from './queueState';

let c: Rendered | undefined;
afterEach(() => {
  c?.unmount();
  c = undefined;
});

const config: TrainConfig = {
  name: 'a',
  dataset_id: 'ds_shapes',
  arch: 'folo-n',
  epochs: 30,
  batch: 16,
  imgsz: 640,
  device: 'auto',
  optimizer: 'auto',
  lr0: 0.01,
  weights: null,
  augment: { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 },
  seed: 0,
};
const queued = (id: string, place: number, seed: number) => ({ id, name: id, status: 'queued', queue_position: place, config: { ...config, seed } }) as unknown as Run;

describe('queueState', () => {
  // #354 review: the run in front, made over the API with seed 18446744073709551615 (read here as …552000), was
  // estimated with the rounded seed, which the server refuses (422): the wait went unknown. The seed does not change
  // a run's length, so one Studio does not hold exactly is left out (lib/seed, rule B).
  it("asks for the estimate of a run in front without a seed Studio cannot send exactly", async () => {
    const api = new MockApi('normal', 0);
    const estimate = vi.spyOn(api, 'estimateRun');
    hub.api = api;
    hub.mode = 'mock';
    const runs = [queued('run_front', 1, 18446744073709551615), queued('run_exact', 2, 9007199254740991), queued('run_me', 3, 0)];
    c = render(() => {
      const others = new Resource(async () => runs);
      const system = new Resource(async () => ({ features: ['queue_control', 'run_estimate'] }) as unknown as SystemInfo);
      queueState({ run: () => runs[2], live: new LiveRun(() => null), others, system });
      return <span />;
    });
    await vi.waitFor(() => {
      flush();
      expect(estimate).toHaveBeenCalledTimes(2);
    });
    const sent = estimate.mock.calls.map((call) => call[0]);
    expect(sent.map((b) => b.name)).toEqual(['run_front', 'run_exact']);
    expect('seed' in sent[0]!).toBe(false);
    expect(sent[1]!.seed).toBe(9007199254740991);
  });
});
