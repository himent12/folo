// @vitest-environment happy-dom
// A tile says what a person did to its image: "Edited" (boxes saved by hand) and "Reviewed" (checked) are two facts,
// and an image can be both (v-models-label N4: saving by hand reviews it, and its tile said only "Edited").
import { flush } from 'fw';
import { afterEach, describe, expect, it } from 'vitest';
import { render } from '../../../framework/testing';
import type { Api } from '$lib/api/client';
import { hub } from '$lib/api/index';
import type { LabelImage } from '$lib/api/types';
import LabelTile from './LabelTile';

const img = (o: Partial<LabelImage>): LabelImage => ({
  id: 'im1',
  file: 'im1.jpg',
  width: 640,
  height: 480,
  status: 'labeled',
  boxes: [{ class: 0, x1: 1, y1: 1, x2: 9, y2: 9 }],
  source: 'agent',
  error: null,
  ...o,
});

describe('LabelTile', () => {
  const saved = hub.api;
  afterEach(() => (hub.api = saved));

  it.each<[string, Partial<LabelImage>, string[]]>([
    ['the agent labels, unchecked', {}, []],
    ['the agent labels, marked right', { reviewed: true }, ['Reviewed']],
    ['edited by hand (which reviews it)', { source: 'manual', reviewed: true }, ['Edited', 'Reviewed']],
    ['edited, then marked not reviewed', { source: 'manual', reviewed: false }, ['Edited']],
  ])('%s', (_, o, marks) => {
    hub.api = { labelImageUrl: async () => 'data:,' } as unknown as Api;
    const target = document.createElement('div');
    const view = render(() => <LabelTile img={img(o)} jobId="lb_1" classes={['bolt']} boxes density="m" />, target);
    flush();
    expect([...target.querySelectorAll('.mark')].map((m) => m.textContent?.trim())).toEqual(marks);
    view.unmount();
  });
});
