import { describe, expect, it } from 'vitest';
import type { LabelJob } from '$lib/api/types';
import { Instructions } from './instructions';

function deferred() {
  let resolve!: () => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<void>((res, rej) => ((resolve = res), (reject = rej)));
  return { promise, resolve, reject };
}

/** Instructions over a job whose saves wait until the test answers them, in order. */
function setup(status: LabelJob['status'] = 'idle') {
  const job = { id: 'lb_1', status, instructions: 'old' } as LabelJob;
  const sent: { text: string; answer: ReturnType<typeof deferred> }[] = [];
  const ins = new Instructions(
    () => job,
    (_id, text) => {
      const answer = deferred();
      sent.push({ text, answer });
      return answer.promise;
    },
  );
  ins.sync('old');
  return { job, ins, sent };
}
const settle = () => new Promise((r) => setTimeout(r, 0));

describe('Instructions', () => {
  it('marks as saved what it sent, so text typed meanwhile stays an edit and is saved next', async () => {
    const { ins, sent } = setup();
    ins.text = 'A';
    const saving = ins.save();
    await settle();
    ins.text = 'B';
    sent[0]!.answer.resolve();
    expect(await saving).toBe(true);
    // The server's answer (and its job event) says A: the field keeps B.
    ins.sync('A');
    expect(ins.text).toBe('B');
    void ins.save();
    await settle();
    expect(sent.map((s) => s.text)).toEqual(['A', 'B']);
  });

  it('runs one save at a time, and a save asked meanwhile sends the latest text once', async () => {
    const { ins, sent } = setup();
    ins.text = 'A';
    const first = ins.save();
    await settle();
    ins.text = 'AB';
    const second = ins.save();
    ins.text = 'ABC';
    const third = ins.save();
    await settle();
    expect(sent.map((s) => s.text)).toEqual(['A']);
    sent[0]!.answer.resolve();
    await settle();
    expect(sent.map((s) => s.text)).toEqual(['A', 'ABC']);
    sent[1]!.answer.resolve();
    expect(await Promise.all([first, second, third])).toEqual([true, true, true]);
    expect(sent).toHaveLength(2);
  });

  it('lets a run start only once the latest text is saved', async () => {
    const { ins, sent } = setup();
    const order: string[] = [];
    ins.text = 'A';
    void ins.save().then(() => order.push('saved A'));
    await settle();
    ins.text = 'B';
    // What the page's start() does: wait for the save, then start.
    const start = ins.save().then((ok) => ok && order.push('start'));
    sent[0]!.answer.resolve();
    await settle();
    expect(order).toEqual(['saved A']);
    expect(sent.map((s) => s.text)).toEqual(['A', 'B']);
    sent[1]!.answer.resolve();
    await start;
    expect(order).toEqual(['saved A', 'start']);
  });

  it('says why the text is not saved, and does not hold the next save up', async () => {
    const { ins, sent } = setup();
    ins.text = 'x'.repeat(5000);
    expect(await ins.save()).toBe(false);
    expect(ins.error).toMatch(/^At most 4,000 characters/);
    ins.text = 'A';
    const saving = ins.save();
    await settle();
    sent[0]!.answer.reject(new Error('the server is gone'));
    expect(await saving).toBe(false);
    expect(ins.error).toBe('Not saved: the server is gone');
    const again = ins.save();
    await settle();
    sent[1]!.answer.resolve();
    expect(await again).toBe(true);
    expect(ins.error).toBeNull();
  });

  it('takes the server’s text unless the field holds an edit, and saves nothing while a run is active', async () => {
    const { ins, sent } = setup('running');
    ins.sync('new');
    expect(ins.text).toBe('new');
    ins.text = 'mine';
    ins.sync('newer');
    expect(ins.text).toBe('mine');
    expect(await ins.save()).toBe(true);
    expect(sent).toHaveLength(0);
  });
});
