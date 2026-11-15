// @vitest-environment happy-dom
// New run's dataset picker and its train-only note (round s2-runs-6): rendered, picked and read as a user meets it.
import { flush, signal } from 'fw';
import { afterEach, describe, expect, it } from 'vitest';
import type { Resource } from '$lib/api/resource';
import type { ListedDataset } from '../datasets/problem';
import { render, type Rendered } from '../../../framework/testing';
import DatasetSection from './DatasetSection';
import { FIELD, type FormErrors } from './jump';

const ds = (id: string, train: number, val: number) =>
  ({ id, name: id, format: 'yolo', path: `D:/data/${id}`, splits: { train, val, test: 0 }, classes: ['bolt'] }) as unknown as ListedDataset;
const list = [ds('train-only', 192, 0), ds('with-val', 192, 48)];
const noErrors = Object.fromEntries(Object.keys(FIELD).map((k) => [k, null])) as FormErrors;

let c: Rendered | undefined;
let target: HTMLElement;
afterEach(() => {
  c?.unmount();
  c = undefined;
  target.remove();
});

function mount(first: string) {
  target = document.createElement('div');
  document.body.appendChild(target);
  const id = signal(first);
  const datasets = { data: list } as unknown as Resource<ListedDataset[]>;
  c = render(() => <DatasetSection datasets={datasets} bind:datasetId={id} dataset={list.find((d) => d.id === id.get())} errors={noErrors} />, target);
  flush();
  return id;
}

const note = () => target.querySelector('#dataset-note')?.textContent?.trim() ?? null;
const describedBy = (sel: string) => target.querySelector(sel)?.getAttribute('aria-describedby') ?? null;

describe('New run: the dataset picker’s train-only note', () => {
  it('shows and announces the note for a train-only dataset and drops it for one with a val split', () => {
    const id = mount('with-val');
    expect(note()).toBeNull();
    // The region stays in the page, so the note is announced when a pick brings it.
    expect(target.querySelector('[aria-live="polite"]')).not.toBeNull();
    // Picked as a user does: the train-only dataset's card.
    target.querySelector<HTMLInputElement>('input[name="dataset"][value="train-only"]')!.click();
    flush();
    expect(id.get()).toBe('train-only');
    expect(note()).toMatch(/^No validation images: each epoch is scored on the training images/);
    expect(target.querySelector('[aria-live="polite"] #dataset-note')).not.toBeNull();
    // Both pickers (cards from 768 px, the select on phones) are described by it.
    expect(describedBy('[role="radiogroup"]')).toBe('dataset-note');
    expect(describedBy('select#dataset-pick')).toContain('dataset-note');
    target.querySelector<HTMLInputElement>('input[name="dataset"][value="with-val"]')!.click();
    flush();
    expect(note()).toBeNull();
    expect(describedBy('[role="radiogroup"]')).toBeNull();
    expect(describedBy('select#dataset-pick') ?? '').not.toContain('dataset-note');
  });
});
