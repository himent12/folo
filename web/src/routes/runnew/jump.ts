// The New run form's errors, one per field, and the way from a review row or an error back to its control.
import { effect, signal, type Signal } from 'fw';
import type { Arch, Augment, Dataset, Model } from '$lib/api/types';
import type { ListedDataset } from '../datasets/problem';
import type { CloudPick } from './cloudPick';
import { devWord } from './device';
import { MAX_RUN_NAME, nameLength, runName } from './name';
import { num } from '$lib/format';
import { COPIED_SEED_ERROR, exactSeed, seedError } from '$lib/seed';

/** A review row's or a form error's way back to its control (RunNew's `jump`). */
export type Jump = 'name' | 'data' | 'model' | 'epochs' | 'batch' | 'imgsz' | 'device' | 'amp' | 'weights' | 'opt' | 'lr0' | 'seed' | 'head' | 'close' | 'runon' | 'offer' | 'hours';

/** The form's errors, one per field; null where the field is fine. */
export type FormErrors = Record<'name' | 'epochs' | 'lr0' | 'seed' | 'close' | 'dataset' | 'weights' | 'device' | 'amp' | 'runon' | 'offer' | 'hours', string | null>;

export const FIELD: Record<keyof FormErrors, string> = {
  name: 'Run name',
  epochs: 'Epochs',
  lr0: 'Learning rate',
  seed: 'Seed',
  close: 'Close mosaic',
  dataset: 'Dataset',
  weights: 'Start from',
  device: 'Device',
  amp: 'Mixed precision',
  runon: 'Run on',
  offer: 'GPU',
  hours: 'Stop after',
};
export const JUMP: Record<keyof FormErrors, Jump> = {
  name: 'name',
  epochs: 'epochs',
  lr0: 'lr0',
  seed: 'seed',
  close: 'close',
  dataset: 'data',
  weights: 'weights',
  device: 'device',
  amp: 'amp',
  runon: 'runon',
  offer: 'offer',
  hours: 'hours',
};

/** The control a review row returns to; the picker that is on screen (cards or the phone select). */
export function jumpTarget(key: Jump): HTMLElement | null {
  const visible = (el: HTMLElement | null) => (el && el.getClientRects().length ? el : null);
  switch (key) {
    case 'data':
      return visible(document.getElementById('dataset-pick')) ?? document.querySelector<HTMLElement>('input[name="dataset"]:checked');
    case 'model':
      return visible(document.querySelector<HTMLElement>('#arch-pick [aria-checked="true"]')) ?? document.querySelector<HTMLElement>('input[name="arch"]:checked');
    case 'batch':
    case 'imgsz':
      return document.querySelector<HTMLElement>(`#${key} [aria-checked="true"]`);
    default:
      return document.getElementById(key);
  }
}

/**
 * A dataset with nothing in its train split cannot be trained on (the server refuses it): where its images are
 * instead, "No training images (72 in test)."
 */
export function noTrainImages(dataset: Dataset): string | null {
  const s = dataset.splits;
  if (s.train > 0) return null;
  const rest = [s.val ? `${num(s.val)} in val` : '', s.test ? `${num(s.test)} in test` : ''].filter(Boolean).join(', ');
  return rest ? `No training images (${rest}).` : 'No images to train on.';
}

/**
 * What stops training on the picked dataset, said at the picker: none picked, or the one asked for (a draft's, a
 * link's) gone (`gone`: the line that says so), its folder missing or unreadable, or no train split.
 */
export function datasetError(dataset: ListedDataset | undefined, gone: string | null): string | null {
  if (!dataset) return gone ?? 'Pick a dataset.';
  if (dataset.problem === 'missing') return 'Folder missing: put it back, or pick another dataset.';
  if (dataset.problem === 'unreadable') return "Can't read its folder as a dataset now: pick another dataset.";
  return noTrainImages(dataset);
}

/** Number inputs bind `null` when cleared; accept only finite whole numbers in range. */
export const isInt = (v: unknown, lo: number, hi: number) => typeof v === 'number' && Number.isInteger(v) && v >= lo && v <= hi;

/** The form's errors, one per field, in the order Start names the first. */
export function formErrors(
  name: string,
  epochs: number,
  lr0: number,
  seed: string,
  augment: Augment,
  dataset: ListedDataset | undefined,
  gone: string | null,
  weights: string,
  picked: Model | undefined,
  arch: Arch,
  deviceErr: { device: string; gone: boolean; detail: string } | null,
  onCloud: boolean,
  device: string,
  ampErr: { device: string; message: string } | null,
  ampOn: boolean,
  cloudOn: boolean,
  cloudPick: CloudPick,
): FormErrors {
  const n = runName(name);
  return {
    // Other characters become dashes (runnew/name), so only an empty or a long name is wrong.
    name: !name.trim()
      ? 'Give the run a name.'
    : !n
          ? 'Use letters or digits.'
        : nameLength(n) > MAX_RUN_NAME
          ? `At most ${MAX_RUN_NAME} characters (this one has ${nameLength(n)}).`
          : null,
    epochs: !isInt(epochs, 1, 10000) ? 'A whole number from 1 to 10,000.' : null,
    // As the server reads it, a 32-bit float: 1e-46 is 0 there and 9.999999999 is 10, both refused.
    lr0: !(typeof lr0 === 'number' && Math.fround(lr0) > 0 && Math.fround(lr0) < 10) ? 'Greater than 0 and below 10.' : null,
    seed: seedError(seed),
    close: !isInt(augment.close_mosaic, 0, Number.isFinite(epochs) ? epochs : 0) ? 'A whole number up to the epoch count.' : null,
    dataset: datasetError(dataset, gone),
    weights: weights !== 'none' && picked?.arch !== arch ? 'Pick weights of the same model size.' : null,
    // The field says what to pick; the line beside Start only names the problem and leads there.
    device: deviceErr && !onCloud ? `${devWord(device)} is not available.` : null,
    amp: ampErr && ampOn && !onCloud ? 'Not available on this server now.' : null,
    runon: cloudOn ? cloudPick.errors.runon : null,
    offer: onCloud ? cloudPick.errors.offer : null,
    hours: onCloud ? cloudPick.errors.hours : null,
  };
}

/**
 * The Seed field's copy of a run's seed (`?from=`). One over 2^53 − 1 arrived rounded by JSON parsing (lib/seed, rule
 * B): another seed, so it is never offered as the run's; the field is left empty and says why (`over`, kept in the
 * draft) until a seed is typed.
 */
export function copiedSeed(seed: Signal<string>, wasOver: boolean) {
  const over = signal(wasOver);
  effect(() => {
    if (seed.get() !== '') over.set(false);
  });
  return {
    over: () => over.get(),
    copy(v: number) {
      seed.set(exactSeed(v) ? String(v) : '');
      over.set(!exactSeed(v));
    },
    errors: (e: FormErrors): FormErrors => (over.get() ? { ...e, seed: COPIED_SEED_ERROR } : e),
  };
}

/** The first problem, named by its field (shown beside a disabled Start, as a way to that field). */
export function firstError(errors: FormErrors): { text: string; key: Jump } | null {
  for (const [k, v] of Object.entries(errors)) if (v) return { text: `${FIELD[k as keyof typeof errors]}: ${v}`, key: JUMP[k as keyof typeof errors] };
  return null;
}

/** Brings a jump's control into view and focuses it, marking the spot where the browser keeps the focus ring hidden. */
export function landOn(key: Jump) {
  const el = jumpTarget(key);
  if (!el) return;
  const box = el.closest<HTMLElement>('.field, .card, .arch, .ds-pick, .a-pick') ?? el;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  box.scrollIntoView({ block: 'center', behavior: reduce ? 'auto' : 'smooth' });
  el.focus({ preventScroll: true, focusVisible: true } as FocusOptions);
  // Where the browser keeps the focus ring hidden after a mouse click, mark the landing spot until focus leaves.
  if (el.matches(':focus-visible')) return;
  box.dataset.jumped = '';
  el.addEventListener('blur', () => delete box.dataset.jumped, { once: true });
}
