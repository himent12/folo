import type { Arch, Augment, Dataset, Model } from '$lib/api/types';
import type { CloudPick } from './cloudPick';
import { devWord } from './device';
import { MAX_RUN_NAME, runName } from './name';
import { num } from '$lib/format';

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

/** Number inputs bind `null` when cleared; accept only finite whole numbers in range. */
export const isInt = (v: unknown, lo: number, hi: number) => typeof v === 'number' && Number.isInteger(v) && v >= lo && v <= hi;

/** The form's errors, one per field, in the order Start names the first. */
export function formErrors(
  name: string,
  epochs: number,
  lr0: number,
  seed: number,
  augment: Augment,
  dataset: Dataset | undefined,
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
        : n.length > MAX_RUN_NAME
          ? `At most ${MAX_RUN_NAME} characters (this one has ${n.length}).`
          : null,
    epochs: !isInt(epochs, 1, 10000) ? 'A whole number from 1 to 10,000.' : null,
    lr0: !(typeof lr0 === 'number' && lr0 > 0 && lr0 < 10) ? 'Greater than 0 and below 10.' : null,
    seed: !isInt(seed, 0, 2 ** 32 - 1) ? 'A whole number, 0 or more.' : null,
    close: !isInt(augment.close_mosaic, 0, Number.isFinite(epochs) ? epochs : 0) ? 'A whole number up to the epoch count.' : null,
    dataset: !dataset ? 'Pick a dataset.' : noTrainImages(dataset),
    weights: weights !== 'none' && picked?.arch !== arch ? 'Pick weights of the same model size.' : null,
    // The field says what to pick; the line beside Start only names the problem and leads there.
    device: deviceErr && !onCloud ? `${devWord(device)} is not available.` : null,
    amp: ampErr && ampOn && !onCloud ? 'Not available on this server now.' : null,
    runon: cloudOn ? cloudPick.errors.runon : null,
    offer: onCloud ? cloudPick.errors.offer : null,
    hours: onCloud ? cloudPick.errors.hours : null,
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
