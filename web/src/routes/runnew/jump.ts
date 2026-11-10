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
