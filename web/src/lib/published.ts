// Published weights (`folo weights pull`) in plain words. The server's own description (`Model.about`: "folo-n trained
// from scratch on COCO train2017 at 640 px (EMA weights, f32, unfused: fine-tune or predict)") is provenance, kept for
// an ⓘ; what a person reads first is what the weights are for.

import type { Model } from '$lib/api/types';

/** The dataset the description names ("COCO"), when it names one the usual way ("… on COCO train2017 …"). */
export function pretrainedOn(about: string | null | undefined): string | null {
  return /\bon ([A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)\b/.exec(about ?? '')?.[1] ?? null;
}

/** "Pretrained on COCO (80 classes) · good for fine-tuning". */
export function publishedSummary(m: Pick<Model, 'about' | 'classes'>): string {
  const on = pretrainedOn(m.about);
  const classes = `${m.classes.length} ${m.classes.length === 1 ? 'class' : 'classes'}`;
  return `${on ? `Pretrained on ${on}` : 'Pretrained'} (${classes}) · good for fine-tuning`;
}
