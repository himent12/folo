/**
 * One validation sample, large: its file name and epoch in the title, Previous / Next (buttons, and ← / → anywhere in
 * the open dialog but a text field or slider), and the image fitted to the viewport, never scrolled. The file name comes
 * from the dataset's image list (an image id is its position there, docs/API.md), asked once per image; until it
 * arrives, or when the dataset is gone, the sample is named by its place among the samples.
 */
import ChevronLeft from '$icons/chevron-left';
import ChevronRight from '$icons/chevron-right';
import { computed, css, effect, signal } from 'fw';
import { hub } from '$lib/api/index';
import type { Sample, ScoredBox, Box } from '$lib/api/types';
import DatasetImage from '$lib/detect/DatasetImage';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';

export interface SampleInspectorProps {
  samples: Sample[];
  /** The sample shown (an index into `samples`); null: closed. */
  index: number | null;
  onindex: (i: number | null) => void;
  epoch: number | undefined;
  classes: string[];
  preds: (s: Sample) => ScoredBox[];
  truth: (s: Sample) => Box[];
  /** "3 predictions ≥ 0.25 · 6 ground-truth boxes". */
  describe: (s: Sample) => string;
  /** The run's dataset was removed: no image to show. */
  datasetGone?: boolean;
}

/**
 * File names asked for, per API instance (another server, or the demo, has other datasets under the same ids: as
 * lib/api/metricsCache), by dataset and image id (the request, so a second open asks nothing).
 */
const caches = new WeakMap<object, Map<string, Promise<string | null>>>();

/** The file's base name of a dataset image, or null when the dataset cannot say. */
export function imageFileName(datasetId: string, imageId: string): Promise<string | null> {
  const api = hub.api;
  let names = caches.get(api);
  if (!names) caches.set(api, (names = new Map()));
  const key = `${datasetId}/${imageId}`;
  let p = names.get(key);
  if (!p) {
    const at = Number(imageId.replace(/^\D+/, ''));
    const cache = names;
    p = Number.isFinite(at)
      ? api.datasetImages(datasetId, { offset: at, limit: 1 }).then(
          (r) => {
            const img = r.images[0];
            return img && img.id === imageId ? (img.file.split(/[\\/]/).pop() ?? img.file) : null;
          },
          () => null,
        )
      : Promise.resolve(null);
    cache.set(key, p);
    p.then((n) => n === null && cache.delete(key));
  }
  return p;
}

export default function SampleInspector(props: SampleInspectorProps) {
  const sample = computed(() => (props.index === null ? null : (props.samples[props.index] ?? null)));
  const file = signal<string | null>(null);
  effect(() => {
    const s = sample.get();
    file.set(null);
    if (!s || props.datasetGone) return;
    let live = true;
    void imageFileName(s.dataset_id, s.image_id).then((n) => live && file.set(n));
    return () => {
      live = false;
    };
  });
  const count = () => props.samples.length;
  const step = (d: number) => {
    const i = props.index;
    if (i === null) return;
    const next = i + d;
    if (next >= 0 && next < count()) props.onindex(next);
  };
  // ← / → step through the samples while the dialog is open, except where the keys move something else.
  effect(() => {
    if (props.index === null) return;
    const onkey = (e: KeyboardEvent) => {
      if (e.altKey || e.ctrlKey || e.metaKey || (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight')) return;
      const t = e.target;
      if (t instanceof Element && t.closest('input, select, textarea, [role="slider"]')) return;
      e.preventDefault();
      step(e.key === 'ArrowLeft' ? -1 : 1);
    };
    window.addEventListener('keydown', onkey);
    return () => window.removeEventListener('keydown', onkey);
  });

  const title = computed(() => {
    const i = props.index;
    if (i === null) return 'Sample';
    const at = `${file.get() ?? `Sample ${i + 1} of ${count()}`}`;
    return props.epoch ? `${at} · epoch ${props.epoch}` : at;
  });

  return (
    <Dialog
      open={props.index !== null}
      onclose={() => props.onindex(null)}
      title={title.get()}
      description={sample.get() ? props.describe(sample.get()!) : undefined}
      size="lg"
      footer={() => (
        <>
          <span class="pos num">
            {(props.index ?? 0) + 1} of {count()}
          </span>
          <Button icon={ChevronLeft} disabled={!props.index} onclick={() => step(-1)} title="Previous sample (←)">
            Previous
          </Button>
          <Button iconRight={ChevronRight} disabled={props.index === null || props.index >= count() - 1} onclick={() => step(1)} title="Next sample (→)">
            Next
          </Button>
        </>
      )}
    >
      {sample.get() && props.datasetGone ? (
        <p class="none">The dataset was removed, so this image is gone; the counts above are from the run.</p>
      ) : sample.get() ? (
        // As wide as the dialog allows and as tall as the dialog (at most 720 px, ui/Dialog) leaves between its header
        // and footer.
        <div
          class="big"
          style:aspect-ratio={`${sample.get()!.width} / ${sample.get()!.height}`}
          style:width={`min(100%, calc((min(720px, 100dvh - 48px) - 200px) * ${sample.get()!.width / sample.get()!.height}))`}
        >
          <DatasetImage
            datasetId={sample.get()!.dataset_id}
            imageId={sample.get()!.image_id}
            width={sample.get()!.width}
            height={sample.get()!.height}
            boxes={props.preds(sample.get()!)}
            truth={props.truth(sample.get()!)}
            classes={props.classes}
          />
        </div>
      ) : null}
    </Dialog>
  );
}

css`
  .big {
    margin-inline: auto;
    border-radius: var(--radius-md);
    overflow: hidden;
  }
  .none {
    color: var(--text-tertiary);
  }
  .pos {
    margin-right: auto;
    align-self: center;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
`;
