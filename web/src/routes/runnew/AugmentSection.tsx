// fw port of AugmentSection.svelte.
/**
 * New run, section 4: the augmentation settings, folded behind a one-line summary until opened, beside a preview of
 * them on the dataset's own images.
 */
import ChevronRight from '$icons/chevron-right';
import Shuffle from '$icons/shuffle';
import RotateCcw from '$icons/rotate-ccw';
import { bindable, computed, css, deep, For, signal, type Signal } from 'fw';
import type { Resource } from '$lib/api/resource';
import type { Augment, Dataset, DatasetImage } from '$lib/api/types';
import { num } from '$lib/format';
import Button from '$lib/ui/Button';
import Field from '$lib/ui/Field.tsx';
import Input from '$lib/ui/Input';
import Panel from '$lib/ui/Panel';
import Slider from '$lib/ui/Slider';
import AugPreview, { type AugKey } from './AugPreview';
import type { FormErrors } from './jump.fw';

export interface AugmentSectionProps {
  augment?: Augment;
  augOpen?: boolean;
  closeEdited?: boolean;
  defaults: Augment;
  errors: FormErrors;
  closeEff: number | null;
  epochs: number;
  dataset: Dataset | undefined;
  sampleImages: Resource<DatasetImage[]>;
}

const trim = (v: number) => String(+v.toFixed(3));

export default function AugmentSection(props: AugmentSectionProps) {
  // Bound by the page (`$bindable` in the Svelte original): never undefined. `augment` is a `deep` object, edited in
  // place by the controls and replaced whole by Defaults.
  const augment = bindable(props, 'augment') as Signal<Augment>;
  const augOpen = bindable(props, 'augOpen') as Signal<boolean>;
  const closeEdited = bindable(props, 'closeEdited') as Signal<boolean>;

  const focus = signal<AugKey | null>(null);
  const shuffle = signal(1);
  // A folded section must not hide its own error: while one is shown the section stays open whatever was toggled.
  const augShown = computed(() => augOpen.get() || props.errors.close !== null);

  /** `bind:` to one setting of whichever augment object the page holds now. */
  const at = <K extends 'mosaic' | 'mixup' | 'fliplr' | 'scale' | 'translate' | 'close_mosaic'>(k: K) => ({
    get: () => augment.get()[k],
    set: (v: Augment[K]) => {
      augment.get()[k] = v;
    },
  });
  const saturation = {
    get: () => augment.get().hsv[1],
    set: (v: number) => {
      augment.get().hsv[1] = v;
    },
  };

  /** The augmentation settings as one line; values that differ from the defaults are emphasised. */
  const augLine = computed(() => {
    const a = augment.get();
    const d = props.defaults;
    return [
      { k: 'mosaic', v: trim(a.mosaic), changed: a.mosaic !== d.mosaic },
      { k: 'mixup', v: trim(a.mixup), changed: a.mixup !== d.mixup },
      { k: 'flip', v: trim(a.fliplr), changed: a.fliplr !== d.fliplr },
      { k: 'scale', v: `±${trim(a.scale)}`, changed: a.scale !== d.scale },
      { k: 'translate', v: `±${trim(a.translate)}`, changed: a.translate !== d.translate },
      { k: 'colour', v: trim(a.hsv[1]), changed: a.hsv[1] !== d.hsv[1] },
      { k: 'close mosaic', v: String(a.close_mosaic), changed: a.close_mosaic !== d.close_mosaic },
    ];
  });
  const augChanged = computed(() => augLine.get().filter((it) => it.changed));

  // The Svelte original's `ctl` snippet, as a local component (six consecutive snippet calls would be one text run to
  // fw's compiler, which renders the first one twice to find out it is not text).
  const Ctl = (p: { k: AugKey; children: JSX.Element }) => (
    <div
      class="aug-row"
      class:focused={focus.get() === p.k}
      onpointerenter={() => focus.set(p.k)}
      onpointerleave={() => focus.set(null)}
      onfocusin={() => focus.set(p.k)}
      onfocusout={() => focus.set(null)}
    >
      {p.children}
    </div>
  );

  return (
    <Panel
      title="4 · Augmentation"
      id="sec-aug"
      actions={() => (
        <span class="aug-actions" class:shut={!augShown.get()}>
          <Button size="sm" variant="ghost" icon={Shuffle} onclick={() => shuffle.set((s) => s + 1)}>
            New samples
          </Button>
          <Button
            size="sm"
            variant="ghost"
            icon={RotateCcw}
            onclick={() => (augment.set(deep(structuredClone(props.defaults))), closeEdited.set(false))}
          >
            Defaults
          </Button>
        </span>
      )}
    >
      <button type="button" class="aug-toggle" aria-expanded={augShown.get()} aria-controls="aug-body" onclick={() => augOpen.set(!augShown.get())}>
        <ChevronRight size={14} aria-hidden="true" />
        <span class="aug-line num wide-only">
          <For each={augLine.get()} key={(it) => it.k}>
            {(it, i) => (
              <>
                {i() > 0 ? (
                  <span class="sep" aria-hidden="true">
                    ·
                  </span>
                ) : null}
                <span class:changed={it().changed}>
                  {it().k} {it().v}
                </span>
              </>
            )}
          </For>
        </span>
        {/* Phones: the seven settings never fit one line, so the line says what was changed, wrapping whole items. */}
        <span class="aug-line num phone-only">
          {augChanged.get().length ? (
            <>
              <span>{augChanged.get().length} changed</span>
              <For each={augChanged.get()} key={(it) => it.k}>
                {(it) => (
                  <>
                    <span class="sep" aria-hidden="true">
                      ·
                    </span>
                    <span class="changed">
                      {it().k} {it().v}
                    </span>
                  </>
                )}
              </For>
            </>
          ) : (
            <span>All {augLine.get().length} settings at their defaults</span>
          )}
        </span>
        <span class="aug-verb">{augShown.get() ? 'Hide' : 'Edit'}</span>
      </button>
      <div class="aug" id="aug-body" class:shut={!augShown.get()}>
        <div class="aug-controls">
          <Ctl k="mosaic">
            <Slider label="Mosaic" bind:value={at('mosaic')} min={0} max={1} step={0.05} hint="Stitch 4 images into one: more objects per sample." />
          </Ctl>
          <Ctl k="mixup">
            <Slider label="Mixup" bind:value={at('mixup')} min={0} max={1} step={0.05} hint="Blend two images and keep both sets of boxes." />
          </Ctl>
          <Ctl k="fliplr">
            <Slider label="Horizontal flip" bind:value={at('fliplr')} min={0} max={1} step={0.05} hint="Chance to mirror left-right." />
          </Ctl>
          <Ctl k="scale">
            <Slider label="Scale" bind:value={at('scale')} min={0} max={0.9} step={0.05} format={(v) => `±${v.toFixed(2)}`} hint="Random zoom in and out." />
          </Ctl>
          <Ctl k="translate">
            <Slider
              label="Translate"
              bind:value={at('translate')}
              min={0}
              max={0.5}
              step={0.01}
              format={(v) => `±${v.toFixed(2)}`}
              hint="Random shift, as a fraction of the size."
            />
          </Ctl>
          <Ctl k="hsv">
            <Slider
              label="Colour jitter (saturation)"
              bind:value={saturation}
              min={0}
              max={1}
              step={0.05}
              hint={`Hue ±${augment.get().hsv[0]}, value ±${augment.get().hsv[2]}: robustness to lighting.`}
            />
          </Ctl>
          <div class="aug-row">
            <Field
              label="Close mosaic"
              for="close"
              hint={
                props.closeEff !== null && props.closeEff < augment.get().close_mosaic && augment.get().mosaic > 0
                  ? `At most half the run: mosaic closes for the last ${num(props.closeEff)} ${props.closeEff === 1 ? 'epoch' : 'epochs'}.`
                  : 'Train the last N epochs without mosaic to settle the boxes.'
              }
              error={props.errors.close}
            >
              <Input
                id="close"
                type="number"
                min="0"
                step="1"
                max={props.epochs}
                bind:value={at('close_mosaic')}
                oninput={() => closeEdited.set(true)}
                suffix="epochs"
                invalid={!!props.errors.close}
              />
            </Field>
          </div>
        </div>
        <div class="aug-preview">
          <p class="overline">{focus.get() ? `Preview · ${focus.get() === 'fliplr' ? 'flip' : focus.get()} only` : 'Preview · all augmentations'}</p>
          {props.dataset ? (
            <AugPreview
              datasetId={props.dataset.id}
              images={props.sampleImages.data ?? []}
              classes={props.dataset.classes}
              augment={augment.get()}
              focus={focus.get()}
              seed={shuffle.get()}
            />
          ) : null}
          <p class="aug-hint">Hover or focus a setting to see it alone. Boxes move with the pixels, as in training.</p>
        </div>
      </div>
    </Panel>
  );
}

css`
  .aug {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: var(--space-5);
  }
  .aug-controls {
    display: grid;
    gap: 2px;
  }
  .aug-row {
    padding: var(--space-2) var(--space-3);
    margin: 0 calc(-1 * var(--space-3));
    border-radius: var(--radius-md);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .aug-row.focused {
    background: var(--bg-hover);
  }
  .aug-preview {
    display: grid;
    gap: var(--space-2);
    align-content: start;
    align-self: start;
    position: sticky;
    top: calc(var(--size-topbar) + var(--space-4));
  }
  .aug-hint {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* Helper text keeps a 12 px measure of at most ~60 characters, however wide the column. */
  .aug-hint,
  .aug-controls :global(.hint),
  .aug-controls :global(.msg) {
    max-width: 360px;
    text-wrap: pretty;
  }
  .aug-toggle {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: var(--control-lg);
    margin-top: calc(-1 * var(--space-1));
    padding: 0 var(--space-2);
    margin-inline: calc(-1 * var(--space-2));
    width: calc(100% + 2 * var(--space-2));
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    text-align: left;
  }
  .aug-toggle:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .aug-toggle > :global(svg) {
    flex-shrink: 0;
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .aug-toggle[aria-expanded='true'] > :global(svg) {
    transform: rotate(90deg);
  }
  .aug-line {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .aug-line .sep {
    margin-inline: var(--space-1-5);
    color: var(--text-tertiary);
  }
  .aug-line.phone-only {
    display: none;
  }
  @media (max-width: 767px) {
    .aug-line.wide-only {
      display: none;
    }
    /* Whole items wrap; none is cut. */
    .aug-line.phone-only {
      display: block;
      padding-block: var(--space-2);
      white-space: normal;
    }
    .aug-line.phone-only > span {
      white-space: nowrap;
    }
  }
  .aug-line .changed {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .aug-verb {
    flex-shrink: 0;
    color: var(--text-tertiary);
  }
  .aug-actions {
    display: contents;
  }
  .aug.shut,
  .aug-actions.shut {
    display: none;
  }
  .aug-toggle[aria-expanded='true'] {
    margin-bottom: var(--space-3);
  }
  @media (max-width: 1279px) {
    /* The preview takes a little more of the narrower form, so its frames stay large enough to read. */
    .aug {
      grid-template-columns: minmax(0, 5fr) minmax(0, 6fr);
      gap: var(--space-4);
    }
  }
  @media (max-width: 767px) {
    .aug {
      grid-template-columns: minmax(0, 1fr);
    }

    .aug-preview {
      position: static;
      order: -1;
    }
  }
`;
