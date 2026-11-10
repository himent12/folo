<script lang="ts">
  /**
   * New run, section 4: the augmentation settings, folded behind a one-line summary until opened, beside a preview of
   * them on the dataset's own images.
   */
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import Shuffle from '@lucide/svelte/icons/shuffle';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import type { Resource } from '$lib/api/resource.svelte';
  import type { Augment, Dataset, DatasetImage } from '$lib/api/types';
  import { num } from '$lib/format';
  import Button from '$lib/ui/Button.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Slider from '$lib/ui/Slider.svelte';
  import AugPreview, { type AugKey } from './AugPreview.svelte';
  import type { FormErrors } from './jump';

  let {
    augment = $bindable(),
    augOpen = $bindable(),
    closeEdited = $bindable(),
    defaults,
    errors,
    closeEff,
    epochs,
    dataset,
    sampleImages,
  }: {
    augment: Augment;
    augOpen: boolean;
    closeEdited: boolean;
    defaults: Augment;
    errors: FormErrors;
    closeEff: number | null;
    epochs: number;
    dataset: Dataset | undefined;
    sampleImages: Resource<DatasetImage[]>;
  } = $props();

  let focus = $state<AugKey | null>(null);
  let shuffle = $state(1);
  // A folded section must not hide its own error: while one is shown the section stays open whatever was toggled.
  const augShown = $derived(augOpen || errors.close !== null);

  const trim = (v: number) => String(+v.toFixed(3));
  /** The augmentation settings as one line; values that differ from the defaults are emphasised. */
  const augLine = $derived([
    { k: 'mosaic', v: trim(augment.mosaic), changed: augment.mosaic !== defaults.mosaic },
    { k: 'mixup', v: trim(augment.mixup), changed: augment.mixup !== defaults.mixup },
    { k: 'flip', v: trim(augment.fliplr), changed: augment.fliplr !== defaults.fliplr },
    { k: 'scale', v: `±${trim(augment.scale)}`, changed: augment.scale !== defaults.scale },
    { k: 'translate', v: `±${trim(augment.translate)}`, changed: augment.translate !== defaults.translate },
    { k: 'colour', v: trim(augment.hsv[1]), changed: augment.hsv[1] !== defaults.hsv[1] },
    { k: 'close mosaic', v: String(augment.close_mosaic), changed: augment.close_mosaic !== defaults.close_mosaic },
  ]);
  const augChanged = $derived(augLine.filter((it) => it.changed));
</script>

<Panel title="4 · Augmentation" id="sec-aug">
  {#snippet actions()}
    <span class="aug-actions" class:shut={!augShown}>
      <Button size="sm" variant="ghost" icon={Shuffle} onclick={() => shuffle++}>New samples</Button>
      <Button size="sm" variant="ghost" icon={RotateCcw} onclick={() => ((augment = structuredClone(defaults)), (closeEdited = false))}>Defaults</Button>
    </span>
  {/snippet}
  <button type="button" class="aug-toggle" aria-expanded={augShown} aria-controls="aug-body" onclick={() => (augOpen = !augShown)}>
    <ChevronRight size={14} aria-hidden="true" />
    <span class="aug-line num wide-only">
      {#each augLine as it, i (it.k)}{#if i > 0}<span class="sep" aria-hidden="true">·</span>{/if}<span class:changed={it.changed}>{it.k} {it.v}</span>{/each}
    </span>
    <!-- Phones: the seven settings never fit one line, so the line says what was changed, wrapping whole items. -->
    <span class="aug-line num phone-only">
      {#if augChanged.length}
        <span>{augChanged.length} changed</span>{#each augChanged as it (it.k)}<span class="sep" aria-hidden="true">·</span><span class="changed">{it.k} {it.v}</span>{/each}
      {:else}
        <span>All {augLine.length} settings at their defaults</span>
      {/if}
    </span>
    <span class="aug-verb">{augShown ? 'Hide' : 'Edit'}</span>
  </button>
  <div class="aug" id="aug-body" class:shut={!augShown}>
    <div class="aug-controls">
      {#snippet ctl(k: AugKey, children: import('svelte').Snippet)}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div
          class="aug-row"
          class:focused={focus === k}
          onpointerenter={() => (focus = k)}
          onpointerleave={() => (focus = null)}
          onfocusin={() => (focus = k)}
          onfocusout={() => (focus = null)}
        >
          {@render children()}
        </div>
      {/snippet}
      {#snippet mosaicCtl()}<Slider label="Mosaic" bind:value={augment.mosaic} min={0} max={1} step={0.05} hint="Stitch 4 images into one: more objects per sample." />{/snippet}
      {#snippet mixupCtl()}<Slider label="Mixup" bind:value={augment.mixup} min={0} max={1} step={0.05} hint="Blend two images and keep both sets of boxes." />{/snippet}
      {#snippet flipCtl()}<Slider label="Horizontal flip" bind:value={augment.fliplr} min={0} max={1} step={0.05} hint="Chance to mirror left-right." />{/snippet}
      {#snippet scaleCtl()}<Slider label="Scale" bind:value={augment.scale} min={0} max={0.9} step={0.05} format={(v) => `±${v.toFixed(2)}`} hint="Random zoom in and out." />{/snippet}
      {#snippet translateCtl()}<Slider label="Translate" bind:value={augment.translate} min={0} max={0.5} step={0.01} format={(v) => `±${v.toFixed(2)}`} hint="Random shift, as a fraction of the size." />{/snippet}
      {#snippet hsvCtl()}<Slider label="Colour jitter (saturation)" bind:value={augment.hsv[1]} min={0} max={1} step={0.05} hint="Hue ±{augment.hsv[0]}, value ±{augment.hsv[2]}: robustness to lighting." />{/snippet}
      {@render ctl('mosaic', mosaicCtl)}
      {@render ctl('mixup', mixupCtl)}
      {@render ctl('fliplr', flipCtl)}
      {@render ctl('scale', scaleCtl)}
      {@render ctl('translate', translateCtl)}
      {@render ctl('hsv', hsvCtl)}
      <div class="aug-row">
        <Field
          label="Close mosaic"
          for="close"
          hint={closeEff !== null && closeEff < augment.close_mosaic && augment.mosaic > 0
            ? `At most half the run: mosaic closes for the last ${num(closeEff)} ${closeEff === 1 ? 'epoch' : 'epochs'}.`
            : 'Train the last N epochs without mosaic to settle the boxes.'}
          error={errors.close}
        >
          <Input id="close" type="number" min="0" step="1" max={epochs} bind:value={augment.close_mosaic} oninput={() => (closeEdited = true)} suffix="epochs" invalid={!!errors.close} />
        </Field>
      </div>
    </div>
    <div class="aug-preview">
      <p class="overline">{focus ? `Preview · ${focus === 'fliplr' ? 'flip' : focus} only` : 'Preview · all augmentations'}</p>
      {#if dataset}
        <AugPreview datasetId={dataset.id} images={sampleImages.data ?? []} classes={dataset.classes} {augment} {focus} seed={shuffle} />
      {/if}
      <p class="aug-hint">Hover or focus a setting to see it alone. Boxes move with the pixels, as in training.</p>
    </div>
  </div>
</Panel>

<style>
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
</style>
