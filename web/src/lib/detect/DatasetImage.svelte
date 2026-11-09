<script lang="ts">
  /**
   * A dataset image (resolved through the API, so it works with the mock too) with optional boxes on top. The tile
   * keeps its geometry throughout: a skeleton until the image has loaded, the boxes only over a loaded image (never
   * floating on an empty frame), and a failed URL or image says so, with Retry.
   */
  import ImageOff from '@lucide/svelte/icons/image-off';
  import { hub } from '$lib/api/index.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import DetectionView, { type ViewBox } from './DetectionView.svelte';

  interface Props {
    datasetId: string;
    imageId: string;
    width: number;
    height: number;
    thumb?: number;
    boxes?: ViewBox[];
    truth?: ViewBox[];
    classes: readonly string[];
    labels?: boolean;
    scores?: boolean;
    only?: ReadonlySet<number> | null;
    active?: number | null;
    onhover?: (i: number | null) => void;
    fit?: 'contain' | 'cover';
    compact?: boolean;
    alt?: string;
  }

  let { datasetId, imageId, thumb, boxes = [], truth = [], ...rest }: Props = $props();

  let src = $state<string | null>(null);
  let phase = $state<'loading' | 'ready' | 'error'>('loading');
  let attempt = $state(0);
  let el = $state<HTMLElement>();
  // Inside a button or link (a picker thumbnail) Retry cannot be a button of its own. While the image is unavailable
  // the tile itself retries instead of picking (click, Enter or Space alike; an unavailable image cannot be picked),
  // and "Retry" is part of the tile's accessible name, so keyboard and screen-reader users get the same action.
  const host = $derived(el?.parentElement?.closest<HTMLElement>('button, a[href]') ?? null);
  const nested = $derived(!!host);
  $effect(() => {
    const h = host;
    if (!h || phase !== 'error') return;
    h.addEventListener('click', retry, { capture: true });
    // A tile with its own aria-label ("Use …", "Try …") is named by that, not by the text inside: say it there.
    const label = h.getAttribute('aria-label');
    if (label !== null) h.setAttribute('aria-label', `${label}: image unavailable, activate to retry`);
    return () => {
      h.removeEventListener('click', retry, { capture: true });
      if (label !== null) h.setAttribute('aria-label', label);
    };
  });

  $effect(() => {
    void attempt;
    let live = true;
    src = null;
    phase = 'loading';
    hub.api.imageUrl(datasetId, imageId, thumb).then(
      (u) => {
        if (live) src = u;
      },
      () => {
        if (live) phase = 'error';
      },
    );
    return () => {
      live = false;
    };
  });

  /** Native (not delegated) capture listeners, so the tile around it never sees the click. */
  function retry(e: Event) {
    e.stopPropagation();
    e.preventDefault();
    attempt++;
  }
</script>

<div class="dsimg" bind:this={el} aria-busy={phase === 'loading'}>
  <DetectionView
    src={phase === 'error' ? null : src}
    boxes={phase === 'ready' ? boxes : []}
    truth={phase === 'ready' ? truth : []}
    onload={(ok) => (phase = ok ? 'ready' : 'error')}
    {...rest}
  />
  {#if phase === 'loading'}
    <span class="cover"><Skeleton w="100%" h="100%" r="0" image /></span>
  {:else if phase === 'error'}
    <span class="cover failed">
      <ImageOff size={16} strokeWidth={1.75} aria-hidden="true" />
      <span class="msg">Image unavailable</span>
      {#if nested}
        <!-- The tile is the control (see `host` above): this is its label. -->
        <span class="retry">Retry</span>
      {:else}
        <button type="button" class="retry" onclickcapture={retry} onkeydowncapture={(e) => e.stopPropagation()}>Retry</button>
      {/if}
    </span>
  {/if}
</div>

<style>
  .dsimg {
    position: relative;
    width: 100%;
    height: 100%;
  }
  .cover {
    position: absolute;
    inset: 0;
    display: flex;
  }
  .cover > :global(.sk) {
    flex: 1;
  }
  .failed {
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-1);
    padding: var(--space-2);
    background: var(--bg-inset);
    color: var(--text-tertiary);
    text-align: center;
  }
  .msg {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .retry {
    display: inline-flex;
    align-items: center;
    height: var(--control-sm);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
    cursor: pointer;
  }
  .retry:hover {
    background: var(--bg-hover);
    text-decoration-color: currentColor;
  }
  /* Touch: a 44 px target. Inside a button the whole tile is the target, so the pointer-only Retry stays small. */
  @media (pointer: coarse) {
    button.retry {
      min-height: 44px;
    }
  }
</style>
