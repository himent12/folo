<script lang="ts">
  /** A dataset image (resolved through the API, so it works with the mock too) with optional boxes on top. */
  import { hub } from '$lib/api/index.svelte';
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

  let { datasetId, imageId, thumb, ...rest }: Props = $props();

  let src = $state<string | null>(null);

  $effect(() => {
    let live = true;
    src = null;
    hub.api.imageUrl(datasetId, imageId, thumb).then(
      (u) => {
        if (live) src = u;
      },
      () => {},
    );
    return () => {
      live = false;
    };
  });
</script>

<DetectionView {src} {...rest} />
