<script lang="ts">
  /**
   * A fresh install's Overview, under the three steps: the live card's frame (epoch bar, the four stats, the validation
   * chart's axes) in placeholder geometry, muted and still, so the page shows what it becomes once a run trains instead
   * of three quarters of nothing. Decoration: hidden from assistive tech, inert, never animated.
   */
  import ChartSkeleton from '$lib/charts/ChartSkeleton.svelte';
  import Progress from '$lib/ui/Progress.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Stat from '$lib/ui/Stat.svelte';

  let narrow = $state(typeof matchMedia !== 'undefined' && matchMedia('(max-width: 767px)').matches);
  $effect(() => {
    const mq = matchMedia('(max-width: 767px)');
    const on = () => (narrow = mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  });
</script>

<section class="preview" aria-labelledby="preview-note">
  <p class="note" id="preview-note">Once a run starts, it trains here: its epoch, mAP and validation curve, live.</p>
  <div class="card" aria-hidden="true" inert>
    <header>
      <Skeleton w="80px" h="20px" r="var(--radius-full)" />
      <Skeleton w="min(160px, 30vw)" h="16px" />
    </header>
    <div class="body">
      <div class="stats">
        <div class="epoch">
          <Stat label="Epoch" value="" size="lg" loading />
          <Progress value={0} size="sm" label="Training progress" />
        </div>
        <div class="grid">
          <Stat label="mAP50-95" value="" loading />
          <Stat label="Throughput" value="" unit="img/s" loading />
          <Stat label="Learning rate" value="" loading />
          <Stat label="Loss" value="" loading />
        </div>
      </div>
      <div class="chart">
        <p class="chart-title">Validation mAP <span class="faint">by epoch</span></p>
        <ChartSkeleton height={narrow ? 120 : 160} />
      </div>
    </div>
  </div>
</section>

<style>
  .preview {
    display: grid;
    gap: var(--space-3);
  }
  .note {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .card {
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
  }
  /* Muted and still: placeholder blocks with no shimmer; the labels keep their contrast (they are still text). */
  .card :global(.sk) {
    animation: none;
  }
  header {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-4) var(--space-5) var(--space-3);
  }
  .body {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);
    gap: var(--space-5);
    padding: var(--space-2) var(--space-5) var(--space-5);
  }
  .stats {
    display: grid;
    gap: var(--space-5);
    align-content: start;
  }
  .epoch {
    display: grid;
    gap: var(--space-2);
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-4) var(--space-3);
  }
  .chart {
    min-width: 0;
    display: grid;
    gap: var(--space-2);
    align-content: start;
  }
  .chart-title {
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    font-weight: var(--weight-medium);
  }
  .chart-title .faint {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
  }
  @media (max-width: 1023px) {
    .body {
      grid-template-columns: minmax(0, 1fr);
    }
    .stats {
      grid-template-columns: minmax(200px, 1fr) 2fr;
      align-items: start;
    }
    .grid {
      grid-template-columns: repeat(4, 1fr);
    }
  }
  @media (max-width: 767px) {
    header {
      padding: var(--space-3) var(--space-4) var(--space-2);
    }
    .body {
      gap: var(--space-4);
      padding: var(--space-1) var(--space-4) var(--space-4);
    }
    .stats {
      grid-template-columns: minmax(0, 1fr);
    }
    .grid {
      grid-template-columns: 1fr 1fr;
    }
    .grid > :global(:nth-child(n + 3)) {
      display: none;
    }
  }
</style>
