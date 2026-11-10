<script lang="ts">
  /**
   * SEAM (built next): "New loop", `POST /api/bench/benchmarks/{id}/loops` (`hub.api.startBenchLoop`). Its form: the
   * first teacher (a model whose classes are the benchmark's, same names in the same order: `models` filtered), the pool
   * (a dataset split, default train, or a labeling job: `datasets`, `jobs`), generations (1–5, $lib/bench/rules
   * GENERATIONS_MAX), the confidence labels are kept from (default: the teacher's own threshold), and the student
   * (architecture, epochs 30, image size). On success it calls `onstarted` with the loop (the page puts it on the
   * Loops tab through `LiveBench.putLoop`) and closes. Not mounted yet: the Loops tab has no "New loop" action until
   * this exists.
   */
  import type { Bench, BenchLoop, Dataset, LabelJob, Model } from '$lib/api/types';
  import Dialog from '$lib/ui/Dialog.svelte';

  let {
    open = $bindable(false),
    bench,
    models,
    datasets,
    jobs,
    onstarted,
  }: {
    open?: boolean;
    bench: Bench;
    models: readonly Model[] | undefined;
    datasets: readonly Dataset[] | undefined;
    jobs: readonly LabelJob[] | undefined;
    onstarted: (loop: BenchLoop) => void;
  } = $props();

</script>

<Dialog bind:open title="New loop" description="Test whether {bench.name}'s models can teach their successors." />
