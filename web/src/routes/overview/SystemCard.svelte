<script lang="ts">
  import Cpu from '@lucide/svelte/icons/cpu';
  import { untrack } from 'svelte';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import Ring from '$lib/charts/Ring.svelte';
  import Sparkline from '$lib/charts/Sparkline.svelte';
  import { bytes, pct } from '$lib/format';
  import { prefs } from '$lib/prefs.svelte';
  import Badge from '$lib/ui/Badge.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Progress from '$lib/ui/Progress.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';

  const sys = new Resource((s) => hub.api.system(s), { poll: 2000 });

  // Client-side history for the utilisation sparklines (the API reports only "now").
  let gpuHist = $state<number[]>([]);
  let cpuHist = $state<number[]>([]);
  $effect(() => {
    const d = sys.data;
    if (!d) return;
    const g = d.gpus[0]?.util;
    untrack(() => {
      if (g !== null && g !== undefined) gpuHist = [...gpuHist.slice(-59), g];
      cpuHist = [...cpuHist.slice(-59), d.cpu.util];
    });
  });
</script>

<Panel title="System" id="system">
  {#snippet actions()}
    {#if sys.data?.gpus[0]}<Badge tone="outline" mono>{sys.data.gpus[0].backend}</Badge>{/if}
  {/snippet}
  {#if sys.error && !sys.data}
    <ErrorState error={sys.error} retry={() => sys.refresh()} compact title="Could not read system status" />
  {:else if !sys.data}
    <div class="rows">
      {#each [0, 1, 2] as i (i)}<div class="sk"><Skeleton w="44px" h="44px" r="50%" /><div class="skl"><Skeleton w="60%" /><Skeleton w="40%" /></div></div>{/each}
    </div>
  {:else}
    {@const d = sys.data}
    <div class="rows">
      {#each d.gpus as g (g.ordinal)}
        <div class="row">
          <Ring value={g.util} label="GPU {g.ordinal} utilisation" />
          <div class="text">
            <p class="name truncate">{g.name}</p>
            <p class="sub num">GPU {g.ordinal} · {bytes(g.memory_bytes, prefs.units)} · {g.util === null ? 'util n/a' : `${pct(g.util, 0)} busy`}</p>
          </div>
          {#if gpuHist.length >= 5}<Sparkline values={gpuHist} width={72} height={24} domain={[0, 1]} length={60} color="var(--text-secondary)" />{/if}
        </div>
      {:else}
        <p class="nogpu">No GPU adapter found; training runs on the CPU.</p>
      {/each}
      <div class="row">
        <Ring value={d.cpu.util} label="CPU utilisation" />
        <div class="text">
          <p class="name truncate"><Cpu size={12} class="ico" />{d.cpu.name}</p>
          <p class="sub num">{d.cpu.threads} threads · {pct(d.cpu.util, 0)} busy</p>
        </div>
        {#if cpuHist.length >= 5}<Sparkline values={cpuHist} width={72} height={24} domain={[0, 1]} length={60} color="var(--text-secondary)" />{/if}
      </div>
      <div class="mem">
        <div class="mem-top">
          <span class="faint">Memory</span>
          <span class="num">{bytes(d.memory.used_bytes, prefs.units)} <span class="faint">/ {bytes(d.memory.total_bytes, prefs.units)}</span></span>
        </div>
        <Progress value={d.memory.used_bytes / d.memory.total_bytes} tone="neutral" label="Memory in use" />
      </div>
    </div>
  {/if}
</Panel>

<style>
  .rows {
    display: grid;
    gap: var(--space-4);
  }
  .row,
  .sk {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
  }
  .skl {
    flex: 1;
    display: grid;
    gap: 6px;
  }
  .text {
    flex: 1;
    min-width: 0;
    display: grid;
    gap: 2px;
  }
  .name {
    font-weight: var(--weight-medium);
  }
  .name :global(.ico) {
    display: none;
  }
  .sub {
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
  }
  .mem {
    display: grid;
    gap: 6px;
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  .mem-top {
    display: flex;
    justify-content: space-between;
    font-size: var(--type-sm-size);
  }
  .nogpu {
    color: var(--text-tertiary);
  }
</style>
