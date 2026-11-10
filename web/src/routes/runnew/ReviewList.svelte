<script lang="ts">
  /** The Summary's settings that shape the run, each row a way back to its control (`jump`). */
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import type { FormErrors, Jump } from './jump';

  let {
    withName,
    name,
    errors,
    review,
    jump,
  }: {
    withName: boolean;
    name: string;
    errors: FormErrors;
    review: { key: Jump; label: string; value: string; mono?: boolean; warn?: boolean }[];
    jump: (key: Jump) => void;
  } = $props();
</script>

<!-- A value stays in ink; a problem with it is the danger mark before it (DESIGN_DIRECTION §2.6: status hue on the
     mark, never on the value), and the row's reason is the line under the summary. -->
{#snippet rvValue(value: string, warn: boolean, mono: boolean)}
  <span class="rv-end">
    {#if warn}<CircleAlert class="rv-mark" size={14} strokeWidth={2} aria-hidden="true" />{/if}<span class="rv-v" class:mono>{value}</span>
  </span>
{/snippet}

<ul class="review" aria-label="Settings">
  {#if withName}
    <li><button type="button" class="rv" class:warn={!!errors.name} onclick={() => jump('name')} aria-label="Run name: {name.trim() || 'none'}{errors.name ? ', needs attention' : ''}. Edit"><span class="rv-k">Run name</span>{@render rvValue(name.trim() || '—', !!errors.name, false)}</button></li>
  {/if}
  {#each review as r (r.key)}
    <li>
      <button type="button" class="rv" class:warn={r.warn} onclick={() => jump(r.key)} aria-label="{r.label}: {r.value}{r.warn ? ', needs attention' : ''}. Edit">
        <span class="rv-k">{r.label}</span>{@render rvValue(r.value, !!r.warn, !!r.mono)}
      </button>
    </li>
  {/each}
</ul>

<style>
  /* Settings to review: 28 px rows, each a way back to its control. */
  .review {
    display: grid;
    margin: calc(-1 * var(--space-1)) calc(-1 * var(--space-2)) 0;
  }
  /* Touch: each review row is a full 44 px target (after the base rule, so it wins). */
  @media (pointer: coarse) {
    .rv.rv {
      height: 44px;
    }
  }
  .rv {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    width: 100%;
    height: 28px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    text-align: left;
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .rv:hover {
    background: var(--bg-hover);
  }
  .rv:focus-visible {
    outline-offset: -2px;
  }
  .rv-k {
    flex-shrink: 0;
    color: var(--text-tertiary);
  }
  .rv-v {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text-primary);
  }
  .rv-end {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    min-width: 0;
  }
  .rv-end :global(.rv-mark) {
    flex-shrink: 0;
    color: var(--danger-solid);
  }
</style>
