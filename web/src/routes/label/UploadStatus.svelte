<script lang="ts">
  /**
   * An addition of images, said where the images are: its progress while files are sent ("Uploading 34 of 120…"), then
   * how it went ("118 added · 2 skipped"), the files left out with their reasons one click away; or why it failed (when
   * it failed part way, both: what was added, then how many files were not sent and why).
   */
  import X from '@lucide/svelte/icons/x';
  import { UploadIncomplete } from '$lib/api/upload';
  import { num } from '$lib/format';
  import Button from '$lib/ui/Button.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import Progress from '$lib/ui/Progress.svelte';
  import Spinner from '$lib/ui/Spinner.svelte';
  import type { Uploads } from './uploads.svelte';

  let { uploads }: { uploads: Uploads } = $props();

  const uid = $props.id();
  let open = $state(false);
  const result = $derived(uploads.result);
  /** An upload that stopped part way: what it added shows, then why the rest was not sent. */
  const incomplete = $derived(uploads.error instanceof UploadIncomplete ? uploads.error : null);
  $effect(() => {
    void result;
    open = false;
  });
</script>

{#if uploads.progress}
  {@const p = uploads.progress}
  <div class="status" role="status">
    <p class="line num"><Spinner size={12} /> Uploading {num(p.sent)} of {num(p.total)}…</p>
    <Progress value={p.total ? p.sent / p.total : 0} tone="accent" label="{p.sent} of {p.total} files uploaded" />
  </div>
{:else if uploads.error && !incomplete}
  <ErrorState error={uploads.error} title="Could not upload the images" compact>
    {#snippet actions()}<Button size="sm" onclick={() => uploads.dismiss()}>Dismiss</Button>{/snippet}
  </ErrorState>
{:else if result}
  <div class="status" role="status">
    <div class="head">
      <p class="line num">
        <Meta>
          <span>{num(result.added.length)} added</span>
          {#if result.skipped.length}<span>{num(result.skipped.length)} skipped</span>{/if}
        </Meta>
      </p>
      {#if result.skipped.length}
        <button type="button" class="more" aria-expanded={open} aria-controls="{uid}-skipped" onclick={() => (open = !open)}>{open ? 'Hide skipped' : 'Show skipped'}</button>
      {/if}
      {#if !incomplete}<span class="close"><Button variant="ghost" size="sm" icon={X} aria-label="Dismiss" onclick={() => uploads.dismiss()} /></span>{/if}
    </div>
    {#if result.skipped.length}
      <ul class="skipped scroll-shade" id="{uid}-skipped" hidden={!open}>
        {#each result.skipped as s, i (i)}<li><span class="mono f">{s.file}</span><span class="r">{s.reason}</span></li>{/each}
      </ul>
    {/if}
  </div>
  {#if incomplete}
    <ErrorState error={incomplete.cause} title="{num(incomplete.unsent)} {incomplete.unsent === 1 ? 'file was' : 'files were'} not sent" compact>
      {#snippet actions()}<Button size="sm" onclick={() => uploads.dismiss()}>Dismiss</Button>{/snippet}
    </ErrorState>
  {/if}
{/if}

<style>
  .status {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
  }
  .head {
    display: flex;
    align-items: center;
    gap: var(--space-3);
  }
  .line {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .close {
    margin-left: auto;
  }
  .more {
    font-size: var(--type-sm-size);
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  @media (pointer: coarse) {
    .more {
      min-height: 44px;
    }
  }
  .skipped {
    display: grid;
    gap: var(--space-1);
    max-height: 160px;
    overflow-y: auto;
    font-size: var(--type-sm-size);
  }
  .skipped[hidden] {
    display: none;
  }
  .skipped li {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);
    gap: var(--space-3);
  }
  .f {
    overflow-wrap: anywhere;
  }
  .r {
    color: var(--text-secondary);
  }
</style>
