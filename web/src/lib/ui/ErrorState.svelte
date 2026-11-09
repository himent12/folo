<script lang="ts">
  import type { Snippet } from 'svelte';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import RotateCw from '@lucide/svelte/icons/rotate-cw';
  import Settings from '@lucide/svelte/icons/settings';
  import { ApiError } from '$lib/api/client';
  import { hub } from '$lib/api/index.svelte';
  import { sentence } from '$lib/format';
  import { isOom, oomDetail } from '$lib/oom';
  import { href } from '$lib/router.svelte';
  import Button from './Button.svelte';
  import { isOffline } from './offline';

  let {
    error,
    retry,
    title = 'Could not load this',
    compact = false,
    actions,
    hint: ownHint,
  }: {
    error: unknown;
    retry?: () => void;
    title?: string;
    compact?: boolean;
    actions?: Snippet;
    /** What to do, in this screen's terms, instead of the server's hint (which then goes with the technical details). */
    hint?: string;
  } = $props();

  // Connection failures look the same everywhere: what was tried, and both ways out (retry, settings).
  const offline = $derived(isOffline(error));
  const heading = $derived(offline ? "Can't reach the FOLO server" : title);
  // Out of GPU memory says so plainly; the driver's report goes with the technical details.
  const oom = $derived(!offline && isOom(error));
  const raw = $derived(error instanceof Error ? error.message : String(error));
  const message = $derived(
    offline ? `Nothing answered at ${hub.serverUrl}.` : oom ? 'The GPU ran out of memory.' : error instanceof ApiError ? sentence(raw) : raw,
  );
  const serverHint = $derived(error instanceof ApiError && error.hint ? sentence(error.hint) : undefined);
  const hint = $derived(offline ? 'Start folo serve, or check the server URL.' : (ownHint ?? serverHint));
  /** The hint in parts, identifiers (in backticks, as the server writes them) set in mono. */
  const hintParts = $derived((hint ?? '').split('`').map((text, i) => ({ text, code: i % 2 === 1 })));
  const code = $derived(error instanceof ApiError ? `${error.code}${error.status ? ` · HTTP ${error.status}` : ''}` : undefined);
  const detail = $derived(oom ? oomDetail(raw) : null);
  /** The server's own hint, kept with the details when the screen says what applies here. */
  const hiddenHint = $derived(!offline && ownHint && serverHint ? serverHint.replaceAll('`', '') : null);
</script>

<div class="err" class:compact role="alert">
  <div class="icon"><TriangleAlert size={16} strokeWidth={1.75} /></div>
  <div class="text">
    <p class="title">{heading}</p>
    <p class="msg">{message}</p>
    {#if hint}<p class="hint">{#each hintParts as p, i (i)}{#if p.code}<code>{p.text}</code>{:else}{p.text}{/if}{/each}</p>{/if}
    {#if code || detail || hiddenHint}
      <details class="code">
        <summary>Technical details</summary>
        {#if detail}<span class="line">{detail}</span>{/if}
        {#if hiddenHint}<span class="line">Server hint: {hiddenHint}</span>{/if}
        {#if code}<span class="line">{code}</span>{/if}
      </details>
    {/if}
    <!-- The ways out sit under the cause they answer, not at the far edge of the panel. -->
    {#if retry || offline || actions}
      <div class="acts">
        {@render actions?.()}
        {#if retry}<Button size="sm" icon={RotateCw} onclick={retry}>Retry</Button>{/if}
        {#if offline}<Button size="sm" variant="ghost" icon={Settings} href={href('/settings')}>Connection settings</Button>{/if}
      </div>
    {/if}
  </div>
</div>

<style>
  /* In place of the data, inside the panel that would hold it: no second box, the red is in the icon and title. */
  .err {
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-4);
  }
  .compact {
    padding: var(--space-3) 0;
  }
  /* Inside a padded panel the panel's own padding is the edge (as for EmptyState), so a titled panel holding the error
     is its header, the message and 16 px, not 28. */
  :global(.pad-md > .body) > .err.compact,
  :global(.pad-sm > .body) > .err.compact {
    padding: var(--space-1) 0 0;
  }
  .icon {
    color: var(--danger-text);
    padding-top: 2px;
  }
  .text {
    flex: 1;
    min-width: 0;
    display: grid;
    gap: 2px;
  }
  .title {
    font-weight: var(--weight-medium);
    color: var(--danger-text);
  }
  .msg {
    color: var(--text-secondary);
  }
  .hint {
    color: var(--text-tertiary);
  }
  .hint code {
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
  }
  .code {
    margin-top: 2px;
    font-family: var(--font-mono);
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .code .line {
    display: block;
    overflow-wrap: anywhere;
  }
  .code summary {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    cursor: pointer;
    font-family: var(--font-sans);
    width: max-content;
    list-style: none;
  }
  .code summary::-webkit-details-marker {
    display: none;
  }
  /* The system disclosure chevron (as in Config), not the native triangle. */
  .code summary::before {
    content: '';
    width: 5px;
    height: 5px;
    margin: 0 2px;
    border-right: 1.5px solid currentColor;
    border-bottom: 1.5px solid currentColor;
    transform: rotate(-45deg);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .code[open] summary::before {
    transform: rotate(45deg);
  }
  /* Touch: the disclosure is a 44 px target, as Outcome's driver report is. */
  @media (pointer: coarse) {
    .code summary {
      min-height: 44px;
    }
  }
  .acts {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    margin-top: var(--space-2);
  }
  .title {
    color: var(--danger-text);
  }
</style>
