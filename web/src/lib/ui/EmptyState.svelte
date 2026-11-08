<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { IconType } from './types';

  let {
    icon: Icon,
    title,
    description,
    actions,
    compact = false,
  }: { icon?: IconType; title: string; description?: string; actions?: Snippet; compact?: boolean } = $props();
</script>

<div class="empty" class:compact>
  {#if Icon}
    <div class="mark" aria-hidden="true">
      <span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span
        class="corner br"
      ></span>
      <Icon size={16} strokeWidth={1.75} />
    </div>
  {/if}
  <div class="text">
    <p class="title">{title}</p>
    {#if description}<p class="desc">{description}</p>{/if}
  </div>
  {#if actions}<div class="actions">{@render actions()}</div>{/if}
</div>

<style>
  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: var(--space-3);
    padding: var(--space-10) var(--space-6);
  }
  .compact {
    padding: var(--space-8) var(--space-4);
    gap: var(--space-3);
  }
  .mark {
    position: relative;
    display: grid;
    place-items: center;
    width: 32px;
    height: 32px;
    color: var(--text-tertiary);
  }
  .corner {
    position: absolute;
    width: 7px;
    height: 7px;
    border-color: var(--line-strong);
    border-style: solid;
    border-width: 0;
  }
  .tl {
    top: 0;
    left: 0;
    border-top-width: 1.5px;
    border-left-width: 1.5px;
    border-top-left-radius: 3px;
  }
  .tr {
    top: 0;
    right: 0;
    border-top-width: 1.5px;
    border-right-width: 1.5px;
    border-top-right-radius: 3px;
  }
  .bl {
    bottom: 0;
    left: 0;
    border-bottom-width: 1.5px;
    border-left-width: 1.5px;
    border-bottom-left-radius: 3px;
  }
  .br {
    bottom: 0;
    right: 0;
    border-bottom-width: 1.5px;
    border-right-width: 1.5px;
    border-bottom-right-radius: 3px;
  }
  .text {
    display: grid;
    gap: var(--space-1);
    max-width: 360px;
  }
  .title {
    font-size: var(--type-md-size);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .desc {
    color: var(--text-secondary);
    text-wrap: balance;
  }
  .actions {
    display: flex;
    gap: var(--space-2);
    flex-wrap: wrap;
    justify-content: center;
  }
</style>
