<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { HTMLButtonAttributes } from 'svelte/elements';
  import type { IconType } from './types';
  import Spinner from './Spinner.svelte';

  interface Props extends Omit<HTMLButtonAttributes, 'children'> {
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
    size?: 'sm' | 'md' | 'lg';
    icon?: IconType;
    iconRight?: IconType;
    href?: string;
    loading?: boolean;
    block?: boolean;
    children?: Snippet;
  }

  let {
    variant = 'secondary',
    size = 'md',
    icon: Icon,
    iconRight: IconRight,
    href,
    loading = false,
    block = false,
    disabled,
    children,
    class: klass = '',
    type = 'button',
    ...rest
  }: Props = $props();

  const iconSize = $derived(size === 'lg' ? 16 : 14);
  const cls = $derived(`btn ${variant} ${size} ${block ? 'block' : ''} ${children ? '' : 'square'} ${klass}`);
</script>

{#snippet inner()}
  {#if loading}
    <Spinner size={iconSize} />
  {:else if Icon}
    <Icon size={iconSize} strokeWidth={1.75} aria-hidden="true" />
  {/if}
  {#if children}<span class="label">{@render children()}</span>{/if}
  {#if IconRight}<IconRight size={iconSize} strokeWidth={1.75} aria-hidden="true" />{/if}
{/snippet}

{#if href && !disabled}
  <a {href} class={cls} {...rest as Record<string, unknown>}>{@render inner()}</a>
{:else}
  <button {type} class={cls} disabled={disabled || loading} aria-busy={loading || undefined} {...rest}>
    {@render inner()}
  </button>
{/if}

<style>
  .btn {
    --h: var(--control-md);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-1-5);
    height: var(--h);
    padding: 0 var(--space-3);
    border-radius: var(--radius-md);
    font-size: var(--type-base-size);
    font-weight: var(--weight-medium);
    line-height: 1;
    white-space: nowrap;
    user-select: none;
    border: 1px solid transparent;
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      border-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard),
      box-shadow var(--dur-fast) var(--ease-standard),
      transform var(--dur-instant) var(--ease-standard);
    flex-shrink: 0;
  }
  .btn:active:not(:disabled) {
    transform: translateY(0.5px);
  }
  .btn:focus-visible {
    border-radius: var(--radius-md);
  }
  .sm {
    --h: var(--control-sm);
    padding: 0 var(--space-2);
    font-size: var(--type-sm-size);
    border-radius: var(--radius-sm);
    gap: var(--space-1);
  }
  .lg {
    --h: var(--control-lg);
    padding: 0 var(--space-4);
  }
  .square {
    width: var(--h);
    padding: 0;
  }
  .block {
    width: 100%;
  }

  .primary {
    background: var(--accent-solid);
    color: var(--accent-on);
    box-shadow: inset 0 1px 0 #ffffff40, 0 0 0 1px #00000014;
  }
  .primary:hover:not(:disabled) {
    background: var(--accent-solid-hover);
  }

  .secondary {
    background: var(--bg-raised);
    border-color: var(--line-default);
    color: var(--text-primary);
    box-shadow: var(--shadow-sm);
  }
  .secondary:hover:not(:disabled) {
    border-color: var(--line-strong);
    background: color-mix(in srgb, var(--bg-raised), var(--text-primary) 4%);
  }

  .ghost {
    color: var(--text-secondary);
  }
  .ghost:hover:not(:disabled) {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .ghost:active:not(:disabled) {
    background: var(--bg-active);
  }

  .danger {
    background: var(--danger-muted);
    color: var(--danger-text);
    border-color: color-mix(in srgb, var(--danger-solid) 30%, transparent);
  }
  .danger:hover:not(:disabled) {
    background: color-mix(in srgb, var(--danger-solid) 22%, transparent);
  }

  .btn:disabled {
    opacity: 0.45;
  }
  .label {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1-5);
  }
</style>
