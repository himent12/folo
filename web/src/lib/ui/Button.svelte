<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { HTMLButtonAttributes } from 'svelte/elements';
  import type { IconType } from './types';
  import Spinner from './Spinner.svelte';

  interface Props extends Omit<HTMLButtonAttributes, 'children'> {
    /** `danger` is the quiet destructive control on a page; `destructive` is the solid confirm inside a dialog. */
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'destructive';
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

{#if href && !disabled && !loading}
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

  /* The quiet destructive control on a page (Stop): a secondary button whose label and icon carry the danger hue,
     no tinted fill, so it never outshouts the page's focal object. The dialog it opens holds the solid confirm. */
  .danger {
    background: var(--bg-raised);
    border-color: var(--line-default);
    color: var(--danger-text);
    box-shadow: var(--shadow-sm);
  }
  .danger:hover:not(:disabled) {
    border-color: color-mix(in srgb, var(--danger-solid) 45%, var(--line-default));
    background: color-mix(in srgb, var(--bg-raised), var(--danger-solid) 6%);
  }

  .destructive {
    background: var(--danger-solid);
    color: var(--text-inverse);
    font-weight: var(--weight-medium);
  }
  .destructive:hover:not(:disabled) {
    background: color-mix(in srgb, var(--danger-solid), var(--text-inverse) 10%);
  }

  /* Light theme: lime on white needs an edge to read as a control. A neutral 1 px line.default edge (design/README.md),
     drawn outside the fill (background-clip), so the translucent line reads grey on the page, never olive on lime. */
  :global([data-theme='light']) .primary {
    border-color: var(--line-default);
    background-clip: padding-box;
    box-shadow: 0 1px 2px #0a0e1414;
  }
  /* Disabled, one treatment in both themes for every filled button: the raised surface, a disabled label, the
     default edge, no shadow. It reads as off, not as a faded primary (no pale lime that still looks pressable). A
     loading button is busy, not unavailable: it keeps its look and the spinner says it is working. */
  .btn:disabled:not(.ghost, [aria-busy='true']) {
    background: var(--bg-raised);
    border-color: var(--line-default);
    color: var(--text-disabled);
    box-shadow: none;
  }
  .ghost:disabled:not([aria-busy='true']) {
    color: var(--text-disabled);
  }
  /* Touch: grow the hit area to at least 44 px without changing the visual size. */
  @media (pointer: coarse) {
    .btn {
      position: relative;
    }
    .btn::after {
      content: '';
      position: absolute;
      inset: min(0px, calc((var(--h) - 44px) / 2)) 0;
    }
    /* Icon buttons get a real 44 × 44 box (no overlapping invisible targets). */
    .btn.square {
      --h: 44px;
    }
    .btn.square::after {
      display: none;
    }
  }
  .label {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1-5);
  }
</style>
