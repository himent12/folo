<script lang="ts">
  import type { Snippet } from 'svelte';

  type Tone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger' | 'info' | 'outline';
  let { tone = 'neutral', children, mono = false }: { tone?: Tone; children: Snippet; mono?: boolean } = $props();
</script>

<span class="badge {tone}" class:mono>{@render children()}</span>

<style>
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 20px;
    padding: 0 6px;
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    line-height: 1;
    white-space: nowrap;
    background: var(--bg-active);
    color: var(--text-secondary);
  }
  .mono {
    font-family: var(--font-mono);
    font-weight: var(--weight-regular);
    letter-spacing: -0.01em;
  }
  .accent {
    background: var(--accent-muted);
    color: var(--accent-text);
  }
  .success {
    background: var(--success-muted);
    color: var(--success-text);
  }
  .warning {
    background: var(--warning-muted);
    color: var(--warning-text);
  }
  .danger {
    background: var(--danger-muted);
    color: var(--danger-text);
  }
  .info {
    background: var(--info-muted);
    color: var(--info-text);
  }
  .outline {
    background: transparent;
    box-shadow: inset 0 0 0 1px var(--line-default);
  }
</style>
