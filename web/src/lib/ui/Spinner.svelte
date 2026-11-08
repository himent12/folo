<script lang="ts">
  let { size = 14, label }: { size?: number; label?: string } = $props();
</script>

<svg
  class="spinner"
  width={size}
  height={size}
  viewBox="0 0 16 16"
  role={label ? 'status' : undefined}
  aria-label={label}
  aria-hidden={label ? undefined : 'true'}
>
  <circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" stroke-opacity="0.2" stroke-width="1.5" />
  <path d="M8 1.75a6.25 6.25 0 0 1 6.25 6.25" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" />
</svg>

<style>
  .spinner {
    animation: spin 700ms linear infinite;
    flex-shrink: 0;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
