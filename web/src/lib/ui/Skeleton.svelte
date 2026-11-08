<script lang="ts">
  let {
    w = '100%',
    h = '12px',
    r = 'var(--radius-sm)',
    inline = false,
    class: klass = '',
  }: {
    w?: string;
    h?: string;
    r?: string;
    /** Sits in a line of text (after a zero-width space), so the line keeps its real height. */
    inline?: boolean;
    class?: string;
  } = $props();
</script>

<span class="sk {klass}" class:inline style:width={w} style:height={h} style:border-radius={r} aria-hidden="true"></span>

<style>
  .sk {
    display: block;
    background: linear-gradient(90deg, var(--bg-active) 0%, var(--bg-hover) 50%, var(--bg-active) 100%);
    background-size: 200% 100%;
    animation: shimmer 1.4s linear infinite;
  }
  .inline {
    display: inline-block;
    vertical-align: middle;
  }
  @keyframes shimmer {
    from {
      background-position: 100% 0;
    }
    to {
      background-position: -100% 0;
    }
  }
</style>
