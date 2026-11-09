<script lang="ts">
  /**
   * A model, checkpoint or run identifier: an architecture (`folo-s`), the checkpoint a run started from
   * (`folo-s-coco80`, `best.folo`) or a server id (`r_…`). Always mono, wherever it appears (DESIGN_DIRECTION §2.3);
   * a run's, model's or dataset's own name, as its row, title or chip label, stays in sans. With `href` it is a link.
   */
  let {
    value,
    href,
    title,
    class: cls = '',
    onclick,
  }: { value: string; href?: string; title?: string; class?: string; onclick?: (e: MouseEvent) => void } = $props();
</script>

{#if href}
  <a class="ident mono {cls}" {href} {title} {onclick}>{value}</a>
{:else}
  <span class="ident mono {cls}" {title}>{value}</span>
{/if}
