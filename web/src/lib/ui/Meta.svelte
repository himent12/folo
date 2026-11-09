<script lang="ts">
  /**
   * A meta line: its child elements separated by " · " (a tertiary dot with 8 px on each side), the one separator
   * every meta line uses (DESIGN_DIRECTION §2.3). Each child is one item; give it an element of its own (a span),
   * never a bare text node, and never write the dot by hand.
   *
   * The dot hangs at the start of every item, so a line that wraps never ends on a "·": the dot that would lead the
   * next line falls outside the line's left edge and is clipped. `line` keeps the items on one line, cut with an
   * ellipsis at the end (table rows, list rows).
   */
  import type { Snippet } from 'svelte';

  let {
    children,
    line = false,
    class: cls = '',
    title,
  }: { children: Snippet; line?: boolean; class?: string; title?: string } = $props();
</script>

<span class="ui-meta {cls}" class:line {title}>{@render children()}</span>

<style>
  /* Wrapping: a flex row shifted one separator (20 px) left and clipped back at the content edge, so each line's
     leading dot is cut off; focus rings keep 4 px of room on that side. */
  .ui-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    row-gap: var(--space-1);
    min-width: 0;
    margin-left: -20px;
    clip-path: inset(-8px -8px -8px 16px);
  }
  /* A line breaks between items, never inside one; an item that may be too long for the line truncates itself. */
  .ui-meta > :global(*) {
    min-width: 0;
    white-space: nowrap;
  }
  .ui-meta > :global(*)::before {
    content: '·';
    display: inline-block;
    flex: none;
    width: 20px;
    text-align: center;
    font-family: var(--font-sans);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    text-decoration: none;
  }
  /* An item's content is inline (it follows its dot); a one-line meta nested in an item sits after the dot too. */
  :global(.ui-meta) .ui-meta.line {
    display: inline-block;
    max-width: calc(100% - 20px);
    vertical-align: top;
  }
  /* One line: inline flow cut with an ellipsis; the first item has no dot. */
  .ui-meta.line {
    display: block;
    margin-left: 0;
    clip-path: none;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .ui-meta.line > :global(*:first-child)::before {
    content: none;
  }
</style>
