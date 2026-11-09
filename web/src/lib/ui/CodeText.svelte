<script lang="ts" module>
  /** `text` in parts: the server's code spans (in backticks) apart from the prose around them. */
  export function codeParts(text: string): { text: string; code: boolean }[] {
    return text
      .split('`')
      .map((t, i) => ({ text: t, code: i % 2 === 1 }))
      .filter((p) => p.text);
  }
</script>

<script lang="ts">
  /**
   * Prose that may hold code, as the server writes it (`folo train --amp`): each code span in mono, never broken inside
   * a word (DESIGN_DIRECTION 2.3: a command is code, and a fact is never cut), so `--amp` never splits into "-" and
   * "-amp" on a phone. Each word of a span is one unbreakable run; a long span wraps at its spaces only. Copying it
   * gives the real characters.
   */
  let { text }: { text: string } = $props();
</script>

{#each codeParts(text) as p, i (i)}{#if p.code}<code class="mono">{#each p.text.split(' ') as w, j (j)}{#if j > 0}{' '}{/if}<span
          class="w">{w}</span
        >{/each}</code
    >{:else}{p.text}{/if}{/each}

<style>
  code {
    font-family: var(--font-mono);
  }
  .w {
    white-space: nowrap;
  }
</style>
