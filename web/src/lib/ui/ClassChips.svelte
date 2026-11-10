<script lang="ts" module>
  /** `classes` with the names in `raw` (split at commas and line breaks) added, each once, at most `max` in all. */
  export function addClasses(classes: readonly string[], raw: string, max: number): string[] {
    const next = raw
      .split(/[,\n]/)
      .map((c) => c.trim())
      .filter((c, i, all) => c && !classes.includes(c) && all.indexOf(c) === i);
    return [...classes, ...next].slice(0, max);
  }
</script>

<script lang="ts">
  /**
   * Class names as chips with their class colour, typed into one field: Enter or a comma adds what was typed (so does
   * leaving the field), Backspace in an empty field takes the last one off, × removes one. The enclosing Field's
   * message describes the field (ui/field); `draft` is what is typed and not added yet, for the form to add on submit.
   */
  import X from '@lucide/svelte/icons/x';
  import { classColor } from '$lib/design/tokens';
  import { describedBy, getField } from './field';

  let {
    id,
    classes = $bindable(),
    draft = $bindable(''),
    max,
    placeholder,
    invalid = false,
  }: {
    id: string;
    classes: string[];
    draft?: string;
    max: number;
    placeholder: string;
    invalid?: boolean;
  } = $props();

  const field = getField();

  function add() {
    classes = addClasses(classes, draft, max);
    draft = '';
  }

  function onkeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' || e.key === ',') {
      if (!draft.trim()) return;
      e.preventDefault();
      add();
    } else if (e.key === 'Backspace' && !draft && classes.length) {
      classes = classes.slice(0, -1);
    }
  }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions: a click on the frame focuses its field, which has the keys. -->
<div class="chips" class:invalid onclick={(e) => (e.currentTarget.querySelector('input') as HTMLInputElement | null)?.focus()}>
  {#each classes as c, i (c)}
    <span class="chip">
      <span class="sw" style:background={classColor(i)}></span>
      {c}
      <button type="button" aria-label="Remove {c}" onclick={() => (classes = classes.filter((x) => x !== c))}><X size={12} /></button>
    </span>
  {/each}
  <input
    {id}
    bind:value={draft}
    {onkeydown}
    onblur={() => draft.trim() && add()}
    placeholder={classes.length ? '' : placeholder}
    aria-invalid={invalid || field?.invalid || undefined}
    aria-describedby={describedBy(field?.describedby)}
    autocomplete="off"
    spellcheck={false}
  />
</div>

<style>
  .chips {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1);
    min-height: var(--control-md);
    padding: 4px;
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    /* An input's edge (line.control), drawn like ui/Input: an invalid draft turns the edge and the ring danger. */
    border: 1px solid var(--line-control);
    cursor: text;
    transition: border-color var(--dur-fast) var(--ease-standard);
  }
  :global([data-theme='light']) .chips {
    background: var(--bg-panel);
  }
  .chips:hover {
    border-color: var(--line-control-hover);
  }
  /* The field's frame takes the global ring while its draft input has keyboard focus, like ui/Input (§2.9); a chip's
     remove button keeps its own ring, so the frame is not ringed for it. */
  .chips:has(input:focus-visible) {
    outline: var(--size-focus-ring) solid var(--focus);
    outline-offset: var(--size-focus-offset);
  }
  .chips.invalid {
    --line-control: var(--danger-solid);
    --line-control-hover: var(--danger-solid);
    --focus: var(--danger-solid);
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 20px;
    padding: 0 2px 0 6px;
    border-radius: var(--radius-xs);
    background: var(--bg-active);
    font-size: var(--type-sm-size);
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
  }
  :global([data-theme='light']) .sw {
    box-shadow: inset 0 0 0 1px #0a0e1433;
  }
  .chip button {
    display: grid;
    place-items: center;
    width: 16px;
    height: 16px;
    border-radius: 2px;
    color: var(--text-tertiary);
  }
  .chip button:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  /* Touch: a 44 px hit area around the small remove glyph. */
  @media (pointer: coarse) {
    .chip button {
      position: relative;
    }
    .chip button::after {
      content: '';
      position: absolute;
      inset: -12px;
    }
  }
  .chips input {
    flex: 1;
    min-width: 80px;
    height: 20px;
    padding: 0 4px;
    border: 0;
    background: none;
    outline: none;
  }
  /* Touch: the typed class at 16 px (type.lg), under which iOS zooms into the field (sourceRules.test.ts); the chips keep
     their size. */
  @media (pointer: coarse) {
    .chips input {
      height: 24px;
      font-size: var(--type-lg-size);
    }
  }
  .chips input::placeholder {
    color: var(--text-tertiary);
  }
</style>
