/**
 * Class names as chips with their class colour, typed into one field: Enter or a comma adds what was typed (so does
 * leaving the field), Backspace in an empty field takes the last one off, × removes one. The enclosing Field's
 * message describes the field (ui/field); `draft` is what is typed and not added yet, for the form to add on submit.
 */
import X from '$icons/x';
import { bindable, css, For } from 'fw';
import { classColor } from '$lib/design/tokens';
import { describedBy, getField } from './field';

/** The names in `raw` (split at commas and line breaks) that `classes` does not have yet, each once. */
const newClasses = (classes: readonly string[], raw: string) =>
  raw
    .split(/[,\n]/)
    .map((c) => c.trim())
    .filter((c, i, all) => c && !classes.includes(c) && all.indexOf(c) === i);

/** `classes` with the names in `raw` (split at commas and line breaks) added, each once, at most `max` in all. */
export function addClasses(classes: readonly string[], raw: string, max: number): string[] {
  return [...classes, ...newClasses(classes, raw)].slice(0, max);
}

/** The new names in `raw` that `addClasses` leaves out: those past `max`. */
export function classesLeftOut(classes: readonly string[], raw: string, max: number): string[] {
  return newClasses(classes, raw).slice(Math.max(0, max - classes.length));
}

export interface ClassChipsProps {
  id: string;
  classes: string[];
  draft?: string;
  max: number;
  placeholder: string;
  invalid?: boolean;
  /** After each addition: the names that did not fit under `max` (none: an empty list), for the form to say. */
  onleftout?: (names: string[]) => void;
}

export default function ClassChips(props: ClassChipsProps) {
  const classes = bindable(props, 'classes');
  const draft = bindable(props, 'draft', '');
  const field = getField();

  function add() {
    props.onleftout?.(classesLeftOut(classes.get(), draft.get(), props.max));
    classes.set(addClasses(classes.get(), draft.get(), props.max));
    draft.set('');
  }

  function onkeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' || e.key === ',') {
      if (!draft.get().trim()) return;
      e.preventDefault();
      add();
    } else if (e.key === 'Backspace' && !draft.get() && classes.get().length) {
      classes.set(classes.get().slice(0, -1));
    }
  }

  // A click on the frame focuses its field, which has the keys.
  return (
    <div
      class="chips"
      class:invalid={props.invalid ?? false}
      onclick={(e) => (e.currentTarget.querySelector('input') as HTMLInputElement | null)?.focus()}
    >
      <For each={classes.get()} key={(c) => c}>
        {(c, i) => (
          <span class="chip">
            <span class="sw" style:background={classColor(i())}></span>
            {` ${c()} `}
            <button type="button" aria-label={`Remove ${c()}`} onclick={() => classes.set(classes.get().filter((x) => x !== c()))}>
              <X size={12} />
            </button>
          </span>
        )}
      </For>
      <input
        id={props.id}
        bind:value={draft}
        onkeydown={onkeydown}
        onblur={() => draft.get().trim() && add()}
        placeholder={classes.get().length ? '' : props.placeholder}
        aria-invalid={props.invalid || field?.invalid || undefined}
        aria-describedby={describedBy(field?.describedby)}
        autocomplete="off"
        spellcheck="false"
      />
    </div>
  );
}

css`
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
`;
