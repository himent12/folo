// fw port of Select.svelte.
/**
 * A styled native <select>: the platform picker is the most accessible and fastest option on every device. An option
 * may carry a `short` label: the closed select shows it (a model's name), while the open list and the accessible value
 * keep the full one (the name with its score and what the score is on), so a closed select never cuts a fact off at
 * its edge. Say the rest under the select, in a meta line it is described by (DESIGN_DIRECTION §2.23).
 */
// One select affordance app-wide: a single down chevron at the field's end, the one Dataset's class picker (a list
// button) has too, so two pickers side by side never wear two different marks.
import ChevronDown from '$icons/chevron-down';
import { bindable, computed, css, defaults, For, Index } from 'fw';
import { describedBy, getField } from './field.fw';

export interface SelectOption<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
  /** Consecutive options with the same group are listed under it (`<optgroup>`); none, at the top level. */
  group?: string;
  /** What the option is, as its tooltip (a native option holds no markup). */
  title?: string;
  /** The closed select's text while this option is chosen; the open list shows `label`. */
  short?: string;
}

export interface SelectProps<T extends string> {
  value: T;
  options: SelectOption<T>[];
  id?: string;
  size?: 'sm' | 'md';
  /** Accessible name when there is no visible <label for>. */
  label?: string;
  onchange?: (v: T) => void;
  disabled?: boolean;
  /** The field shows an error: announced, and tied to its message by `describedby` (Field's `<for>-error`). */
  invalid?: boolean;
  /** More ids to describe it by; an enclosing Field's hint or error is added on its own (`ui/field`). */
  describedby?: string;
}

export default function Select<T extends string>(props: SelectProps<T>) {
  const p = defaults(props, { size: 'md', disabled: false, invalid: false });
  const value = bindable(props, 'value');
  const field = getField();

  const shown = computed(() => p.options.find((o) => o.value === value.get())?.short);
  /** Runs of options: the ungrouped ones as they are, each group's under its label. */
  const runs = computed(() =>
    p.options.reduce<{ group?: string; options: SelectOption<T>[] }[]>((out, o) => {
      const last = out.at(-1);
      if (last && last.group === o.group) last.options.push(o);
      else out.push({ group: o.group, options: [o] });
      return out;
    }, []),
  );

  const option = (o: () => SelectOption<T>) => (
    <option value={o().value} disabled={o().disabled} title={o().title}>
      {o().label}
    </option>
  );
  const list = (r: () => { options: SelectOption<T>[] }) => (
    <For each={r().options} key={(o) => o.value}>
      {(o) => option(o)}
    </For>
  );

  return (
    <div class={`select ${p.size}`} class:closed-short={shown.get() !== undefined}>
      <select
        id={p.id}
        bind:value={value}
        aria-label={p.label}
        disabled={p.disabled}
        aria-invalid={p.invalid || field?.invalid || undefined}
        aria-describedby={describedBy(p.describedby, field?.describedby)}
        onchange={() => p.onchange?.(value.get())}
      >
        <Index each={runs.get()}>{(r) => <>{r().group ? <optgroup label={r().group}>{list(r)}</optgroup> : list(r)}</>}</Index>
      </select>
      {shown.get() !== undefined ? (
        <span class="shown" aria-hidden="true">
          {shown.get()}
        </span>
      ) : null}
      <span class="chev" aria-hidden="true">
        <ChevronDown size={14} strokeWidth={1.75} />
      </span>
    </div>
  );
}

css`
  .select {
    position: relative;
    display: flex;
    min-width: 0;
  }
  select {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: var(--control-md);
    padding: 0 28px 0 12px;
    border-radius: var(--radius-md);
    background: var(--bg-raised);
    border: 1px solid var(--line-control);
    color: var(--text-primary);
    cursor: pointer;
    transition: border-color var(--dur-fast) var(--ease-standard);
    text-overflow: ellipsis;
  }
  .sm select {
    height: var(--control-sm);
    padding: 0 28px 0 8px;
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
  }
  select:hover {
    border-color: var(--line-control-hover);
  }
  /* Focus: the global :focus-visible ring (base.css), never a halo or a lime edge of its own (§2.9). */
  select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  option {
    background: var(--bg-overlay);
    color: var(--text-primary);
  }
  /* A short label: the select's own text is hidden and the short one drawn in its place; the open list keeps ink. */
  .closed-short select {
    color: transparent;
  }
  .shown {
    position: absolute;
    /* Where the select's own text starts: its 1 px edge and 12 px padding. */
    inset: 0 28px 0 12px;
    padding-inline-start: 1px;
    overflow: hidden;
    color: var(--text-primary);
    line-height: var(--control-md);
    white-space: nowrap;
    text-overflow: ellipsis;
    pointer-events: none;
  }
  .select:has(select:disabled) .shown {
    opacity: 0.5;
  }
  .sm .shown {
    inset-inline-start: 8px;
    font-size: var(--type-sm-size);
    line-height: var(--control-sm);
  }
  @media (pointer: coarse) {
    .shown,
    .sm .shown {
      line-height: 44px;
      font-size: var(--type-lg-size);
    }
  }
  .chev {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    display: flex;
    color: var(--text-tertiary);
    pointer-events: none;
  }
  .sm .chev {
    right: 6px;
  }
  /* Touch: 44 px controls, and the select's text at 16 px (type.lg), under which iOS zooms the page into it on focus
     (sourceRules.test.ts). */
  @media (pointer: coarse) {
    select,
    .sm select {
      height: 44px;
      font-size: var(--type-lg-size);
    }
  }
`;
