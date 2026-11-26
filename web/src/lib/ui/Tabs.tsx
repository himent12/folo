/**
 * Underline tabs for switching views inside a page. Arrow keys move between tabs (WAI-ARIA tabs pattern). The 1 px
 * track spans the whole column the tabs sit in, whatever holds them (a block, a grid cell or a flex row, as Run's
 * sticky phone picker is), never ending at the last tab.
 */
import { bindable, computed, css, defaults, For } from 'fw';

export interface Tab<T extends string> {
  value: T;
  label: string;
  meta?: string;
}

export interface TabsProps<T extends string> {
  tabs: Tab<T>[];
  value: T;
  label: string;
  onchange?: (v: T) => void;
  idPrefix?: string;
}

export default function Tabs<T extends string>(props: TabsProps<T>) {
  const p = defaults(props, { idPrefix: 'tab' });
  const value = bindable(props, 'value');
  let root!: HTMLDivElement;

  // WAI-ARIA tabs roving tabindex: keep the first tab tabbable if `value` does not match any tab.
  const tabTarget = computed(() => {
    const cur = p.tabs.findIndex((t) => t.value === value.get());
    return cur >= 0 ? cur : p.tabs.length > 0 ? 0 : -1;
  });

  function select(v: T) {
    value.set(v);
    p.onchange?.(v);
  }

  function onkeydown(e: KeyboardEvent) {
    const tabs = p.tabs;
    if (tabs.length === 0) return;
    const buttons = [...root.querySelectorAll<HTMLButtonElement>('[role=tab]')];
    const focused = buttons.indexOf(document.activeElement as HTMLButtonElement);
    const i = focused >= 0 ? focused : tabs.findIndex((t) => t.value === value.get());
    let j = i;
    if (e.key === 'ArrowRight') j = i < 0 ? 0 : (i + 1) % tabs.length;
    else if (e.key === 'ArrowLeft') j = i < 0 ? tabs.length - 1 : (i - 1 + tabs.length) % tabs.length;
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = tabs.length - 1;
    else return;
    e.preventDefault();
    const next = tabs[j];
    if (!next) return;
    select(next.value);
    buttons[j]?.focus();
  }

  return (
    <div class="tabs" role="tablist" aria-label={p.label} ref={(el) => (root = el)} onkeydown={onkeydown} tabindex={-1}>
      <For each={p.tabs} key={(t) => t.value}>
        {(t, idx) => {
          const on = () => t().value === value.get();
          return (
            <button
              type="button"
              role="tab"
              id={`${p.idPrefix}-${t().value}`}
              aria-selected={on()}
              aria-controls={`${p.idPrefix}-panel`}
              tabindex={idx() === tabTarget.get() ? 0 : -1}
              class:on={on()}
              onclick={() => select(t().value)}
            >
              {t().label}
              {t().meta ? <span class="meta">{t().meta}</span> : null}
            </button>
          );
        }}
      </For>
    </div>
  );
}

css`
  .tabs {
    display: flex;
    width: 100%;
    min-width: 0;
    gap: var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
    overflow-x: auto;
    scrollbar-width: none;
  }
  .tabs:focus {
    outline: none;
  }
  button {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 36px;
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
    white-space: nowrap;
    transition: color var(--dur-fast) var(--ease-standard);
  }
  button::after {
    content: '';
    position: absolute;
    left: 0;
    right: 0;
    bottom: -1px;
    height: 2px;
    border-radius: 2px 2px 0 0;
    background: var(--text-primary);
    transform: scaleX(0);
    transition: transform var(--dur-base) var(--ease-out);
  }
  button:hover {
    color: var(--text-primary);
  }
  button.on {
    color: var(--text-primary);
  }
  button.on::after {
    transform: scaleX(1);
  }
  @media (pointer: coarse) {
    button {
      height: 44px;
    }
  }
  .meta {
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
    font-variant-numeric: tabular-nums;
  }
  button:focus-visible {
    outline-offset: -2px;
  }
`;
