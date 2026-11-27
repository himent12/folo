/**
 * Quick class-selection chips (keys 1–9) and compact keyboard hint strip for the labeling editor.
 * Lets users see and switch the active drawing/selected class in one click or keypress, with horizontal fade-scroll
 * on narrow screens so long class lists never overflow the dialog.
 */
import { css, effect, For, signal } from 'fw';
import { classInk } from '$lib/design/classInk';
import { fadeScroll } from '$lib/ui/fadeScroll';
import Kbd from '$lib/ui/Kbd';

export interface EditorHintsProps {
  classes: readonly string[];
  /** Active class index (the selected box's class when a box is selected, else the active draw class). */
  activeClass: number;
  /** True when a box is currently selected (keys/chips reclassify it; otherwise they set the draw class). */
  hasSelection: boolean;
  /** True while the agent is labeling this image. */
  disabled: boolean;
  /** Touch-only viewport: hides desktop keyboard hints and keycap badges. */
  touch: boolean;
  onselectclass: (cls: number) => void;
}

export default function EditorHints(props: EditorHintsProps) {
  const quick = () => props.classes.slice(0, 9).map((name, cls) => ({ cls, name }));
  const keySpan = () => {
    const n = Math.min(props.classes.length, 9);
    return n <= 1 ? '1' : `1–${n}`;
  };
  const stripEl = signal<HTMLDivElement | null>(null);
  const keepStrip = (el: HTMLDivElement) => {
    stripEl.set(el);
    return {
      destroy() {
        if (stripEl.peek() === el) stripEl.set(null);
      },
    };
  };

  effect(() => {
    const strip = stripEl.get();
    const cls = props.activeClass;
    const count = props.classes.length;
    if (!strip || count === 0) return;
    const btn = strip.querySelector<HTMLButtonElement>(`button[data-cls="${cls}"]`);
    if (!btn) return;
    const maxScroll = Math.max(0, strip.scrollWidth - strip.clientWidth);
    if (maxScroll <= 0) return;
    const sr = strip.getBoundingClientRect();
    const br = btn.getBoundingClientRect();
    if (br.width > 0 && sr.width > 0) {
      // Keep the active chip whole and clear of the 48 px edge fade mask without scrolling outer dialog containers.
      const pad = Math.max(0, Math.min(48, Math.floor((sr.width - br.width) / 2)));
      let next = strip.scrollLeft;
      if (br.left < sr.left + pad) next = Math.max(0, strip.scrollLeft + (br.left - sr.left) - pad);
      else if (br.right > sr.right - pad) next = Math.min(maxScroll, strip.scrollLeft + (br.right - sr.right) + pad);
      if (next !== strip.scrollLeft) {
        strip.scrollLeft = next;
        strip.dispatchEvent(new Event('scroll'));
      }
    } else {
      btn.scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
    }
  });

  return (
    <div class="ed-hints">
      {quick().length > 0 ? (
        <div
          class="class-keys"
          role="group"
          aria-label={props.hasSelection ? 'Set selected box class' : 'Active drawing class'}
          data-scrolls
          use:fadeScroll
          use:keepStrip
        >
          <For each={quick()} key={(item) => item.cls}>
            {(item) => {
              const active = () => props.activeClass === item().cls;
              return (
                <button
                  type="button"
                  class="ck"
                  class:active={active()}
                  data-cls={item().cls}
                  aria-pressed={active()}
                  disabled={props.disabled}
                  onclick={() => props.onselectclass(item().cls)}
                >
                  {props.touch ? null : <Kbd keys={String(item().cls + 1)} subtle />}
                  <span class="sw" style:background={classInk(item().cls)} aria-hidden="true"></span>
                  <span class="cname truncate">{item().name}</span>
                </button>
              );
            }}
          </For>
        </div>
      ) : null}

      {!props.touch ? (
        <div class="key-strip" role="group" aria-label="Keyboard shortcuts">
          {quick().length > 0 ? (
            <span class="kh">
              <Kbd keys={keySpan()} subtle />
              <span>{props.hasSelection ? 'Set class' : 'Draw class'}</span>
            </span>
          ) : null}
          <span class="kh">
            <Kbd keys="Mod+Z" subtle />
            <span>Undo</span>
          </span>
          <span class="kh">
            <Kbd keys="J" subtle />
            <Kbd keys="K" subtle />
            <span>Next / Prev</span>
          </span>
          <span class="kh">
            <Kbd keys="U" subtle />
            <span>Next unlabeled</span>
          </span>
          <span class="kh">
            <Kbd keys="D" subtle />
            <span>Duplicate</span>
          </span>
          <span class="kh">
            <Kbd keys="P" subtle />
            <span>Copy prev</span>
          </span>
          <span class="kh">
            <Kbd keys="Del" subtle />
            <span>Delete</span>
          </span>
          <span class="kh">
            <Kbd keys="Enter" subtle />
            <span>Review / Save</span>
          </span>
        </div>
      ) : null}
    </div>
  );
}

css`
  .ed-hints {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: var(--space-2);
    min-width: 0;
  }
  .class-keys {
    display: flex;
    align-items: center;
    flex-wrap: nowrap;
    gap: 6px;
    overflow-x: auto;
    scrollbar-width: none;
    padding: 2px 0;
    min-width: 0;
  }
  .class-keys:focus-visible {
    outline-offset: -2px;
  }
  .class-keys:global([data-fade='end']) {
    mask-image: linear-gradient(90deg, #000 calc(100% - 48px), transparent);
  }
  .class-keys:global([data-fade='start']) {
    mask-image: linear-gradient(90deg, transparent, #000 48px);
  }
  .class-keys:global([data-fade='both']) {
    mask-image: linear-gradient(90deg, transparent, #000 48px, #000 calc(100% - 48px), transparent);
  }
  .ck {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
    max-width: 180px;
    height: var(--control-sm);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--line-subtle);
    background: var(--bg-panel);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .ck:focus-visible {
    outline-offset: -2px;
  }
  @media (hover: hover) {
    .ck:hover:not(:disabled, .active) {
      border-color: var(--line-strong);
      background: var(--bg-hover);
      color: var(--text-primary);
    }
  }
  .ck.active {
    border-color: var(--text-tertiary);
    background: var(--bg-active);
    color: var(--text-primary);
  }
  .ck:disabled {
    color: var(--text-disabled);
    cursor: not-allowed;
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
    box-shadow: inset 0 0 0 1px var(--chart-swatch-edge);
  }
  .cname {
    min-width: 0;
    font-weight: var(--weight-medium);
  }
  .key-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-3);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .kh {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    white-space: nowrap;
  }
  @media (pointer: coarse) {
    .ck {
      min-width: 44px;
      min-height: 44px;
      justify-content: center;
    }
  }
  @media (max-width: 767px) {
    .key-strip,
    .ck :global(.kbd) {
      display: none;
    }
  }
`;
