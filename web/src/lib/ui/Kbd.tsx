/** Renders a shortcut like "Mod+K" as key caps; Mod is ⌘ on Apple platforms and Ctrl elsewhere. */
import { computed, css, defaults, Index } from 'fw';
import { formatKeys } from '$lib/keys';

export interface KbdProps {
  keys: string;
  subtle?: boolean;
  /** A key that does nothing now (turned off in Settings): outlined, no fill. */
  off?: boolean;
}

export default function Kbd(props: KbdProps) {
  const p = defaults(props, { subtle: false, off: false });
  const parts = computed(() => formatKeys(p.keys));

  return (
    <span class="kbd" class:subtle={p.subtle} class:off={p.off} aria-label={parts.get().join(' ')}>
      <Index each={parts.get()}>{(k) => <kbd>{k()}</kbd>}</Index>
    </span>
  );
}

css`
  /* A key group never breaks across lines; punctuation after it rides along in a \`.nowrap\` span ("Ctrl V."). */
  .kbd {
    display: inline-flex;
    gap: 2px;
    white-space: nowrap;
  }
  /* 12 px caps (type.sm): the key names are labels, so they meet the 12 px floor (DESIGN_DIRECTION §2.14). */
  kbd {
    display: inline-grid;
    place-items: center;
    min-width: 20px;
    height: 20px;
    padding: 0 4px;
    border-radius: var(--radius-xs);
    border: 1px solid var(--line-default);
    border-bottom-color: var(--line-strong);
    background: var(--bg-raised);
    color: var(--text-secondary);
    font-family: var(--font-sans);
    font-size: var(--type-sm-size);
    line-height: 1;
    font-weight: var(--weight-medium);
  }
  .off kbd {
    background: transparent;
    border-style: dashed;
    border-color: var(--line-strong);
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
  }
  .subtle kbd {
    background: transparent;
    border-color: var(--line-subtle);
    color: var(--text-tertiary);
  }
`;
