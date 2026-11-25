/**
 * Subtle floating indicator shown while a G-chord ("go to") keyboard sequence is armed (./shortcuts `gChord`).
 * Never takes focus or intercepts pointer events; clears automatically on timeout, completion, Escape, or blur.
 */
import { css, cubicOut, fly, Index, Show } from 'fw';
import Kbd from '$lib/ui/Kbd';
import { motion } from '$lib/ui/motion';
import { mainNav, settingsNav } from './nav';
import { gChord } from './shortcuts';

const TARGETS = [...mainNav, settingsNav].map((n) => ({
  key: n.keys.split(' ')[1] ?? '',
  label: n.label,
}));

const STATUS_TEXT = `Go to: press ${TARGETS.map((t) => `${t.key} for ${t.label}`).join(', ')}`;

export default function GChordIndicator() {
  return (
    <div class="g-chord-slot">
      <span class="sr-only" role="status">
        {gChord.get() ? STATUS_TEXT : ''}
      </span>
      <Show when={gChord.get()}>
        <div class="g-chord" aria-hidden="true" transition:fly={{ y: 6, duration: motion(120), easing: cubicOut }}>
          <span class="lead">
            <Kbd keys="G" />
            <span class="go">Go to</span>
          </span>
          <span class="targets">
            <Index each={TARGETS}>
              {(t) => (
                <span class="target">
                  <kbd class="key">{t().key}</kbd>
                  <span class="name">{t().label}</span>
                </span>
              )}
            </Index>
          </span>
        </div>
      </Show>
    </div>
  );
}

css`
  .g-chord-slot {
    position: fixed;
    inset-inline: var(--space-4);
    bottom: calc(var(--dock-h) + var(--space-4));
    display: flex;
    justify-content: center;
    z-index: var(--z-toast);
    pointer-events: none;
  }
  .g-chord {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    max-width: 100%;
    gap: var(--space-2) var(--space-3);
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-lg);
    background: var(--bg-overlay);
    border: 1px solid var(--line-strong);
    box-shadow: var(--shadow-lg);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .lead {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1-5);
    padding-right: var(--space-3);
    border-right: 1px solid var(--line-subtle);
  }
  .go {
    color: var(--text-tertiary);
    font-weight: var(--weight-medium);
    white-space: nowrap;
  }
  .targets {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-3);
  }
  .target {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    white-space: nowrap;
  }
  .key {
    font-family: var(--font-sans);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-semibold);
    color: var(--text-primary);
  }
  .name {
    color: var(--text-secondary);
  }
  @media (max-width: 767px) {
    .g-chord-slot {
      bottom: calc(var(--tabbar-h) + var(--dock-h) + var(--space-3) + env(safe-area-inset-bottom));
    }
  }
`;
