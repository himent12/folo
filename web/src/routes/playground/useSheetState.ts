// Narrow-viewport (tablet popover / phone bottom sheet) threshold-sheet state and click-away / focus-return handling.
import { computed, effect, signal, untrack, type MediaQuery } from 'fw';

export function usePlaygroundSheet(narrow: MediaQuery, phone: MediaQuery, hasSource: () => boolean) {
  const tablet = computed(() => narrow.current && !phone.current);
  const moreOpen = signal(false);
  const sheetEl = signal<HTMLElement>();
  const moreBtn = signal<HTMLElement>();

  const sheetShown = computed(() => moreOpen.get() && narrow.current && hasSource());
  const sheetTitle = computed(() => (tablet.get() ? 'More settings' : 'Thresholds'));

  effect(() => {
    if (sheetShown.get()) sheetEl.get()?.focus();
  });
  effect(() => {
    if (!narrow.current) untrack(() => moreOpen.set(false));
  });
  effect(() => {
    if (!sheetShown.get() || !tablet.get()) return;
    const away = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!sheetEl.peek()?.contains(t) && !moreBtn.peek()?.contains(t)) moreOpen.set(false);
    };
    document.addEventListener('pointerdown', away);
    return () => document.removeEventListener('pointerdown', away);
  });

  function closeSheet() {
    moreOpen.set(false);
    moreBtn.peek()?.querySelector<HTMLElement>('button')?.focus({ preventScroll: true });
  }

  return { tablet, moreOpen, sheetEl, moreBtn, sheetShown, sheetTitle, closeSheet };
}
