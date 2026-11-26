import { computed, effect, signal, untrack } from 'fw';
import { router } from '$lib/router';
import type { EntryKind } from './AddEntryDialog';

const parseKind = (p: string | null): EntryKind => (p === 'agent' || p === 'external' ? p : 'model');

/**
 * Keeps a benchmark page's Add Entry dialog in sync with `?add=model|agent|external`:
 * - Opens (and switches tab) when `?add=` is present at mount or navigated to while already on the page.
 * - Closes a URL-opened dialog when browser Back removes `?add=`.
 * - Strips `?add=` from the URL (`replace: true`) when the dialog closes so a reload does not reopen it.
 */
export function useAddQuery(clearAdd: () => void) {
  const addParam = computed(() => (router.route.name === 'benchmark' ? router.route.query.get('add') : null));
  const first = untrack(() => addParam.get());
  const addOpen = signal(first !== null);
  const addKind = signal<EntryKind>(parseKind(first));
  let openedFromUrl = first !== null;

  function openAdd(kind: EntryKind = 'model', fromUrl = false) {
    openedFromUrl = fromUrl;
    addKind.set(kind);
    addOpen.set(true);
  }

  effect(() => {
    const p = addParam.get();
    if (p !== null) untrack(() => openAdd(parseKind(p), true));
    else if (openedFromUrl && untrack(() => addOpen.get())) {
      openedFromUrl = false;
      untrack(() => addOpen.set(false));
    }
  });

  effect(() => {
    if (addOpen.get()) return;
    openedFromUrl = false;
    const r = untrack(() => router.route);
    const loc = untrack(() => router.location);
    if (r.name !== 'benchmark' || loc.name !== 'benchmark' || !r.query.has('add')) return;
    untrack(clearAdd);
  });

  return { addOpen, addKind, openAdd };
}
