// Detection box selection, table row focus navigation, and Escape/click-away dismissal for Playground.
import { computed, onWindow, type Signal } from 'fw';
import { nextRowIndex, scrollRowIntoView, type Det } from './viewState';

export interface UseSelectionInput {
  compare: Signal<boolean>;
  inspectSide: Signal<'a' | 'b'>;
  tab: Signal<'detections' | 'classes' | 'json'>;
  tbody: Signal<HTMLTableSectionElement | undefined>;
  keptA: () => Det[];
  keptB: () => Det[];
  pickedA: Signal<Det | null>;
  pickedB: Signal<Det | null>;
  rowsA: () => { b: Det; i: number }[];
  rowsB: () => { b: Det; i: number }[];
  resultA: () => { boxes: Det[]; ms: number } | null;
  resultB: () => { boxes: Det[]; ms: number } | null;
  selectedIdxA: () => number | null;
  selectedIdxB: () => number | null;
  activeIdxA: () => number | null;
  activeIdxB: () => number | null;
  isNarrow: () => boolean;
  pickerOpen: Signal<boolean>;
  sheetShown: () => boolean;
  closeSheet: () => void;
  onpaste: (e: ClipboardEvent) => void;
}

export function usePlaygroundSelection(o: UseSelectionInput) {
  /** In Compare mode, whether Model B is currently shown in the Detections table. */
  const showingB = computed(() => o.compare.get() && o.inspectSide.get() === 'b');
  const tableResult = computed(() => (showingB.get() ? o.resultB() : o.resultA()));
  const tableRows = computed(() => (showingB.get() ? o.rowsB() : o.rowsA()));
  const tableSelectedIdx = computed(() => (showingB.get() ? o.selectedIdxB() : o.selectedIdxA()));
  const tableActiveIdx = computed(() => (showingB.get() ? o.activeIdxB() : o.activeIdxA()));
  /** The row that takes Tab focus (roving tabindex): the selected one, else the first. */
  const tabRow = computed(() => tableSelectedIdx.get() ?? tableRows.get()[0]?.i ?? null);

  const rowEl = (i: number) => o.tbody.peek()?.querySelector<HTMLTableRowElement>(`tr[data-det="${i}"]`) ?? null;

  /** A box picked on the image: bring its row into view inside the list without moving the page. */
  function pickFromImage(i: number) {
    const b = o.keptA()[i];
    if (!b) return;
    const switched = (o.compare.peek() && o.inspectSide.peek() !== 'a') || o.tab.peek() !== 'detections';
    if (o.compare.peek() && o.inspectSide.peek() !== 'a') o.inspectSide.set('a');
    const next = o.pickedA.peek() === b ? null : b;
    o.pickedA.set(next);
    if (next) {
      if (o.tab.peek() !== 'detections') o.tab.set('detections');
      if (switched) queueMicrotask(() => scrollRowIntoView(rowEl(i), o.isNarrow()));
      else scrollRowIntoView(rowEl(i), o.isNarrow());
    }
  }

  function pickFromImageB(i: number) {
    const b = o.keptB()[i];
    if (!b) return;
    const switched = o.inspectSide.peek() !== 'b' || o.tab.peek() !== 'detections';
    if (o.inspectSide.peek() !== 'b') o.inspectSide.set('b');
    const next = o.pickedB.peek() === b ? null : b;
    o.pickedB.set(next);
    if (next) {
      if (o.tab.peek() !== 'detections') o.tab.set('detections');
      if (switched) queueMicrotask(() => scrollRowIntoView(rowEl(i), o.isNarrow()));
      else scrollRowIntoView(rowEl(i), o.isNarrow());
    }
  }

  function onrowkey(e: KeyboardEvent, k: number) {
    const list = tableRows.peek();
    const to = nextRowIndex(e.key, k, list.length);
    if (to === null) return;
    e.preventDefault();
    const r = list[to];
    if (r) rowEl(r.i)?.focus();
  }

  function onwindowkey(e: KeyboardEvent) {
    if (e.defaultPrevented || e.key !== 'Escape' || o.pickerOpen.peek()) return;
    if (document.querySelector('dialog[open]')) return;
    if (o.sheetShown()) {
      e.preventDefault();
      e.stopPropagation();
      o.closeSheet();
    } else if (o.pickedA.peek() || o.pickedB.peek()) {
      o.pickedA.set(null);
      o.pickedB.set(null);
    }
  }

  function onwindowdown(e: PointerEvent) {
    if ((o.pickedA.peek() || o.pickedB.peek()) && !(e.target as Element | null)?.closest?.('[data-det], [data-box], .zoom-hud')) {
      o.pickedA.set(null);
      o.pickedB.set(null);
    }
  }

  onWindow('paste', o.onpaste);
  onWindow('keydown', onwindowkey);
  onWindow('pointerdown', onwindowdown);

  return {
    showingB,
    tableResult,
    tableRows,
    tableSelectedIdx,
    tableActiveIdx,
    tabRow,
    pickFromImage,
    pickFromImageB,
    onrowkey,
  };
}
