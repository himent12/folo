/**
 * One image of a labeling job, to review and correct: its boxes in full size and as a list; select a box (on the
 * image, or in the list), change its class (the list's select, or 1–9), delete it (Delete, Backspace or its button),
 * move or resize it by dragging or with the arrow keys (Alt: resize; Shift: ten steps; a step is about a screen pixel,
 * never under an image pixel), draw a new one on empty image or add one with Add box; Esc drops the selection before it
 * closes anything. Undo and Redo (Ctrl+Z, Ctrl+Shift+Z or Ctrl+Y; ⌘ too) step through this image's edits, Discard
 * included. Save (the hand-drawn labels replace the image's; none is a background image; an image not
 * labeled yet can be saved as it is) or Discard; leaving unsaved changes asks first, whichever way (J / K, Close, a
 * link, Back), and closing the tab warns. J / K step through the images in the current filter. "Looks right" (Enter)
 * marks labels nobody changed as reviewed and opens the next image to review; saved boxes are reviewed too, so with
 * unsaved changes Enter saves them and opens the next image to review (Ctrl+S / ⌘S saves and stays). Both are made on
 * the labels shown (`expect`): once another tab, the CLI or an agent saved others, the server refuses them, the edits
 * stay and the stream (for a mark, a load of the image) shows the new labels. "Ask the agent again"
 * runs the agent on this image alone with a note; "Remove image" takes it out of the job. An image the agent has right
 * now cannot be edited, and says so; one that waits for its turn in a run can, and then leaves the run. Its keys work
 * while it is the topmost dialog (shell/pageCommands `dialogKey`), also after a button that had focus turned disabled.
 */
import ChevronLeft from '$icons/chevron-left';
import ChevronRight from '$icons/chevron-right';
import Check from '$icons/check';
import Plus from '$icons/plus';
import Redo from '$icons/redo-2';
import Sparkles from '$icons/sparkles';
import Trash from '$icons/trash-2';
import Undo from '$icons/undo-2';
import X from '$icons/x';
import { computed, createUniqueId, css, effect, Index, MediaQuery, onCleanup, onWindow, signal, untrack } from 'fw';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import type { LiveLabelJob } from '$lib/api/liveLabel';
import type { Box, LabelImage, LabelJob } from '$lib/api/types';
import { classColor } from '$lib/design/tokens';
import { num } from '$lib/format';
import { boxRevision, NOTE_MAX } from '$lib/label';
import { isTyping } from '$lib/keys';
import { router } from '$lib/router';
import { dialogKey } from '$lib/shell/pageCommands';
import { TOUCH_ONLY } from '$lib/shell/shortcuts';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import { DialogStatus } from '$lib/ui/feedback';
import Field from '$lib/ui/Field.tsx';
import Gate from '$lib/ui/Gate';
import Input from '$lib/ui/Input';
import Kbd from '$lib/ui/Kbd';
import Select from '$lib/ui/Select';
import { tooltip } from '$lib/ui/tooltip';
import type { AgentChoice } from './agentChoice';
import { keyStep, newBox, nudge, sameBoxes, stretch, tidy } from './boxEdit';
import { History } from './history';
import EditCanvas from './EditCanvas';
import EditorHints from './EditorHints';
import { askOutcome, imageState } from './words';

export interface EditorProps {
  live: LiveLabelJob;
  job: LabelJob;
  /** The images in the current filter, for J / K. */
  images: readonly LabelImage[];
  imageId: string | null;
  choice: AgentChoice;
  /** Opens another image, or closes the editor (null). */
  onnavigate: (id: string | null) => void;
}

const ARROWS: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };

/** The server refused a save or review made on boxes the image no longer has (`expect`, docs/API.md). */
const changedElsewhere = (e: unknown) => e instanceof ApiError && e.code === 'label_boxes_changed';

export default function Editor(props: EditorProps) {
  const uid = createUniqueId();
  const status = new DialogStatus();
  const open = signal(false);
  effect(() => {
    open.set(props.imageId !== null);
  });

  const img = computed(() => {
    const id = props.imageId;
    return id === null ? undefined : props.live.images.find((i) => i.id === id);
  });
  const idx = computed(() => {
    const cur = img.get();
    return cur ? props.images.findIndex((i) => i.id === cur.id) : -1;
  });

  // Announce image navigation only when the open image changes, not when background agent batches grow `props.images`.
  const navStatus = signal('');
  let prevNavId: string | null = null;
  effect(() => {
    const id = props.imageId;
    const prev = prevNavId;
    prevNavId = id;
    if (id === null) {
      navStatus.set('');
      return;
    }
    if (prev === null || prev === id) return;
    const msg = untrack(() => {
      const cur = props.live.images.find((i) => i.id === id);
      if (!cur) return '';
      const i = props.images.findIndex((x) => x.id === cur.id);
      const pos = i >= 0 ? `${num(i + 1)} of ${num(props.images.length)}` : 'Not in this filter';
      return `${cur.file}, ${pos}`;
    });
    navStatus.set(msg);
  });

  // The draft: the boxes as edited here, over the image's own (`base`). An image that changes on the server while it
  // is not being edited (the agent relabeled it) shows its new boxes; one that changes under unsaved edits (another
  // tab saved it) keeps the edits, takes the new boxes as what Discard goes back to, and says so: Save would replace
  // them, and nothing else would tell.
  const base = signal<Box[]>([]);
  const saving = signal(false);
  const draft = signal<Box[]>([]);
  const selected = signal<number | null>(null);
  const hovered = signal<number | null>(null);
  const lastClass = signal(0);
  /** Screen px per image px as the canvas shows the image. */
  const scale = signal(0);
  let shownId: string | null = null;
  const dirty = computed(() => !sameBoxes(draft.get(), base.get()));
  // Another tab saved this image under the edits: kept apart from the dismissible status line, so the leave dialog's
  // Save (the editor is inert under it) says so too. It goes once the edits do, or this editor's save replaced those labels.
  const elsewhere = signal(false);
  effect(() => {
    if (!dirty.get()) untrack(() => elsewhere.set(false));
  });
  const picked = computed(() => {
    const s = selected.get();
    return s === null ? undefined : draft.get()[s];
  });
  // This image's edits, for Undo and Redo; `rev` tells the buttons it changed.
  let disposed = false;
  onCleanup(() => (disposed = true));
  const history = new History<Box[]>();
  const rev = signal(0);
  const undoCount = computed(() => (rev.get(), history.past.length));
  const redoCount = computed(() => (rev.get(), history.future.length));
  const canUndo = computed(() => undoCount.get() > 0);
  const canRedo = computed(() => redoCount.get() > 0);
  const drawClass = computed(() => Math.min(lastClass.get(), Math.max(0, props.job.classes.length - 1)));
  const activeClass = computed(() => picked.get()?.class ?? drawClass.get());
  function forget() {
    history.clear();
    rev.set(rev.peek() + 1);
  }
  effect(() => {
    const id = props.imageId;
    const cur = img.get();
    if (id === null) {
      hovered.set(null);
      if (shownId !== null) {
        shownId = null;
        base.set([]);
        draft.set([]);
        selected.set(null);
        note.set('');
        asked.set(null);
        status.clear();
        elsewhere.set(false);
        leaving.set(null);
        removing.set(null);
        removeStatus.clear();
        untrack(forget);
      }
      return;
    }
    if (!cur) return void hovered.set(null);
    const rem = removing.peek();
    if (rem && (rem.id !== cur.id || rem.jobId !== props.job.id)) {
      removing.set(null);
      removeStatus.clear();
    }
    if (cur.id !== shownId) {
      shownId = cur.id;
      base.set(cur.boxes);
      draft.set(cur.boxes);
      selected.set(null);
      hovered.set(null);
      note.set('');
      asked.set(null);
      status.clear();
      elsewhere.set(false);
      untrack(forget);
    } else if (!sameBoxes(cur.boxes, base.get())) {
      if (sameBoxes(draft.get(), base.get())) {
        base.set(cur.boxes);
        draft.set(cur.boxes);
        selected.set(null);
        hovered.set(null);
        untrack(forget);
      } else if (!saving.get()) {
        // Not this editor's own save on its way (its answer sets `base`): someone else's.
        base.set(cur.boxes);
        if (!sameBoxes(draft.peek(), cur.boxes))
          untrack(() => {
            elsewhere.set(true);
            status.show('info', 'Changed elsewhere while you edited', 'Its labels were saved from another tab or window. Save replaces them with yours; Discard shows them.');
          });
      }
    }
  });

  const locked = computed(() => (img.get()?.status === 'labeling' ? 'The agent is labeling this image. You can edit it once it is done.' : null));
  const undoTip = computed(() => {
    if (locked.get()) return { text: 'Undo unavailable while the agent is labeling' };
    const n = undoCount.get();
    return n > 0 ? { text: `Undo (${n} ${n === 1 ? 'step' : 'steps'})`, keys: 'Mod+Z' } : { text: 'Nothing to undo', keys: 'Mod+Z' };
  });
  const redoTip = computed(() => {
    if (locked.get()) return { text: 'Redo unavailable while the agent is labeling' };
    const n = redoCount.get();
    return n > 0 ? { text: `Redo (${n} ${n === 1 ? 'step' : 'steps'})`, keys: 'Mod+Shift+Z' } : { text: 'Nothing to redo', keys: 'Mod+Shift+Z' };
  });

  // ---------- editing ----------
  /** Every edit of the draft goes through here, so it can be undone; `kind` joins a run of like edits into one step. */
  function edit(next: Box[], kind: string | null = null) {
    if (sameBoxes(next, draft.peek())) return;
    history.record(draft.peek(), kind);
    draft.set(next);
    rev.set(rev.peek() + 1);
  }
  function undo(redo = false) {
    if (locked.peek()) return;
    const to = redo ? history.redo(draft.peek()) : history.undo(draft.peek());
    if (!to) return;
    // The selection is a place in the list, and a step (Discard most of all) may put another box there: it goes, so
    // Delete or a class key never lands on a box the user did not pick.
    selected.set(null);
    draft.set(to);
    rev.set(rev.peek() + 1);
  }
  /** Sets the active drawing class and, when a box is selected, reclassifies that box too (keys 1–9 or quick chips). */
  function setClass(c: number) {
    if (locked.peek() || c < 0 || c >= props.job.classes.length) return;
    lastClass.set(c);
    const s = selected.peek();
    if (s === null) return;
    edit(draft.peek().map((b, i) => (i === s ? { ...b, class: c } : b)));
  }
  function removeBox() {
    const s = selected.peek();
    if (s === null || locked.peek()) return;
    const root = body.peek();
    const active = document.activeElement;
    const restoreFocus =
      active instanceof Node &&
      !!(root?.querySelector('.boxes')?.contains(active) || root?.querySelector('.sel')?.contains(active));
    edit(draft.peek().filter((_, i) => i !== s));
    selected.set(null);
    if (restoreFocus) {
      queueMicrotask(() => {
        const cur = body.peek();
        const btns = [...(cur?.querySelectorAll<HTMLButtonElement>('.boxes button') ?? [])];
        const addBtn = cur?.querySelector<HTMLButtonElement>('[data-add-box]');
        const next = btns[Math.min(s, btns.length - 1)] ?? (addBtn && !addBtn.disabled ? addBtn : null) ?? cur?.closest('dialog');
        next?.focus();
      });
    }
  }
  // Keep the selected box visible inside `.boxes` only (never `scrollIntoView`, which scrolls `dialog .body` mid-drag on phones).
  effect(() => {
    const s = selected.get();
    if (s === null) return;
    queueMicrotask(() => {
      const list = body.peek()?.querySelector<HTMLElement>('.boxes');
      const btn = list?.querySelectorAll<HTMLButtonElement>('button')[s];
      if (!list || !btn) return;
      const lr = list.getBoundingClientRect();
      const br = btn.getBoundingClientRect();
      if (lr.height <= 0 || br.height <= 0) return;
      if (br.top < lr.top) list.scrollTop += br.top - lr.top;
      else if (br.bottom > lr.bottom) list.scrollTop += br.bottom - lr.bottom;
    });
  });
  /** Add box: a box of the class used last in the middle of the image, selected (the keyboard's way to draw one). */
  function addBox() {
    const cur = img.peek();
    if (!cur || locked.peek()) return;
    edit([...draft.peek(), newBox(cur.width, cur.height, drawClass.peek())]);
    selected.set(draft.peek().length - 1);
  }
  /** Arrow keys on the selected box: move it, or with Alt resize it from its bottom-right corner; Shift: ten steps. */
  function arrow(e: KeyboardEvent): boolean {
    const d = ARROWS[e.key];
    const b = picked.peek();
    const cur = img.peek();
    if (!d || !b || !cur || locked.peek()) return false;
    const by = keyStep(scale.peek()) * (e.shiftKey ? 10 : 1);
    const next = e.altKey ? stretch(b, d[0] * by, d[1] * by, cur.width, cur.height) : nudge(b, d[0] * by, d[1] * by, cur.width, cur.height);
    const s = selected.peek();
    edit(
      draft.peek().map((x, i) => (i === s ? next : x)),
      `${e.altKey ? 'size' : 'move'} ${s}`,
    );
    return true;
  }
  function discard() {
    if (locked.peek() || saving.peek()) return;
    edit(base.peek());
    selected.set(null);
  }

  // An image not labeled yet (waiting, or failed) can be saved as it is: kept as labeled by hand, its boxes or none.
  const savable = computed(() => dirty.get() || (!!img.get() && img.get()!.status !== 'labeled'));
  async function save(): Promise<boolean> {
    const cur = img.peek();
    if (!cur || saving.peek()) return false;
    // The answer is this image's, sent with these boxes: by the time it comes another image may be open, or the
    // boxes edited further, and neither is the answer's to change.
    const id = cur.id;
    const sent = draft.peek();
    // Made on the labels shown here: refused once the image has others (another tab, the CLI or an agent saved some
    // this editor has not shown yet), which the save would otherwise replace unseen. The revision is the newest
    // image's, not `base`'s: boxes that compare equal here (0 and -0) can still be another revision.
    const expect = boxRevision(cur.boxes);
    const mark = props.live.mark();
    // A save ends a run of arrow presses: the next one is a step of its own, back to what was saved.
    history.seal();
    saving.set(true);
    status.clear();
    try {
      const saved = await hub.api.updateLabelImage(props.job.id, id, sent.map(tidy), expect);
      props.live.put([saved], mark);
      if (!disposed && shownId === id) {
        base.set(saved.boxes);
        elsewhere.set(false);
        if (draft.peek() === sent) draft.set(saved.boxes);
        status.success('Saved', imageState(saved));
      }
      return true;
    } catch (e) {
      // The edits stay; the stream brings the labels saved elsewhere, as the base Discard goes back to.
      if (disposed || shownId !== id) return false;
      if (!changedElsewhere(e)) status.error(e, 'Could not save the boxes');
      else status.show('info', 'Changed elsewhere while you edited', 'Its labels were saved from another tab or window first, so yours were not. Save replaces them with yours; Discard shows them.');
      return false;
    } finally {
      saving.set(false);
    }
  }

  // ---------- reviewing ----------
  /** "Looks right" applies: labels nobody has changed here or checked before, not with the agent. */
  const reviewable = computed(() => {
    const cur = img.get();
    return !!cur && cur.status === 'labeled' && !cur.reviewed && !dirty.get() && !locked.get();
  });
  const reviewing = signal(false);
  /** No keyboard to name a key for. */
  const touch = new MediaQuery(TOUCH_ONLY, false);
  /** The next image of the filter whose labels nobody has checked: after this one, then from the start. */
  function nextToReview(fromId: string): string | null {
    const list = props.images;
    const i = list.findIndex((x) => x.id === fromId);
    const order = i < 0 ? list : [...list.slice(i + 1), ...list.slice(0, i)];
    return order.find((x) => x.id !== fromId && x.status === 'labeled' && !x.reviewed)?.id ?? null;
  }
  /** Marks the labels reviewed (and opens the next image to review) or not. */
  async function review(reviewed: boolean) {
    const cur = img.peek();
    if (!cur || reviewing.peek()) return;
    // Read before the answer: under "To review" the image leaves the filter once it is reviewed.
    const next = reviewed ? nextToReview(cur.id) : null;
    // The mark is for the labels on screen (the newest image's, as in `save`): once the image has others, the server
    // refuses it.
    const expect = boxRevision(cur.boxes);
    const mark = props.live.mark();
    reviewing.set(true);
    status.clear();
    try {
      const saved = await hub.api.reviewLabelImage(props.job.id, cur.id, reviewed, expect);
      props.live.put([saved], mark);
      if (disposed || shownId !== cur.id || !reviewed) return;
      if (next && router.route.name === 'label-job' && router.route.params.id === props.job.id && router.route.query.get('image') === cur.id) props.onnavigate(next);
      else if (!next) status.success('Reviewed', 'Every labeled image here is reviewed.');
    } catch (e) {
      if (!changedElsewhere(e)) { if (!disposed && shownId === cur.id) status.error(e, 'Could not mark the image'); } else if (!disposed) await reload(cur.id);
    } finally {
      reviewing.set(false);
    }
  }
  /** After a refused mark: the image's labels as the server has them now (the stream may tell later), then the ask. */
  async function reload(id: string) {
    const at = props.live.mark();
    const now = await hub.api.labelImages(props.job.id, { ids: [id] }).then(
      (p) => p.images.find((i) => i.id === id),
      () => undefined,
    );
    if (now) props.live.put([now], at);
    if (shownId !== id) return;
    const why = 'Its labels were changed from another tab or window before your mark got there, so nothing was marked.';
    if (now) status.show('info', 'Changed elsewhere', `${why} Check the labels shown now.`);
    else status.show('error', 'Changed elsewhere', `${why} The new labels could not be loaded, so the ones shown may be out of date.`);
  }

  /** Enter with unsaved changes: saves them (which marks the image reviewed) and opens the next image to review. */
  async function saveAndNext() {
    const cur = img.peek();
    if (!cur) return;
    const next = nextToReview(cur.id);
    if ((await save()) && !disposed && shownId === cur.id && next && router.route.name === 'label-job' && router.route.params.id === props.job.id && router.route.query.get('image') === cur.id) props.onnavigate(next);
  }

  // ---------- leaving with changes ----------
  /**
   * Where to go once unsaved changes are answered: another image, or away (null); or a navigation the router holds
   * (`proceed`: a link, Back, the palette, another `?image=`), which goes on after Save or Discard and is dropped on Cancel.
   */
  const leaving = signal<{ to: string | null; proceed?: () => void } | null>(null);
  const unsaved = () => !!img.peek() && dirty.peek() && !locked.peek();
  function go(to: string | null) {
    if (unsaved()) leaving.set({ to });
    else props.onnavigate(to);
  }
  async function answer(keep: 'save' | 'discard') {
    const ask = leaving.peek();
    if (!ask) return;
    if (keep === 'save' && (!(await save()) || leaving.peek() !== ask)) {
      if (leaving.peek() === ask) leaving.set(null);
      return;
    }
    leaving.set(null);
    draft.set(base.peek());
    if (ask.proceed) ask.proceed();
    else props.onnavigate(ask.to ?? null);
  }
  // Any other way off this image (the route changes, or its `?image=` does) asks first too.
  effect(() =>
    router.guard((to, proceed) => {
      if (!unsaved() || (to.path === router.route.path && to.query.get('image') === props.imageId)) return true;
      leaving.set({ to: null, proceed });
      return false;
    }),
  );
  function onbeforeunload(e: BeforeUnloadEvent) {
    if (unsaved()) e.preventDefault();
  }
  const step = (d: number) => {
    const next = props.images[idx.peek() + d];
    if (next) go(next.id);
  };

  const body = signal<HTMLElement>();
  function onkeydown(e: KeyboardEvent) {
    if (!open.peek() || leaving.peek() || removing.peek()) return;
    const dialog = body.peek()?.closest('dialog');
    // A button that turns disabled under focus (Save once saved, Discard, Ask, Next on the last image) drops focus to
    // the page's body, where no key reaches the dialog: with no other dialog open, the editor takes focus back.
    const dropped = !!dialog?.open && e.target === document.body && [...document.querySelectorAll('dialog[open]')].every((d) => d === dialog);
    // Ctrl+S (⌘S) saves from anywhere in the editor, a field included, and is never the browser's Save page.
    const inEditor = dropped || (!!dialog?.open && e.target instanceof Node && dialog.contains(e.target));
    if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 's' && inEditor && !e.defaultPrevented) {
      e.preventDefault();
      if (savable.peek() && !locked.peek()) void save();
      return;
    }
    // Undo / redo the boxes; in a field they are the field's own.
    const z = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && !e.altKey && (z === 'z' || (z === 'y' && !e.shiftKey)) && inEditor && !isTyping(e) && !e.defaultPrevented) {
      e.preventDefault();
      undo(z === 'y' || e.shiftKey);
      return;
    }
    if (dropped ? e.defaultPrevented || e.metaKey || e.ctrlKey : !dialogKey(e, dialog)) return;
    if (dropped) dialog!.focus();
    const k = e.key;
    // Esc drops the selection first (its default, the dialog's close, is kept for the next press).
    if (k === 'Escape') {
      if (selected.peek() === null) return;
      selected.set(null);
    } else if (k in ARROWS) {
      if (!arrow(e)) return;
    } else if (e.altKey) return;
    else if (k === 'j' || k === 'J') step(1);
    else if (k === 'k' || k === 'K') step(-1);
    else if (/^[1-9]$/.test(k)) setClass(Number(k) - 1);
    // Enter on a control is that control's; anywhere else in the editor it is "Looks right", or with changes, Save and
    // on to the next image to review.
    else if (k === 'Enter') {
      if (e.target instanceof Element && e.target.closest('button, a, select, input, textarea')) return;
      if (reviewable.peek()) void review(true);
      else if (dirty.peek() && !locked.peek()) void saveAndNext();
      else return;
    }
    else if (k === 'Delete' || k === 'Backspace') removeBox();
    else return;
    e.preventDefault();
  }
  onWindow('keydown', onkeydown);
  // A held arrow repeats as one step; a released one ends it, so separate presses are separate steps.
  onWindow('keyup', (e) => e.key in ARROWS && history.seal());
  onWindow('beforeunload', onbeforeunload);

  // ---------- asking the agent again ----------
  const note = signal('');
  const asking = signal(false);
  /**
   * The run "Ask the agent again" started (when it began, its `started`) and the image that asked, followed until it
   * ends while that image is shown; null: none. Another image opened forgets it, as it does the line.
   */
  const asked = signal<{ image: string; started: string } | null>(null);
  // Once that run ends, the status line says what came of it: the new boxes, the agent's error, or the stop.
  effect(() => {
    const a = asked.get();
    const run = props.job.run;
    const cur = img.get();
    if (a === null || !run || run.started !== a.started || run.outcome === null || cur?.id !== a.image) return;
    const out = askOutcome(run, cur);
    if (!out) return;
    asked.set(null);
    status.show(out.tone, out.title, out.detail);
  });
  const askWhy = computed(() =>
    props.job.status !== 'idle'
      ? 'The agent is busy with a run; ask once it is done'
      : !props.choice.agent
        ? 'No agent is installed on the server’s machine'
        : dirty.get()
          ? 'Save or discard your changes first'
          : note.get().length > NOTE_MAX
            ? 'Shorten the note first'
            : null,
  );
  async function ask() {
    const req = props.choice.request();
    const cur = img.peek();
    if (!cur || !req) return;
    asking.set(true);
    status.clear();
    try {
      const job = await props.live.act(() => hub.api.startLabeling(props.job.id, { ...req, image_ids: [cur.id], note: note.peek().trim() }));
      // The job's own update is applied (`live.act`) wherever the editor is; the line only on the image that asked.
      if (shownId !== cur.id || !open.peek()) return;
      status.info(`${props.choice.agent?.name ?? 'The agent'} is labeling this image again`, 'Its new boxes show here when it is done.');
      // After the line above: a run that has already ended (an agent that fails at once) replaces it.
      if (job.run) asked.set({ image: cur.id, started: job.run.started });
    } catch (e) {
      if (shownId === cur.id && open.peek()) status.error(e, 'Could not ask the agent');
    } finally {
      asking.set(false);
    }
  }

  // ---------- removing ----------
  const removing = signal<{ jobId: string; id: string; file: string } | null>(null);
  const removeBusy = signal(false);
  const removeStatus = new DialogStatus();
  function openRemove() {
    const cur = img.peek();
    if (!cur) return;
    removeStatus.clear();
    removing.set({ jobId: props.job.id, id: cur.id, file: cur.file });
  }
  async function remove() {
    const target = removing.peek();
    if (!target || removeBusy.peek()) return;
    removeBusy.set(true);
    const i = idx.peek();
    const next = props.images[i + 1] ?? props.images[i - 1];
    try {
      await hub.api.deleteLabelImage(target.jobId, target.id);
      if (disposed) return;
      if (props.job.id === target.jobId) props.live.drop(target.id);
      const r = router.route;
      if (shownId === target.id && props.job.id === target.jobId && open.peek() && r.name === 'label-job' && r.params.id === target.jobId && r.query.get('image') === target.id) {
        draft.set(base.peek());
        removing.set(null);
        props.onnavigate(next && next.id !== target.id ? next.id : null);
      }
    } catch (e) {
      if (!disposed && shownId === target.id && props.job.id === target.jobId && removing.peek()?.id === target.id) removeStatus.error(e, 'Could not remove the image');
    } finally {
      if (!disposed) removeBusy.set(false);
    }
  }

  const size = (b: Box) => `${Math.round(b.x2 - b.x1)}×${Math.round(b.y2 - b.y1)}`;
  const name = (c: number) => props.job.classes[c] ?? `class ${c + 1}`;

  return (
    <>
      <Dialog
        bind:open={open}
        title={img.get()?.file ?? 'Image'}
        description={img.get() ? imageState(img.get()!) : undefined}
        size="lg"
        class="label-editor"
        status={status}
        onbeforeclose={() => {
          if (!dirty.peek() || locked.peek()) return true;
          leaving.set({ to: null });
          return false;
        }}
        onclose={() => props.imageId !== null && !leaving.peek() && props.onnavigate(null)}
        footer={() => (
          <>
            {dirty.get() ? (
              <p class="state" data-gate-why>
                Unsaved changes
                {touch.current ? null : (
                  <>
                    . <Kbd keys="Enter" subtle /> saves them and opens the next image to review.
                  </>
                )}
              </p>
            ) : savable.get() && !locked.get() ? (
              <p class="state">Save keeps it as labeled by hand, {draft.get().length ? 'with these boxes' : 'with no boxes: a background image'}.</p>
            ) : reviewable.get() ? (
              <p class="state">
                Labels right? {touch.current ? 'Looks right' : <Kbd keys="Enter" subtle />} marks them reviewed and opens the next image to review.
              </p>
            ) : img.get()?.reviewed && !locked.get() ? (
              <p class="state">Reviewed. Edit and save to change the labels.</p>
            ) : !locked.get() ? (
              <p class="state" data-gate-why>
                No changes yet
              </p>
            ) : null}
            <Button variant="ghost" disabled={!dirty.get() || !!locked.get() || saving.get()} onclick={discard}>
              Discard
            </Button>
            {img.get()?.reviewed && !dirty.get() && !locked.get() ? (
              <Button variant="ghost" loading={reviewing.get()} onclick={() => review(false)}>
                Mark not reviewed
              </Button>
            ) : null}
            {reviewable.get() ? (
              <Button variant="primary" icon={Check} loading={reviewing.get()} onclick={() => review(true)}>
                Looks right
              </Button>
            ) : (
              <Gate reason={locked.get() ? 'The image is with the agent' : savable.get() ? null : 'No changes yet'}>
                {(why) => (
                  <Button variant="primary" loading={saving.get()} disabled={!!why()} aria-describedby={why()} onclick={save}>
                    Save
                  </Button>
                )}
              </Gate>
            )}
          </>
        )}
      >
        {img.get() ? (
          <div class="ed" ref={body}>
            <div class="stage">
              <div class="nav">
                <span use:tooltip={{ text: 'Previous', keys: 'K' }}>
                  <Button variant="ghost" size="sm" icon={ChevronLeft} aria-label="Previous image" disabled={idx.get() <= 0} onclick={() => step(-1)} />
                </span>
                <span class="pos num">
                  {idx.get() >= 0 ? `${num(idx.get() + 1)} of ${num(props.images.length)}` : 'Not in this filter'}
                </span>
                <span class="sr-only" aria-live="polite" data-nav-status>
                  {navStatus.get()}
                </span>
                <span use:tooltip={{ text: 'Next', keys: 'J' }}>
                  <Button
                    variant="ghost"
                    size="sm"
                    icon={ChevronRight}
                    aria-label="Next image"
                    disabled={idx.get() < 0 || idx.get() >= props.images.length - 1}
                    onclick={() => step(1)}
                  />
                </span>
                {locked.get() ? (
                  <p class="lock" role="status">
                    <span class="dot" aria-hidden="true"></span>
                    {locked.get()}
                  </p>
                ) : (
                  <p class="tip">Drag to draw, move or resize a box. Arrows move the selected box, Alt+arrows resize it.</p>
                )}
              </div>
              <div class="canvas-box">
                <EditCanvas
                  jobId={props.job.id}
                  img={img.get()!}
                  classes={props.job.classes}
                  boxes={draft.get()}
                  bind:selected={selected}
                  bind:scale={scale}
                  hovered={hovered.get()}
                  cls={drawClass.get()}
                  disabled={!!locked.get()}
                  onchange={edit}
                />
              </div>
              <EditorHints
                classes={props.job.classes}
                activeClass={activeClass.get()}
                hasSelection={selected.get() !== null}
                disabled={!!locked.get()}
                touch={touch.current}
                onselectclass={setClass}
              />
            </div>

            <aside class="side">
              <section>
                <div class="head">
                  <h3>Boxes <span class="meta num">{draft.get().length}</span></h3>
                  <span class="tools">
                    <span class:off={!canUndo.get() || !!locked.get()} use:tooltip={undoTip.get()}>
                      <Button variant="ghost" size="sm" icon={Undo} aria-label="Undo" disabled={!canUndo.get() || !!locked.get()} onclick={() => undo()} />
                    </span>
                    <span class:off={!canRedo.get() || !!locked.get()} use:tooltip={redoTip.get()}>
                      <Button variant="ghost" size="sm" icon={Redo} aria-label="Redo" disabled={!canRedo.get() || !!locked.get()} onclick={() => undo(true)} />
                    </span>
                    <Button variant="ghost" size="sm" icon={Plus} disabled={!!locked.get()} data-add-box onclick={addBox}>Add box</Button>
                  </span>
                </div>
                {undoCount.get() > 0 || redoCount.get() > 0 ? (
                  <p class="hist-depth num">History: {undoCount.get()} undo · {redoCount.get()} redo</p>
                ) : null}
                {draft.get().length ? (
                  <ul class="boxes scroll-shade" onpointerleave={() => hovered.set(null)}>
                    <Index each={draft.get()}>
                      {(b, i) => (
                        <li>
                          <button
                            type="button"
                            class:on={selected.get() === i}
                            aria-current={selected.get() === i ? 'true' : undefined}
                            aria-label={`${name(b().class)}, ${Math.round(b().x2 - b().x1)} by ${Math.round(b().y2 - b().y1)} pixels`}
                            onpointerenter={(e) => e.pointerType !== 'touch' && hovered.set(i)}
                            onclick={() => selected.set(selected.peek() === i ? null : i)}
                          >
                            <span class="sw" style:background={classColor(b().class)}></span>
                            <span class="cn truncate">{name(b().class)}</span>
                            <span class="sz num">{size(b())}</span>
                          </button>
                        </li>
                      )}
                    </Index>
                  </ul>
                ) : (
                  <p class="none">No boxes. Saved like this, the image is a background image: no objects.</p>
                )}
                {picked.get() ? (
                  <div class="sel">
                    <Field label="Class" for={`${uid}-class`} hint="Keys 1–9 set it too.">
                      <Select
                        id={`${uid}-class`}
                        value={String(picked.get()!.class)}
                        options={props.job.classes.map((c, i) => ({ value: String(i), label: `${i + 1} · ${c}` }))}
                        disabled={!!locked.get()}
                        onchange={(v) => setClass(Number(v))}
                      />
                    </Field>
                    <Button size="sm" icon={X} disabled={!!locked.get()} onclick={removeBox}>Delete box</Button>
                  </div>
                ) : null}
              </section>

              <section>
                <Field label="Ask the agent again" for={`${uid}-note`} hint="It labels this image alone; its new boxes replace these.">
                  <Input id={`${uid}-note`} bind:value={note} placeholder="What should it do differently?" autocomplete="off" />
                </Field>
                <Gate reason={askWhy.get()}>
                  {(why) => (
                    <Button icon={Sparkles} loading={asking.get()} disabled={!!why()} aria-describedby={why()} onclick={ask}>
                      Ask {props.choice.agent?.name ?? 'the agent'}
                    </Button>
                  )}
                </Gate>
                {askWhy.get() ? <p class="why" data-gate-why>{askWhy.get()}</p> : null}
              </section>

              <section class="foot">
                <Button variant="danger" icon={Trash} disabled={!!locked.get()} onclick={openRemove}>Remove image</Button>
              </section>
            </aside>
          </div>
        ) : null}
      </Dialog>

      <Dialog
        open={!!leaving.get()}
        title="Save your changes?"
        description={`${img.get()?.file ?? 'This image'} has boxes you have not saved.${
          elsewhere.get() ? ' Its labels were also saved from another tab or window while you edited: Save replaces them with yours; Discard keeps those.' : ''
        }`}
        size="sm"
        onclose={() => leaving.set(null)}
        footer={() => (
          <>
            <Button onclick={() => answer('discard')}>Discard changes</Button>
            <Button variant="primary" loading={saving.get()} onclick={() => answer('save')}>Save</Button>
          </>
        )}
      />

      <Dialog
        open={!!removing.get()}
        title={`Remove ${removing.get()?.file ?? 'this image'}?`}
        description="Its labels go with it. An uploaded file is deleted; an image added from a folder stays on disk."
        size="sm"
        status={removeStatus}
        onbeforeclose={() => !removeBusy.peek()}
        onclose={() => removing.set(null)}
        footer={() => (
          <>
            <Button disabled={removeBusy.get()} onclick={() => removing.set(null)}>Keep it</Button>
            <Button variant="destructive" icon={Trash} loading={removeBusy.get()} onclick={remove}>Remove image</Button>
          </>
        )}
      />
    </>
  );
}

css`
  /* From 768 px: as wide and tall as the window allows, the image beside its list. */
  @media (min-width: 768px) {
    :global(dialog.lg.label-editor) {
      width: min(1160px, calc(100vw - 48px));
      max-height: calc(100dvh - 48px);
    }
  }
  .ed {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 280px;
    gap: var(--space-4);
  }
  .stage {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    align-self: start;
    align-content: start;
    gap: var(--space-2);
    min-width: 0;
  }
  .canvas-box {
    height: min(576px, calc(100dvh - 324px));
    min-height: 240px;
  }
  .nav {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    min-height: var(--control-sm);
  }
  .pos {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
    min-width: 72px;
    text-align: center;
  }
  .tip,
  .lock {
    margin-left: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .lock {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--text-secondary);
  }
  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-solid);
    flex-shrink: 0;
  }
  .side {
    display: grid;
    align-content: start;
    gap: var(--space-4);
    min-width: 0;
  }
  .side section {
    display: grid;
    gap: var(--space-2);
  }
  .side section + section {
    padding-top: var(--space-4);
    border-top: 1px solid var(--line-subtle);
  }
  h3 {
    font-size: var(--type-base-size);
    font-weight: var(--weight-medium);
  }
  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
  }
  .tools {
    display: flex;
    align-items: center;
    gap: var(--space-1);
  }
  .tools > span {
    display: inline-flex;
  }
  /* A disabled button takes no pointer events in some browsers: the wrapper takes the hover so contextual undo/redo tips show. */
  .tools > span.off {
    cursor: not-allowed;
  }
  .tools > span.off > :global(:disabled) {
    pointer-events: none;
  }
  .hist-depth {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .meta {
    margin-left: var(--space-1);
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
  }
  .boxes {
    max-height: 220px;
    overflow-y: auto;
  }
  .boxes button {
    display: grid;
    grid-template-columns: 8px minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2);
    width: 100%;
    height: 28px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    text-align: left;
  }
  @media (hover: hover) {
    .boxes button:hover {
      background: var(--bg-hover);
    }
  }
  .boxes button.on {
    background: var(--bg-active);
    box-shadow: var(--select-bar);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  }
  @media (pointer: coarse) {
    .boxes button {
      height: 44px;
    }
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
  }
  .sz {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .none,
  .why,
  .state {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .sel {
    display: grid;
    gap: var(--space-2);
    justify-items: start;
  }
  .sel :global(.field) {
    width: 100%;
  }
  .foot {
    justify-items: start;
  }
  .state {
    margin-right: auto;
    align-self: center;
  }
  @media (max-width: 767px) {
    .ed {
      grid-template-columns: minmax(0, 1fr);
    }
    .canvas-box {
      height: min(56dvh, 420px);
      min-height: 220px;
    }
    .tip {
      display: none;
    }
    .state {
      flex-basis: 100%;
    }
  }
`;
