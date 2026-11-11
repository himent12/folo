/**
 * One image of a labeling job, to review and correct: its boxes in full size and as a list; select a box (on the
 * image, or in the list), change its class (the list's select, or 1–9), delete it (Delete, Backspace or its button),
 * move or resize it by dragging or with the arrow keys (Alt: resize; Shift: 10 px), draw a new one on empty image or
 * add one with Add box. Save (the hand-drawn labels replace the image's; none is a background image; an image not
 * labeled yet can be saved as it is) or Discard; leaving unsaved changes asks first, whichever way (J / K, Close, a
 * link, Back), and closing the tab warns. J / K step through the images in the current filter. "Looks right" (Enter)
 * marks labels nobody changed as reviewed and opens the next image to review; saved boxes are reviewed too. "Ask the agent again"
 * runs the agent on this image alone with a note; "Remove image" takes it out of the job. An image the agent has right
 * now cannot be edited, and says so; one that waits for its turn in a run can, and then leaves the run. Its keys work
 * while it is the topmost dialog (shell/pageCommands `dialogKey`), also after a button that had focus turned disabled.
 */
import ChevronLeft from '$icons/chevron-left';
import ChevronRight from '$icons/chevron-right';
import Check from '$icons/check';
import Plus from '$icons/plus';
import Sparkles from '$icons/sparkles';
import Trash from '$icons/trash-2';
import X from '$icons/x';
import { computed, createUniqueId, css, effect, Index, onWindow, signal } from 'fw';
import { hub } from '$lib/api/index';
import type { LiveLabelJob } from '$lib/api/liveLabel';
import type { Box, LabelImage, LabelJob } from '$lib/api/types';
import { classColor } from '$lib/design/tokens';
import { num } from '$lib/format';
import { NOTE_MAX } from '$lib/label';
import { router } from '$lib/router';
import { dialogKey } from '$lib/shell/pageCommands';
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
import { newBox, nudge, sameBoxes, stretch, tidy } from './boxEdit';
import EditCanvas from './EditCanvas';
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

  // The draft: the boxes as edited here, over the image's own (`base`). An image that changes on the server while it
  // is not being edited (the agent relabeled it) shows its new boxes.
  const base = signal<Box[]>([]);
  const draft = signal<Box[]>([]);
  const selected = signal<number | null>(null);
  const hovered = signal<number | null>(null);
  const lastClass = signal(0);
  let shownId: string | null = null;
  const dirty = computed(() => !sameBoxes(draft.get(), base.get()));
  const picked = computed(() => {
    const s = selected.get();
    return s === null ? undefined : draft.get()[s];
  });
  effect(() => {
    const cur = img.get();
    if (!cur) return;
    if (cur.id !== shownId) {
      shownId = cur.id;
      base.set(cur.boxes);
      draft.set(cur.boxes);
      selected.set(null);
      note.set('');
      asked.set(null);
      status.clear();
    } else if (!sameBoxes(cur.boxes, base.get()) && sameBoxes(draft.get(), base.get())) {
      base.set(cur.boxes);
      draft.set(cur.boxes);
      selected.set(null);
    }
  });

  const locked = computed(() => (img.get()?.status === 'labeling' ? 'The agent is labeling this image. You can edit it once it is done.' : null));

  // ---------- editing ----------
  function edit(next: Box[]) {
    draft.set(next);
  }
  function setClass(c: number) {
    const s = selected.peek();
    if (s === null || locked.peek() || c >= props.job.classes.length) return;
    lastClass.set(c);
    draft.set(draft.peek().map((b, i) => (i === s ? { ...b, class: c } : b)));
  }
  function removeBox() {
    const s = selected.peek();
    if (s === null || locked.peek()) return;
    draft.set(draft.peek().filter((_, i) => i !== s));
    selected.set(null);
  }
  /** Add box: a box of the class used last in the middle of the image, selected (the keyboard's way to draw one). */
  function addBox() {
    const cur = img.peek();
    if (!cur || locked.peek()) return;
    draft.set([...draft.peek(), newBox(cur.width, cur.height, Math.min(lastClass.peek(), props.job.classes.length - 1))]);
    selected.set(draft.peek().length - 1);
  }
  /** Arrow keys on the selected box: move it, or with Alt resize it from its bottom-right corner; Shift: 10 px. */
  function arrow(e: KeyboardEvent): boolean {
    const d = ARROWS[e.key];
    const b = picked.peek();
    const cur = img.peek();
    if (!d || !b || !cur || locked.peek()) return false;
    const by = e.shiftKey ? 10 : 1;
    const next = e.altKey ? stretch(b, d[0] * by, d[1] * by, cur.width, cur.height) : nudge(b, d[0] * by, d[1] * by, cur.width, cur.height);
    const s = selected.peek();
    draft.set(draft.peek().map((x, i) => (i === s ? next : x)));
    return true;
  }

  const saving = signal(false);
  // An image not labeled yet (waiting, or failed) can be saved as it is: kept as labeled by hand, its boxes or none.
  const savable = computed(() => dirty.get() || (!!img.get() && img.get()!.status !== 'labeled'));
  async function save(): Promise<boolean> {
    const cur = img.peek();
    if (!cur || saving.peek()) return false;
    // The answer is this image's, sent with these boxes: by the time it comes another image may be open, or the
    // boxes edited further, and neither is the answer's to change.
    const id = cur.id;
    const sent = draft.peek();
    const mark = props.live.mark();
    saving.set(true);
    status.clear();
    try {
      const saved = await hub.api.updateLabelImage(props.job.id, id, sent.map(tidy));
      props.live.put([saved], mark);
      if (shownId === id) {
        base.set(saved.boxes);
        if (draft.peek() === sent) draft.set(saved.boxes);
        status.success('Saved', imageState(saved));
      }
      return true;
    } catch (e) {
      status.error(e, 'Could not save the boxes');
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
    const mark = props.live.mark();
    reviewing.set(true);
    status.clear();
    try {
      const saved = await hub.api.reviewLabelImage(props.job.id, cur.id, reviewed);
      props.live.put([saved], mark);
      if (shownId !== cur.id || !reviewed) return;
      if (next) props.onnavigate(next);
      else status.success('Reviewed', 'Every labeled image here is reviewed.');
    } catch (e) {
      status.error(e, 'Could not mark the image');
    } finally {
      reviewing.set(false);
    }
  }

  // ---------- leaving with changes ----------
  /**
   * Where to go once unsaved changes are answered: another image, or away (null); or a navigation the router holds
   * (`proceed`: a link, Back, the palette, another `?image=`), which goes on after Save or Discard and is dropped on Cancel.
   */
  const leaving = signal<{ to: string | null; proceed?: () => void } | null>(null);
  const unsaved = () => props.imageId !== null && dirty.peek() && !locked.peek();
  function go(to: string | null) {
    if (unsaved()) leaving.set({ to });
    else props.onnavigate(to);
  }
  async function answer(keep: 'save' | 'discard') {
    const ask = leaving.peek();
    if (keep === 'save' && !(await save())) {
      leaving.set(null);
      return;
    }
    leaving.set(null);
    draft.set(base.peek());
    if (ask?.proceed) ask.proceed();
    else props.onnavigate(ask?.to ?? null);
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
    if (dropped ? e.defaultPrevented || e.metaKey || e.ctrlKey : !dialogKey(e, dialog)) return;
    if (dropped) dialog!.focus();
    const k = e.key;
    if (k in ARROWS) {
      if (!arrow(e)) return;
    } else if (e.altKey) return;
    else if (k === 'j' || k === 'J') step(1);
    else if (k === 'k' || k === 'K') step(-1);
    else if (/^[1-9]$/.test(k)) setClass(Number(k) - 1);
    // Enter on a control is that control's; anywhere else in the editor it is "Looks right".
    else if (k === 'Enter') {
      if (!reviewable.peek() || (e.target instanceof Element && e.target.closest('button, a, select, input, textarea'))) return;
      void review(true);
    }
    else if (k === 'Delete' || k === 'Backspace') removeBox();
    else return;
    e.preventDefault();
  }
  onWindow('keydown', onkeydown);
  onWindow('beforeunload', onbeforeunload);

  // ---------- asking the agent again ----------
  const note = signal('');
  const asking = signal(false);
  /** When the run "Ask the agent again" started began (its `started`), followed until it ends; null: none. */
  const asked = signal<string | null>(null);
  // Once that run ends, the status line says what came of it: the new boxes, the agent's error, or the stop.
  effect(() => {
    const started = asked.get();
    const run = props.job.run;
    if (started === null || !run || run.started !== started || run.outcome === null) return;
    const out = askOutcome(run, img.get());
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
      status.info(`${props.choice.agent?.name ?? 'The agent'} is labeling this image again`, 'Its new boxes show here when it is done.');
      // After the line above: a run that has already ended (an agent that fails at once) replaces it.
      if (job.run) asked.set(job.run.started);
    } catch (e) {
      status.error(e, 'Could not ask the agent');
    } finally {
      asking.set(false);
    }
  }

  // ---------- removing ----------
  const removing = signal(false);
  const removeBusy = signal(false);
  const removeStatus = new DialogStatus();
  async function remove() {
    const cur = img.peek();
    if (!cur) return;
    removeBusy.set(true);
    const i = idx.peek();
    const next = props.images[i + 1] ?? props.images[i - 1];
    try {
      await hub.api.deleteLabelImage(props.job.id, cur.id);
      props.live.drop(cur.id);
      removing.set(false);
      props.onnavigate(next && next.id !== cur.id ? next.id : null);
    } catch (e) {
      removeStatus.error(e, 'Could not remove the image');
    } finally {
      removeBusy.set(false);
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
              </p>
            ) : savable.get() && !locked.get() ? (
              <p class="state">Save keeps it as labeled by hand, {draft.get().length ? 'with these boxes' : 'with no boxes: a background image'}.</p>
            ) : reviewable.get() ? (
              <p class="state">
                Labels right? <Kbd keys="Enter" subtle /> marks them reviewed and opens the next image to review.
              </p>
            ) : img.get()?.reviewed && !locked.get() ? (
              <p class="state">Reviewed. Edit and save to change the labels.</p>
            ) : !locked.get() ? (
              <p class="state" data-gate-why>
                No changes yet
              </p>
            ) : null}
            <Button variant="ghost" disabled={!dirty.get()} onclick={() => (draft.set(base.peek()), selected.set(null))}>
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
                <span class="pos num">{idx.get() >= 0 ? `${num(idx.get() + 1)} of ${num(props.images.length)}` : 'Not in this filter'}</span>
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
                  hovered={hovered.get()}
                  cls={lastClass.get()}
                  disabled={!!locked.get()}
                  onchange={edit}
                />
              </div>
            </div>

            <aside class="side">
              <section>
                <div class="head">
                  <h3>
                    Boxes <span class="meta num">{draft.get().length}</span>
                  </h3>
                  <Button variant="ghost" size="sm" icon={Plus} disabled={!!locked.get()} onclick={addBox}>
                    Add box
                  </Button>
                </div>
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
                    <Button size="sm" icon={X} disabled={!!locked.get()} onclick={removeBox}>
                      Delete box
                    </Button>
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
                {askWhy.get() ? (
                  <p class="why" data-gate-why>
                    {askWhy.get()}
                  </p>
                ) : null}
              </section>

              <section class="foot">
                <Button variant="danger" icon={Trash} disabled={!!locked.get()} onclick={() => removing.set(true)}>
                  Remove image
                </Button>
              </section>
            </aside>
          </div>
        ) : null}
      </Dialog>

      <Dialog
        open={!!leaving.get()}
        title="Save your changes?"
        description={`${img.get()?.file ?? 'This image'} has boxes you have not saved.`}
        size="sm"
        onclose={() => leaving.set(null)}
        footer={() => (
          <>
            <Button onclick={() => answer('discard')}>Discard changes</Button>
            <Button variant="primary" loading={saving.get()} onclick={() => answer('save')}>
              Save
            </Button>
          </>
        )}
      />

      <Dialog
        bind:open={removing}
        title={`Remove ${img.get()?.file ?? 'this image'}?`}
        description="Its labels go with it. An uploaded file is deleted; an image added from a folder stays on disk."
        size="sm"
        status={removeStatus}
        footer={() => (
          <>
            <Button onclick={() => removing.set(false)}>Keep it</Button>
            <Button variant="destructive" icon={Trash} loading={removeBusy.get()} onclick={remove}>
              Remove image
            </Button>
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
    grid-template-rows: auto minmax(0, 1fr);
    gap: var(--space-2);
    min-width: 0;
  }
  .canvas-box {
    height: min(640px, calc(100dvh - 260px));
    min-height: 280px;
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
