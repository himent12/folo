<script lang="ts">
  /**
   * One image of a labeling job, to review and correct: its boxes in full size and as a list; select a box (on the
   * image, or in the list), change its class (the list's select, or 1–9), delete it (Delete, Backspace or its button),
   * move or resize it by dragging, draw a new one on empty image. Save (the hand-drawn labels replace the image's; none
   * is a background image) or Discard; stepping away from unsaved changes asks first. J / K step through the images in
   * the current filter. "Ask the agent again" runs the agent on this image alone with a note; "Remove image" takes it
   * out of the job. An image the agent has right now cannot be edited, and says so; one that waits for its turn in a run
   * can, and then leaves the run.
   */
  import ChevronLeft from '@lucide/svelte/icons/chevron-left';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import Sparkles from '@lucide/svelte/icons/sparkles';
  import Trash from '@lucide/svelte/icons/trash-2';
  import X from '@lucide/svelte/icons/x';
  import { hub } from '$lib/api/index.svelte';
  import type { LiveLabelJob } from '$lib/api/liveLabel.svelte';
  import type { Box, LabelImage, LabelJob } from '$lib/api/types';
  import { classColor } from '$lib/design/tokens';
  import { num } from '$lib/format';
  import { isTyping } from '$lib/keys';
  import { NOTE_MAX } from '$lib/label';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import { DialogStatus } from '$lib/ui/feedback.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Select from '$lib/ui/Select.svelte';
  import { tooltip } from '$lib/ui/tooltip';
  import type { AgentChoice } from './agentChoice.svelte';
  import { sameBoxes, tidy } from './boxEdit';
  import EditCanvas from './EditCanvas.svelte';
  import { imageState } from './words';

  let {
    live,
    job,
    images,
    imageId,
    choice,
    onnavigate,
  }: {
    live: LiveLabelJob;
    job: LabelJob;
    /** The images in the current filter, for J / K. */
    images: readonly LabelImage[];
    imageId: string | null;
    choice: AgentChoice;
    /** Opens another image, or closes the editor (null). */
    onnavigate: (id: string | null) => void;
  } = $props();

  const uid = $props.id();
  const status = new DialogStatus();
  let open = $state(false);
  $effect(() => {
    open = imageId !== null;
  });

  const img = $derived(imageId === null ? undefined : live.images.find((i) => i.id === imageId));
  const idx = $derived(img ? images.findIndex((i) => i.id === img.id) : -1);

  // The draft: the boxes as edited here, over the image's own (`base`). An image that changes on the server while it
  // is not being edited (the agent relabeled it) shows its new boxes.
  let base = $state.raw<Box[]>([]);
  let draft = $state.raw<Box[]>([]);
  let selected = $state<number | null>(null);
  let hovered = $state<number | null>(null);
  let lastClass = $state(0);
  let shownId: string | null = null;
  const dirty = $derived(!sameBoxes(draft, base));
  const picked = $derived(selected === null ? undefined : draft[selected]);
  $effect(() => {
    const cur = img;
    if (!cur) return;
    if (cur.id !== shownId) {
      shownId = cur.id;
      base = cur.boxes;
      draft = cur.boxes;
      selected = null;
      note = '';
      status.clear();
    } else if (!sameBoxes(cur.boxes, base) && sameBoxes(draft, base)) {
      base = cur.boxes;
      draft = cur.boxes;
      selected = null;
    }
  });

  const locked = $derived(img?.status === 'labeling' ? 'The agent is labeling this image. You can edit it once it is done.' : null);

  // ---------- editing ----------
  function edit(next: Box[]) {
    draft = next;
  }
  function setClass(c: number) {
    if (selected === null || locked || c >= job.classes.length) return;
    lastClass = c;
    draft = draft.map((b, i) => (i === selected ? { ...b, class: c } : b));
  }
  function removeBox() {
    if (selected === null || locked) return;
    draft = draft.filter((_, i) => i !== selected);
    selected = null;
  }

  let saving = $state(false);
  async function save(): Promise<boolean> {
    if (!img || saving) return false;
    saving = true;
    status.clear();
    try {
      const saved = await hub.api.updateLabelImage(job.id, img.id, draft.map(tidy));
      live.put([saved], live.mark());
      base = saved.boxes;
      draft = saved.boxes;
      status.success('Saved', imageState(saved));
      return true;
    } catch (e) {
      status.error(e, 'Could not save the boxes');
      return false;
    } finally {
      saving = false;
    }
  }

  // ---------- leaving with changes ----------
  /** Where to go once unsaved changes are answered: another image, or away (null). */
  let leaving = $state<{ to: string | null } | null>(null);
  function go(to: string | null) {
    if (dirty && !locked) leaving = { to };
    else onnavigate(to);
  }
  async function answer(keep: 'save' | 'discard') {
    const to = leaving?.to ?? null;
    if (keep === 'save' && !(await save())) {
      leaving = null;
      return;
    }
    leaving = null;
    draft = base;
    onnavigate(to);
  }
  const step = (d: number) => {
    const next = images[idx + d];
    if (next) go(next.id);
  };

  function onkeydown(e: KeyboardEvent) {
    if (!open || leaving || removing || e.metaKey || e.ctrlKey || e.altKey || isTyping(e)) return;
    const k = e.key;
    if (k === 'j' || k === 'J') step(1);
    else if (k === 'k' || k === 'K') step(-1);
    else if (/^[1-9]$/.test(k)) setClass(Number(k) - 1);
    else if (k === 'Delete' || k === 'Backspace') removeBox();
    else return;
    e.preventDefault();
  }

  // ---------- asking the agent again ----------
  let note = $state('');
  let asking = $state(false);
  const askWhy = $derived(
    job.status !== 'idle'
      ? 'The agent is busy with a run; ask once it is done'
      : !choice.agent
        ? 'No agent is installed on the server’s machine'
        : dirty
          ? 'Save or discard your changes first'
          : note.length > NOTE_MAX
            ? 'Shorten the note first'
            : null,
  );
  async function ask() {
    const req = choice.request();
    if (!img || !req) return;
    asking = true;
    status.clear();
    try {
      await live.act(() => hub.api.startLabeling(job.id, { ...req, image_ids: [img.id], note: note.trim() }));
      status.info(`${choice.agent?.name ?? 'The agent'} is labeling this image again`, 'Its new boxes show here when it is done.');
    } catch (e) {
      status.error(e, 'Could not ask the agent');
    } finally {
      asking = false;
    }
  }

  // ---------- removing ----------
  let removing = $state(false);
  let removeBusy = $state(false);
  const removeStatus = new DialogStatus();
  async function remove() {
    if (!img) return;
    removeBusy = true;
    const next = images[idx + 1] ?? images[idx - 1];
    try {
      await hub.api.deleteLabelImage(job.id, img.id);
      live.drop(img.id);
      removing = false;
      onnavigate(next && next.id !== img.id ? next.id : null);
    } catch (e) {
      removeStatus.error(e, 'Could not remove the image');
    } finally {
      removeBusy = false;
    }
  }

  const size = (b: Box) => `${Math.round(b.x2 - b.x1)}×${Math.round(b.y2 - b.y1)}`;
  const name = (c: number) => job.classes[c] ?? `class ${c + 1}`;
</script>

<svelte:window {onkeydown} />

<Dialog
  bind:open
  title={img?.file ?? 'Image'}
  description={img ? imageState(img) : undefined}
  size="lg"
  class="label-editor"
  {status}
  onbeforeclose={() => {
    if (!dirty || locked) return true;
    leaving = { to: null };
    return false;
  }}
  onclose={() => imageId !== null && !leaving && onnavigate(null)}
>
  {#if img}
    <div class="ed">
      <div class="stage">
        <div class="nav">
          <span use:tooltip={{ text: 'Previous', keys: 'K' }}><Button variant="ghost" size="sm" icon={ChevronLeft} aria-label="Previous image" disabled={idx <= 0} onclick={() => step(-1)} /></span>
          <span class="pos num">{#if idx >= 0}{num(idx + 1)} of {num(images.length)}{:else}Not in this filter{/if}</span>
          <span use:tooltip={{ text: 'Next', keys: 'J' }}><Button variant="ghost" size="sm" icon={ChevronRight} aria-label="Next image" disabled={idx < 0 || idx >= images.length - 1} onclick={() => step(1)} /></span>
          {#if locked}<p class="lock" role="status"><span class="dot" aria-hidden="true"></span>{locked}</p>{:else}<p class="tip">Drag on the image to draw a box; drag a box to move it, its handles to resize it.</p>{/if}
        </div>
        <div class="canvas-box">
          <EditCanvas jobId={job.id} {img} classes={job.classes} boxes={draft} bind:selected {hovered} cls={lastClass} disabled={!!locked} onchange={edit} />
        </div>
      </div>

      <aside class="side">
        <section>
          <h3>Boxes <span class="meta num">{draft.length}</span></h3>
          {#if draft.length}
            <ul class="boxes scroll-shade" onpointerleave={() => (hovered = null)}>
              {#each draft as b, i (i)}
                <li>
                  <button
                    type="button"
                    class:on={selected === i}
                    aria-current={selected === i ? 'true' : undefined}
                    aria-label="{name(b.class)}, {Math.round(b.x2 - b.x1)} by {Math.round(b.y2 - b.y1)} pixels"
                    onpointerenter={(e) => e.pointerType !== 'touch' && (hovered = i)}
                    onclick={() => (selected = selected === i ? null : i)}
                  >
                    <span class="sw" style:background={classColor(b.class)}></span>
                    <span class="cn truncate">{name(b.class)}</span>
                    <span class="sz num">{size(b)}</span>
                  </button>
                </li>
              {/each}
            </ul>
          {:else}
            <p class="none">No boxes. Saved like this, the image is a background image: no objects.</p>
          {/if}
          {#if picked}
            <div class="sel">
              <Field label="Class" for="{uid}-class" hint="Keys 1–9 set it too.">
                <Select id="{uid}-class" value={String(picked.class)} options={job.classes.map((c, i) => ({ value: String(i), label: `${i + 1} · ${c}` }))} disabled={!!locked} onchange={(v) => setClass(Number(v))} />
              </Field>
              <Button size="sm" icon={X} disabled={!!locked} onclick={removeBox}>Delete box</Button>
            </div>
          {/if}
        </section>

        <section>
          <Field label="Ask the agent again" for="{uid}-note" hint="It labels this image alone; its new boxes replace these.">
            <Input id="{uid}-note" bind:value={note} placeholder="What should it do differently?" autocomplete="off" />
          </Field>
          <Gate reason={askWhy}>
            {#snippet children(why)}
              <Button icon={Sparkles} loading={asking} disabled={!!why} aria-describedby={why} onclick={ask}>Ask {choice.agent?.name ?? 'the agent'}</Button>
            {/snippet}
          </Gate>
          {#if askWhy}<p class="why" data-gate-why>{askWhy}</p>{/if}
        </section>

        <section class="foot">
          <Button variant="danger" icon={Trash} disabled={!!locked} onclick={() => (removing = true)}>Remove image</Button>
        </section>
      </aside>
    </div>
  {/if}
  {#snippet footer()}
    {#if dirty}<p class="state" data-gate-why>Unsaved changes</p>{:else if !locked}<p class="state" data-gate-why>No changes yet</p>{/if}
    <Button variant="ghost" disabled={!dirty} onclick={() => ((draft = base), (selected = null))}>Discard</Button>
    <Gate reason={locked ? 'The image is with the agent' : dirty ? null : 'No changes yet'}>
      {#snippet children(why)}
        <Button variant="primary" loading={saving} disabled={!!why} aria-describedby={why} onclick={save}>Save</Button>
      {/snippet}
    </Gate>
  {/snippet}
</Dialog>

<Dialog open={!!leaving} title="Save your changes?" description="{img?.file ?? 'This image'} has boxes you have not saved." size="sm" onclose={() => (leaving = null)}>
  {#snippet footer()}
    <Button onclick={() => answer('discard')}>Discard changes</Button>
    <Button variant="primary" loading={saving} onclick={() => answer('save')}>Save</Button>
  {/snippet}
</Dialog>

<Dialog
  bind:open={removing}
  title="Remove {img?.file ?? 'this image'}?"
  description="Its labels go with it. An uploaded file is deleted; an image added from a folder stays on disk."
  size="sm"
  status={removeStatus}
>
  {#snippet footer()}
    <Button onclick={() => (removing = false)}>Keep it</Button>
    <Button variant="destructive" icon={Trash} loading={removeBusy} onclick={remove}>Remove image</Button>
  {/snippet}
</Dialog>

<style>
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
</style>
