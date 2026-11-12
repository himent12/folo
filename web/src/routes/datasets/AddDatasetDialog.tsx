/** "Add dataset": register a YOLO/COCO folder on the server's disk, or generate a synthetic shapes set. */
import FolderOpen from '$icons/folder-open';
import Sparkles from '$icons/sparkles';
import Tags from '$icons/tags';
import { bindable, computed, createUniqueId, css, effect, signal } from 'fw';
import { hub } from '$lib/api/index';
import type { CreateDataset } from '$lib/api/types';
import { count, duration } from '$lib/format';
import { href, router } from '$lib/router';
import Button from '$lib/ui/Button';
import ClassChips, { addClasses } from '$lib/ui/ClassChips';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import Field from '$lib/ui/Field.tsx';
import Gate from '$lib/ui/Gate';
import Input from '$lib/ui/Input';
import Segmented from '$lib/ui/Segmented';
import Tabs from '$lib/ui/Tabs';
import { toast } from '$lib/ui/toast';
import { noTraining } from '../dataset/health';
import { datasetsChanged } from './changes';
import { isUnlabeled } from './problem';
import { cleanPath, isFullPath, NAME_RE, nameFromPath, suggestName, uniqueName } from './names';

type Mode = 'folder' | 'synthetic';

/** Most classes of a generated set: each class is one of the generator's 8 shapes (docs/API.md). */
const MAX_SHAPES = 8;

export interface AddDatasetDialogProps {
  open?: boolean;
  mode?: Mode;
  /** The datasets already listed, to tell a new registration from an existing one and to keep names apart (fetched when
   *  absent). */
  known?: readonly { id: string; name: string }[];
}

const nameError = (v: string) => (!v.trim() ? 'Required.' : NAME_RE.test(v.trim()) ? null : 'Letters, digits, ".", "_" and "-" only (max 64).');
/** A name in error offers its nearest valid form as one click ("Try my-dataset"). */
const nameFix = (v: string, set: (n: string) => void) => {
  const n = suggestName(v);
  return n && n !== v.trim() ? { label: `Try ${n}`, apply: () => set(n) } : null;
};
/** A typed name another dataset has: allowed (the server keeps both), but the two would read the same everywhere. */
const takenHint = (n: string) => `A dataset is already called ${n}. Another name tells them apart.`;
/** The sample set's default name, numbered when taken ("shapes-synth-2"). */
const SYNTH_NAME = 'shapes-synth';
const FIX: Record<string, string> = { name: 'Fix the name first', path: 'Fix the folder path first', images: 'Fix the image count first', classes: 'Add at least one class', seed: 'Fix the seed first' };

export default function AddDatasetDialog(props: AddDatasetDialogProps) {
  const open = bindable(props, 'open', false);
  const mode = bindable(props, 'mode', 'folder' as Mode);

  const uid = createUniqueId();
  /** The request in flight, if any: it outlives the dialog (closing does not stop the server), so it is not reset. */
  const work = signal<{ what: string; since: number } | null>(null);
  const busy = computed(() => !!work.get());
  /** Seconds since the request started, ticking while it runs (a long generation says it is still going). */
  const elapsed = signal(0);
  effect(() => {
    const w = work.get();
    if (!w) return;
    elapsed.set(0);
    const t = setInterval(() => elapsed.set((Date.now() - w.since) / 1000), 1000);
    return () => clearInterval(t);
  });
  const error = signal<unknown>(undefined);
  const touched = signal(false);

  // Register a folder
  const name = signal('');
  const path = signal('');
  const format = signal<CreateDataset['format']>('auto');

  // Synthetic
  const sName = signal('');
  const sImages = signal<number | null>(1000);
  const classes = signal<string[]>(['circle', 'square', 'triangle']);
  const classDraft = signal('');
  const imgsz = signal<'320' | '480' | '640' | '1024'>('640');
  const seed = signal<number | null>(0);

  /** The path as it will be sent: a pasted "Copy as path" loses its quotes. */
  const cleaned = computed(() => cleanPath(path.get()));
  /** The datasets there are, when the page did not give them: for names, and to tell a registered folder. */
  const fetched = signal<readonly { id: string; name: string }[] | null>(null);
  effect(() => {
    if (open.get() && !props.known) void hub.api.datasets().then((l) => fetched.set(l), () => {});
  });
  /** Names in use (lower case): a default never repeats one, and a typed one that does is pointed out. */
  const taken = computed(() => new Set((props.known ?? fetched.get() ?? []).map((d) => d.name.toLowerCase())));
  const isTaken = (n: string) => !!n && taken.get().has(n.toLowerCase());
  /** The folder's name, tidied and numbered when taken, as the default name. */
  const derivedName = computed(() => uniqueName(nameFromPath(cleaned.get()), taken.get()));
  const folderName = computed(() => name.get().trim() || derivedName.get());
  const synthName = computed(() => sName.get().trim() || uniqueName(SYNTH_NAME, taken.get()));

  const errs = computed<Partial<Record<string, string | null>>>(() => {
    if (mode.get() === 'folder') {
      const n = folderName.get();
      return {
        name: n ? nameError(n) : null,
        path: !cleaned.get() ? 'Required: an absolute folder path on the server.' : isFullPath(cleaned.get()) ? null : 'Use the full path, such as D:/data/my-dataset.',
      };
    }
    const images = sImages.get();
    const s = seed.get();
    return {
      name: nameError(synthName.get()),
      images: images === null || !Number.isFinite(images) || images < 10 || images > 20000 ? 'Between 10 and 20,000.' : null,
      classes: classes.get().length === 0 ? 'Add at least one class.' : null,
      seed: s === null || !Number.isInteger(s) || s < 0 ? 'A non-negative integer.' : null,
    };
  });
  const valid = computed(() => Object.values(errs.get()).every((e) => !e));
  /** Nothing to submit yet: the action is disabled and says what it needs (as New run's Start does). Mistakes in what
   *  was typed are still checked on submit, next to their fields. */
  const missing = computed(() =>
    mode.get() === 'folder'
      ? cleaned.get()
        ? null
        : 'Enter a folder path'
      : classes.get().length || classDraft.get().trim()
        ? null
        : 'Add at least one class',
  );
  const show = (e: string | null | undefined) => (touched.get() ? (e ?? null) : null);
  /** Once a field shows its error, the action is gated on it too (a lime button beside a red field invites a press
   *  that cannot work), saying which; it lifts as soon as the field is fixed. */
  const shownError = computed(() => (touched.get() ? Object.entries(errs.get()).find(([, e]) => e)?.[0] : undefined));
  const reason = computed(() => {
    const s = shownError.get();
    return missing.get() ?? (s ? (FIX[s] ?? null) : null);
  });

  function reset() {
    error.set(undefined);
    touched.set(false);
  }

  /** Closed while the request runs: it goes on in the server, and the answer comes as a toast, never a jump. */
  function onclose() {
    // page-level: the dialog has just closed; the request goes on behind it.
    const w = work.peek();
    if (w) toast.info(`Still ${w.what[0]!.toLowerCase()}${w.what.slice(1)}`, 'It goes on in the background. A note here will say when it is done.');
  }

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    if (mode.get() === 'synthetic' && classDraft.get().trim()) {
      classes.set(addClasses(classes.get(), classDraft.get(), MAX_SHAPES));
      classDraft.set('');
    }
    touched.set(true);
    error.set(undefined);
    if (!valid.get() || busy.get()) return;
    const m = mode.get();
    const images = Math.round(sImages.get() ?? 0);
    work.set({ what: m === 'folder' ? `Reading ${folderName.get()}` : `Generating ${count(images, 'image')}`, since: Date.now() });
    try {
      // Registering a folder that is already registered answers with that dataset (docs/API.md), not a copy.
      const before = m === 'folder' ? (props.known ?? fetched.peek() ?? (await hub.api.datasets().catch(() => []))) : [];
      const ds =
        m === 'folder'
          ? await hub.api.createDataset({ name: folderName.get(), path: cleaned.get(), format: format.get() })
          : await hub.api.createSyntheticDataset({
              name: synthName.get(),
              images,
              classes: classes.get(),
              imgsz: Number(imgsz.get()),
              seed: seed.get() ?? 0,
            });
      const known = before.some((d) => d.id === ds.id);
      const title = known ? `Already registered as ${ds.name}` : m === 'folder' ? 'Dataset registered' : 'Synthetic dataset ready';
      // Where the images went, when none can train (a COCO test file registered alone puts them all in test).
      const nt = noTraining(ds);
      const detail = nt ? `${ds.name} · ${nt.where}, none to train on` : `${ds.name} · ${count(ds.splits.train + ds.splits.val + ds.splits.test, 'image')}`;
      work.set(null);
      if (!known) datasetsChanged();
      if (!open.peek()) {
        // Dismissed while it ran: wherever the person is now, they stay there; the toast offers the dataset.
        const go = { label: 'Open', run: () => router.go(`/datasets/${ds.id}`) };
        if (known) toast.info(title, 'That folder is in the list.', go);
        else toast.success(title, detail, go);
        return;
      }
      // The dialog closes first: what was added is said over the dataset's page it opens.
      open.set(false);
      if (known) toast.info(title, 'That folder is in the list; opening it.');
      else toast.success(title, detail);
      router.go(`/datasets/${ds.id}`);
    } catch (err) {
      work.set(null);
      if (open.peek()) error.set(err);
      else toast.error(err, m === 'folder' ? 'Could not register the folder' : 'Could not generate the dataset');
    }
  }

  effect(() => {
    if (!open.get()) reset();
  });

  return (
    <Dialog
      bind:open={open}
      onclose={onclose}
      title="Add dataset"
      description="Register a folder the server can read, or generate a sample set of shapes to try FOLO."
      footer={() => (
        <>
          {/* What Register still needs, in words (a disabled button's tooltip never shows to touch or the keyboard). A
              field in error says its own reason beside it, so this line is only for what has not been entered yet. */}
          {work.get() ? (
            /* What runs, how long it has, and that closing does not stop it (the server has no cancel). The seconds are
               left out of what a screen reader hears, so it is not read out every second. */
            <p class="why" role="status">
              {work.get()!.what}… <span class="num" aria-hidden="true">{duration(elapsed.get())}</span>
              <span class="then"> Closing this does not stop it.</span>
            </p>
          ) : missing.get() ? (
            <p class="why" data-gate-why>
              {missing.get()}
            </p>
          ) : null}{' '}
          <Button variant="ghost" onclick={() => open.set(false)}>
            {busy.get() ? 'Close' : 'Cancel'}
          </Button>{' '}
          <Gate reason={reason.get()}>
            {(why) => (
              <Button
                variant="primary"
                type="submit"
                form={`${uid}-form`}
                loading={busy.get()}
                disabled={!!why()}
                aria-describedby={why()}
                icon={mode.get() === 'synthetic' ? Sparkles : FolderOpen}
              >
                {mode.get() === 'folder' ? 'Register' : 'Generate'}
              </Button>
            )}
          </Gate>
        </>
      )}
    >
      <div class="tabs" inert={busy.get()}>
        <Tabs
          label="How to add"
          idPrefix={`${uid}-mode`}
          tabs={[
            { value: 'folder', label: 'Register folder' },
            { value: 'synthetic', label: 'Generate sample set' },
          ]}
          bind:value={mode}
          onchange={reset}
        />
      </div>{' '}
      <div id={`${uid}-mode-panel`} class="form" role="tabpanel" aria-labelledby={`${uid}-mode-${mode.get()}`}>
        <form id={`${uid}-form`} onsubmit={submit} novalidate>
          {/* Locked while the request runs: what is on screen is what was sent. */}
          <fieldset class="panel" disabled={busy.get()}>
            {mode.get() === 'folder' ? (
              <>
                <Field label="Folder" for={`${uid}-path`} error={show(errs.get().path)} hint="Absolute path on the machine running `folo serve`. Files are never modified.">
                  <Input
                    id={`${uid}-path`}
                    bind:value={path}
                    placeholder="e.g. D:/data/my-dataset"
                    icon={FolderOpen}
                    class="mono-input"
                    autocomplete="off"
                    spellcheck={false}
                    invalid={!!show(errs.get().path)}
                  />
                </Field>{' '}
                <Field
                  label="Name"
                  for={`${uid}-name`}
                  error={show(errs.get().name)}
                  fix={show(errs.get().name) ? nameFix(folderName.get(), (n) => name.set(n)) : null}
                  hint={isTaken(name.get().trim()) ? takenHint(name.get().trim()) : undefined}
                >
                  <Input
                    id={`${uid}-name`}
                    bind:value={name}
                    placeholder={derivedName.get() || 'my-dataset'}
                    autocomplete="off"
                    spellcheck={false}
                    invalid={!!show(errs.get().name)}
                  />
                </Field>{' '}
                <Field label="Format" hint={format.get() === 'auto' ? 'Detects data.yaml (YOLO) or annotations/*.json (COCO).' : undefined}>
                  <Segmented
                    label="Format"
                    options={[
                      { value: 'auto', label: 'Auto-detect' },
                      { value: 'yolo', label: 'YOLO' },
                      { value: 'coco', label: 'COCO' },
                    ]}
                    bind:value={format}
                  />
                </Field>
              </>
            ) : (
              <>
                <Field
                  label="Name"
                  for={`${uid}-sname`}
                  error={show(errs.get().name)}
                  fix={show(errs.get().name) ? nameFix(synthName.get(), (n) => sName.set(n)) : null}
                  hint={isTaken(sName.get().trim()) ? takenHint(sName.get().trim()) : undefined}
                >
                  <Input
                    id={`${uid}-sname`}
                    bind:value={sName}
                    placeholder={uniqueName(SYNTH_NAME, taken.get())}
                    autocomplete="off"
                    spellcheck={false}
                    invalid={!!show(errs.get().name)}
                  />
                </Field>{' '}
                <Field label="Classes" for={`${uid}-classes`} error={show(errs.get().classes)} hint={classes.get().length >= MAX_SHAPES ? `${MAX_SHAPES} of ${MAX_SHAPES}: one shape each, so no more fit.` : `Press Enter or comma to add. Up to ${MAX_SHAPES}, one shape each.`}>
                  <ClassChips id={`${uid}-classes`} bind:classes={classes} bind:draft={classDraft} max={MAX_SHAPES} placeholder="circle, square, …" />
                </Field>{' '}
                <div class="row">
                  <Field label="Images" for={`${uid}-images`} error={show(errs.get().images)}>
                    <Input id={`${uid}-images`} type="number" min={10} max={20000} step={100} bind:value={sImages} invalid={!!show(errs.get().images)} />
                  </Field>{' '}
                  <Field label="Seed" for={`${uid}-seed`} error={show(errs.get().seed)}>
                    <Input id={`${uid}-seed`} type="number" min={0} step={1} bind:value={seed} invalid={!!show(errs.get().seed)} />
                  </Field>
                </div>{' '}
                <Field label="Image size" hint={`Square images, ${imgsz.get()} × ${imgsz.get()} px. 80% train, 20% val.`}>
                  <Segmented
                    label="Image size"
                    options={[
                      { value: '320', label: '320' },
                      { value: '480', label: '480' },
                      { value: '640', label: '640' },
                      { value: '1024', label: '1024' },
                    ]}
                    bind:value={imgsz}
                  />
                </Field>
              </>
            )}{' '}
            {error.get() ? (
              <ErrorState
                error={error.get()}
                title={mode.get() === 'folder' ? 'Could not register this folder' : 'Could not generate the dataset'}
                compact
                // Images without labels: FOLO's own answer for exactly this is labeling them.
                hint={isUnlabeled(error.get()) ? 'Label them first: an agent can draw the boxes in Label images.' : undefined}
                actions={
                  isUnlabeled(error.get()) ? (
                    <Button size="sm" icon={Tags} href={href('/datasets/label')} onclick={() => open.set(false)}>
                      Label images
                    </Button>
                  ) : undefined
                }
              />
            ) : null}
          </fieldset>
        </form>
      </div>
    </Dialog>
  );
}

css`
  /* The second sentence under the first: what runs and for how long, then that Close leaves it running. */
  .then {
    display: block;
    color: var(--text-tertiary);
  }
  .why {
    margin-right: auto;
    align-self: center;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  /* Phones: the footer's buttons share its width; the line goes above them. */
  @media (max-width: 767px) {
    .why {
      flex-basis: 100%;
    }
  }
  .tabs {
    margin: 0 calc(var(--space-5) * -1);
    padding: 0 var(--space-5);
    border-bottom: 1px solid var(--line-subtle);
  }
  .tabs :global(.tabs) {
    border-bottom: 0;
  }
  .form {
    padding-top: var(--space-4);
  }
  .panel {
    display: grid;
    gap: var(--space-4);
    min-width: 0;
    margin: 0;
    padding: 0;
    border: 0;
  }
  .row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-3);
  }
  .panel :global(.seg) {
    justify-self: start;
  }
  .panel :global(.mono-input input) {
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
  }
  /* The example is a sentence, not a value: Geist in text.tertiary, so it never reads as a path already filled in. */
  .panel :global(.mono-input input::placeholder) {
    font-family: var(--font-sans);
    font-size: var(--type-base-size);
    color: var(--text-tertiary);
  }
`;
