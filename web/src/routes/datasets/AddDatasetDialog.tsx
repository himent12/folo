/** "Add dataset": register a YOLO/COCO folder on the server's disk, or generate a synthetic shapes set. */
import FolderOpen from '$icons/folder-open';
import Sparkles from '$icons/sparkles';
import { bindable, computed, createUniqueId, css, effect, signal } from 'fw';
import { hub } from '$lib/api/index';
import type { CreateDataset } from '$lib/api/types';
import { router } from '$lib/router';
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
import { NAME_RE, suggestName } from './names';

type Mode = 'folder' | 'synthetic';

export interface AddDatasetDialogProps {
  open?: boolean;
  mode?: Mode;
  /** The datasets already listed, to tell a new registration from an existing one (fetched when absent). */
  known?: readonly { id: string }[];
}

const nameError = (v: string) => (!v.trim() ? 'Required.' : NAME_RE.test(v.trim()) ? null : 'Letters, digits, ".", "_" and "-" only (max 64).');
/** A name in error offers its nearest valid form as one click ("Try my-dataset"). */
const nameFix = (v: string, set: (n: string) => void) => {
  const n = suggestName(v);
  return n && n !== v.trim() ? { label: `Try ${n}`, apply: () => set(n) } : null;
};
const FIX: Record<string, string> = { name: 'Fix the name first', path: 'Enter a folder path', images: 'Fix the image count first', classes: 'Add at least one class', seed: 'Fix the seed first' };

export default function AddDatasetDialog(props: AddDatasetDialogProps) {
  const open = bindable(props, 'open', false);
  const mode = bindable(props, 'mode', 'folder' as Mode);

  const uid = createUniqueId();
  const busy = signal(false);
  const error = signal<unknown>(undefined);
  const touched = signal(false);

  // Register a folder
  const name = signal('');
  const path = signal('');
  const format = signal<CreateDataset['format']>('auto');

  // Synthetic
  const sName = signal('shapes-synth');
  const sImages = signal<number | null>(1000);
  const classes = signal<string[]>(['circle', 'square', 'triangle']);
  const classDraft = signal('');
  const imgsz = signal<'320' | '480' | '640' | '1024'>('640');
  const seed = signal<number | null>(0);

  /** A folder's last path segment, as a default name. */
  const derivedName = computed(
    () =>
      path
        .get()
        .trim()
        .replace(/[\\/]+$/, '')
        .split(/[\\/]/)
        .pop()
        ?.replace(/[^A-Za-z0-9._-]/g, '-') ?? '',
  );
  const folderName = computed(() => name.get().trim() || derivedName.get());

  const errs = computed<Partial<Record<string, string | null>>>(() => {
    if (mode.get() === 'folder') {
      const n = folderName.get();
      return {
        name: n ? nameError(n) : null,
        path: path.get().trim() ? null : 'Required: an absolute folder path on the server.',
      };
    }
    const images = sImages.get();
    const s = seed.get();
    return {
      name: nameError(sName.get()),
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
      ? path.get().trim()
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
    busy.set(false);
  }

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    if (mode.get() === 'synthetic' && classDraft.get().trim()) {
      classes.set(addClasses(classes.get(), classDraft.get(), 16));
      classDraft.set('');
    }
    touched.set(true);
    error.set(undefined);
    if (!valid.get() || busy.get()) return;
    busy.set(true);
    const m = mode.get();
    try {
      // Registering a folder that is already registered answers with that dataset (docs/API.md), not a copy.
      const before = m === 'folder' ? (props.known ?? (await hub.api.datasets().catch(() => []))) : [];
      const ds =
        m === 'folder'
          ? await hub.api.createDataset({ name: folderName.get(), path: path.get().trim(), format: format.get() })
          : await hub.api.createSyntheticDataset({
              name: sName.get().trim(),
              images: Math.round(sImages.get() ?? 0),
              classes: classes.get(),
              imgsz: Number(imgsz.get()),
              seed: seed.get() ?? 0,
            });
      // The dialog closes first: what was added is said over the dataset's page it opens.
      open.set(false);
      if (before.some((d) => d.id === ds.id)) toast.info(`Already registered as ${ds.name}`, 'That folder is in the list; opening it.');
      else toast.success(m === 'folder' ? 'Dataset registered' : 'Synthetic dataset ready', `${ds.name} · ${ds.splits.train + ds.splits.val + ds.splits.test} images`);
      router.go(`/datasets/${ds.id}`);
    } catch (err) {
      error.set(err);
    } finally {
      busy.set(false);
    }
  }

  effect(() => {
    if (!open.get()) reset();
  });

  return (
    <Dialog
      bind:open={open}
      title="Add dataset"
      description="Register a folder the server can read, or generate a sample set of shapes to try FOLO."
      footer={() => (
        <>
          {/* What Register still needs, in words (a disabled button's tooltip never shows to touch or the keyboard). A
              field in error says its own reason beside it, so this line is only for what has not been entered yet. */}
          {missing.get() ? (
            <p class="why" data-gate-why>
              {missing.get()}
            </p>
          ) : null}{' '}
          <Button variant="ghost" onclick={() => open.set(false)}>
            Cancel
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
      <div class="tabs">
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
        <form id={`${uid}-form`} class="panel" onsubmit={submit} novalidate>
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
                fix={show(errs.get().name) ? nameFix(sName.get(), (n) => sName.set(n)) : null}
              >
                <Input id={`${uid}-sname`} bind:value={sName} autocomplete="off" spellcheck={false} invalid={!!show(errs.get().name)} />
              </Field>{' '}
              <Field label="Classes" for={`${uid}-classes`} error={show(errs.get().classes)} hint="Press Enter or comma to add. Up to 16.">
                <ClassChips id={`${uid}-classes`} bind:classes={classes} bind:draft={classDraft} max={16} placeholder="circle, square, …" />
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
            <ErrorState error={error.get()} title={mode.get() === 'folder' ? 'Could not register this folder' : 'Could not generate the dataset'} compact />
          ) : null}
        </form>
      </div>
    </Dialog>
  );
}

css`
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
