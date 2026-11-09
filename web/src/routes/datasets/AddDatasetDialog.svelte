<script lang="ts">
  /** "Add dataset": register a YOLO/COCO folder on the server's disk, or generate a synthetic shapes set. */
  import FolderOpen from '@lucide/svelte/icons/folder-open';
  import Sparkles from '@lucide/svelte/icons/sparkles';
  import X from '@lucide/svelte/icons/x';
  import { hub } from '$lib/api/index.svelte';
  import type { CreateDataset } from '$lib/api/types';
  import { classColor } from '$lib/design/tokens';
  import { router } from '$lib/router.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Tabs from '$lib/ui/Tabs.svelte';
  import { toast } from '$lib/ui/toast.svelte';

  type Mode = 'folder' | 'synthetic';
  let {
    open = $bindable(false),
    mode = $bindable('folder'),
    known,
  }: {
    open?: boolean;
    mode?: Mode;
    /** The datasets already listed, to tell a new registration from an existing one (fetched when absent). */
    known?: readonly { id: string }[];
  } = $props();

  const uid = $props.id();
  let busy = $state(false);
  let error = $state.raw<unknown>(undefined);
  let touched = $state(false);

  // Register a folder
  let name = $state('');
  let path = $state('');
  let format = $state<CreateDataset['format']>('auto');

  // Synthetic
  let sName = $state('shapes-synth');
  let sImages = $state<number | null>(1000);
  let classes = $state<string[]>(['circle', 'square', 'triangle']);
  let classDraft = $state('');
  let imgsz = $state<'320' | '480' | '640' | '1024'>('640');
  let seed = $state<number | null>(0);

  const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
  const nameError = (v: string) =>
    !v.trim() ? 'Required.' : NAME_RE.test(v.trim()) ? null : 'Letters, digits, ".", "_" and "-" only (max 64).';

  /** A folder's last path segment, as a default name. */
  const derivedName = $derived(path.trim().replace(/[\\/]+$/, '').split(/[\\/]/).pop()?.replace(/[^A-Za-z0-9._-]/g, '-') ?? '');
  const folderName = $derived(name.trim() || derivedName);

  const errs = $derived(
    mode === 'folder'
      ? {
          name: folderName ? nameError(folderName) : null,
          path: path.trim() ? null : 'Required: an absolute folder path on the server.',
        }
      : {
          name: nameError(sName),
          images: sImages === null || !Number.isFinite(sImages) || sImages < 10 || sImages > 20000 ? 'Between 10 and 20,000.' : null,
          classes: classes.length === 0 ? 'Add at least one class.' : null,
          seed: seed === null || !Number.isInteger(seed) || seed < 0 ? 'A non-negative integer.' : null,
        },
  );
  const valid = $derived(Object.values(errs).every((e) => !e));
  const show = (e: string | null | undefined) => (touched ? (e ?? null) : null);

  function addClasses(raw: string) {
    const next = raw
      .split(/[,\n]/)
      .map((c) => c.trim())
      .filter((c) => c && !classes.includes(c));
    if (next.length) classes = [...classes, ...next].slice(0, 16);
    classDraft = '';
  }

  function onClassKey(e: KeyboardEvent) {
    if (e.key === 'Enter' || e.key === ',') {
      if (!classDraft.trim()) return;
      e.preventDefault();
      addClasses(classDraft);
    } else if (e.key === 'Backspace' && !classDraft && classes.length) {
      classes = classes.slice(0, -1);
    }
  }

  function reset() {
    error = undefined;
    touched = false;
    busy = false;
  }

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    if (mode === 'synthetic' && classDraft.trim()) addClasses(classDraft);
    touched = true;
    error = undefined;
    if (!valid || busy) return;
    busy = true;
    try {
      // Registering a folder that is already registered answers with that dataset (docs/API.md), not a copy.
      const before = mode === 'folder' ? (known ?? (await hub.api.datasets().catch(() => []))) : [];
      const ds =
        mode === 'folder'
          ? await hub.api.createDataset({ name: folderName, path: path.trim(), format })
          : await hub.api.createSyntheticDataset({
              name: sName.trim(),
              images: Math.round(sImages ?? 0),
              classes,
              imgsz: Number(imgsz),
              seed: seed ?? 0,
            });
      if (before.some((d) => d.id === ds.id)) toast.info(`Already registered as ${ds.name}`, 'That folder is in the list; opening it.');
      else toast.success(mode === 'folder' ? 'Dataset registered' : 'Synthetic dataset ready', `${ds.name} · ${ds.splits.train + ds.splits.val + ds.splits.test} images`);
      open = false;
      router.go(`/datasets/${ds.id}`);
    } catch (err) {
      error = err;
    } finally {
      busy = false;
    }
  }

  $effect(() => {
    if (!open) reset();
  });
</script>

<Dialog bind:open title="Add dataset" description="Register a folder the server can read, or generate a sample set of shapes to try FOLO.">
  <div class="tabs">
    <Tabs
      label="How to add"
      idPrefix="{uid}-mode"
      tabs={[
        { value: 'folder', label: 'Register folder' },
        { value: 'synthetic', label: 'Generate sample set' },
      ]}
      bind:value={mode}
      onchange={reset}
    />
  </div>
  <div id="{uid}-mode-panel" class="form" role="tabpanel" aria-labelledby="{uid}-mode-{mode}">
    <form id="{uid}-form" class="panel" onsubmit={submit} novalidate>
      {#if mode === 'folder'}
        <Field label="Folder" for="{uid}-path" error={show(errs.path)} hint="Absolute path on the machine running folo serve. Files are never modified.">
          <Input id="{uid}-path" bind:value={path} placeholder="D:/data/my-dataset" icon={FolderOpen} class="mono-input" autocomplete="off" spellcheck={false} invalid={!!show(errs.path)} aria-describedby={show(errs.path) ? `${uid}-path-error` : `${uid}-path-hint`} />
        </Field>
        <Field label="Name" for="{uid}-name" error={show(errs.name)}>
          <Input id="{uid}-name" bind:value={name} placeholder={derivedName || 'my-dataset'} autocomplete="off" spellcheck={false} invalid={!!show(errs.name)} aria-describedby={show(errs.name) ? `${uid}-name-error` : undefined} />
        </Field>
        <Field label="Format" hint={format === 'auto' ? 'Detects data.yaml (YOLO) or annotations/*.json (COCO).' : undefined}>
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
      {:else}
        <Field label="Name" for="{uid}-sname" error={show(errs.name)}>
          <Input id="{uid}-sname" bind:value={sName} autocomplete="off" spellcheck={false} invalid={!!show(errs.name)} aria-describedby={show(errs.name) ? `${uid}-sname-error` : undefined} />
        </Field>
        <Field label="Classes" for="{uid}-classes" error={show(errs.classes)} hint="Press Enter or comma to add. Up to 16.">
          <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
          <div class="chips" class:invalid={!!show(errs.classes)} onclick={(e) => (e.currentTarget.querySelector('input') as HTMLInputElement | null)?.focus()}>
            {#each classes as c, i (c)}
              <span class="chip">
                <span class="sw" style:background={classColor(i)}></span>
                {c}
                <button type="button" aria-label="Remove {c}" onclick={() => (classes = classes.filter((x) => x !== c))}><X size={12} /></button>
              </span>
            {/each}
            <input
              id="{uid}-classes"
              bind:value={classDraft}
              onkeydown={onClassKey}
              onblur={() => classDraft.trim() && addClasses(classDraft)}
              placeholder={classes.length ? '' : 'circle, square, …'}
              aria-invalid={!!show(errs.classes) || undefined}
              aria-describedby={show(errs.classes) ? `${uid}-classes-error` : `${uid}-classes-hint`}
              autocomplete="off"
              spellcheck={false}
            />
          </div>
        </Field>
        <div class="row">
          <Field label="Images" for="{uid}-images" error={show(errs.images)}>
            <Input id="{uid}-images" type="number" min={10} max={20000} step={100} bind:value={sImages} invalid={!!show(errs.images)} aria-describedby={show(errs.images) ? `${uid}-images-error` : undefined} />
          </Field>
          <Field label="Seed" for="{uid}-seed" error={show(errs.seed)}>
            <Input id="{uid}-seed" type="number" min={0} step={1} bind:value={seed} invalid={!!show(errs.seed)} aria-describedby={show(errs.seed) ? `${uid}-seed-error` : undefined} />
          </Field>
        </div>
        <Field label="Image size" hint="Square images, {imgsz} × {imgsz} px. 80% train, 20% val.">
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
      {/if}
      {#if error}
        <ErrorState {error} title={mode === 'folder' ? 'Could not register this folder' : 'Could not generate the dataset'} compact />
      {/if}
    </form>
  </div>
  {#snippet footer()}
    <Button variant="ghost" onclick={() => (open = false)}>Cancel</Button>
    <Button variant="primary" type="submit" form="{uid}-form" loading={busy} icon={mode === 'synthetic' ? Sparkles : FolderOpen}>
      {mode === 'folder' ? 'Register' : 'Generate'}
    </Button>
  {/snippet}
</Dialog>

<style>
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
  .chips {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1);
    min-height: var(--control-md);
    padding: 4px;
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    /* An input's edge (line.control), drawn like ui/Input: an invalid draft turns the edge and the ring danger. */
    border: 1px solid var(--line-control);
    cursor: text;
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      box-shadow var(--dur-fast) var(--ease-standard);
  }
  :global([data-theme='light']) .chips {
    background: var(--bg-panel);
  }
  .chips:hover {
    border-color: var(--line-control-hover);
  }
  .chips:focus-within {
    border-color: var(--focus);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--focus) 18%, transparent);
  }
  .chips.invalid {
    --line-control: var(--danger-solid);
    --line-control-hover: var(--danger-solid);
    --focus: var(--danger-solid);
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 20px;
    padding: 0 2px 0 6px;
    border-radius: var(--radius-xs);
    background: var(--bg-active);
    font-size: var(--type-sm-size);
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
  }
  :global([data-theme='light']) .sw {
    box-shadow: inset 0 0 0 1px #0a0e1433;
  }
  .chip button {
    display: grid;
    place-items: center;
    width: 16px;
    height: 16px;
    border-radius: 2px;
    color: var(--text-tertiary);
  }
  .chip button:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  /* Touch: a 44 px hit area around the small remove glyph. */
  @media (pointer: coarse) {
    .chip button {
      position: relative;
    }
    .chip button::after {
      content: '';
      position: absolute;
      inset: -12px;
    }
  }
  .chips input {
    flex: 1;
    min-width: 80px;
    height: 20px;
    padding: 0 4px;
    border: 0;
    background: none;
    outline: none;
  }
  .chips input::placeholder {
    color: var(--text-tertiary);
  }
</style>
