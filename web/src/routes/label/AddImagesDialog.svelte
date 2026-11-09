<script lang="ts">
  /**
   * "Add images": upload files from this device, or add a folder on the server's machine (its images are read in
   * place). An upload closes the dialog at once and shows its progress over the images; a folder is answered here.
   */
  import FolderOpen from '@lucide/svelte/icons/folder-open';
  import Upload from '@lucide/svelte/icons/upload';
  import { MediaQuery } from 'svelte/reactivity';
  import { TOUCH_ONLY } from '$lib/shell/shortcuts';
  import { media } from '$lib/design/tokens';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Tabs from '$lib/ui/Tabs.svelte';
  import Viewfinder from '$lib/ui/Viewfinder.svelte';
  import type { Uploads } from './uploads.svelte';

  let {
    open = $bindable(false),
    tab = $bindable('upload'),
    uploads,
    onbrowse,
  }: {
    open?: boolean;
    tab?: 'upload' | 'folder';
    uploads: Uploads;
    /** Opens the file picker (the page's: its files go to the same upload). */
    onbrowse: () => void;
  } = $props();

  const uid = $props.id();
  const touch = new MediaQuery(TOUCH_ONLY, false);
  const phone = new MediaQuery(media.phone, false);
  let path = $state('');
  let busy = $state(false);
  let error = $state.raw<unknown>(undefined);
  let dragging = $state(false);

  $effect(() => {
    if (!open) {
      error = undefined;
      busy = false;
      dragging = false;
    }
  });

  function drop(e: DragEvent) {
    e.preventDefault();
    dragging = false;
    const files = [...(e.dataTransfer?.files ?? [])];
    if (!files.length) return;
    open = false;
    void uploads.files(files);
  }

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    if (!path.trim() || busy) return;
    busy = true;
    error = undefined;
    try {
      await uploads.folder(path.trim());
      open = false;
      path = '';
    } catch (err) {
      error = err;
    } finally {
      busy = false;
    }
  }
</script>

<Dialog bind:open title="Add images" description="They join the job as images to label. Duplicates and files that are not images are skipped.">
  <div class="tabs">
    <Tabs
      label="Where the images are"
      idPrefix="{uid}-src"
      tabs={[
        { value: 'upload', label: 'Upload' },
        { value: 'folder', label: 'Server folder' },
      ]}
      bind:value={tab}
    />
  </div>
  <div id="{uid}-src-panel" class="panel" role="tabpanel" aria-labelledby="{uid}-src-{tab}">
    {#if tab === 'upload'}
      <!-- svelte-ignore a11y_no_static_element_interactions: a drop target; its button is the keyboard's way in. -->
      <div
        class="drop"
        class:dragging
        ondragover={(e) => {
          e.preventDefault();
          dragging = true;
        }}
        ondragleave={() => (dragging = false)}
        ondrop={drop}
      >
        <Viewfinder {dragging} title={touch.current || phone.current ? 'Choose images from this device' : 'Drop images here'}>
          {#snippet description()}JPEG, PNG, BMP or WebP; large sets go up in parts.{/snippet}
          {#snippet actions()}
            <Button variant="primary" icon={Upload} onclick={() => ((open = false), onbrowse())}>{touch.current || phone.current ? 'Choose images' : 'Browse files'}</Button>
          {/snippet}
        </Viewfinder>
      </div>
    {:else}
      <form id="{uid}-form" class="form" onsubmit={submit} novalidate>
        <Field label="Folder" for="{uid}-path" hint="On the machine running `folo serve`. Its images, in subfolders too, are read in place, never copied or changed.">
          <Input id="{uid}-path" bind:value={path} placeholder="e.g. D:/photos/street" icon={FolderOpen} class="mono-input" autocomplete="off" spellcheck={false} />
        </Field>
        {#if error}<ErrorState {error} title="Could not add this folder" compact />{/if}
      </form>
    {/if}
  </div>
  {#snippet footer()}
    {#if tab === 'folder'}
      {#if !path.trim()}<p class="why" data-gate-why>Enter a folder path</p>{/if}
      <Button variant="ghost" onclick={() => (open = false)}>Cancel</Button>
      <Gate reason={path.trim() ? null : 'Enter a folder path'}>
        {#snippet children(why)}
          <Button variant="primary" type="submit" form="{uid}-form" icon={FolderOpen} loading={busy} disabled={!!why} aria-describedby={why}>Add folder</Button>
        {/snippet}
      </Gate>
    {:else}
      <Button variant="ghost" onclick={() => (open = false)}>Cancel</Button>
    {/if}
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
  .panel {
    padding-top: var(--space-4);
  }
  .drop {
    display: grid;
    border-radius: var(--radius-lg);
    background: var(--bg-inset);
    --vf-pad: var(--space-6);
  }
  .drop.dragging {
    background: color-mix(in srgb, var(--bg-inset), var(--accent-solid) 6%);
  }
  .form {
    display: grid;
    gap: var(--space-4);
  }
  .form :global(.mono-input input) {
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
  }
  .form :global(.mono-input input::placeholder) {
    font-family: var(--font-sans);
    font-size: var(--type-base-size);
    color: var(--text-tertiary);
  }
  .why {
    margin-right: auto;
    align-self: center;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  @media (max-width: 767px) {
    .why {
      flex-basis: 100%;
    }
  }
</style>
