/**
 * "Add images": upload files from this device, or add a folder on the server's machine (its images are read in
 * place). An upload closes the dialog at once and shows its progress over the images; a folder is answered here.
 */
import FolderOpen from '$icons/folder-open';
import Upload from '$icons/upload';
import { bindable, computed, createUniqueId, css, effect, MediaQuery, signal } from 'fw';
import { folderPath, folderPathError } from '$lib/label';
import { TOUCH_ONLY } from '$lib/shell/shortcuts';
import { media } from '$lib/design/tokens';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import Field from '$lib/ui/Field.tsx';
import Gate from '$lib/ui/Gate';
import Input from '$lib/ui/Input';
import Tabs from '$lib/ui/Tabs';
import Viewfinder from '$lib/ui/Viewfinder';
import { whyNothingAdded, type Uploads } from './uploads';

export interface AddImagesDialogProps {
  open?: boolean;
  tab?: 'upload' | 'folder';
  uploads: Uploads;
  /** Opens the file picker (the page's: its files go to the same upload). */
  onbrowse: () => void;
}

export default function AddImagesDialog(props: AddImagesDialogProps) {
  const open = bindable(props, 'open', false);
  const tab = bindable(props, 'tab', 'upload' as 'upload' | 'folder');
  const uid = createUniqueId();
  const touch = new MediaQuery(TOUCH_ONLY, false);
  const phone = new MediaQuery(media.phone, false);
  const path = signal('');
  /** The path without the quotes and spaces around it (Explorer's "Copy as path"). */
  const folder = computed(() => folderPath(path.get()));
  /** Set by the first submit: from then on a relative path says so as it is typed. */
  const tried = signal(false);
  const pathError = computed(() => (tried.get() ? folderPathError(folder.get()) : null));
  const busy = signal(false);
  const error = signal<unknown>(undefined);
  /** Why the last folder added nothing (no images there, a file that is not one, all in the job already). */
  const nothing = signal<string | null>(null);
  const dragging = signal(false);

  effect(() => {
    if (!open.get()) {
      error.set(undefined);
      nothing.set(null);
      tried.set(false);
      busy.set(false);
      dragging.set(false);
    }
  });

  function drop(e: DragEvent) {
    e.preventDefault();
    dragging.set(false);
    const files = [...(e.dataTransfer?.files ?? [])];
    if (!files.length) return;
    open.set(false);
    void props.uploads.files(files);
  }

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    tried.set(true);
    if (!folder.peek() || pathError.peek() || busy.peek()) return;
    busy.set(true);
    error.set(undefined);
    nothing.set(null);
    try {
      const answer = await props.uploads.folder(folder.peek());
      if (answer && !answer.added.length) {
        nothing.set(whyNothingAdded(answer.skipped));
        return;
      }
      open.set(false);
      path.set('');
    } catch (err) {
      error.set(err);
    } finally {
      busy.set(false);
    }
  }

  return (
    <Dialog
      bind:open={open}
      title="Add images"
      description="They join the job as images to label. Duplicates and files that are not images are skipped."
      footer={() => (
        <>
          {tab.get() === 'folder' ? (
            <>
              {!folder.get() ? (
                <p class="why" data-gate-why>
                  Enter a folder path
                </p>
              ) : null}
              <Button variant="ghost" onclick={() => open.set(false)}>
                Cancel
              </Button>
              <Gate reason={folder.get() ? null : 'Enter a folder path'}>
                {(why) => (
                  <Button
                    variant="primary"
                    type="submit"
                    form={`${uid}-form`}
                    icon={FolderOpen}
                    loading={busy.get()}
                    disabled={!!why()}
                    aria-describedby={why()}
                  >
                    Add folder
                  </Button>
                )}
              </Gate>
            </>
          ) : (
            <Button variant="ghost" onclick={() => open.set(false)}>
              Cancel
            </Button>
          )}
        </>
      )}
    >
      <div class="tabs">
        <Tabs
          label="Where the images are"
          idPrefix={`${uid}-src`}
          tabs={[
            { value: 'upload', label: 'Upload' },
            { value: 'folder', label: 'Server folder' },
          ]}
          bind:value={tab}
        />
      </div>
      <div id={`${uid}-src-panel`} class="panel" role="tabpanel" aria-labelledby={`${uid}-src-${tab.get()}`}>
        {tab.get() === 'upload' ? (
          /* A drop target; its button is the keyboard's way in. */
          <div
            class="drop"
            class:dragging={dragging.get()}
            ondragover={(e) => {
              e.preventDefault();
              dragging.set(true);
            }}
            ondragleave={() => dragging.set(false)}
            ondrop={drop}
          >
            <Viewfinder
              dragging={dragging.get()}
              title={touch.current || phone.current ? 'Choose images from this device' : 'Drop images here'}
              description={() => <>JPEG, PNG, BMP or WebP; large sets go up in parts.</>}
              actions={() => (
                <Button variant="primary" icon={Upload} onclick={() => (open.set(false), props.onbrowse())}>
                  {touch.current || phone.current ? 'Choose images' : 'Browse files'}
                </Button>
              )}
            />
          </div>
        ) : (
          <form id={`${uid}-form`} class="form" onsubmit={submit} novalidate>
            <Field
              label="Folder"
              for={`${uid}-path`}
              hint="On the machine running `folo serve`. Its images, in subfolders too, are read in place, never copied or changed."
              error={pathError.get()}
            >
              <Input
                id={`${uid}-path`}
                bind:value={path}
                placeholder="e.g. D:/photos/street"
                icon={FolderOpen}
                class="mono-input"
                autocomplete="off"
                spellcheck={false}
              />
            </Field>
            {error.get() ? <ErrorState error={error.get()} title="Could not add this folder" compact /> : null}
            {nothing.get() ? (
              <div class="nothing" role="status">
                <p class="title">Nothing was added</p>
                <p>{nothing.get()}</p>
              </div>
            ) : null}
          </form>
        )}
      </div>
    </Dialog>
  );
}

css`
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
  .nothing {
    padding: var(--space-3) var(--space-4);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    overflow-wrap: anywhere;
  }
  .nothing .title {
    font-weight: var(--weight-medium);
    color: var(--text-primary);
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
`;
