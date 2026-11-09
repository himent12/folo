<script lang="ts">
  /**
   * "New labeling job" (name, classes, instructions for the agent) and "Edit job" (name and classes). Once the job's
   * images have boxes, which refer to classes by their place in the list, the edit form renames and appends classes
   * and never removes one, and says why. Validation, gating and feedback as Add dataset's.
   */
  import { untrack } from 'svelte';
  import Plus from '@lucide/svelte/icons/plus';
  import X from '@lucide/svelte/icons/x';
  import { hub } from '$lib/api/index.svelte';
  import type { LabelJob } from '$lib/api/types';
  import { classColor } from '$lib/design/tokens';
  import { CLASSES_MAX, classChangeError, classesError, INSTRUCTIONS_MAX, JOB_NAME_MAX } from '$lib/label';
  import { router } from '$lib/router.svelte';
  import Button from '$lib/ui/Button.svelte';
  import ClassChips, { addClasses } from '$lib/ui/ClassChips.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import Input from '$lib/ui/Input.svelte';
  import TextArea from '$lib/ui/TextArea.svelte';
  import { toast } from '$lib/ui/toast.svelte';

  let {
    open = $bindable(false),
    job,
    hasBoxes = false,
    onsave,
  }: {
    open?: boolean;
    /** The job to edit; none: a new one. */
    job?: LabelJob;
    /**
     * Edit: some image of the job has boxes, whatever its status (one whose relabeling failed keeps its old ones, and
     * the job's box count leaves those out). Boxes name classes by their place, so none can then be removed.
     */
    hasBoxes?: boolean;
    /** Edit: saves the change (the page applies the answer). */
    onsave?: (change: { name: string; classes: string[] }) => Promise<unknown>;
  } = $props();

  const uid = $props.id();
  const editing = $derived(!!job);
  let busy = $state(false);
  let error = $state.raw<unknown>(undefined);
  let touched = $state(false);
  let name = $state('');
  let classes = $state<string[]>([]);
  let draft = $state('');
  let instructions = $state('');

  // Every opening (or another job) starts from the job as it is (or blank); a change of the job while it is open (an
  // upload, a run's progress) keeps the edits, and a submit in flight.
  const jobId = $derived(job?.id);
  $effect(() => {
    if (!open) return;
    void jobId;
    untrack(() => {
      name = job?.name ?? '';
      classes = job ? [...job.classes] : [];
      draft = '';
      instructions = '';
      error = undefined;
      touched = false;
      busy = false;
    });
  });

  const all = $derived(editing ? classes.map((c) => c.trim()) : addClasses(classes, draft, CLASSES_MAX));
  const errs = $derived({
    name: !name.trim() ? 'Required.' : name.trim().length > JOB_NAME_MAX ? `At most ${JOB_NAME_MAX} characters.` : null,
    classes: classesError(all) ?? (job ? classChangeError(job.classes, all, hasBoxes) : null),
    instructions: instructions.length > INSTRUCTIONS_MAX ? `At most ${INSTRUCTIONS_MAX.toLocaleString('en-US')} characters (now ${instructions.length.toLocaleString('en-US')}).` : null,
  });
  const show = (e: string | null) => (touched ? e : null);
  const missing = $derived(!name.trim() ? 'Enter a name' : all.length === 0 ? 'Add at least one class' : null);
  const FIX: Record<string, string> = { name: 'Fix the name first', classes: 'Fix the classes first', instructions: 'Shorten the instructions first' };
  const shownError = $derived(touched ? Object.entries(errs).find(([, e]) => e)?.[0] : undefined);
  const reason = $derived(missing ?? (shownError ? FIX[shownError] : null));

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    touched = true;
    error = undefined;
    if (Object.values(errs).some((x) => x) || busy) return;
    busy = true;
    try {
      if (job) {
        await onsave?.({ name: name.trim(), classes: all });
        open = false;
      } else {
        const created = await hub.api.createLabelJob({ name: name.trim(), classes: all, instructions: instructions.trim() });
        // The dialog closes first: what was made is said over the job's page it opens.
        open = false;
        toast.success('Labeling job created', `${created.name} · ${created.classes.length} ${created.classes.length === 1 ? 'class' : 'classes'}`);
        router.go(`/datasets/label/${created.id}`);
      }
    } catch (err) {
      error = err;
    } finally {
      busy = false;
    }
  }
</script>

<Dialog
  bind:open
  title={editing ? 'Edit job' : 'New labeling job'}
  description={editing ? undefined : 'Name the job and the classes to find; you add the images next. An agent draws the boxes, and you review them.'}
>
  <form id="{uid}-form" class="form" onsubmit={submit} novalidate>
    <Field label="Name" for="{uid}-name" error={show(errs.name)}>
      <Input id="{uid}-name" bind:value={name} placeholder="e.g. street-scenes" autocomplete="off" spellcheck={false} invalid={!!show(errs.name)} />
    </Field>
    {#if editing}
      <Field
        label="Classes"
        error={show(errs.classes)}
        hint={hasBoxes ? 'Images have boxes of these classes, so you can rename a class or add one, not remove one.' : 'Rename, add or remove classes.'}
      >
        <ul class="rows">
          {#each classes as _, i (i)}
            <li>
              <span class="sw" style:background={classColor(i)}></span>
              <Input bind:value={classes[i]} aria-label="Class {i + 1}" autocomplete="off" spellcheck={false} invalid={!!show(errs.classes)} />
              {#if !hasBoxes || i >= (job?.classes.length ?? 0)}
                <Button variant="ghost" size="sm" icon={X} aria-label="Remove {classes[i] || `class ${i + 1}`}" onclick={() => (classes = classes.filter((__, k) => k !== i))} />
              {/if}
            </li>
          {/each}
        </ul>
        <span class="add"><Button size="sm" icon={Plus} disabled={classes.length >= CLASSES_MAX} onclick={() => (classes = [...classes, ''])}>Add class</Button></span>
      </Field>
    {:else}
      <Field label="Classes" for="{uid}-classes" error={show(errs.classes)} hint="Press Enter or comma to add. Up to {CLASSES_MAX}.">
        <ClassChips id="{uid}-classes" bind:classes bind:draft max={CLASSES_MAX} placeholder="car, person, …" invalid={!!show(errs.classes)} />
      </Field>
      <Field label="Instructions for the agent" for="{uid}-instructions" error={show(errs.instructions)} hint="Optional. What counts as each class and what to leave out, in your own words.">
        <TextArea
          id="{uid}-instructions"
          bind:value={instructions}
          rows={4}
          placeholder="Label every car and pedestrian, also partly hidden ones. Ignore reflections and posters."
          invalid={!!show(errs.instructions)}
        />
      </Field>
    {/if}
    {#if error}
      <ErrorState {error} title={editing ? 'Could not save the job' : 'Could not create the job'} compact />
    {/if}
  </form>
  {#snippet footer()}
    {#if missing}<p class="why" data-gate-why>{missing}</p>{/if}
    <Button variant="ghost" onclick={() => (open = false)}>Cancel</Button>
    <Gate {reason}>
      {#snippet children(why)}
        <Button variant="primary" type="submit" form="{uid}-form" loading={busy} disabled={!!why} aria-describedby={why} icon={editing ? undefined : Plus}>
          {editing ? 'Save' : 'Create job'}
        </Button>
      {/snippet}
    </Gate>
  {/snippet}
</Dialog>

<style>
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
  .form {
    display: grid;
    gap: var(--space-4);
  }
  .rows {
    display: grid;
    gap: var(--space-2);
  }
  .rows li {
    display: grid;
    grid-template-columns: 8px minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2);
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
  }
  .add {
    justify-self: start;
    margin-top: var(--space-1);
  }
</style>
