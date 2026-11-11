/**
 * "New labeling job" (name, classes, instructions for the agent) and "Edit job" (name and classes). Once the job's
 * images have boxes, which refer to classes by their place in the list, the edit form renames and appends classes
 * and never removes one, and says why. Validation, gating and feedback as Add dataset's.
 */
import Plus from '$icons/plus';
import X from '$icons/x';
import { bindable, computed, createUniqueId, css, effect, Index, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import type { LabelJob } from '$lib/api/types';
import { classColor } from '$lib/design/tokens';
import { CLASSES_MAX, classChangeError, classesError, INSTRUCTIONS_MAX, JOB_NAME_MAX } from '$lib/label';
import { router } from '$lib/router';
import Button from '$lib/ui/Button';
import ClassChips, { addClasses } from '$lib/ui/ClassChips';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import Field from '$lib/ui/Field.tsx';
import Gate from '$lib/ui/Gate';
import Input from '$lib/ui/Input';
import TextArea from '$lib/ui/TextArea';
import { toast } from '$lib/ui/toast';
import { leftOutText } from './words';

export interface JobDialogProps {
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
}

const FIX: Record<string, string> = { name: 'Fix the name first', classes: 'Fix the classes first', instructions: 'Shorten the instructions first' };

export default function JobDialog(props: JobDialogProps) {
  const open = bindable(props, 'open', false);
  const uid = createUniqueId();
  const editing = computed(() => !!props.job);
  const hasBoxes = () => props.hasBoxes ?? false;
  const busy = signal(false);
  const error = signal<unknown>(undefined);
  const touched = signal(false);
  const name = signal('');
  const classes = signal<string[]>([]);
  const draft = signal('');
  /** The names the last addition could not take: past the limit of classes. */
  const leftOut = signal<string[]>([]);
  const instructions = signal('');

  // Every opening (or another job) starts from the job as it is (or blank); a change of the job while it is open (an
  // upload, a run's progress) keeps the edits, and a submit in flight.
  const jobId = computed(() => props.job?.id);
  effect(() => {
    if (!open.get()) return;
    void jobId.get();
    untrack(() => {
      const job = props.job;
      name.set(job?.name ?? '');
      classes.set(job ? [...job.classes] : []);
      draft.set('');
      leftOut.set([]);
      instructions.set('');
      error.set(undefined);
      touched.set(false);
      busy.set(false);
    });
  });

  const all = computed(() => (editing.get() ? classes.get().map((c) => c.trim()) : addClasses(classes.get(), draft.get(), CLASSES_MAX)));
  const errs = computed(() => {
    const n = name.get();
    const ins = instructions.get();
    const job = props.job;
    return {
      name: !n.trim() ? 'Required.' : n.trim().length > JOB_NAME_MAX ? `At most ${JOB_NAME_MAX} characters.` : null,
      classes: classesError(all.get()) ?? (job ? classChangeError(job.classes, all.get(), hasBoxes()) : null),
      instructions:
        ins.length > INSTRUCTIONS_MAX ? `At most ${INSTRUCTIONS_MAX.toLocaleString('en-US')} characters (now ${ins.length.toLocaleString('en-US')}).` : null,
    };
  });
  const show = (e: string | null) => (touched.get() ? e : null);
  const missing = computed(() => (!name.get().trim() ? 'Enter a name' : all.get().length === 0 ? 'Add at least one class' : null));
  const shownError = computed(() => (touched.get() ? Object.entries(errs.get()).find(([, e]) => e)?.[0] : undefined));
  const reason = computed(() => {
    const s = shownError.get();
    return missing.get() ?? (s ? FIX[s] : null);
  });

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    touched.set(true);
    error.set(undefined);
    if (Object.values(errs.peek()).some((x) => x) || busy.peek()) return;
    busy.set(true);
    try {
      const job = props.job;
      if (job) {
        await props.onsave?.({ name: name.peek().trim(), classes: all.peek() });
        open.set(false);
      } else {
        const created = await hub.api.createLabelJob({ name: name.peek().trim(), classes: all.peek(), instructions: instructions.peek().trim() });
        // The dialog closes first: what was made is said over the job's page it opens.
        open.set(false);
        toast.success('Labeling job created', `${created.name} · ${created.classes.length} ${created.classes.length === 1 ? 'class' : 'classes'}`);
        router.go(`/datasets/label/${created.id}`);
      }
    } catch (err) {
      error.set(err);
    } finally {
      busy.set(false);
    }
  }

  /** `bind:value={classes[i]}`: the class at `i`, written back as a new list. */
  const classAt = (i: number) => ({
    get: () => classes.get()[i] ?? '',
    set: (v: string) => classes.set(classes.peek().map((c, k) => (k === i ? v : c))),
  });

  return (
    <Dialog
      bind:open={open}
      title={editing.get() ? 'Edit job' : 'New labeling job'}
      description={editing.get() ? undefined : 'Name the job and the classes to find; you add the images next. An agent draws the boxes, and you review them.'}
      footer={() => (
        <>
          {missing.get() ? (
            <p class="why" data-gate-why>
              {missing.get()}
            </p>
          ) : null}
          <Button variant="ghost" onclick={() => open.set(false)}>
            Cancel
          </Button>
          <Gate reason={reason.get()}>
            {(why) => (
              <Button
                variant="primary"
                type="submit"
                form={`${uid}-form`}
                loading={busy.get()}
                disabled={!!why()}
                aria-describedby={why()}
                icon={editing.get() ? undefined : Plus}
              >
                {editing.get() ? 'Save' : 'Create job'}
              </Button>
            )}
          </Gate>
        </>
      )}
    >
      <form id={`${uid}-form`} class="form" onsubmit={submit} novalidate>
        <Field label="Name" for={`${uid}-name`} error={show(errs.get().name)}>
          <Input
            id={`${uid}-name`}
            bind:value={name}
            placeholder="e.g. street-scenes"
            autocomplete="off"
            spellcheck={false}
            invalid={!!show(errs.get().name)}
          />
        </Field>
        {editing.get() ? (
          <Field
            label="Classes"
            error={show(errs.get().classes)}
            hint={hasBoxes() ? 'Images have boxes of these classes, so you can rename a class or add one, not remove one.' : 'Rename, add or remove classes.'}
          >
            <ul class="rows">
              <Index each={classes.get()}>
                {(c, i) => (
                  <li>
                    <span class="sw" style:background={classColor(i)}></span>
                    <Input
                      bind:value={classAt(i)}
                      aria-label={`Class ${i + 1}`}
                      autocomplete="off"
                      spellcheck={false}
                      invalid={!!show(errs.get().classes)}
                    />
                    {!hasBoxes() || i >= (props.job?.classes.length ?? 0) ? (
                      <Button
                        variant="ghost"
                        size="sm"
                        icon={X}
                        aria-label={`Remove ${c() || `class ${i + 1}`}`}
                        onclick={() => classes.set(classes.peek().filter((__, k) => k !== i))}
                      />
                    ) : null}
                  </li>
                )}
              </Index>
            </ul>
            <span class="add">
              <Button size="sm" icon={Plus} disabled={classes.get().length >= CLASSES_MAX} onclick={() => classes.set([...classes.peek(), ''])}>
                Add class
              </Button>
            </span>
          </Field>
        ) : (
          <>
            <Field
              label="Classes"
              for={`${uid}-classes`}
              error={show(errs.get().classes)}
              hint={leftOut.get().length ? leftOutText(leftOut.get()) : `Press Enter or comma to add. Up to ${CLASSES_MAX}.`}
            >
              <ClassChips
                id={`${uid}-classes`}
                bind:classes={classes}
                bind:draft={draft}
                max={CLASSES_MAX}
                onleftout={(names) => leftOut.set(names)}
                placeholder="car, person, …"
                invalid={!!show(errs.get().classes)}
              />
            </Field>
            <Field
              label="Instructions for the agent"
              for={`${uid}-instructions`}
              error={show(errs.get().instructions)}
              hint="Optional. What counts as each class and what to leave out, in your own words."
            >
              <TextArea
                id={`${uid}-instructions`}
                bind:value={instructions}
                rows={4}
                placeholder="Label every car and pedestrian, also partly hidden ones. Ignore reflections and posters."
                invalid={!!show(errs.get().instructions)}
              />
            </Field>
          </>
        )}
        {error.get() ? <ErrorState error={error.get()} title={editing.get() ? 'Could not save the job' : 'Could not create the job'} compact /> : null}
      </form>
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
`;
