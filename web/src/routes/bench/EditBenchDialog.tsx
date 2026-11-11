/**
 * "Edit name and description": what a benchmark is called and what it stands for (where the camera is, what makes the
 * job hard). Both can change whatever its entries; the description is what an agent reads when it designs the plan.
 */
import { bindable, computed, createUniqueId, css, effect, signal, untrack } from 'fw';
import type { Bench } from '$lib/api/types';
import { benchNameError, descriptionError } from '$lib/bench/rules';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import Field from '$lib/ui/Field.tsx';
import Gate from '$lib/ui/Gate';
import Input from '$lib/ui/Input';
import TextArea from '$lib/ui/TextArea';

export interface EditBenchDialogProps {
  open?: boolean;
  bench: Bench;
  /** Saves the change (the page applies the answer). */
  onsave: (change: { name: string; description: string }) => Promise<unknown>;
}

export default function EditBenchDialog(props: EditBenchDialogProps) {
  const open = bindable(props, 'open', false);

  const uid = createUniqueId();
  const name = signal('');
  const description = signal('');
  const busy = signal(false);
  const error = signal<unknown>(undefined);
  const touched = signal(false);

  effect(() => {
    if (!open.get()) return;
    untrack(() => {
      name.set(props.bench.name);
      description.set(props.bench.description);
      busy.set(false);
      error.set(undefined);
      touched.set(false);
    });
  });

  const errs = computed(() => ({ name: benchNameError(name.get()), description: descriptionError(description.get()) }));
  const show = (e: string | null) => (touched.get() ? e : null);
  const unchanged = computed(() => name.get().trim() === props.bench.name && description.get().trim() === props.bench.description.trim());
  const reason = computed(() =>
    !name.get().trim()
      ? 'Enter a name'
      : touched.get() && errs.get().name
        ? 'Fix the name first'
        : touched.get() && errs.get().description
          ? 'Shorten the description first'
          : unchanged.get()
            ? 'No changes yet'
            : null,
  );

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    touched.set(true);
    error.set(undefined);
    const ers = errs.peek();
    if (ers.name || ers.description || busy.peek() || unchanged.peek()) return;
    busy.set(true);
    try {
      await props.onsave({ name: name.peek().trim(), description: description.peek().trim() });
      open.set(false);
    } catch (err) {
      error.set(err);
    } finally {
      busy.set(false);
    }
  }

  return (
    <Dialog
      bind:open={open}
      title="Edit name and description"
      footer={() => (
        <>
          {reason.get() ? (
            <p class="why" data-gate-why>
              {reason.get()}
            </p>
          ) : null}
          <Button variant="ghost" onclick={() => open.set(false)}>
            Cancel
          </Button>
          <Gate reason={reason.get()}>
            {(why) => (
              <Button variant="primary" type="submit" form={`${uid}-form`} loading={busy.get()} disabled={!!why()} aria-describedby={why()}>
                Save
              </Button>
            )}
          </Gate>
        </>
      )}
    >
      <form id={`${uid}-form`} class="form" onsubmit={submit} novalidate>
        <Field label="Name" for={`${uid}-name`} error={show(errs.get().name)}>
          <Input id={`${uid}-name`} bind:value={name} autocomplete="off" invalid={!!show(errs.get().name)} />
        </Field>
        <Field
          label="What the camera sees"
          for={`${uid}-desc`}
          error={show(errs.get().description)}
          hint="Where the camera is and what makes the job hard. An agent reads it when it designs the plan."
        >
          <TextArea id={`${uid}-desc`} bind:value={description} rows={4} invalid={!!show(errs.get().description)} />
        </Field>
        {error.get() ? <ErrorState error={error.get()} title="Could not save the benchmark" compact /> : null}
      </form>
    </Dialog>
  );
}

css`
  .form {
    display: grid;
    gap: var(--space-4);
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
