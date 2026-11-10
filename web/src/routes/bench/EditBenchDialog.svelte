<script lang="ts">
  /**
   * "Edit name and description": what a benchmark is called and what it stands for (where the camera is, what makes the
   * job hard). Both can change whatever its entries; the description is what an agent reads when it designs the plan.
   */
  import { untrack } from 'svelte';
  import type { Bench } from '$lib/api/types';
  import { benchNameError, descriptionError } from '$lib/bench/rules';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import Input from '$lib/ui/Input.svelte';
  import TextArea from '$lib/ui/TextArea.svelte';

  let {
    open = $bindable(false),
    bench,
    onsave,
  }: {
    open?: boolean;
    bench: Bench;
    /** Saves the change (the page applies the answer). */
    onsave: (change: { name: string; description: string }) => Promise<unknown>;
  } = $props();

  const uid = $props.id();
  let name = $state('');
  let description = $state('');
  let busy = $state(false);
  let error = $state.raw<unknown>(undefined);
  let touched = $state(false);

  $effect(() => {
    if (!open) return;
    untrack(() => {
      name = bench.name;
      description = bench.description;
      busy = false;
      error = undefined;
      touched = false;
    });
  });

  const errs = $derived({ name: benchNameError(name), description: descriptionError(description) });
  const show = (e: string | null) => (touched ? e : null);
  const unchanged = $derived(name.trim() === bench.name && description.trim() === bench.description.trim());
  const reason = $derived(!name.trim() ? 'Enter a name' : touched && errs.name ? 'Fix the name first' : touched && errs.description ? 'Shorten the description first' : unchanged ? 'No changes yet' : null);

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    touched = true;
    error = undefined;
    if (errs.name || errs.description || busy || unchanged) return;
    busy = true;
    try {
      await onsave({ name: name.trim(), description: description.trim() });
      open = false;
    } catch (err) {
      error = err;
    } finally {
      busy = false;
    }
  }
</script>

<Dialog bind:open title="Edit name and description">
  <form id="{uid}-form" class="form" onsubmit={submit} novalidate>
    <Field label="Name" for="{uid}-name" error={show(errs.name)}>
      <Input id="{uid}-name" bind:value={name} autocomplete="off" invalid={!!show(errs.name)} />
    </Field>
    <Field label="What the camera sees" for="{uid}-desc" error={show(errs.description)} hint="Where the camera is and what makes the job hard. An agent reads it when it designs the plan.">
      <TextArea id="{uid}-desc" bind:value={description} rows={4} invalid={!!show(errs.description)} />
    </Field>
    {#if error}<ErrorState {error} title="Could not save the benchmark" compact />{/if}
  </form>
  {#snippet footer()}
    {#if reason}<p class="why" data-gate-why>{reason}</p>{/if}
    <Button variant="ghost" onclick={() => (open = false)}>Cancel</Button>
    <Gate {reason}>
      {#snippet children(why)}
        <Button variant="primary" type="submit" form="{uid}-form" loading={busy} disabled={!!why} aria-describedby={why}>Save</Button>
      {/snippet}
    </Gate>
  {/snippet}
</Dialog>

<style>
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
</style>
