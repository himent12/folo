<script lang="ts">
  /**
   * "Create dataset": the job's labeled images written as a YOLO dataset that trains like any other. Says plainly what
   * goes in and what is left out; the name follows the Datasets rules (one click to the nearest valid one).
   */
  import Database from '@lucide/svelte/icons/database';
  import { hub } from '$lib/api/index.svelte';
  import type { Dataset, LabelJob } from '$lib/api/types';
  import { num } from '$lib/format';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import Field from '$lib/ui/Field.svelte';
  import { DialogStatus } from '$lib/ui/feedback.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import { NAME_RE, suggestName } from '../datasets/names';
  import { images } from './words';

  let { open = $bindable(false), job, oncreated }: { open?: boolean; job: LabelJob; oncreated: (ds: Dataset) => void } = $props();

  const uid = $props.id();
  const status = new DialogStatus();
  let name = $state('');
  let val = $state<'0.1' | '0.2' | '0.3'>('0.2');
  let busy = $state(false);
  let touched = $state(false);

  $effect(() => {
    if (!open) return;
    name = NAME_RE.test(job.name) ? job.name : (suggestName(job.name) ?? '');
    val = '0.2';
    touched = false;
    busy = false;
  });

  const nameError = $derived(!name.trim() ? 'Required.' : NAME_RE.test(name.trim()) ? null : 'Letters, digits, ".", "_" and "-" only (max 64).');
  const fixed = $derived(nameError && name.trim() ? suggestName(name) : null);
  const labeled = $derived(job.counts.labeled);
  const left = $derived(job.counts.pending + job.counts.failed);
  const valImages = $derived(Math.min(Math.max(0, labeled - 1), Math.round(labeled * Number(val))));
  const reason = $derived(!name.trim() ? 'Enter a name' : touched && nameError ? 'Fix the name first' : null);

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    touched = true;
    if (nameError || busy) return;
    busy = true;
    status.clear();
    try {
      const ds = await hub.api.createLabelDataset(job.id, { name: name.trim(), val_fraction: Number(val) });
      open = false;
      oncreated(ds);
    } catch (err) {
      status.error(err, 'Could not create the dataset');
    } finally {
      busy = false;
    }
  }
</script>

<Dialog bind:open title="Create dataset" description="The labeled images become a YOLO dataset you can train on. The job stays, so you can keep labeling." {status}>
  <form id="{uid}-form" class="form" onsubmit={submit} novalidate>
    <Field label="Name" for="{uid}-name" error={touched ? nameError : null} fix={touched && fixed ? { label: `Try ${fixed}`, apply: () => (name = fixed) } : null}>
      <Input id="{uid}-name" bind:value={name} autocomplete="off" spellcheck={false} invalid={touched && !!nameError} />
    </Field>
    <Field label="Validation share" hint="About {images(labeled - valImages)} to train on and {images(valImages)} to check the model with.">
      <Segmented
        label="Validation share"
        bind:value={val}
        options={[
          { value: '0.1', label: '10%' },
          { value: '0.2', label: '20%' },
          { value: '0.3', label: '30%' },
        ]}
      />
    </Field>
    <p class="facts">
      {images(labeled)} with labels, {num(job.counts.boxes)} {job.counts.boxes === 1 ? 'box' : 'boxes'}.{#if left}{' '}{images(left)} without labels {left === 1 ? 'is' : 'are'} left out.{/if}
    </p>
  </form>
  {#snippet footer()}
    {#if !name.trim()}<p class="why" data-gate-why>Enter a name</p>{/if}
    <Button variant="ghost" onclick={() => (open = false)}>Cancel</Button>
    <Gate {reason}>
      {#snippet children(why)}
        <Button variant="primary" type="submit" form="{uid}-form" icon={Database} loading={busy} disabled={!!why} aria-describedby={why}>Create dataset</Button>
      {/snippet}
    </Gate>
  {/snippet}
</Dialog>

<style>
  .form {
    display: grid;
    gap: var(--space-4);
  }
  .form :global(.seg) {
    justify-self: start;
  }
  .facts {
    color: var(--text-secondary);
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
