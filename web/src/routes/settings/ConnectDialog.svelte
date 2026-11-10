<script lang="ts">
  /**
   * Settings › Cloud › Connect: a provider and its credentials, the form built from `GET /api/cloud/providers` (one
   * field per `fields` entry, secrets masked), checked with the provider by the server before anything is stored.
   * Secrets live in this form only while it is open: never in preferences, a log, a toast or an error line, and nothing
   * of one provider's survives switching to the other.
   */
  import ExternalLink from '@lucide/svelte/icons/external-link';
  import PlugZap from '@lucide/svelte/icons/plug-zap';
  import { ApiError, hub } from '$lib/api/index.svelte';
  import type { CloudConnection, CloudField, CloudProvider } from '$lib/api/types';
  import { sentence } from '$lib/format';
  import Button from '$lib/ui/Button.svelte';
  import CodeText from '$lib/ui/CodeText.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import { toast } from '$lib/ui/toast.svelte';

  let {
    open = $bindable(false),
    providers,
    onconnected,
  }: {
    open?: boolean;
    providers: CloudProvider[];
    /** The new connection, once the dialog has closed. */
    onconnected: (c: CloudConnection) => void;
  } = $props();

  const uid = $props.id();
  let providerId = $state('');
  let values = $state<Record<string, string>>({});
  let name = $state('');
  let busy = $state(false);
  /** Errors at their fields (by key, `name` for the name), from the check on submit or the server's answer. */
  let fieldErrors = $state<Record<string, string>>({});
  /** Anything the server said that no field is about. */
  let failure = $state.raw<unknown>(undefined);

  const provider = $derived(providers.find((p) => p.id === providerId));
  $effect(() => {
    if (!provider && providers[0]) providerId = providers[0].id;
  });
  /** The first secret field: where a rejected credential is said, and where the provider's console is linked. */
  const firstSecret = $derived(provider?.fields.find((f) => f.secret));

  /** Back to an empty form: every value dropped, the secrets with them. */
  function reset() {
    values = {};
    name = '';
    fieldErrors = {};
    failure = undefined;
    busy = false;
  }
  $effect(() => {
    if (!open) reset();
  });

  /** A field's label inside a sentence: "the server URL", "the API key" (an acronym keeps its case). */
  const inSentence = (f: CloudField) => (/^[A-Z]{2}/.test(f.label) ? f.label : f.label.charAt(0).toLowerCase() + f.label.slice(1));
  /** What a credential is called after "this": "API key" → "key", "Token" → "token". */
  const noun = (f: CloudField) => f.label.toLowerCase().replace(/^api /, '');

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    const p = provider;
    if (!p || busy) return;
    failure = undefined;
    fieldErrors = Object.fromEntries(p.fields.filter((f) => f.required && !values[f.key]?.trim()).map((f) => [f.key, `Enter the ${inSentence(f)}.`]));
    if (Object.keys(fieldErrors).length) return;
    busy = true;
    try {
      const conn = await hub.api.connectCloud({
        provider: p.id,
        name: name.trim(),
        values: Object.fromEntries(p.fields.map((f) => [f.key, values[f.key]?.trim() ?? ''])),
      });
      // The dialog closes first: the new row says the rest.
      open = false;
      toast.success(`Connected ${p.name}`);
      onconnected(conn);
    } catch (err) {
      const field = err instanceof ApiError && err.code === 'invalid_body' ? p.fields.find((f) => err.message.includes(`\`${f.key}\``)) : undefined;
      if (err instanceof ApiError && err.code === 'cloud_auth_failed' && firstSecret) fieldErrors = { [firstSecret.key]: `${p.name} rejected this ${noun(firstSecret)}.` };
      else if (field && err instanceof ApiError) fieldErrors = { [field.key]: sentence(err.message, true) };
      else failure = err;
    } finally {
      busy = false;
    }
  }
</script>

<Dialog bind:open title="Connect a cloud" description="Train on a rented GPU, or on another FOLO server of yours.">
  {#if provider}
    <form id="{uid}-form" class="form" onsubmit={submit} novalidate autocomplete="off">
      {#if providers.length > 1}
        <Field label="Provider" hint={provider.about}>
          <!-- Another provider starts an empty form: nothing typed for one is kept for the other. -->
          <Segmented label="Provider" bind:value={providerId} onchange={reset} options={providers.map((p) => ({ value: p.id, label: p.name }))} />
        </Field>
      {:else}
        <p class="about"><CodeText text={provider.about} /></p>
      {/if}
      {#each provider.fields as f (`${provider.id}/${f.key}`)}
        <Field label={f.label} for="{uid}-{f.key}" hint={f.help ?? undefined} error={fieldErrors[f.key] ?? null}>
          {#snippet aside()}{#if f === firstSecret && provider.docs_url}<a class="docs" href={provider.docs_url} target="_blank" rel="noreferrer"
                >Where to find it<ExternalLink size={12} strokeWidth={1.75} aria-hidden="true" /></a
              >{/if}{/snippet}
          <Input
            id="{uid}-{f.key}"
            type={f.secret ? 'password' : 'text'}
            class={f.secret ? '' : 'mono-input'}
            bind:value={values[f.key]}
            placeholder={f.placeholder ?? undefined}
            autocomplete="off"
            spellcheck={false}
            aria-required={f.required || undefined}
            invalid={!!fieldErrors[f.key]}
            oninput={() => delete fieldErrors[f.key]}
          />
        </Field>
      {/each}
      <Field label="Name" for="{uid}-name" hint="How Settings and New run list this connection.">
        <Input id="{uid}-name" bind:value={name} placeholder={provider.name} autocomplete="off" spellcheck={false} />
      </Field>
      {#if failure}<ErrorState error={failure} title="Could not connect" compact />{/if}
    </form>
  {/if}
  {#snippet footer()}
    <Button variant="ghost" onclick={() => (open = false)}>Cancel</Button>
    <Button variant="primary" type="submit" form="{uid}-form" icon={PlugZap} loading={busy}>Connect</Button>
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
  .about {
    max-inline-size: var(--measure);
    color: var(--text-secondary);
  }
  /* A URL is an identifier (mono); its example is a sentence's placeholder, so it stays in Geist. */
  .form :global(.mono-input input) {
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
  }
  .form :global(.mono-input input::placeholder) {
    font-family: var(--font-sans);
    font-size: var(--type-base-size);
  }
  /* Touch: the typed text at 16 px, as every field's (ui/Input), so iOS never zooms into it. */
  @media (pointer: coarse) {
    .form :global(.mono-input input),
    .form :global(.mono-input input::placeholder) {
      font-size: var(--type-lg-size);
    }
  }
  .docs {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .docs:hover {
    text-decoration-color: var(--text-secondary);
  }
  /* Touch: the link is a 44 px target in the label's row. */
  @media (pointer: coarse) {
    .docs {
      min-height: 44px;
    }
  }
</style>
