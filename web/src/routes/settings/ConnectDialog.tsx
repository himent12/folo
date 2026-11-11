// fw port of ConnectDialog.svelte.
/**
 * Settings › Cloud › Connect: a provider and its credentials, the form built from `GET /api/cloud/providers` (one
 * field per `fields` entry, secrets masked), checked with the provider by the server before anything is stored.
 * Secrets live in this form only while it is open: never in preferences, a log, a toast or an error line, and nothing
 * of one provider's survives switching to the other.
 */
import ExternalLink from '$icons/external-link';
import PlugZap from '$icons/plug-zap';
import { bindable, computed, createUniqueId, css, effect, For, signal, untrack, type Signal } from 'fw';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import type { CloudConnection, CloudField, CloudProvider } from '$lib/api/types';
import { sentence } from '$lib/format';
import Button from '$lib/ui/Button';
import CodeText from '$lib/ui/CodeText';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import Field from '$lib/ui/Field.tsx';
import Input from '$lib/ui/Input';
import Segmented from '$lib/ui/Segmented';
import { toast } from '$lib/ui/toast';

export interface ConnectDialogProps {
  open?: boolean;
  providers: CloudProvider[];
  /** The new connection, once the dialog has closed. */
  onconnected: (c: CloudConnection) => void;
}

/** A field's label inside a sentence: "the server URL", "the API key" (an acronym keeps its case). */
const inSentence = (f: CloudField) => (/^[A-Z]{2}/.test(f.label) ? f.label : f.label.charAt(0).toLowerCase() + f.label.slice(1));
/** What a credential is called after "this": "API key" → "key", "Token" → "token". */
const noun = (f: CloudField) => f.label.toLowerCase().replace(/^api /, '');

export default function ConnectDialog(props: ConnectDialogProps) {
  const open = bindable(props, 'open', false);
  const uid = createUniqueId();
  const providerId = signal('');
  const values = signal<Record<string, string>>({});
  const name = signal('');
  const busy = signal(false);
  /** Errors at their fields (by key, `name` for the name), from the check on submit or the server's answer. */
  const fieldErrors = signal<Record<string, string>>({});
  /** Anything the server said that no field is about. */
  const failure = signal<unknown>(undefined);

  const provider = computed(() => props.providers.find((p) => p.id === providerId.get()));
  effect(() => {
    const first = props.providers[0];
    if (!provider.get() && first) providerId.set(first.id);
  });
  /** The first secret field: where a rejected credential is said, and where the provider's console is linked. */
  const firstSecret = computed(() => provider.get()?.fields.find((f) => f.secret));

  /** A field's typed value, as `bind:value={values[f.key]}` was. */
  const valueOf = (key: string): Signal<string | undefined> => ({
    get: () => values.get()[key],
    peek: () => values.peek()[key],
    set: (v) => {
      const next = typeof v === 'function' ? v(values.peek()[key]) : v;
      values.set({ ...values.peek(), [key]: next as string });
    },
  });

  /** Back to an empty form: every value dropped, the secrets with them. */
  function reset() {
    values.set({});
    name.set('');
    fieldErrors.set({});
    failure.set(undefined);
    busy.set(false);
  }
  effect(() => {
    if (!open.get()) untrack(reset);
  });

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    const p = provider.peek();
    if (!p || busy.peek()) return;
    failure.set(undefined);
    const typed = values.peek();
    const missing = Object.fromEntries(p.fields.filter((f) => f.required && !typed[f.key]?.trim()).map((f) => [f.key, `Enter the ${inSentence(f)}.`]));
    fieldErrors.set(missing);
    if (Object.keys(missing).length) return;
    busy.set(true);
    try {
      const conn = await hub.api.connectCloud({
        provider: p.id,
        name: name.peek().trim(),
        values: Object.fromEntries(p.fields.map((f) => [f.key, typed[f.key]?.trim() ?? ''])),
      });
      // The dialog closes first: the new row says the rest.
      open.set(false);
      toast.success(`Connected ${p.name}`);
      props.onconnected(conn);
    } catch (err) {
      const field = err instanceof ApiError && err.code === 'invalid_body' ? p.fields.find((f) => err.message.includes(`\`${f.key}\``)) : undefined;
      const secret = firstSecret.peek();
      if (err instanceof ApiError && err.code === 'cloud_auth_failed' && secret) fieldErrors.set({ [secret.key]: `${p.name} rejected this ${noun(secret)}.` });
      else if (field && err instanceof ApiError) fieldErrors.set({ [field.key]: sentence(err.message, true) });
      else failure.set(err);
    } finally {
      busy.set(false);
    }
  }

  return (
    <Dialog
      bind:open={open}
      title="Connect a cloud"
      description="Train on a rented GPU, or on another FOLO server of yours."
      footer={() => (
        <>
          <Button variant="ghost" onclick={() => open.set(false)}>
            Cancel
          </Button>
          <Button variant="primary" type="submit" form={`${uid}-form`} icon={PlugZap} loading={busy.get()}>
            Connect
          </Button>
        </>
      )}
    >
      {provider.get() ? (
        <form id={`${uid}-form`} class="form" onsubmit={submit} novalidate autocomplete="off">
          {props.providers.length > 1 ? (
            <Field label="Provider" hint={provider.get()?.about}>
              {/* Another provider starts an empty form: nothing typed for one is kept for the other. */}
              <Segmented label="Provider" bind:value={providerId} onchange={reset} options={props.providers.map((p) => ({ value: p.id, label: p.name }))} />
            </Field>
          ) : (
            <p class="about">
              <CodeText text={provider.get()?.about ?? ''} />
            </p>
          )}
          <For each={provider.get()?.fields ?? []} key={(f) => `${provider.peek()?.id}/${f.key}`}>
            {(f) => (
              <Field
                label={f().label}
                for={`${uid}-${f().key}`}
                hint={f().help ?? undefined}
                error={fieldErrors.get()[f().key] ?? null}
                aside={() => (
                  <>
                    {f() === firstSecret.get() && provider.get()?.docs_url ? (
                      <a class="docs" href={provider.get()?.docs_url ?? undefined} target="_blank" rel="noreferrer">
                        Where to find it
                        <ExternalLink size={12} strokeWidth={1.75} aria-hidden="true" />
                      </a>
                    ) : null}
                  </>
                )}
              >
                <Input
                  id={`${uid}-${f().key}`}
                  type={f().secret ? 'password' : 'text'}
                  class={f().secret ? '' : 'mono-input'}
                  bind:value={valueOf(f().key)}
                  placeholder={f().placeholder ?? undefined}
                  autocomplete="off"
                  spellcheck={false}
                  aria-required={f().required || undefined}
                  invalid={!!fieldErrors.get()[f().key]}
                  oninput={() => {
                    const key = f().key;
                    const errs = fieldErrors.peek();
                    if (!(key in errs)) return;
                    const next = { ...errs };
                    delete next[key];
                    fieldErrors.set(next);
                  }}
                />
              </Field>
            )}
          </For>
          <Field label="Name" for={`${uid}-name`} hint="How Settings and New run list this connection.">
            <Input id={`${uid}-name`} bind:value={name} placeholder={provider.get()?.name} autocomplete="off" spellcheck={false} />
          </Field>
          {failure.get() ? <ErrorState error={failure.get()} title="Could not connect" compact /> : null}
        </form>
      ) : null}
    </Dialog>
  );
}

css`
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
`;
