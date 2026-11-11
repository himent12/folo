/** Settings › Connection: whether Studio talks to a server or shows demo data, and the server's address (test, save). */
import Check from '$icons/check';
import CircleCheck from '$icons/circle-check';
import FlaskConical from '$icons/flask-conical';
import PlugZap from '$icons/plug-zap';
import { computed, css, signal } from 'fw';
import { HttpApi } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { prefs } from '$lib/prefs';
import Button from '$lib/ui/Button';
import Field from '$lib/ui/Field.tsx';
import Input from '$lib/ui/Input';
import Panel from '$lib/ui/Panel';
import { toast } from '$lib/ui/toast';

type Test = { ok: true; version: string; mock: boolean; ms: number } | { ok: false; message: string };

export default function ConnectionPanel() {
  // ---- connection ----
  const origin = typeof location === 'undefined' ? '' : location.origin;
  const url = signal(prefs.serverUrl);
  const testing = signal(false);
  const test = signal<Test | null>(null);
  /** The last test, by outcome. */
  const passed = computed(() => {
    const t = test.get();
    return t?.ok ? t : null;
  });
  const failed = computed(() => {
    const t = test.get();
    return t && !t.ok ? t : null;
  });
  const forced =typeof location !== 'undefined' && new URLSearchParams(location.search).get('mock') === '1';

  const normalized = computed(() => url.get().trim().replace(/\/+$/, ''));
  const invalid = computed(() => normalized.get() !== '' && !/^https?:\/\/[^\s/]+/i.test(normalized.get()));
  const dirty = computed(() => normalized.get() !== prefs.serverUrl);

  async function runTest() {
    if (invalid.peek()) return;
    testing.set(true);
    test.set(null);
    const t0 = performance.now();
    try {
      const h = await new HttpApi(normalized.peek()).health(AbortSignal.timeout(3000));
      test.set({ ok: true, version: h.version, mock: h.mock, ms: performance.now() - t0 });
    } catch (e) {
      const timeout = e instanceof DOMException && (e.name === 'TimeoutError' || e.name === 'AbortError');
      test.set({ ok: false, message: timeout ? 'No answer within 3 s.' : e instanceof Error ? e.message : String(e) });
    } finally {
      testing.set(false);
    }
  }

  async function save() {
    if (invalid.peek()) return;
    prefs.serverUrl = normalized.peek();
    url.set(prefs.serverUrl);
    await hub.connect();
    if (hub.mode === 'live') toast.success('Connected', `folo serve ${hub.version} at ${hub.serverUrl}`);
    else if (forced) toast.info('Saved', 'Demo data stays on while the address has ?mock=1.');
    else toast.error(new Error(`Could not reach ${hub.serverUrl}; showing demo data.`), 'Saved, but not connected');
  }

  return (
    <Panel title="Connection" id="connection" divided>
      <div class="stack">
        <div class="status" data-mode={hub.mode}>
          {/* Demo data is a mode, not a degraded state: the sidebar's neutral flask, never the warning hue. A server
              that answers gets the mark and the word Settings > Cloud gives a working connection (a check in a
              circle, "Connected"), not a ringed dot and "Live". */}
          {hub.mode === 'mock' ? (
            <FlaskConical class="led-glyph" size={16} strokeWidth={1.75} aria-hidden="true" />
          ) : hub.mode === 'live' ? (
            <CircleCheck class="led-glyph ok" size={16} strokeWidth={1.75} aria-hidden="true" />
          ) : (
            <span class="led" aria-hidden="true"></span>
          )}
          <div class="status-text">
            {hub.mode === 'live' ? (
              <>
                <p class="s-title">
                  Connected <span class="mono faint">folo serve {hub.version}</span>
                </p>
                <p class="s-desc">
                  <span class="mono">{hub.serverUrl}</span>
                </p>
              </>
            ) : hub.mode === 'mock' ? (
              <>
                {/* What it means for the user, in one line; why (the address, the endpoint) is in Technical details. */}
                <p class="s-title">Using demo data</p>
                <p class="s-desc">
                  {hub.reason === 'forced' ? (
                    'Sample runs, datasets and models.'
                  ) : (
                    <>
                      No server answered. Start <span class="mono">folo serve</span> or enter its address below.
                    </>
                  )}
                </p>
              </>
            ) : (
              <>
                <p class="s-title">Connecting…</p>
                <p class="s-desc">
                  Looking for <span class="mono">{hub.serverUrl}</span>.
                </p>
              </>
            )}
          </div>
        </div>

        <Field label="Server URL" for="server-url" hint="Leave blank to use this server." error={invalid.get() ? 'Use a full address like http://127.0.0.1:7878.' : null}>
          <div class="url-row">
            <Input
              id="server-url"
              bind:value={url}
              placeholder="Same as this page"
              spellcheck={false}
              autocomplete="off"
              invalid={invalid.get()}
              class="mono-input"
              onkeydown={(e: KeyboardEvent) => {
                if (e.key === 'Enter') void runTest();
              }}
            />
            <Button icon={PlugZap} loading={testing.get()} disabled={invalid.get()} onclick={runTest}>
              Test
            </Button>
          </div>
        </Field>

        {test.get() ? (
          <div class="test" class:ok={!!passed.get()} role="status">
            {passed.get() ? (
              <>
                <Check size={14} strokeWidth={2.25} aria-hidden="true" />
                <span>
                  Reachable: <span class="mono">folo serve {passed.get()?.version}</span>
                  {passed.get()?.mock ? ' (mock mode)' : ''}
                </span>
                <span class="num faint">{Math.round(passed.get()?.ms ?? 0)} ms</span>
              </>
            ) : (
              <>
                <span class="x" aria-hidden="true">
                  ×
                </span>
                <span>{failed.get()?.message}</span>
              </>
            )}
          </div>
        ) : null}

        <details class="tech">
          <summary>Technical details</summary>
          <dl>
            <dt>Requests go to</dt>
            <dd class="mono">{hub.serverUrl}/api</dd>
            {hub.mode === 'mock' ? (
              <>
                <dt>Demo data</dt>
                <dd>
                  {hub.reason === 'forced' ? (
                    <>
                      On because the address has <span class="mono">?mock=1</span>; remove it to connect to a server.
                    </>
                  ) : (
                    <>
                      Shown because <span class="mono">{hub.serverUrl}</span> did not answer.
                    </>
                  )}
                </dd>
              </>
            ) : null}
            <dt>A blank URL</dt>
            <dd>
              Uses the address this page was served from
              {origin ? (
                <>
                  {' '}(<span class="mono">{origin}</span>)
                </>
              ) : null}
              .
            </dd>
          </dl>
        </details>

        <div class="row-actions">
          {/* The primary action only once there is something to save; reconnecting as is stays secondary. */}
          <Button variant={dirty.get() ? 'primary' : 'secondary'} disabled={invalid.get()} onclick={save}>
            {dirty.get() ? 'Save and reconnect' : 'Reconnect'}
          </Button>
        </div>
      </div>
    </Panel>
  );
}

css`
  .stack {
    display: grid;
    gap: var(--space-4);
  }
  .status {
    display: flex;
    gap: var(--space-3);
    padding: var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    border: 1px solid var(--line-subtle);
  }
  :global([data-theme='light']) .status {
    background: var(--bg-raised);
  }
  .led {
    flex-shrink: 0;
    width: 8px;
    height: 8px;
    margin-top: 6px;
    border-radius: 50%;
    background: var(--text-tertiary);
  }
  .status :global(.led-glyph) {
    flex-shrink: 0;
    margin-top: 2px;
    color: var(--text-tertiary);
  }
  .status :global(.led-glyph.ok) {
    color: var(--success-text);
  }
  .status-text {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .s-title {
    font-weight: var(--weight-medium);
    display: flex;
    gap: var(--space-2);
    align-items: baseline;
  }
  .s-desc {
    color: var(--text-secondary);
    overflow-wrap: anywhere;
  }
  .url-row {
    display: flex;
    gap: var(--space-2);
  }
  .url-row :global(.input) {
    flex: 1;
  }
  /* The field shrinks to the row, never the row to the field's default width (a 320 px phone held it 4 px wide). */
  .url-row :global(.input input) {
    width: 0;
  }
  /* The typed URL is an identifier (mono); the placeholder is a sentence, so it stays in Geist. */
  .url-row :global(.mono-input input) {
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
  }
  .url-row :global(.mono-input input::placeholder) {
    font-family: var(--font-sans);
    font-size: var(--type-base-size);
  }
  .test {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--danger-text);
  }
  .test.ok {
    color: var(--success-text);
  }
  .test.ok > span:not(.faint) {
    color: var(--text-primary);
  }
  .test .faint {
    margin-left: auto;
  }
  .x {
    font-weight: var(--weight-semibold);
  }
  /* The mechanics (endpoint, why demo data is on), folded: the first screen says what it means, not how. */
  .tech {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .tech summary {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    width: max-content;
    cursor: pointer;
    list-style: none;
  }
  .tech summary::-webkit-details-marker {
    display: none;
  }
  .tech summary::before {
    content: '';
    width: 5px;
    height: 5px;
    margin: 0 2px;
    border-right: 1.5px solid currentColor;
    border-bottom: 1.5px solid currentColor;
    transform: rotate(-45deg);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .tech[open] summary::before {
    transform: rotate(45deg);
  }
  .tech summary:hover {
    color: var(--text-secondary);
  }
  .tech dl {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: var(--space-1) var(--space-4);
    margin-top: var(--space-2);
  }
  .tech dd {
    color: var(--text-secondary);
    overflow-wrap: anywhere;
  }
  @media (pointer: coarse) {
    .tech summary {
      min-height: 44px;
    }
  }
  @media (max-width: 767px) {
    .tech dl {
      grid-template-columns: minmax(0, 1fr);
      gap: 0;
    }
    .tech dd + dt {
      margin-top: var(--space-2);
    }
  }
  .row-actions {
    display: flex;
    justify-content: flex-end;
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
  }
  @media (max-width: 767px) {
    .row-actions :global(.btn) {
      width: 100%;
    }
  }
`;
