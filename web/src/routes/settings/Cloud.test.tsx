// @vitest-environment happy-dom
// Settings › Cloud: the connections (account, balance, the last check), the instances FOLO launched, and the Connect,
// Disconnect and Terminate flows with the errors the API documents (docs/API.md, Cloud compute).
import { flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { Resource } from '$lib/api/resource';
import { watched } from '$lib/shell/runList';
import { toast } from '$lib/ui/toast';
import { createRoot } from 'fw';
import { render } from '../../../framework/testing';
import Cloud from './Cloud';

/** Port of `mount(Cloud, { target })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
const mount = (_: typeof Cloud, o: { target: HTMLElement }) => render(() => <Cloud />, o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

async function until<T>(get: () => T | null | undefined | false, what = 'the condition'): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 3000, interval: 10 },
  );
  return v as T;
}

beforeEach(() => {
  hub.mode = 'mock';
  toast.items = [];
  target = document.createElement('div');
  document.body.appendChild(target);
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

function open(api: MockApi) {
  hub.api = api;
  c = mount(Cloud, { target });
}

const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, ' ').trim() ?? '';
const button = (root: ParentNode, name: string) => [...root.querySelectorAll<HTMLButtonElement>('button')].find((b) => text(b) === name);
const dialog = () => document.querySelector<HTMLDialogElement>('dialog[open]');
function type(input: HTMLInputElement, value: string) {
  input.value = value;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
}

describe('Settings › Cloud', () => {
  it('lists the connection with its account, balance and state, and the instances, an ended one quiet', async () => {
    open(new MockApi('normal', 0));
    const row = await until(() => target.querySelector('.cx-row .cx-name'), 'the connection');
    const li = row.closest('li')!;
    expect(text(row)).toBe('RunPod');
    expect([...li.querySelectorAll('.cx-meta .ui-meta > span')].map(text)).toEqual(['you@example.com', 'Balance $42.17', 'checked 3m ago']);
    expect(button(li, 'Check')).toBeDefined();
    expect(button(li, 'Disconnect')).toBeDefined();
    const inst = await until(() => target.querySelector('table.inst tbody tr'), 'the instances');
    expect(inst.classList.contains('ended')).toBe(true);
    expect(text(inst.querySelector('.run a'))).toBe('shapes-m');
    expect(text(inst.querySelector('.st'))).toBe('Terminated');
    expect(button(inst, 'Terminate')).toBeUndefined();
  });

  it('says there is no cloud, with one way to connect, and lists no instances', async () => {
    open(new MockApi('cloud-none', 0));
    await until(() => text(target.querySelector('#cloud')).includes('No cloud connected'), 'the empty state');
    expect([...target.querySelectorAll('#cloud button')].filter((b) => text(b) === 'Connect')).toHaveLength(1);
    expect(target.querySelector('#cloud-instances')).toBeNull();
  });

  it('shows a failed check in words, and a termination that is not confirmed as a problem', async () => {
    open(new MockApi('cloud-trouble', 0));
    const err = await until(() => target.querySelector('.cx-error'), 'the error');
    expect(text(err)).toContain('RunPod rejected this API key');
    expect(button(err.closest('li')!, 'Check again')).toBeDefined();
    const callout = await until(() => target.querySelector('#cloud-instances .callout'), 'the callout');
    expect(callout.getAttribute('role')).toBe('alert');
    // The instance by its name and its provider's id, to look up in the provider's console.
    expect([...callout.querySelectorAll('.ident')].map(text)).toEqual(['folo-run_c3d5-4c1a', 'k2x9wq7m3fz1']);
    const row = target.querySelector('table.inst tbody tr')!;
    expect(text(row.querySelector('.st'))).toBe('Terminating');
    expect(button(row, 'Terminate')).toBeDefined();
  });

  it('checks a connection again, busy while it runs, then shows the refreshed row', async () => {
    const api = new MockApi('normal', 0);
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const real = api.checkCloudConnection.bind(api);
    vi.spyOn(api, 'checkCloudConnection').mockImplementation(async (id) => {
      await gate;
      return { ...(await real(id)), balance_usd: 40.5 };
    });
    open(api);
    const li = (await until(() => target.querySelector('.cx-row .cx-name'), 'the connection')).closest('li')!;
    button(li, 'Check')!.click();
    flushSync();
    expect(button(li, 'Check')!.getAttribute('aria-busy')).toBe('true');
    release();
    await until(() => text(li).includes('Balance $40.50'), 'the refreshed row');
  });

  it('says a rejected key at its field, never anywhere else, and keeps nothing of it', async () => {
    open(new MockApi('normal', 0));
    await until(() => target.querySelector('#cloud header .btn'), 'Connect');
    button(target, 'Connect')!.click();
    const d = await until(() => dialog()?.querySelector('input[type="password"]') && dialog(), 'the dialog');
    const key = d.querySelector<HTMLInputElement>('input[type="password"]')!;
    // The key is what to type first: it has focus on open; the worker URL is marked optional (shell F18).
    await until(() => document.activeElement === key, 'focus on the key');
    expect([...d.querySelectorAll('label')].map(text)).toEqual(expect.arrayContaining(['API key', 'Worker download URL Optional']));
    key.blur();
    // Required fields are checked on submit, the message naming the field, and focus goes to it.
    d.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await until(() => text(d.querySelector('.msg.error')) === 'Enter the API key.', 'the required message');
    await until(() => document.activeElement === key, 'focus on the field in error');
    expect(key.getAttribute('autocomplete')).toBe('off');
    type(key, 'rpa_bad_SECRET_0042');
    d.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    const msg = await until(() => d.querySelector(`#${key.id}-error`), 'the field error');
    expect(text(msg)).toBe('RunPod rejected this key.');
    expect(key.getAttribute('aria-invalid')).toBe('true');
    expect(toast.items).toEqual([]);
    expect(document.body.textContent).not.toContain('SECRET_0042');
    expect(JSON.stringify(localStorage)).not.toContain('SECRET_0042');
  });

  it('connects: the dialog closes, a toast says so, the list has the new row', async () => {
    open(new MockApi('normal', 0));
    await until(() => target.querySelector('#cloud header .btn'), 'Connect');
    button(target, 'Connect')!.click();
    const d = await until(() => dialog()?.querySelector('input[type="password"]') && dialog(), 'the dialog');
    type(d.querySelector<HTMLInputElement>('input[type="password"]')!, 'rpa_live_key_7f3a');
    d.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await until(() => toast.items.length, 'the toast');
    expect(toast.items[0]).toMatchObject({ tone: 'success', title: 'Connected RunPod' });
    expect(dialog()).toBeNull();
    await until(() => [...target.querySelectorAll('.cx-name')].map(text).includes('RunPod 2'), 'the new row');
    expect(document.body.textContent).not.toContain('rpa_live_key_7f3a');
  });

  it('keeps nothing of one provider’s secret when the other is chosen', async () => {
    open(new MockApi('normal', 0));
    await until(() => target.querySelector('#cloud header .btn'), 'Connect');
    button(target, 'Connect')!.click();
    const d = await until(() => dialog()?.querySelector('input[type="password"]') && dialog(), 'the dialog');
    type(d.querySelector<HTMLInputElement>('input[type="password"]')!, 'rpa_typed_first');
    [...d.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((r) => text(r) === 'Remote FOLO server')!.click();
    flushSync();
    expect([...d.querySelectorAll('label')].map(text)).toEqual(expect.arrayContaining(['Server URL', 'Token Optional', 'Name Optional']));
    [...d.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((r) => text(r) === 'RunPod')!.click();
    flushSync();
    expect(d.querySelector<HTMLInputElement>('input[type="password"]')!.value).toBe('');
  });

  it('says what a server wants of a token left empty, not that it rejected one', async () => {
    const api = new MockApi('normal', 0);
    open(api);
    vi.spyOn(api, 'connectCloud').mockRejectedValue(new ApiError(422, 'cloud_auth_failed', 'the server at 10.0.0.5 needs a token (it runs with `--token`)'));
    await until(() => target.querySelector('#cloud header .btn'), 'Connect');
    button(target, 'Connect')!.click();
    const d = await until(() => dialog()?.querySelector('input[type="password"]') && dialog(), 'the dialog');
    [...d.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((r) => text(r) === 'Remote FOLO server')!.click();
    flushSync();
    type(d.querySelector<HTMLInputElement>('input[type="text"]')!, 'http://10.0.0.5:7878');
    const token = d.querySelector<HTMLInputElement>('input[type="password"]')!;
    d.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    const msg = await until(() => d.querySelector(`#${token.id}-error`), 'the field error');
    expect(text(msg)).toBe('The server at 10.0.0.5 needs a token (it runs with --token).');
    // A token that was typed is the one rejected.
    vi.mocked(api.connectCloud).mockRejectedValue(new ApiError(422, 'cloud_auth_failed', 'the server at 10.0.0.5 rejected the token'));
    type(token, 'wrong');
    d.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await until(() => text(d.querySelector(`#${token.id}-error`)) === 'Remote FOLO server rejected this token.', 'the rejection');
  });

  it('confirms Disconnect, says why a connection in use stays, and forgets one that is not', async () => {
    open(new MockApi('cloud-training', 0));
    const li = (await until(() => target.querySelector('.cx-row .cx-name'), 'the connection')).closest('li')!;
    button(li, 'Disconnect')!.click();
    const d = await until(() => dialog(), 'the dialog');
    button(d.querySelector('footer')!, 'Disconnect')!.click();
    const err = await until(() => d.querySelector('[role="alert"] [data-notice]'), 'the refusal');
    expect(text(err)).toContain('RunPod has a live instance, folo-run_c7b1-9d03');
    expect(toast.items).toEqual([]);
  });

  it('confirms Terminate with what happens to the run, then shows the instance terminating', async () => {
    open(new MockApi('cloud-training', 0));
    const row = await until(() => [...target.querySelectorAll('table.inst tbody tr')].find((r) => button(r, 'Terminate')), 'the live instance');
    expect(text(row.querySelector('.st'))).toBe('Running');
    expect(text(row)).toContain('stops by');
    button(row, 'Terminate')!.click();
    const d = await until(() => dialog(), 'the dialog');
    // What happens to the run is said before it does, without falsely claiming immediate shutdown.
    const desc = text(d.querySelector('header p'));
    expect(desc).toContain('fasteners-l');
    expect(desc).toContain('Billing continues until RunPod confirms it is stopped');
    expect(desc).not.toContain('is shut down now');
    await until(() => document.activeElement?.hasAttribute('data-safe'), 'the safe button focused');
    button(d.querySelector('footer')!, 'Terminate')!.click();
    await until(() => toast.items.length, 'the toast');
    expect(toast.items[0]).toMatchObject({ tone: 'info', title: 'Terminating the RTX 4090' });
    await until(() => text(target.querySelector('table.inst tbody tr .st')) === 'Terminating', 'the new state');
  });

  it('frames Terminate on an already-terminating instance as a retry where billing may continue', async () => {
    open(new MockApi('cloud-trouble', 0));
    const row = await until(() => target.querySelector('table.inst tbody tr'), 'the unconfirmed instance');
    expect(text(row.querySelector('.st'))).toBe('Terminating');
    button(row, 'Terminate')!.click();
    const d = await until(() => dialog(), 'the retry terminate dialog');
    expect(text(d.querySelector('header h2'))).toBe('Retry termination?');
    const desc = text(d.querySelector('header p'));
    expect(desc).toContain('asks RunPod again to terminate the RTX 4090');
    expect(desc).toContain('Billing may continue until RunPod confirms it is stopped');
    expect(desc).not.toContain('is shut down now');
    expect(button(d.querySelector('footer')!, 'Cancel')).toBeDefined();
  });

  // s6-shell review (round 1): providers of the wrong shape left "No cloud connected" with a disabled Connect and no word.
  it('says why Connect waits when the providers cannot be read, and Retry reads them again', async () => {
    const api = new MockApi('normal', 0);
    const real = api.cloudProviders.bind(api);
    let fail = true;
    api.cloudProviders = () =>
      fail ? Promise.reject(new ApiError(200, 'bad_response', 'The server sent an answer of an unexpected shape (no list of providers).')) : real();
    api.cloudConnections = async () => [];
    open(api);
    const err = await until(() => [...target.querySelectorAll('.state')].find((e) => text(e).includes('Could not load the cloud providers')), 'the error');
    expect(text(err)).toContain('unexpected shape');
    expect(button(target, 'Connect')!.disabled).toBe(true);
    fail = false;
    button(err, 'Retry')!.click();
    await until(() => !text(target).includes('Could not load the cloud providers'), 'the error to go');
    await until(() => button(target, 'Connect') && !button(target, 'Connect')!.disabled, 'Connect enabled');
  });

  it('tracks Check busy state independently per connection when two checks run concurrently', async () => {
    const api = new MockApi('normal', 0);
    const list = await api.cloudConnections();
    const first = list[0]!;
    const second = { ...first, id: 'cc_second', name: 'Remote Box', provider: 'remote' };
    api.cloudConnections = async () => [first, second];
    const gates = new Map<string, () => void>();
    vi.spyOn(api, 'checkCloudConnection').mockImplementation(
      (id) =>
        new Promise((resolve) => {
          gates.set(id, () => resolve(id === first.id ? { ...first, balance_usd: 11 } : { ...second, balance_usd: 22 }));
        }),
    );
    open(api);
    await until(() => target.querySelectorAll('.cx-row .cx-name').length === 2, 'two connections');
    const rows = [...target.querySelectorAll<HTMLElement>('.cx-row')];
    const btnA = button(rows[0]!, 'Check')!;
    const btnB = button(rows[1]!, 'Check')!;
    btnA.click();
    btnB.click();
    flushSync();
    expect(btnA.getAttribute('aria-busy')).toBe('true');
    expect(btnB.getAttribute('aria-busy')).toBe('true');
    // First check finishes: second button must remain busy until its own check finishes.
    gates.get(first.id)!();
    await until(() => btnA.getAttribute('aria-busy') !== 'true', 'first check finished');
    expect(btnB.getAttribute('aria-busy')).toBe('true');
    gates.get(second.id)!();
    await until(() => btnB.getAttribute('aria-busy') !== 'true', 'second check finished');
  });

  it('ignores a stale ConnectDialog submission after switching providers or closing the dialog', async () => {
    const api = new MockApi('normal', 0);
    let rejectConnect!: (e: Error) => void;
    vi.spyOn(api, 'connectCloud').mockImplementation(
      () =>
        new Promise((_, no) => {
          rejectConnect = no;
        }),
    );
    open(api);
    await until(() => target.querySelector('#cloud header .btn'), 'Connect');
    button(target, 'Connect')!.click();
    const d = await until(() => dialog()?.querySelector('input[type="password"]') && dialog(), 'the dialog');
    type(d.querySelector<HTMLInputElement>('input[type="password"]')!, 'rpa_in_flight');
    d.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    flushSync();
    // Switch provider while RunPod connectCloud is still in flight.
    [...d.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((r) => text(r) === 'Remote FOLO server')!.click();
    flushSync();
    // Late rejection from the superseded RunPod submit must not pollute Remote FOLO server's form.
    rejectConnect(new ApiError(422, 'cloud_auth_failed', 'bad runpod key'));
    await Promise.resolve();
    await Promise.resolve();
    flushSync();
    expect(d.querySelector('.msg.error')).toBeNull();
    expect(d.querySelector('.err')).toBeNull();
  });

  it('reconciles a committed ConnectDialog POST even after switching providers without closing the replacement form', async () => {
    const api = new MockApi('normal', 0);
    const realConnect = api.connectCloud.bind(api);
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    vi.spyOn(api, 'connectCloud').mockImplementation(async (req) => {
      await gate;
      return realConnect(req);
    });
    open(api);
    await until(() => target.querySelector('#cloud header .btn'), 'Connect');
    button(target, 'Connect')!.click();
    const d = await until(() => dialog()?.querySelector('input[type="password"]') && dialog(), 'the dialog');
    type(d.querySelector<HTMLInputElement>('input[type="password"]')!, 'rpa_deferred_ok');
    d.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    flushSync();
    // Switch provider while RunPod POST is still in flight.
    [...d.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((r) => text(r) === 'Remote FOLO server')!.click();
    flushSync();
    const urlInput = d.querySelector<HTMLInputElement>('input[type="text"]')!;
    type(urlInput, 'http://10.0.0.9:7878');

    // RunPod POST succeeds on the current server: new row appears and toast fires, while Remote FOLO dialog stays open.
    release();
    await until(() => [...target.querySelectorAll('.cx-name')].map(text).includes('RunPod 2'), 'committed connection row');
    expect(toast.items[0]).toMatchObject({ tone: 'success', title: 'Connected RunPod' });
    expect(dialog()).toBe(d);
    expect(urlInput.value).toBe('http://10.0.0.9:7878');
  });

  it('isolates Disconnect and Terminate confirmation sessions when dialogs switch while a mutation is in flight', async () => {
    const api = new MockApi('normal', 0);
    const base = (await api.cloudConnections())[0]!;
    const connB = { ...base, id: 'cc_b', name: 'Server B', provider: 'remote' };
    const connC = { ...base, id: 'cc_c', name: 'Server C', provider: 'remote' };
    api.cloudConnections = async () => [base, connB, connC];
    const discGates = new Map<string, { ok: () => void; fail: (e: Error) => void }>();
    vi.spyOn(api, 'disconnectCloud').mockImplementation(
      (id) =>
        new Promise<void>((resolve, reject) => {
          discGates.set(id, { ok: () => resolve(), fail: reject });
        }),
    );

    const instList = await api.cloudInstances();
    const instA = { ...instList[0]!, id: 'ci_a', name: 'pod-a', status: 'running' as const };
    const instB = { ...instList[0]!, id: 'ci_b', name: 'pod-b', status: 'running' as const };
    api.cloudInstances = async () => [instA, instB];
    const termGates = new Map<string, { ok: () => void; fail: (e: Error) => void }>();
    vi.spyOn(api, 'terminateCloudInstance').mockImplementation(
      (id) =>
        new Promise((resolve, reject) => {
          termGates.set(id, {
            ok: () => resolve({ ...(id === instA.id ? instA : instB), status: 'terminating' }),
            fail: reject,
          });
        }),
    );

    open(api);
    await until(() => target.querySelectorAll('.cx-row .cx-name').length === 3, 'three connections');

    // Start disconnecting RunPod (base), dismiss dialog, then open Disconnect for Server B.
    const rows = () => [...target.querySelectorAll<HTMLElement>('.cx-row')];
    button(rows()[0]!, 'Disconnect')!.click();
    let d = await until(() => dialog(), 'disconnect dialog A');
    button(d.querySelector('footer')!, 'Disconnect')!.click();
    flushSync();
    button(d.querySelector('footer')!, 'Cancel')!.click();
    flushSync();

    button(rows()[1]!, 'Disconnect')!.click();
    d = await until(() => dialog(), 'disconnect dialog B');
    expect(text(d.querySelector('header'))).toContain('Disconnect Server B?');

    // RunPod disconnect succeeds: removes RunPod from the list without closing Server B's dialog.
    discGates.get(base.id)!.ok();
    await until(() => ![...target.querySelectorAll('.cx-name')].map(text).includes('RunPod'), 'RunPod removed');
    expect(dialog()).toBe(d);
    expect(text(d.querySelector('header'))).toContain('Disconnect Server B?');

    // Start disconnecting Server B, dismiss, open Server C, then fail Server B -> error must NOT appear in Server C's dialog.
    button(d.querySelector('footer')!, 'Disconnect')!.click();
    flushSync();
    button(d.querySelector('footer')!, 'Cancel')!.click();
    flushSync();
    button(rows().find((r) => text(r).includes('Server C'))!, 'Disconnect')!.click();
    d = await until(() => dialog(), 'disconnect dialog C');
    discGates.get(connB.id)!.fail(new ApiError(409, 'connection_in_use', 'Server B busy'));
    await Promise.resolve();
    await Promise.resolve();
    flushSync();
    expect(d.querySelector('[role="alert"] [data-notice]')).toBeNull();
    // The dismissed Server B failure is still surfaced via a page toast instead of being silently swallowed.
    expect(toast.items.some((t) => t.tone === 'error' && t.title === 'Could not disconnect Server B')).toBe(true);
    button(d.querySelector('footer')!, 'Cancel')!.click();
    flushSync();

    // Same session isolation for Terminate on two live instances.
    await until(() => target.querySelectorAll('table.inst tbody tr').length === 2, 'two live instances');
    const instRows = () => [...target.querySelectorAll<HTMLElement>('table.inst tbody tr')];
    button(instRows()[0]!, 'Terminate')!.click();
    d = await until(() => dialog(), 'terminate dialog A');
    button(d.querySelector('footer')!, 'Terminate')!.click();
    flushSync();
    button(d.querySelector('footer')!, 'Keep it running')!.click();
    flushSync();
    button(instRows()[1]!, 'Terminate')!.click();
    d = await until(() => dialog(), 'terminate dialog B');
    termGates.get(instA.id)!.ok();
    await until(() => text(instRows()[0]!.querySelector('.st')) === 'Terminating', 'inst A terminating');
    expect(dialog()).toBe(d);
  });

  it('disables server-bound actions, closes ConnectDialog, and invalidates pending work when reconnecting to another server', async () => {
    const api = new MockApi('cloud-training', 0);
    const live = (await api.cloudInstances())[0]!;
    let finishTerminate!: () => void;
    vi.spyOn(api, 'terminateCloudInstance').mockImplementation(
      () =>
        new Promise((resolve) => {
          finishTerminate = () => resolve({ ...live, status: 'terminating' });
        }),
    );
    open(api);
    const row = await until(() => [...target.querySelectorAll('table.inst tbody tr')].find((r) => button(r, 'Terminate')), 'live instance');
    // Start a Terminate mutation and also open ConnectDialog later.
    button(row, 'Terminate')!.click();
    const d = await until(() => dialog(), 'terminate dialog');
    button(d.querySelector('footer')!, 'Terminate')!.click();
    flushSync();

    // Deferred hub.connect() switches hub.mode to 'connecting' while keeping the old api until handshake succeeds.
    hub.mode = 'connecting';
    flushSync();
    expect(dialog()).toBeNull();
    expect(button(target, 'Connect')!.disabled).toBe(true);
    expect(button(target, 'Check')!.disabled).toBe(true);
    expect(button(target, 'Disconnect')!.disabled).toBe(true);
    expect(button(row, 'Terminate')!.disabled).toBe(true);

    // Late Terminate completion from previous server A is ignored.
    finishTerminate();
    await Promise.resolve();
    await Promise.resolve();
    flushSync();
    expect(toast.items).toEqual([]);
    expect(text(row.querySelector('.st'))).toBe('Running');

    // Also verify ConnectDialog closes across a server epoch change and does not reopen.
    hub.mode = 'mock';
    flushSync();
    button(target, 'Connect')!.click();
    await until(() => dialog(), 'connect dialog');
    hub.epoch++;
    flushSync();
    expect(dialog()).toBeNull();
  });

  it('rejects an invalid Remote FOLO server URL inline in ConnectDialog and keeps the connection list empty', async () => {
    open(new MockApi('cloud-none', 0));
    await until(() => button(target, 'Connect') && !button(target, 'Connect')!.disabled, 'Connect');
    button(target, 'Connect')!.click();
    const d = await until(() => dialog()?.querySelector('[role="radio"]') && dialog(), 'the dialog');
    [...d.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((r) => text(r) === 'Remote FOLO server')!.click();
    flushSync();
    const urlInput = d.querySelector<HTMLInputElement>('input[type="text"]')!;
    type(urlInput, 'http://127.0.0.1:99999');
    d.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    const err = await until(() => d.querySelector(`#${urlInput.id}-error`), 'the URL field error');
    expect(text(err)).toBe('The server URL is not a valid URL.');
    expect(urlInput.getAttribute('aria-invalid')).toBe('true');
    expect(dialog()).toBe(d);
    expect(target.querySelectorAll('.cx-row .cx-name')).toHaveLength(0);
  });

  it('reuses watched.source for runs, skips overlapping or connecting instance polls, and cancels terminate focus rAF on unmount (#309)', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'setTimeout', 'clearTimeout'] });
    const prevSource = watched.source;
    let disposeSource: (() => void) | undefined;
    try {
      const api = new MockApi('cloud-training', 0);
      const runsSpy = vi.spyOn(api, 'runs');
      hub.api = api;
      // Prime watched.source (as RunWatch does in the shell).
      disposeSource = createRoot((dispose) => {
        watched.source = new Resource((s) => hub.api.runs(s));
        return dispose;
      });
      flushSync();
      await vi.waitFor(() => {
        flushSync();
        expect(watched.source?.data).toBeDefined();
      });
      expect(runsSpy).toHaveBeenCalledTimes(1);

      // Hold the second cloudInstances poll open so we can verify no overlapping poll fires.
      let releasePoll!: () => void;
      const realInst = api.cloudInstances.bind(api);
      let instCalls = 0;
      const instSpy = vi.spyOn(api, 'cloudInstances').mockImplementation(async () => {
        instCalls++;
        if (instCalls === 2) {
          await new Promise<void>((r) => (releasePoll = r));
        }
        return realInst();
      });

      open(api);
      await until(() => target.querySelector('table.inst tbody tr'), 'live instance row');
      // Started from watched.source: zero additional GET /api/runs!
      expect(runsSpy).toHaveBeenCalledTimes(1);
      expect(instSpy).toHaveBeenCalledTimes(1);

      // Tick while connecting -> skipped.
      hub.mode = 'connecting';
      flushSync();
      vi.advanceTimersByTime(5000);
      expect(instSpy).toHaveBeenCalledTimes(1);

      // Back to mock -> first poll fires and stays in flight; next tick skips while in flight.
      hub.mode = 'mock';
      flushSync();
      vi.advanceTimersByTime(5000);
      expect(instSpy).toHaveBeenCalledTimes(2);
      vi.advanceTimersByTime(5000);
      expect(instSpy).toHaveBeenCalledTimes(2);
      releasePoll();
      await Promise.resolve();

      // askTerminate schedules rAF; unmounting immediately cancels it so it never steals focus.
      const cancelRaf = vi.spyOn(globalThis, 'cancelAnimationFrame');
      const row = target.querySelector('table.inst tbody tr')!;
      button(row, 'Terminate')!.click();
      flushSync();
      unmount(c!);
      c = undefined;
      expect(cancelRaf).toHaveBeenCalled();
    } finally {
      disposeSource?.();
      watched.source = prevSource;
      vi.useRealTimers();
    }
  });

  it('deduplicates stacked error banners when both providers and connections fail, and Retry refreshes both', async () => {
    const api = new MockApi('normal', 0);
    const realProv = api.cloudProviders.bind(api);
    const realConn = api.cloudConnections.bind(api);
    let fail = true;
    api.cloudProviders = () => (fail ? Promise.reject(new ApiError(503, 'offline', 'server down')) : realProv());
    api.cloudConnections = () => (fail ? Promise.reject(new ApiError(503, 'offline', 'server down')) : realConn());
    open(api);

    const err = await until(() => [...target.querySelectorAll('.state')].find((e) => text(e).includes('Could not load cloud settings')), 'deduplicated cloud error');
    // Only one error banner inside the Connections panel (not two stacked).
    expect(target.querySelectorAll('#cloud .state')).toHaveLength(1);
    fail = false;
    button(err, 'Retry')!.click();
    await until(() => target.querySelectorAll('.cx-row .cx-name').length > 0, 'connections loaded after single retry');
    expect(target.querySelector('#cloud .state')).toBeNull();
  });
});
