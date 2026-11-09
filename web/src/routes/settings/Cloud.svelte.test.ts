// @vitest-environment happy-dom
// Settings › Cloud: the connections (account, balance, the last check), the instances FOLO launched, and the Connect,
// Disconnect and Terminate flows with the errors the API documents (docs/API.md, Cloud compute).
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import { toast } from '$lib/ui/toast.svelte';
import Cloud from './Cloud.svelte';

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
    expect(text(li.querySelector('.cx-state'))).toBe('Connected');
    expect(button(li, 'Check')).toBeDefined();
    expect(button(li, 'Disconnect')).toBeDefined();
    // The header's Connect, the secondary style: not the page's primary action.
    expect(target.querySelector('#cloud header .btn.secondary')?.textContent?.trim()).toBe('Connect');
    const inst = await until(() => target.querySelector('table.inst tbody tr'), 'the instances');
    expect(inst.classList.contains('ended')).toBe(true);
    // The offer as a meta line: one item each, so a wrapped line never ends on a "·".
    expect([...inst.querySelectorAll('.gpu .ui-meta > span')].map(text)).toEqual(['RTX 4090', '24.0 GiB', 'Secure Cloud']);
    expect(text(inst.querySelector('.run a'))).toBe('shapes-m');
    expect(text(inst.querySelector('.st'))).toBe('Terminated');
    expect(button(inst, 'Terminate')).toBeUndefined();
  });

  it('says there is no cloud, with one way to connect, and lists no instances', async () => {
    open(new MockApi('cloud-none', 0));
    await until(() => text(target.querySelector('#cloud')).includes('No cloud connected'), 'the empty state');
    expect(text(target.querySelector('#cloud'))).toContain('No cloud connected');
    expect(text(target.querySelector('#cloud'))).toContain('Connect RunPod to train on a rented GPU, or another FOLO server of yours.');
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
    expect(text(callout.querySelector('.c-title'))).toBe('Termination not confirmed');
    expect(text(callout.querySelector('.c-msg'))).toBe(
      'FOLO could not confirm that folo-run_c3d5-4c1a is terminated: RunPod rejected this API key. It keeps trying. Check the provider’s console for k2x9wq7m3fz1; the instance may still be billed.',
    );
    expect(callout.querySelectorAll('.ident')).toHaveLength(2);
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
    // Required fields are checked on submit, the message naming the field.
    d.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await until(() => text(d.querySelector('.msg.error')) === 'Enter the API key.', 'the required message');
    const key = d.querySelector<HTMLInputElement>('input[type="password"]')!;
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
    expect([...d.querySelectorAll('label')].map(text)).toEqual(expect.arrayContaining(['Server URL', 'Token', 'Name']));
    [...d.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((r) => text(r) === 'RunPod')!.click();
    flushSync();
    expect(d.querySelector<HTMLInputElement>('input[type="password"]')!.value).toBe('');
  });

  it('confirms Disconnect, says why a connection in use stays, and forgets one that is not', async () => {
    open(new MockApi('cloud-training', 0));
    const li = (await until(() => target.querySelector('.cx-row .cx-name'), 'the connection')).closest('li')!;
    button(li, 'Disconnect')!.click();
    const d = await until(() => dialog(), 'the dialog');
    expect(text(d.querySelector('h2'))).toBe('Disconnect RunPod?');
    expect(text(d.querySelector('header p'))).toBe('FOLO forgets the API key. Runs and models stay.');
    expect(d.querySelector('footer .btn.destructive')?.textContent?.trim()).toBe('Disconnect');
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
    expect(text(d.querySelector('h2'))).toBe('Terminate this instance?');
    expect(text(d.querySelector('header p'))).toBe('The RTX 4090 at RunPod is shut down now. Its run fasteners-l fails; the checkpoints already copied to this machine stay.');
    await until(() => document.activeElement?.hasAttribute('data-safe'), 'the safe button focused');
    button(d.querySelector('footer')!, 'Terminate')!.click();
    await until(() => toast.items.length, 'the toast');
    expect(toast.items[0]).toMatchObject({ tone: 'info', title: 'Terminating the RTX 4090' });
    await until(() => text(target.querySelector('table.inst tbody tr .st')) === 'Terminating', 'the new state');
  });
});
