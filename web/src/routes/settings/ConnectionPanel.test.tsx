// @vitest-environment happy-dom
// Settings › Connection's address (shell F12): what people type in an address bar works, and a wrong scheme is said
// in words about the field, not as a format rule.
import { flush } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import type { CloudField } from '$lib/api/types';
import { render } from '../../../framework/testing';
import { fieldMessage } from './ConnectDialog';
import ConnectionPanel, { addressProblem, serverAddress } from './ConnectionPanel';

describe('serverAddress / addressProblem', () => {
  it('assumes http:// for host:port, and keeps a typed scheme', () => {
    expect(serverAddress(' localhost:7915/ ')).toBe('http://localhost:7915');
    expect(serverAddress('127.0.0.1:7915')).toBe('http://127.0.0.1:7915');
    expect(serverAddress('https://folo.example.com')).toBe('https://folo.example.com');
    expect(serverAddress('')).toBe('');
    expect(addressProblem(serverAddress('localhost:7915'))).toBeNull();
    expect(addressProblem('')).toBeNull();
  });

  it('names the field and the schemes it takes, and rejects malformed URLs', () => {
    expect(addressProblem(serverAddress('ftp://x'))).toBe('The server URL must start with http:// or https://.');
    expect(addressProblem(serverAddress('ftp://'))).toBe('The server URL must start with http:// or https://.');
    expect(addressProblem(serverAddress('[::1]:7878'))).toBeNull();
    expect(addressProblem(serverAddress('https://folo.example.com/studio'))).toBeNull();
    expect(addressProblem(serverAddress('https://folo.example.com/@team'))).toBeNull();
    for (const bad of [
      'http://a b',
      'http://:7878',
      'http://',
      'http:///studio',
      '/studio',
      'https:/localhost:7878',
      'http://localhost:',
      'http://[::1]:',
      'http://127.0.0.1:7878\\',
      'http://127.0.0.1:7878\\api',
      'http://@localhost',
      'http://:@localhost',
      'http://localhost:99999',
      'http://localhost:abc',
      'http://localhost:0',
      'http://[::1',
      'http://127.0.0.1:7878?foo=1',
      'http://127.0.0.1:7878?',
      'http://127.0.0.1:7878#section',
      'http://user:pass@127.0.0.1:7878',
      'javascript:alert(1)',
    ]) {
      expect(addressProblem(serverAddress(bad)), bad).toBe('Use a full address like http://127.0.0.1:7878.');
    }
  });
});

describe('fieldMessage', () => {
  it("words the server's message about a field by the field's name (Connect a cloud)", () => {
    const url = { key: 'url', label: 'Server URL' } as CloudField;
    expect(fieldMessage(url, '`url` must start with https:// or http://')).toBe('The server URL must start with https:// or http://.');
    const worker = { key: 'worker_base_url', label: 'Worker download URL' } as CloudField;
    expect(fieldMessage(worker, '`worker_base_url` has no host')).toBe('The worker download URL has no host.');
  });
});

describe('ConnectionPanel', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('drops a test result once the address changes', async () => {
    vi.stubGlobal('fetch', async () => {
      throw new TypeError('Failed to fetch');
    });
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = render(() => <ConnectionPanel />, target);
    const input = target.querySelector<HTMLInputElement>('#server-url')!;
    const type = (v: string) => {
      input.value = v;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      flush();
    };
    type('http://127.0.0.1:1');
    [...target.querySelectorAll('button')].find((b) => b.textContent?.includes('Test'))!.click();
    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.test')?.textContent).toContain('Cannot reach the FOLO server');
    });
    type('localhost:7915');
    expect(target.querySelector('.test')).toBeNull();
    expect(target.querySelector('.msg.error')).toBeNull();
    c.unmount();
    target.remove();
  });

  it("tests an address by FOLO's health payload, and says what else answered (review studio-2 R1, R4)", async () => {
    let answer = Response.json({ service: 'other' });
    vi.stubGlobal('fetch', async () => answer);
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = render(() => <ConnectionPanel />, target);
    flush();
    const testIt = async (expected: string) => {
      [...target.querySelectorAll('button')].find((b) => b.textContent?.includes('Test'))!.click();
      await vi.waitFor(() => {
        flush();
        expect(target.querySelector('.test')?.textContent).toContain(expected);
      });
    };
    await testIt('This address answered, but not as a FOLO server.');
    answer = new Response('', { status: 401 });
    await testIt('This server asks for sign-in (401).');
    answer = Response.json({ status: 'ok', version: '0.1.0', api: 'v1', mock: false });
    await testIt('Reachable: folo serve 0.1.0');
    c.unmount();
    target.remove();
  });

  it('is green only when the token check lets Studio in: refused is red, a check that cannot tell is neither (#323)', async () => {
    // The health is open; the token check (HEAD /api/runs) answers `head`: `folo serve --token` (401), a proxy whose
    // session expired (403), one that resets the connection, then a server Studio can use.
    let head: number | 'reset' = 401;
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (!(url.endsWith('/api/runs') && init?.method === 'HEAD')) return Response.json({ status: 'ok', version: '0.1.0', api: 'v1', mock: false });
      if (head === 'reset') throw new TypeError('Failed to fetch');
      return new Response(null, { status: head });
    });
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = render(() => <ConnectionPanel />, target);
    flush();
    const testIt = async (expected: string) => {
      [...target.querySelectorAll('button')].find((b) => b.textContent?.includes('Test'))!.click();
      await vi.waitFor(() => {
        flush();
        expect(target.querySelector('.test')?.textContent).toContain(expected);
      });
      return target.querySelector('.test')!.classList.contains('ok');
    };
    expect(await testIt("This server needs a token (folo serve --token), which Studio can't send.")).toBe(false);
    head = 403;
    expect(await testIt('This server refuses access (403).')).toBe(false);
    head = 'reset';
    expect(await testIt('Reachable: folo serve 0.1.0, but access could not be confirmed.')).toBe(false);
    head = 404;
    expect(await testIt('but access could not be confirmed.')).toBe(false);
    head = 200;
    expect(await testIt('Reachable: folo serve 0.1.0')).toBe(true);
    c.unmount();
    target.remove();
  });

  it('offers demo data while no server answers, as a choice that reloads with ?mock=1 (v-shell N1)', async () => {
    vi.stubGlobal('fetch', async () => {
      throw new TypeError('Failed to fetch');
    });
    const assign = vi.fn();
    vi.stubGlobal('location', { ...location, href: 'http://127.0.0.1:7878/#/settings', search: '', origin: 'http://127.0.0.1:7878', assign });
    const target = document.createElement('div');
    const c = render(() => <ConnectionPanel />, target);
    const demo = () => [...target.querySelectorAll('button')].find((b) => b.textContent?.includes('Use demo data'));
    expect(demo()).toBeUndefined();
    await hub.connect();
    flush();
    // Offline, still connecting: the state says so, and nothing turned into demo data by itself.
    expect([hub.mode, target.querySelector('.s-title')?.textContent]).toEqual(['connecting', 'Server offline']);
    demo()!.click();
    expect(assign).toHaveBeenCalledWith('http://127.0.0.1:7878/?mock=1#/settings');
    c.unmount();
  });

  it('offers Leave demo data when in forced ?mock=1 mode and reloads without mock or scenario params', () => {
    const prevMode = hub.mode;
    const assign = vi.fn();
    const target = document.createElement('div');
    try {
      vi.stubGlobal('location', {
        ...location,
        href: 'http://127.0.0.1:7878/?mock=1&scenario=error&keep=1#/settings',
        search: '?mock=1&scenario=error&keep=1',
        origin: 'http://127.0.0.1:7878',
        assign,
      });
      hub.mode = 'mock';
      const c = render(() => <ConnectionPanel />, target);
      flush();
      const leave = () => [...target.querySelectorAll<HTMLButtonElement>('.row-actions button')].find((b) => b.textContent?.includes('Leave demo data'));
      expect(leave()).toBeDefined();
      leave()!.click();
      expect(assign).toHaveBeenCalledWith('http://127.0.0.1:7878/?keep=1#/settings');

      // Hidden when live even if the URL query had ?mock=1.
      hub.mode = 'live';
      flush();
      expect(leave()).toBeUndefined();
      c.unmount();

      // Common case: only ?mock=1 present -> no stray '?' left in URL.
      assign.mockClear();
      vi.stubGlobal('location', { ...location, href: 'http://127.0.0.1:7878/?mock=1#/settings', search: '?mock=1', origin: 'http://127.0.0.1:7878', assign });
      hub.mode = 'mock';
      const c2 = render(() => <ConnectionPanel />, target);
      flush();
      leave()!.click();
      expect(assign).toHaveBeenCalledWith('http://127.0.0.1:7878/#/settings');
      c2.unmount();

      // Hidden when hub.mode is mock without ?mock=1 in location.search.
      vi.stubGlobal('location', { ...location, href: 'http://127.0.0.1:7878/#/settings', search: '', origin: 'http://127.0.0.1:7878', assign });
      hub.mode = 'mock';
      const c3 = render(() => <ConnectionPanel />, target);
      flush();
      expect(leave()).toBeUndefined();
      c3.unmount();
    } finally {
      hub.mode = prevMode;
    }
  });
});
