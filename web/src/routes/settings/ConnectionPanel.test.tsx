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

  it('names the field and the schemes it takes', () => {
    expect(addressProblem(serverAddress('ftp://x'))).toBe('The server URL must start with http:// or https://.');
    expect(addressProblem(serverAddress('http://a b'))).toBe('Use a full address like http://127.0.0.1:7878.');
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
});
