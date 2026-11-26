// @vitest-environment happy-dom
import { flush as flushSync, signal } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Model } from '$lib/api/types';
import DeleteRun from './DeleteRun';

const model = (id: string, name: string, run_id: string) => ({ id, name, run_id, bytes: 2 ** 20, source: 'run' }) as Model;

describe('DeleteRun', () => {
  let target: HTMLElement;
  let c: { unmount(): void } | undefined;
  beforeEach(() => {
    hub.api = new MockApi('normal', 0);
    hub.mode = 'mock';
    HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
      this.open = true;
    };
    target = document.createElement('div');
    document.body.appendChild(target);
  });
  afterEach(() => {
    c?.unmount();
    target.remove();
    vi.restoreAllMocks();
  });

  it('counts the models from a list fetched when it opens, saying they go without counting until it arrives', async () => {
    // The run ended while its page was open: the page's list (from before) had none of its models; the server has two.
    let answer!: (m: Model[]) => void;
    const models = vi.spyOn(hub.api, 'models').mockImplementation(() => new Promise((r) => (answer = r)));
    const run = signal<{ id: string; name: string } | null>(null);
    c = render(() => <DeleteRun run={run.get()} onclose={() => run.set(null)} />, target);
    flushSync();
    expect(models).not.toHaveBeenCalled();

    run.set({ id: 'run_f6f1', name: 'shapes-n' });
    flushSync();
    expect(models).toHaveBeenCalledTimes(1);
    const text = () => document.querySelector('dialog')!.textContent!;
    expect(text()).toContain('Its weights and models are deleted too.');
    expect(text()).not.toContain('no saved weights');

    answer([model('m_f6f1', 'shapes-n-best', 'run_f6f1'), model('m_f6f1-last', 'shapes-n-last', 'run_f6f1')]);
    await vi.waitFor(() => {
      flushSync();
      expect(text()).toContain('2 models (shapes-n-best, shapes-n-last)');
    });
    expect(text()).toContain('Delete run and 2 models');

    // Opening again fetches again, and shows nothing counted from the last opening meanwhile.
    run.set(null);
    flushSync();
    run.set({ id: 'run_f6f1', name: 'shapes-n' });
    flushSync();
    expect(models).toHaveBeenCalledTimes(2);
    expect(text()).toContain('Its weights and models are deleted too.');
    answer([]);
    await vi.waitFor(() => {
      flushSync();
      expect(text()).toContain('It has no saved weights.');
    });
  });

  it('keeps the uncounted words when the list cannot be fetched', async () => {
    vi.spyOn(hub.api, 'models').mockRejectedValue(new Error('offline'));
    c = render(() => <DeleteRun run={{ id: 'run_f6f1', name: 'shapes-n' }} onclose={() => {}} />, target);
    await Promise.resolve();
    await Promise.resolve();
    flushSync();
    expect(document.querySelector('dialog')!.textContent).toContain('Its weights and models are deleted too.');
  });

  it('locks dismissal and blocks double-submit while deletion is in flight', async () => {
    let finishDelete!: () => void;
    const del = vi.spyOn(hub.api, 'deleteRun').mockImplementation(() => new Promise<void>((r) => (finishDelete = r)));
    vi.spyOn(hub.api, 'models').mockResolvedValue([]);
    const run = signal<{ id: string; name: string } | null>({ id: 'run_f6f1', name: 'shapes-n' });
    const onclose = vi.fn(() => run.set(null));
    const ondeleted = vi.fn();
    c = render(() => <DeleteRun run={run.get()} onclose={onclose} ondeleted={ondeleted} />, target);
    flushSync();

    const dialog = document.querySelector<HTMLDialogElement>('dialog')!;
    const cancelBtn = [...dialog.querySelectorAll<HTMLButtonElement>('footer button')].find((b) => b.textContent?.trim() === 'Cancel')!;
    const confirmBtn = [...dialog.querySelectorAll<HTMLButtonElement>('footer button')].find((b) => b.textContent?.includes('Delete'))!;

    confirmBtn.click();
    flushSync();
    expect(del).toHaveBeenCalledTimes(1);
    expect(cancelBtn.disabled).toBe(true);
    expect(confirmBtn.disabled).toBe(true);

    // Escape cancel event and Cancel click must be blocked while deleting.
    const cancelEvent = new Event('cancel', { cancelable: true });
    dialog.dispatchEvent(cancelEvent);
    expect(cancelEvent.defaultPrevented).toBe(true);
    cancelBtn.click();
    flushSync();
    expect(onclose).not.toHaveBeenCalled();

    finishDelete();
    await vi.waitFor(() => {
      flushSync();
      expect(run.get()).toBeNull();
      expect(dialog.open).toBe(false);
      expect(onclose).toHaveBeenCalled();
      expect(ondeleted).toHaveBeenCalledTimes(1);
    });
  });
});
