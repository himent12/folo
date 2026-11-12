// @vitest-environment happy-dom
// A dialog's status region: a notice cleared while shown (Dismiss, the dialog closing) leaves without its own parts
// reading the cleared notice. It threw "Cannot read properties of null (reading 'tone')" once a notice had changed
// tone (a copied path, then a failed copy) and the dialog closed.
import { flush } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import DialogStatusRegion from './DialogStatus';
import { DialogStatus } from './feedback';

let target: HTMLElement;
let c: { unmount(): void } | undefined;
afterEach(() => {
  c?.unmount();
  target?.remove();
  vi.restoreAllMocks();
});

describe('DialogStatus', () => {
  it('clears a notice that replaced one of another tone without reading the cleared notice', () => {
    const errors: unknown[] = [];
    vi.spyOn(console, 'error').mockImplementation((...a) => void errors.push(a));
    const status = new DialogStatus();
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <DialogStatusRegion status={status} />, target);
    const shown = () => target.querySelector('[data-notice]');
    for (const show of [() => status.error(new Error('disk on fire'), 'Could not remove'), () => status.success('Done')]) {
      show();
      flush();
      status.error(new Error('again'), 'Could not remove');
      flush();
      expect(target.querySelector('[role=alert] [data-notice]')?.textContent).toContain('again');
      expect(() => {
        status.clear();
        flush();
      }).not.toThrow();
      expect(shown()).toBeNull();
    }
    expect(errors).toEqual([]);
  });
});
