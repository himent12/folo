// @vitest-environment happy-dom
// A disabled option's reason is its accessible description, never only a pointer's tooltip (Playground's Run on:
// WebGPU and CPU off in a browser without them).
import { describe, expect, it } from 'vitest';
import { flush, signal } from 'fw';
import { render } from '../../../framework/testing';
import Segmented from './Segmented';

describe('Segmented', () => {
  it('describes a disabled option by the reason it is off', () => {
    const { container: target, unmount } = render(() => (
      <Segmented
        label="Run the model on"
        value="server"
        options={[
          { value: 'server', label: 'Server' },
          { value: 'webgpu', label: 'WebGPU', disabled: true, tip: 'No WebGPU in this browser' },
          { value: 'cpu', label: 'CPU' },
        ]}
      />
    ));
    flush();
    const [server, webgpu, cpu] = [...target.querySelectorAll<HTMLButtonElement>('button[role="radio"]')];
    expect(webgpu!.disabled).toBe(true);
    expect(document.getElementById(webgpu!.getAttribute('aria-describedby')!)?.textContent).toBe('No WebGPU in this browser');
    expect(server!.hasAttribute('aria-describedby')).toBe(false);
    expect(cpu!.hasAttribute('aria-describedby')).toBe(false);
    unmount();
  });

  it('keeps the first enabled option tabbable and handles ArrowLeft cleanly when value matches no option', () => {
    const val = signal('missing');
    const { container: target, unmount } = render(() => (
      <Segmented
        label="Group"
        bind:value={val}
        options={[
          { value: 'all', label: 'All' },
          { value: 'a', label: 'A' },
          { value: 'b', label: 'B' },
        ]}
      />
    ));
    flush();
    const radios = [...target.querySelectorAll<HTMLButtonElement>('button[role="radio"]')];
    expect(radios.map((r) => r.tabIndex)).toEqual([0, -1, -1]);
    // ArrowLeft from an unmatched value selects the last enabled option ("B"), not skipping it.
    target.querySelector('[role="radiogroup"]')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    flush();
    expect(val.get()).toBe('b');
    expect(radios.map((r) => r.tabIndex)).toEqual([-1, -1, 0]);
    unmount();
  });
});
