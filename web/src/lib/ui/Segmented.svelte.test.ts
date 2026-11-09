// @vitest-environment happy-dom
// A disabled option's reason is its accessible description, never only a pointer's tooltip (Playground's Run on:
// WebGPU and CPU off in a browser without them).
import { flushSync, mount, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';
import Segmented from './Segmented.svelte';

describe('Segmented', () => {
  it('describes a disabled option by the reason it is off', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = mount(Segmented, {
      target,
      props: {
        label: 'Run the model on',
        value: 'server',
        options: [
          { value: 'server', label: 'Server' },
          { value: 'webgpu', label: 'WebGPU', disabled: true, tip: 'No WebGPU in this browser' },
          { value: 'cpu', label: 'CPU' },
        ],
      },
    });
    flushSync();
    const [server, webgpu, cpu] = [...target.querySelectorAll<HTMLButtonElement>('button[role="radio"]')];
    expect(webgpu!.disabled).toBe(true);
    expect(document.getElementById(webgpu!.getAttribute('aria-describedby')!)?.textContent).toBe('No WebGPU in this browser');
    expect(server!.hasAttribute('aria-describedby')).toBe(false);
    expect(cpu!.hasAttribute('aria-describedby')).toBe(false);
    unmount(c);
    target.remove();
  });
});
