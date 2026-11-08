// @vitest-environment happy-dom
import { createRawSnippet, flushSync, mount, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';
import { ApiError } from '$lib/api/client';
import ErrorState from './ErrorState.svelte';

function render(props: Record<string, unknown>) {
  const target = document.createElement('div');
  document.body.appendChild(target);
  const c = mount(ErrorState, { target, props: props as never });
  flushSync();
  return {
    target,
    text: (sel: string) => target.querySelector(sel)?.textContent ?? null,
    done: () => {
      unmount(c);
      target.remove();
    },
  };
}

describe('ErrorState', () => {
  it('always shows the hint, as a sentence, identifiers in mono', () => {
    const r = render({
      error: new ApiError(404, 'device_not_found', 'device gpu:3 is not available', 'this machine has gpu:0, cpu; use device `auto` or `cpu`'),
      title: 'Could not start',
    });
    expect(r.text('.msg')).toBe('Device gpu:3 is not available.');
    expect(r.text('.hint')).toBe('This machine has gpu:0, cpu; use device auto or cpu.');
    expect([...r.target.querySelectorAll('.hint code')].map((e) => e.textContent)).toEqual(['auto', 'cpu']);
    r.done();
  });

  it('says out of GPU memory plainly, with the driver report among the details', () => {
    const r = render({
      error: new ApiError(507, 'gpu_out_of_memory', 'the GPU ran out of memory (wgpu backend: Out of Memory)', 'lower `batch` or `imgsz`, or use a smaller `arch` (folo-n, folo-s)'),
      title: 'Prediction failed',
      actions: createRawSnippet(() => ({ render: () => '<button class="extra">Run in this browser</button>' })),
    });
    expect(r.text('.msg')).toBe('The GPU ran out of memory.');
    expect(r.text('.hint')).toBe('Lower batch or imgsz, or use a smaller arch (folo-n, folo-s).');
    expect(r.text('.code .line')).toBe('wgpu backend: Out of Memory');
    expect(r.text('.acts .extra')).toBe('Run in this browser');
    r.done();
  });

  it("says what applies on this screen, keeping the server's hint with the details", () => {
    const r = render({
      error: new ApiError(507, 'gpu_out_of_memory', 'the GPU ran out of memory (wgpu backend: Out of Memory)', 'lower `batch` or `imgsz`'),
      title: 'Prediction failed',
      hint: 'Run it in this browser instead.',
    });
    expect(r.text('.hint')).toBe('Run it in this browser instead.');
    const lines = [...r.target.querySelectorAll('.code .line')].map((e) => e.textContent);
    expect(lines).toContain('Server hint: Lower batch or imgsz.');
    r.done();
  });

  it('shows no hint when the server had none', () => {
    const r = render({ error: new ApiError(500, 'internal', 'boom', null) });
    expect(r.target.querySelector('.hint')).toBeNull();
    r.done();
  });
});
