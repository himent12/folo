// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { describe, expect, it } from 'vitest';
import { ApiError } from '$lib/api/client';
import { fromHTML, render as mount } from '../../../framework/testing';
import ErrorState, { type ErrorStateProps } from './ErrorState';

function render(props: Record<string, unknown>) {
  const target = document.createElement('div');
  document.body.appendChild(target);
  const c = mount(() => <ErrorState {...(props as unknown as ErrorStateProps)} />, target);
  flushSync();
  return {
    target,
    text: (sel: string) => target.querySelector(sel)?.textContent ?? null,
    done: () => {
      c.unmount();
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
      actions: () => fromHTML('<button class="extra">Run in this browser</button>'),
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

  it('says a deleted thing is gone the same way on every screen: its list, no Retry, no agent hint (shell F7)', () => {
    for (const [code, title, msg, list, path] of [
      ['run_not_found', 'Run not found', 'This run was deleted or the link is wrong.', 'All runs', '#/runs'],
      ['model_not_found', 'Model not found', 'This model was deleted or the link is wrong.', 'All models', '#/models'],
      ['dataset_not_found', 'Dataset not found', 'This dataset was deleted or the link is wrong.', 'All datasets', '#/datasets'],
      ['bench_not_found', 'Benchmark not found', 'This benchmark was deleted or the link is wrong.', 'All benchmarks', '#/benchmarks'],
      ['label_job_not_found', 'Labeling job not found', 'This labeling job was deleted or the link is wrong.', 'All labeling jobs', '#/datasets/label'],
    ]) {
      const r = render({
        error: new ApiError(404, code!, `thing 'x_1' does not exist`, 'list_things (MCP) or GET /api/things lists the things that exist'),
        title: 'Could not load this',
        retry: () => {},
      });
      expect(r.text('.title')).toBe(title);
      expect(r.text('.msg')).toBe(msg);
      expect(r.target.querySelector('.hint')).toBeNull();
      expect(r.target.textContent).not.toContain('MCP');
      const buttons = [...r.target.querySelectorAll('.acts a, .acts button')];
      expect(buttons.map((b) => b.textContent?.trim())).toEqual([list]);
      expect(buttons[0]!.getAttribute('href')).toBe(path);
      expect(r.text('.code .line')).toBe(`${code} · HTTP 404`);
      r.done();
    }
  });

  it('leaves the way to the list to the page header when it has that link already', () => {
    const page = document.createElement('div');
    page.className = 'page';
    page.innerHTML = '<header><a href="#/models">All models</a></header>';
    document.body.appendChild(page);
    const c = mount(() => <ErrorState error={new ApiError(404, 'model_not_found', "model 'm_1' does not exist")} />, page);
    flushSync();
    expect(page.querySelector('.err .acts')).toBeNull();
    c.unmount();
    page.remove();
  });

  it('keeps a 404 that is not a deleted thing (a device) as an error, and an agent-only hint with the details', () => {
    const r = render({ error: new ApiError(404, 'device_not_found', 'no gpu:3', null), retry: () => {}, title: 'Could not start' });
    expect(r.text('.title')).toBe('Could not start');
    expect(r.text('.acts button')).toContain('Retry');
    r.done();
    const agent = render({ error: new ApiError(409, 'run_active', 'a run is active', 'run_status (GET /api/runs/{id}) has its state'), title: 'Could not delete' });
    expect(agent.target.querySelector('.hint')).toBeNull();
    expect([...agent.target.querySelectorAll('.code .line')].map((e) => e.textContent)).toContain('Server hint: Run_status (GET /api/runs/{id}) has its state.');
    agent.done();
  });

  it('shows no hint when the server had none', () => {
    const r = render({ error: new ApiError(500, 'internal', 'boom', null) });
    expect(r.target.querySelector('.hint')).toBeNull();
    r.done();
  });
});
