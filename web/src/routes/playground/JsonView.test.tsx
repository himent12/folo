// @vitest-environment happy-dom
import { flush, signal } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render as mountView } from '../../../framework/testing';
import JsonView from './JsonView';

let mounted: { unmount(): void } | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  document.body.innerHTML = '';
  vi.useRealTimers();
  vi.restoreAllMocks();
});

function render(value: unknown, props: { label?: string; filename?: string } = {}) {
  const val = signal(value);
  const host = document.createElement('div');
  document.body.appendChild(host);
  mounted = mountView(() => <JsonView value={val.get()} label={props.label} filename={props.filename} />, host);
  flush();
  return { host, val };
}

describe('JsonView', () => {
  it('renders syntax-highlighted lines via DocumentFragment and updates on value change', () => {
    const { host, val } = render(
      { ok: true, count: 3, name: 'cat', detections: [{ class: 'cat', score: 0.9 }, { class: 'dog', score: 0.8 }] },
      { label: 'Prediction as JSON' },
    );
    const pre = host.querySelector('pre')!;
    expect(pre.getAttribute('aria-label')).toBe('Prediction as JSON');
    expect(host.querySelectorAll('code .ln').length).toBeGreaterThan(1);
    expect(host.querySelector('code .key')?.textContent).toBe('"ok"');
    expect(host.querySelector('code .lit')?.textContent).toBe('true');
    expect(host.querySelector('code .num')?.textContent).toBe('3');
    expect(host.querySelector('code .str')?.textContent).toBe('"cat"');

    val.set({ empty: null });
    flush();
    expect(host.querySelector('code .key')?.textContent).toBe('"empty"');
    expect(host.querySelector('code .lit')?.textContent).toBe('null');
    expect(host.querySelector('code .str')).toBeNull();
  });

  it('copies formatted JSON and resets the confirmation timer on rapid re-clicks', async () => {
    vi.useFakeTimers();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });

    const { host } = render({ a: 1 });
    const copyBtn = [...host.querySelectorAll('button')].find((b) => b.textContent?.includes('Copy'))!;
    copyBtn.click();
    await Promise.resolve();
    flush();
    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText.mock.calls[0]![0]).toContain('"a": 1');
    expect(copyBtn.textContent).toContain('Copied');

    // Click again at 1200ms: the old 1600ms timer must be cleared so Copied stays active past 1600ms.
    vi.advanceTimersByTime(1200);
    copyBtn.click();
    await Promise.resolve();
    vi.advanceTimersByTime(600);
    flush();
    expect(copyBtn.textContent).toContain('Copied');

    vi.advanceTimersByTime(1100);
    flush();
    expect(copyBtn.textContent).toContain('Copy');
    expect(copyBtn.textContent).not.toContain('Copied');
  });

  it('downloads formatted JSON as a Blob with the configured filename and revokes the object URL', () => {
    vi.useFakeTimers();
    const createObjectURL = vi.fn().mockReturnValue('blob:mock-json');
    const revokeObjectURL = vi.fn();
    vi.spyOn(URL, 'createObjectURL').mockImplementation(createObjectURL);
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(revokeObjectURL);

    let clickedHref = '';
    let clickedDownload = '';
    const origClick = HTMLAnchorElement.prototype.click;
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      clickedHref = this.href;
      clickedDownload = this.download;
    });

    const { host } = render({ detections: [{ class: 'dog', score: 0.91 }] }, { filename: 'sample.json' });
    const dlBtn = host.querySelector<HTMLButtonElement>('button[aria-label="Download JSON"]')!;
    expect(dlBtn).toBeTruthy();
    dlBtn.click();

    expect(createObjectURL).toHaveBeenCalledOnce();
    const blob = createObjectURL.mock.calls[0]![0] as Blob;
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe('application/json');
    expect(clickedHref).toContain('blob:mock-json');
    expect(clickedDownload).toBe('sample.json');
    expect(revokeObjectURL).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1000);
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:mock-json');
    expect(origClick).toBeDefined();
  });

  it('updates horizontal and vertical overflow edge flags on scroll and content changes', () => {
    const { host, val } = render({ wide: 'x'.repeat(200) });
    const pre = host.querySelector('pre')!;

    let metrics = { scrollLeft: 0, scrollWidth: 500, clientWidth: 200, scrollTop: 0, scrollHeight: 600, clientHeight: 200 };
    Object.defineProperties(pre, {
      scrollLeft: { get: () => metrics.scrollLeft, configurable: true },
      scrollWidth: { get: () => metrics.scrollWidth, configurable: true },
      clientWidth: { get: () => metrics.clientWidth, configurable: true },
      scrollTop: { get: () => metrics.scrollTop, configurable: true },
      scrollHeight: { get: () => metrics.scrollHeight, configurable: true },
      clientHeight: { get: () => metrics.clientHeight, configurable: true },
    });

    // Updating value re-runs syncEdges automatically.
    val.set({ wide: 'y'.repeat(200) });
    flush();
    expect('moreX' in pre.dataset).toBe(true);
    expect('moreXStart' in pre.dataset).toBe(false);
    expect('moreY' in pre.dataset).toBe(true);

    // Scrolling partway right shows both start and end fades.
    metrics = { ...metrics, scrollLeft: 40, scrollTop: 400 };
    pre.dispatchEvent(new Event('scroll'));
    expect('moreXStart' in pre.dataset).toBe(true);
    expect('moreX' in pre.dataset).toBe(true);
    expect('moreY' in pre.dataset).toBe(false);

    // Scrolling to the far right hides the end fade and keeps the start fade.
    metrics = { ...metrics, scrollLeft: 300 };
    pre.dispatchEvent(new Event('scroll'));
    expect('moreXStart' in pre.dataset).toBe(true);
    expect('moreX' in pre.dataset).toBe(false);

    // Shrinking content clears all overflow fades.
    metrics = { scrollLeft: 0, scrollWidth: 180, clientWidth: 200, scrollTop: 0, scrollHeight: 100, clientHeight: 200 };
    val.set({ ok: 1 });
    flush();
    expect('moreX' in pre.dataset).toBe(false);
    expect('moreXStart' in pre.dataset).toBe(false);
    expect('moreY' in pre.dataset).toBe(false);
  });
});
