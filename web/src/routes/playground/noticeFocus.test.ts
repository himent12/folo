// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { focusNotice, holdNoticeFocus } from './noticeFocus';

let strip: HTMLElement;

function mountStrip(html = '<p class="title">Prediction failed</p><button>Retry</button>') {
  strip = document.createElement('div');
  strip.innerHTML = html;
  document.body.appendChild(strip);
  return strip;
}

const settle = () => new Promise((r) => setTimeout(r, 0));
const scrollIntoView = Element.prototype.scrollIntoView;

afterEach(() => {
  strip?.remove();
  vi.restoreAllMocks();
  Element.prototype.scrollIntoView = scrollIntoView;
});

describe('focusNotice', () => {
  it('focuses the heading without scrolling the page', () => {
    mountStrip();
    const focus = vi.spyOn(HTMLElement.prototype, 'focus');
    const into = vi.fn();
    Element.prototype.scrollIntoView = into;
    const h = focusNotice(strip)!;
    expect(document.activeElement).toBe(h);
    expect(h.tabIndex).toBe(-1);
    expect(focus).toHaveBeenCalledWith({ preventScroll: true });
    // On screen (or not laid out): no scroll at all.
    expect(into).not.toHaveBeenCalled();
  });

  it('brings the heading into view, at the nearest edge, only when none of it is on screen', () => {
    mountStrip();
    const into = vi.fn();
    Element.prototype.scrollIntoView = into;
    const h = strip.querySelector<HTMLElement>('.title')!;
    vi.spyOn(h, 'getBoundingClientRect').mockReturnValue({ top: window.innerHeight + 40, bottom: window.innerHeight + 60, height: 20 } as DOMRect);
    focusNotice(strip);
    expect(into).toHaveBeenCalledWith({ block: 'nearest' });
  });
});

describe('holdNoticeFocus', () => {
  it('returns focus to the heading when a re-render drops it, not to the page start', async () => {
    mountStrip();
    const stop = holdNoticeFocus(strip);
    expect(document.activeElement?.textContent).toBe('Prediction failed');
    // The notice re-renders: the focused heading is replaced.
    strip.innerHTML = '<p class="title">Prediction failed</p><button>Retry</button>';
    await settle();
    expect(document.activeElement).toBe(strip.querySelector('.title'));
    stop();
  });

  it('lets go once the user moves focus', async () => {
    mountStrip();
    const stop = holdNoticeFocus(strip);
    const retry = strip.querySelector('button')!;
    retry.focus();
    strip.querySelector('.title')!.remove();
    await settle();
    expect(document.activeElement).toBe(retry);
    stop();
  });
});
