// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import { focusRoute, holdRouteFocus, isNewPage, landOn, routeFocusTarget } from './routeFocus';

afterEach(() => {
  document.body.innerHTML = '';
});

function main(html: string): HTMLElement {
  const el = document.createElement('main');
  el.tabIndex = -1;
  el.innerHTML = html;
  document.body.appendChild(el);
  return el;
}

describe('route focus', () => {
  it('treats only a path change as a new page', () => {
    expect(isNewPage('/runs', '/runs/run_7d1e')).toBe(true);
    expect(isNewPage('/runs', '/runs')).toBe(false);
  });

  it("lands on the page's heading", () => {
    const m = main('<div class="page"><header><h1 tabindex="-1" data-route-focus>Runs</h1></header><button>Filter</button></div>');
    const h1 = m.querySelector('h1')!;
    expect(routeFocusTarget(m)).toBe(h1);
    expect(focusRoute(m)).toBe(h1);
    expect(document.activeElement).toBe(h1);
  });

  it('lands on the part the URL asks for (landOn), leaving the scroll where the page put it', () => {
    const m = main('<h1 tabindex="-1" data-route-focus>Settings</h1><section><h2 id="units-title" tabindex="-1">Units</h2></section>');
    const units = m.querySelector<HTMLElement>('#units-title')!;
    landOn(units);
    let top = 0;
    const real = window.scrollTo;
    window.scrollTo = (() => void top++) as typeof window.scrollTo;
    try {
      expect(focusRoute(m)).toBe(units);
      expect(document.activeElement).toBe(units);
      expect(top).toBe(0);
      // Another part marked later is the one landing: one mark at a time.
      const h1 = m.querySelector<HTMLElement>('h1')!;
      landOn(h1);
      expect(units.hasAttribute('data-route-landing')).toBe(false);
    } finally {
      window.scrollTo = real;
    }
  });

  it('falls back to the main region while the page has no heading yet', () => {
    const m = main('<div aria-busy="true">Loading run</div>');
    focusRoute(m);
    expect(document.activeElement).toBe(m);
  });

  it("moves to the loaded page's heading when the loading one is replaced", async () => {
    const m = main('<h1 tabindex="-1" data-route-focus>Model</h1>');
    const stop = holdRouteFocus(m);
    expect(document.activeElement?.textContent).toBe('Model');
    m.innerHTML = '<h1 tabindex="-1" data-route-focus>fasteners-n-best</h1>';
    await new Promise((r) => setTimeout(r, 0));
    expect(document.activeElement?.textContent).toBe('fasteners-n-best');
    stop();
  });

  it('leaves focus the user moved alone', async () => {
    const m = main('<h1 tabindex="-1" data-route-focus>Model</h1><button>Retry</button>');
    const stop = holdRouteFocus(m);
    const button = m.querySelector('button')!;
    button.focus();
    m.querySelector('h1')!.replaceWith(Object.assign(document.createElement('h1'), { textContent: 'loaded', tabIndex: -1 }));
    await new Promise((r) => setTimeout(r, 0));
    expect(document.activeElement).toBe(button);
    stop();
  });
});
