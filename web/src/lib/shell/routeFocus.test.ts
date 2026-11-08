// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import { focusRoute, isNewPage, routeFocusTarget } from './routeFocus';

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

  it('falls back to the main region while the page has no heading yet', () => {
    const m = main('<div aria-busy="true">Loading run</div>');
    focusRoute(m);
    expect(document.activeElement).toBe(m);
  });
});
