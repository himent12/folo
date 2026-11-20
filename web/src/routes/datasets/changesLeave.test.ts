// @vitest-environment happy-dom
// The leave listeners of ./changes (issue #309, item 5): the module is in the shared chunk, so it adds them only when
// this tab first asks for a dataset, the one request whose failure asks whether the page is going away.
import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
});

describe('telling a leave from a failure', () => {
  it('adds no window listener on import, and its three on the first dataset asked for, once', async () => {
    const add = vi.spyOn(window, 'addEventListener');
    const changes = await import('./changes');
    const leaveEvents = () => add.mock.calls.map((c) => c[0]).filter((t) => ['beforeunload', 'pagehide', 'pageshow'].includes(t));
    expect(leaveEvents()).toEqual([]);
    const g = { server: 'http://127.0.0.1:7878', name: 'shapes', since: 0, ids: [] };
    changes.generationStarted(g);
    changes.generationStarted(g);
    expect(leaveEvents()).toEqual(['beforeunload', 'pagehide', 'pageshow']);
    expect(changes.pageLeaving()).toBe(false);
    window.dispatchEvent(new Event('pagehide'));
    expect(changes.pageLeaving()).toBe(true);
    window.dispatchEvent(new Event('pageshow'));
    expect(changes.pageLeaving()).toBe(false);
  });
});
