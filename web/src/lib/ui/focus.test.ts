// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import { rememberFocus } from './focus';

afterEach(() => {
  document.body.innerHTML = '';
});

describe('rememberFocus', () => {
  it('returns focus to the opener when it was lost', () => {
    const opener = document.createElement('button');
    const inside = document.createElement('button');
    document.body.append(opener, inside);
    opener.focus();
    const restore = rememberFocus();
    inside.focus();
    inside.remove();
    restore();
    expect(document.activeElement).toBe(opener);
  });

  it('leaves focus alone when it already went somewhere real', () => {
    const opener = document.createElement('button');
    const elsewhere = document.createElement('button');
    document.body.append(opener, elsewhere);
    opener.focus();
    const restore = rememberFocus();
    elsewhere.focus();
    restore();
    expect(document.activeElement).toBe(elsewhere);
  });

  it('uses fallback when the opener became disabled or disconnected before close', () => {
    const opener = document.createElement('button');
    const fallback = document.createElement('button');
    const inside = document.createElement('button');
    document.body.append(opener, fallback, inside);
    opener.focus();
    const restore = rememberFocus(() => fallback);
    inside.focus();
    opener.disabled = true;
    inside.remove();
    restore();
    expect(document.activeElement).toBe(fallback);

    // Also when the opener was removed from the DOM entirely.
    const opener2 = document.createElement('button');
    const inside2 = document.createElement('button');
    document.body.append(opener2, inside2);
    opener2.focus();
    const restore2 = rememberFocus(() => fallback);
    inside2.focus();
    opener2.remove();
    inside2.remove();
    restore2();
    expect(document.activeElement).toBe(fallback);
  });
});
