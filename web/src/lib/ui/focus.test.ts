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
});
