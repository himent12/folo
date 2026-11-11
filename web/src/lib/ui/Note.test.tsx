// @vitest-environment happy-dom
// A note after a value keeps a real space before it: round 23's Compare table read "0.000833(requested 0.01)".
import { describe, expect, it } from 'vitest';
import { render } from '../../../framework/testing';
import Note from './Note';

describe('Note', () => {
  it('follows its value after a real, breakable space', () => {
    const target = document.createElement('span');
    target.append('0.000833');
    const c = render(() => <Note text="(requested 0.01)" fact />, target);
    expect(target.textContent).toBe('0.000833 (requested 0.01)');
    c.unmount();
  });
});
