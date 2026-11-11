import { describe, expect, it } from 'vitest';
import { count } from './count';

describe('count', () => {
  it('says one of a thing in the singular', () => {
    expect(count(1, 'class')).toBe('1 class');
    expect(count(2, 'class')).toBe('2 classes');
    expect(count(0, 'image')).toBe('0 images');
    expect(count(2400, 'image')).toBe('2,400 images');
    expect(count(1, 'box')).toBe('1 box');
    expect(count(3, 'box')).toBe('3 boxes');
  });
});
