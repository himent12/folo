// scripts/load-faults.mjs's check on its reference (issue #309): the screen's own stylesheet and the shared one.
import { describe, expect, it } from 'vitest';
import { referenceGaps } from './loadFaultsRef.mjs';

describe('the reference of a clean load', () => {
  const full = ['/assets/index-a1.css', '/assets/shared-b2.css', '/assets/Overview-c3.css'];

  it('is whole with the screen own stylesheet and the shared one', () => {
    expect(referenceGaps('Overview', full)).toEqual([]);
  });

  it("names what it lacks: not empty is not enough", () => {
    expect(referenceGaps('Overview', ['/assets/index-a1.css'])).toEqual(['Overview-*.css', 'shared-*.css']);
    expect(referenceGaps('Settings', full)).toEqual(['Settings-*.css']);
    // Another screen's stylesheet whose name ends like this one's is not it.
    expect(referenceGaps('Settings', ['/assets/shared-b2.css', '/assets/AppSettings-d4.css'])).toEqual(['Settings-*.css']);
  });
});
