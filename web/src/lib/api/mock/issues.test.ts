import { describe, expect, it } from 'vitest';
import { issues, specs, stats } from './datasets';

describe('mock dataset issues', () => {
  it('agree with the stats of every demo dataset', () => {
    for (const s of specs) {
      const list = issues(s);
      const st = stats(s);
      expect(list.filter((i) => i.severity === 'error').length, s.name).toBe(st.broken_labels);
      expect(list.filter((i) => i.kind === 'empty_image').length, s.name).toBe(st.empty_images);
      for (const i of list) {
        expect(i.file).toMatch(/^(images|labels)\//);
        if (i.line !== null) expect(i.line).toBeGreaterThan(0);
      }
    }
  });
});
