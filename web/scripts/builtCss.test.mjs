import { describe, expect, it } from 'vitest';
// lightningcss is Vite's CSS minifier (a dependency of vite), the one that builds Studio's CSS.
import { transform } from 'lightningcss';
import { missingTwins } from './builtCss.mjs';

describe('missingTwins', () => {
  it('flags a prefixed declaration whose rule lost the unprefixed one', () => {
    expect(missingTwins('.bar{-webkit-backdrop-filter:blur(12px)saturate(1.4)}')).toEqual(['.bar: -webkit-backdrop-filter']);
  });

  it('accepts both, the unprefixed alone, and twins inside at-rules', () => {
    expect(missingTwins('.a{-webkit-backdrop-filter:blur(2px);backdrop-filter:blur(2px)}')).toEqual([]);
    expect(missingTwins('.a{backdrop-filter:blur(2px)}')).toEqual([]);
    expect(missingTwins('@media (max-width:767px){.t{-webkit-backdrop-filter:blur(16px);backdrop-filter:blur(16px)}}')).toEqual([]);
    expect(missingTwins('@media (max-width:767px){.t{-webkit-backdrop-filter:blur(16px)}}')).toHaveLength(1);
    expect(missingTwins('input::-webkit-slider-thumb{-webkit-appearance:none}')).toEqual([]);
  });

  it('catches what the minifier does to a hand-written prefix after the unprefixed property', () => {
    // The see-through top bar: lightningcss keeps only the later, prefixed declaration.
    const built = (code) => transform({ filename: 'a.css', code: Buffer.from(code), minify: true, targets: { chrome: 94 << 16, safari: 16 << 16 } }).code.toString();
    expect(missingTwins(built('.a{backdrop-filter:blur(12px) saturate(1.4);-webkit-backdrop-filter:blur(12px) saturate(1.4)}'))).toHaveLength(1);
    // Written once, unprefixed: the minifier adds the prefix and keeps both.
    expect(missingTwins(built('.a{backdrop-filter:blur(12px) saturate(1.4)}'))).toEqual([]);
  });
});
