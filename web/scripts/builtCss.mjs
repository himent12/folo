// A check of the built stylesheets, run by every production build (`npm run build`, and the scripts' builds through
// vite.config.ts): a vendor-prefixed declaration must keep its unprefixed twin in the same rule.
//
// Why: the CSS minifier (lightningcss) adds the prefixes the build's targets need by itself, and it reads a prefixed
// declaration written after the unprefixed one as an override, so `backdrop-filter: X; -webkit-backdrop-filter: X`
// built to the prefixed one alone, which Chrome ignores (the see-through top bar). Write the unprefixed property only.

const PREFIXED = /-(?:webkit|moz|ms)-(backdrop-filter|mask-image|user-select|appearance|text-size-adjust)\s*:/g;

/**
 * Rules of `css` that declare a prefixed property without its unprefixed twin, as "selector: -webkit-prop".
 * @param {string} css a built (or any) stylesheet
 * @returns {string[]}
 */
export function missingTwins(css) {
  const out = [];
  // Every innermost block: a selector (or at-rule prelude) and its declarations.
  for (const m of css.replace(/\/\*[\s\S]*?\*\//g, '').matchAll(/([^{}]*)\{([^{}]*)\}/g)) {
    const [, selector, body] = m;
    // A vendor's own pseudo-element (::-webkit-slider-thumb) takes that vendor's prefixed properties only.
    if (/::-(?:webkit|moz|ms)-/.test(selector)) continue;
    for (const p of body.matchAll(PREFIXED)) {
      const twin = new RegExp(`(?:^|[;\\s])${p[1]}\\s*:`);
      if (!twin.test(body)) out.push(`${selector.trim()}: ${p[0].replace(/\s*:$/, '')}`);
    }
  }
  return out;
}

/**
 * The Vite plugin: fails the build when an emitted stylesheet has a prefixed declaration without its twin.
 * @returns {import('vite').Plugin}
 */
export function builtCssCheck() {
  return {
    name: 'folo-built-css-check',
    apply: 'build',
    enforce: 'post',
    generateBundle(_options, bundle) {
      const bad = [];
      for (const file of Object.values(bundle)) {
        if (file.type !== 'asset' || !file.fileName.endsWith('.css')) continue;
        for (const rule of missingTwins(String(file.source))) bad.push(`${file.fileName}: ${rule}`);
      }
      if (bad.length) {
        this.error(
          `prefixed declaration without its unprefixed twin (write the unprefixed property only; the build adds the prefix):\n  ${bad.join('\n  ')}`,
        );
      }
    },
  };
}
