// The runtime's size: everything fw exports (and what compiled JSX imports), minified and gzipped; and the part a
// typical page pulls in (signals, effects, templates, insert, events, components, For/Show).
//   node framework/bench/size.mjs
import { brotliCompressSync, gzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { rolldown } from 'rolldown';

const runtime = fileURLToPath(new URL('../runtime/', import.meta.url)).replace(/\\/g, '/');
const entries = {
  'minimal (signals, effects, templates, insert, attributes, delegated events, components)': `export { signal, computed, effect, flush, render } from '${runtime}index.ts'; export { createComponent, insert, template, attr, delegateEvents } from '${runtime}internal.ts';`,
  'whole runtime (every export)': `export * from '${runtime}index.ts'; export * from '${runtime}internal.ts';`,
  'typical page (signals, JSX, For, Show, events, props)': `export { signal, computed, effect, flush, For, Show, render, defaults, splitProps } from '${runtime}index.ts'; export { createComponent, insert, template, attr, prop, className, toggleClass, delegateEvents, setEvent, memo, cond, mergeProps, spread } from '${runtime}internal.ts';`,
};
for (const [name, code] of Object.entries(entries)) {
  const bundle = await rolldown({
    input: 'entry',
    plugins: [{ name: 'entry', resolveId: (id) => (id === 'entry' ? id : null), load: (id) => (id === 'entry' ? code : null) }],
    logLevel: 'silent',
  });
  const { output } = await bundle.generate({ format: 'esm', minify: true });
  const js = Buffer.from(output[0].code);
  const kb = (n) => (n / 1024).toFixed(2);
  console.log(`${name}: ${kb(js.length)} KB min, ${kb(gzipSync(js, { level: 9 }).length)} KB gzip, ${kb(brotliCompressSync(js).length)} KB brotli`);
}
