// Appends a Svelte file's <style> content, verbatim, as css`` at the end of its .tsx port (backslashes, backticks and
// ${ escaped for the template literal; the fw compiler undoes the escapes).
//   node framework/tools/copy-style.cjs src/lib/ui/Foo.svelte src/lib/ui/Foo.tsx
const fs = require('fs');
const [svelte, tsx] = process.argv.slice(2);
const src = fs.readFileSync(svelte, 'utf8');
const m = src.match(/<style[^>]*>([\s\S]*?)<\/style>/);
if (!m) throw new Error('no style');
const body = m[1].replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${');
fs.appendFileSync(tsx, '\ncss`' + body + '`;\n');
