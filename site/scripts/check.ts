// `npm run check`: generates every page in memory (nothing is written) and fails on a broken link, a missing image,
// an orphaned doc or a broken llms.txt link. `npm run build` runs the same checks.
import { generate, SiteError } from './generate.ts';

try {
  const { inputs } = await generate({ write: false });
  console.log(`docs site ok: ${Object.keys(inputs).length} pages, llms.txt verified, no orphaned docs`);
} catch (e) {
  console.error(e instanceof SiteError ? e.message : e);
  process.exit(1);
}
