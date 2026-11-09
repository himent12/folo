// `npm run check`: generates every page in memory (nothing is written) and fails on a broken link, a missing image
// or a doc missing from the navigation. `npm run build` runs the same checks.
import { generate, SiteError } from './generate.ts';

try {
  const { inputs } = await generate({ write: false });
  console.log(`docs site ok: ${Object.keys(inputs).length} pages, every link resolves`);
} catch (e) {
  console.error(e instanceof SiteError ? e.message : e);
  process.exit(1);
}
