// Builds the ui gallery twice, fw (gallery-fw.html) and Svelte (gallery-svelte.html), for gallery.mjs:
// `GALLERY_OUT=<dir> npx vite build -c framework/parity/gallery.vite.config.ts` (default dir: framework/parity/dist-gallery).
import { fileURLToPath } from 'node:url';
import { mergeConfig, type UserConfig } from 'vite';
import base from '../../vite.config.ts';

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));

const config: UserConfig = mergeConfig(base, {
  root: here('../..'),
  base: './',
  build: { outDir: process.env.GALLERY_OUT ?? here('./dist-gallery'), emptyOutDir: true, sourcemap: false },
});
config.build!.rollupOptions = { input: { fw: here('./gallery-fw.html'), svelte: here('./gallery-svelte.html') } };
export default config;
