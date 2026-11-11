// Builds the benchmark (fw.html) for run.mjs:
// `npx vite build -c framework/bench/vite.config.ts`.
import { fileURLToPath } from 'node:url';
import { mergeConfig, type UserConfig } from 'vite';
import base from '../../vite.config.ts';

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));

const config: UserConfig = mergeConfig(base, { root: here('../..'), base: './', build: { outDir: here('./dist'), emptyOutDir: true } });
config.build!.rollupOptions = { input: { fw: here('./fw.html') } };
export default config;
