// The docs site: scripts/generate.ts renders the repository's Markdown into .gen/ (the Vite root) and Vite bundles
// the client script and styles. `vite` regenerates and reloads when a source Markdown file changes.
import { resolve } from 'node:path';
import { defineConfig, type Plugin } from 'vite';
import { DIST_DIR, GEN_DIR, REPO_DIR } from './scripts/config.ts';
import { generate, SiteError } from './scripts/generate.ts';

function docs(sources: string[]): Plugin {
  return {
    name: 'folo-docs',
    configureServer(server) {
      const watched = new Set(sources.map((s) => resolve(REPO_DIR, s)));
      for (const f of watched) server.watcher.add(f);
      server.watcher.on('change', async (file) => {
        if (!watched.has(resolve(file))) return;
        try {
          await generate({ write: true });
          server.ws.send({ type: 'full-reload' });
        } catch (e) {
          server.config.logger.error(e instanceof SiteError ? e.message : String(e));
        }
      });
    },
  };
}

export default defineConfig(async () => {
  const { inputs, sources } = await generate({ write: true });
  return {
    root: GEN_DIR,
    // Relative asset URLs, so the built site works from any folder or subpath.
    base: './',
    publicDir: resolve(GEN_DIR, 'public'),
    plugins: [docs(sources)],
    server: { fs: { allow: [REPO_DIR] } },
    build: {
      outDir: DIST_DIR,
      emptyOutDir: true,
      rolldownOptions: { input: inputs },
    },
  };
});
