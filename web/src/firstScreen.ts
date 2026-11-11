// The first screen's chunk, asked for before anything else of the app runs: main.tsx imports this module first, so the
// request starts as soon as the entry chunk has loaded, not after the shell's modules have all been evaluated.
import { parse } from '$lib/router';
import { preload } from './routes';

const first = parse(location.hash).name;
if (first !== 'not-found') void preload(first).catch(() => {});
