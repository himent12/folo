import { flushSync, mount } from 'svelte';
import Bench from './Bench.svelte';
import type { Ops } from './data';

mount(Bench, {
  target: document.getElementById('app')!,
  props: {
    api: (ops: Record<string, (i: number) => void>) => {
      (window as unknown as { bench: Ops }).bench = Object.fromEntries(
        Object.entries(ops).map(([k, fn]) => [k, (i: number) => (fn(i), flushSync())]),
      ) as unknown as Ops;
    },
  },
});
