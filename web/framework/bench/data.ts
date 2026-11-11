// js-framework-benchmark's data generator (same word lists, a seeded RNG so both apps build the same rows).
const A = ['pretty', 'large', 'big', 'small', 'tall', 'short', 'long', 'handsome', 'plain', 'quaint', 'clean', 'elegant', 'easy', 'angry', 'crazy', 'helpful', 'mushy', 'odd', 'unsightly', 'adorable', 'important', 'inexpensive', 'cheap', 'expensive', 'fancy'];
const C = ['red', 'yellow', 'blue', 'green', 'pink', 'brown', 'purple', 'brown', 'white', 'black', 'orange'];
const N = ['table', 'chair', 'house', 'bbq', 'desk', 'car', 'pony', 'cookie', 'sandwich', 'burger', 'pizza', 'mouse', 'keyboard'];

let seed = 1;
const random = (max: number) => {
  seed = (seed * 16807) % 2147483647;
  return seed % max;
};

export function labels(count: number): string[] {
  const out = new Array<string>(count);
  for (let i = 0; i < count; i++) out[i] = `${A[random(A.length)]} ${C[random(C.length)]} ${N[random(N.length)]}`;
  return out;
}

/** The operations both apps expose on `window.bench`; each runs synchronously, DOM updated, when it returns. */
export interface Ops {
  run(): void;
  runLots(): void;
  add(): void;
  update(): void;
  clear(): void;
  swapRows(): void;
  select(i: number): void;
  remove(i: number): void;
}
