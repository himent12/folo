// Counts with their noun, as every dataset screen says them.

import { num } from '$lib/format';

/** "1 class", "2 classes", "2,400 images": the number formatted as everywhere (lib/format), the noun singular for one. */
export function count(n: number, one: string, many = `${one}${/(s|x|sh|ch)$/.test(one) ? 'es' : 's'}`): string {
  return `${num(n)} ${n === 1 ? one : many}`;
}
