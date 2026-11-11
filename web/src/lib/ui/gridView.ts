// // How image grids are shown (a dataset's, a labeling job's): boxes on or off and the thumbnail size, one choice for
// every grid, remembered per browser.

import { signal } from 'fw';
/** TileGrid's `Density` (TileGrid.tsx exports it too; a .ts module here cannot import a .tsx under svelte-check). */
type Density = 's' | 'm' | 'l';

const BOXES = 'folo.datasets.boxes';
const DENSITY = 'folo.datasets.density';

function read(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, v: string) {
  try {
    localStorage.setItem(key, v);
  } catch {
    /* private mode: not remembered */
  }
}

class GridView {
  #boxes = signal(read(BOXES) !== 'off');
  #density = signal<Density>(((d) => (d === 's' || d === 'l' ? d : 'm'))(read(DENSITY)));

  get boxes() {
    return this.#boxes.get();
  }
  set boxes(on: boolean) {
    this.#boxes.set(on);
    write(BOXES, on ? 'on' : 'off');
  }
  get density() {
    return this.#density.get();
  }
  set density(d: Density) {
    this.#density.set(d);
    write(DENSITY, d);
  }
}

export const gridView = new GridView();
