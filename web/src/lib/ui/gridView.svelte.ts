// How image grids are shown (a dataset's, a labeling job's): boxes on or off and the thumbnail size, one choice for
// every grid, remembered per browser.

import type { Density } from './TileGrid.svelte';

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
  #boxes = $state(read(BOXES) !== 'off');
  #density = $state<Density>(((d) => (d === 's' || d === 'l' ? d : 'm'))(read(DENSITY)));

  get boxes() {
    return this.#boxes;
  }
  set boxes(on: boolean) {
    this.#boxes = on;
    write(BOXES, on ? 'on' : 'off');
  }
  get density() {
    return this.#density;
  }
  set density(d: Density) {
    this.#density = d;
    write(DENSITY, d);
  }
}

export const gridView = new GridView();
