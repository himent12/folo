// Where a chart's ⓘ opens on a phone (ui/InfoButton `chart`): in place, between the chart's heading and its readout,
// never as a popover over the plot it explains (round 25: Box size's definition covered the histogram). A frame that
// holds a chart (ui/Panel, Run's metrics chart) provides a slot and renders it (ui/InfoRegion) under its heading; a
// `chart` InfoButton inside it fills the slot while open, as a disclosure (`aria-controls` the region).

import { getContext, setContext, type Snippet } from 'svelte';

const KEY = Symbol('info-slot');

export class InfoSlot {
  /** The region's id, for the button's `aria-controls`. */
  readonly id: string;
  /** What the open button shows: its text or its snippet, and who opened it. */
  content = $state.raw<{ owner: object; text?: string; children?: Snippet } | null>(null);
  constructor(id: string) {
    this.id = id;
  }
}

/** Makes this component's region the place its descendants' chart ⓘs open on phones. */
export function provideInfoSlot(id: string): InfoSlot {
  return setContext(KEY, new InfoSlot(id));
}

/** The nearest frame's slot, if any. */
export function infoSlot(): InfoSlot | undefined {
  return getContext<InfoSlot | undefined>(KEY);
}
