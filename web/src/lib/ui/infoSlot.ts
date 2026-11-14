// Where a chart's ⓘ opens on a phone (ui/InfoButton `chart`): in place, between the chart's heading and its readout,
// never as a popover over the plot it explains (round 25: Box size's definition covered the histogram). A frame that
// holds a chart (ui/Panel, Run's metrics chart) provides a slot and renders it (ui/InfoRegion) under its heading; a
// `chart` InfoButton inside it fills the slot while open, as a disclosure (`aria-controls` the region).

import { getContext, setContext, signal } from 'fw';

const KEY = Symbol('info-slot');

/** What an open button shows in the slot: its text or its snippet, and who opened it. */
export interface InfoSlotContent {
  owner: object;
  text?: string;
  children?: () => JSX.Element;
}

export class InfoSlot {
  /** The region's id, for the button's `aria-controls`. */
  readonly id: string;
  #content = signal<InfoSlotContent | null>(null);
  /** What the open button shows: its text or its snippet, and who opened it. */
  get content(): InfoSlotContent | null {
    return this.#content.get();
  }
  set content(v: InfoSlotContent | null) {
    this.#content.set(v);
  }
  constructor(id: string) {
    this.id = id;
  }
}

/** Makes this component's region the place its descendants' chart ⓘs open on phones. */
export function provideInfoSlot(id: string): InfoSlot {
  const slot = new InfoSlot(id);
  setContext(KEY, slot);
  return slot;
}

/** The nearest frame's slot, if any. */
export function infoSlot(): InfoSlot | undefined {
  return getContext<InfoSlot | undefined>(KEY);
}
