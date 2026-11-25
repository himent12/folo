// Shared types and static Settings section index for the command palette (`./index.tsx`).
import type { RunStatus } from '$lib/api/types';
import type { IconType } from '$lib/ui/types';

export interface Item {
  id: string;
  group: string;
  label: string;
  /** Facts about the entity, one meta item each; `{ ident }` is an identifier (an architecture), set mono (ui/Ident). */
  hint?: string | (string | { ident: string })[];
  /**
   * The hint restates what the label already says (an action's description), where other hints are facts about the
   * entity (status, model, size). Phones, short of width, leave it out instead of cutting it to a fragment.
   */
  describes?: boolean;
  keys?: string;
  icon: IconType;
  /** Other phrasings it answers to (scoring.rank). */
  match?: string[];
  /** The run's, dataset's, model's ... id, which a query may name (scoring.idScore). */
  entityId?: string;
  /** An action about one run, dataset or model: offered once the query names it. */
  scoped?: boolean;
  /** A Recent entry: the id of the item it repeats (what is remembered when it is chosen). */
  origin?: string;
  /** A run's status, from the API's enum: what filters read, never the hint's display word. */
  status?: RunStatus;
  run: () => void;
}

export interface PaletteGroup {
  group: string;
  items: { item: Item; n: number }[];
}

export interface CommandPaletteProps {
  open?: boolean;
  onshortcuts: () => void;
  /** Consumes any query text typed while the palette chunk was lazy-loading on first open. */
  takeInitialQuery?: () => string;
}

/** Settings' sections (routes/Settings: `?section=`), with what each holds, as people may look for it. */
export const SETTINGS_SECTIONS: { id: string; label: string; match: string[] }[] = [
  { id: 'connection', label: 'Connection', match: ['Server URL', 'Server address', 'folo serve'] },
  { id: 'cloud', label: 'Cloud', match: ['Cloud compute', 'Connect a cloud', 'RunPod', 'Rented GPU'] },
  { id: 'appearance', label: 'Appearance', match: ['Theme', 'Dark theme', 'Light theme'] },
  { id: 'units', label: 'Units', match: ['Memory units', 'GiB', 'GB'] },
  { id: 'shortcuts', label: 'Keyboard shortcuts', match: ['Single-key shortcuts'] },
  { id: 'system', label: 'System', match: ['GPU', 'Device', 'Driver', 'Throughput'] },
  { id: 'about', label: 'About', match: ['Version'] },
];
