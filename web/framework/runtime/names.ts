// Names the compiler and the runtime read the same way. No imports: the compiler (Node) loads this file too.

/**
 * The event of an `on<type>` attribute, given what follows `on`: `clickcapture` is `click` in the capture phase,
 * while `gotpointercapture` and `lostpointercapture` are events of their own (Svelte's exceptions).
 */
export function eventOf(name: string): { type: string; capture: boolean } {
  const capture = name.endsWith('capture') && name !== 'gotpointercapture' && name !== 'lostpointercapture';
  return { type: capture ? name.slice(0, -7) : name, capture };
}
