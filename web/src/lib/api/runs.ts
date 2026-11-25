// Notifies the shell's shared run list (`watched.source`) after a run mutation (create, delete, stop, resume, queue
// move) so newly mounted routes never borrow a pre-mutation snapshot via `startFrom` and the sidebar updates at once.
import { watched } from '$lib/shell/runList';

/** Invalidates the shared run list snapshot and triggers a fresh background read. */
export function runsChanged(): void {
  void watched.source?.invalidate();
}
