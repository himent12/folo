// The run list as the shell polls it (RunWatch, every 2 s): the sidebar's live card reads it, so the shell asks once.

import type { Run } from '$lib/api/types';

/** `since`: when the list last arrived, while its polls fail (the list is then a snapshot, not the current state). */
export const watched = $state<{ runs: Run[] | undefined; since: number | null }>({ runs: undefined, since: null });
