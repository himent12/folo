// The run list as the shell polls it (RunWatch, every 2 s): the sidebar's live card reads it, so the shell asks once.

import type { Run } from '$lib/api/types';

export const watched = $state<{ runs: Run[] | undefined }>({ runs: undefined });
