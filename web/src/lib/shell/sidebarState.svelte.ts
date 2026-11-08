// Sidebar state. Up to 1100 px wide the sidebar collapses to icons on its own, so content keeps its width; toggling it
// there opens it for this session only and leaves the saved preference (which is about wide windows) alone.

import { prefs } from '$lib/prefs.svelte';

const NARROW = '(max-width: 1100px)';

class SidebarState {
  narrow = $state(typeof matchMedia !== 'undefined' && matchMedia(NARROW).matches);
  opened = $state(false);
  collapsed = $derived(this.narrow ? !this.opened : prefs.sidebarCollapsed);

  constructor() {
    if (typeof matchMedia === 'undefined') return;
    const mq = matchMedia(NARROW);
    mq.addEventListener('change', () => {
      this.narrow = mq.matches;
      this.opened = false;
    });
  }

  toggle() {
    if (this.narrow) this.opened = !this.opened;
    else prefs.sidebarCollapsed = !prefs.sidebarCollapsed;
  }
}

export const sidebar = new SidebarState();
