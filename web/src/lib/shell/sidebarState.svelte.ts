// Sidebar state. Below the desktop tier (tablets and laptops) the sidebar collapses to icons on its own, so content
// keeps its width; toggling it there opens it for this session only and leaves the saved preference (which is about
// desktop windows) alone. Phones have no sidebar (the tab bar).

import { media } from '$lib/design/tokens';
import { prefs } from '$lib/prefs.svelte';

class SidebarState {
  narrow = $state(typeof matchMedia !== 'undefined' && matchMedia(media.upToDesktop).matches);
  opened = $state(false);
  collapsed = $derived(this.narrow ? !this.opened : prefs.sidebarCollapsed);

  constructor() {
    if (typeof matchMedia === 'undefined') return;
    const mq = matchMedia(media.upToDesktop);
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
