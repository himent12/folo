// Sidebar state. Below the desktop tier (tablets and laptops) the sidebar collapses to icons on its own, so content
// keeps its width; toggling it there opens it for this session only and leaves the saved preference (which is about
// desktop windows) alone. Phones have no sidebar (the tab bar).

import { computed, signal } from 'fw';
import { media } from '$lib/design/tokens';
import { prefs } from '$lib/prefs';

class SidebarState {
  #narrow = signal(typeof matchMedia !== 'undefined' && matchMedia(media.upToDesktop).matches);
  #opened = signal(false);
  #collapsed = computed(() => (this.narrow ? !this.opened : prefs.sidebarCollapsed));

  get narrow(): boolean {
    return this.#narrow.get();
  }
  set narrow(v: boolean) {
    this.#narrow.set(v);
  }
  get opened(): boolean {
    return this.#opened.get();
  }
  set opened(v: boolean) {
    this.#opened.set(v);
  }
  get collapsed(): boolean {
    return this.#collapsed.get();
  }

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
