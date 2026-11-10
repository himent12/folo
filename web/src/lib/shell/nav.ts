import Gauge from '@lucide/svelte/icons/gauge';
import Images from '@lucide/svelte/icons/images';
import Activity from '@lucide/svelte/icons/activity';
import Box from '@lucide/svelte/icons/box';
import Target from '@lucide/svelte/icons/target';
import ScanSearch from '@lucide/svelte/icons/scan-search';
import Settings from '@lucide/svelte/icons/settings';
import type { IconType } from '$lib/ui/types';

export interface NavItem {
  id: string;
  label: string;
  path: string;
  icon: IconType;
  /** Two-key "go to" shortcut, e.g. "G R". */
  keys: string;
  /**
   * Not in the phone tab bar: five tabs is what 320 px holds with their names whole (a sixth cut "Benchmarks" and
   * "Playground"); the section is in the sidebar from 768 px up and in the palette (search) everywhere.
   */
  noTab?: true;
}

export const mainNav: NavItem[] = [
  { id: 'overview', label: 'Overview', path: '/', icon: Gauge, keys: 'G O' },
  { id: 'datasets', label: 'Datasets', path: '/datasets', icon: Images, keys: 'G D' },
  { id: 'runs', label: 'Runs', path: '/runs', icon: Activity, keys: 'G R' },
  { id: 'models', label: 'Models', path: '/models', icon: Box, keys: 'G M' },
  { id: 'benchmarks', label: 'Benchmarks', path: '/benchmarks', icon: Target, keys: 'G B', noTab: true },
  { id: 'playground', label: 'Playground', path: '/playground', icon: ScanSearch, keys: 'G P' },
];

/** The phone tab bar's sections. */
export const tabNav: NavItem[] = mainNav.filter((n) => !n.noTab);

export const settingsNav: NavItem = { id: 'settings', label: 'Settings', path: '/settings', icon: Settings, keys: 'G S' };
