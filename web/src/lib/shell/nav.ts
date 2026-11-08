import Gauge from '@lucide/svelte/icons/gauge';
import Images from '@lucide/svelte/icons/images';
import Activity from '@lucide/svelte/icons/activity';
import Box from '@lucide/svelte/icons/box';
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
}

export const mainNav: NavItem[] = [
  { id: 'overview', label: 'Overview', path: '/', icon: Gauge, keys: 'G O' },
  { id: 'datasets', label: 'Datasets', path: '/datasets', icon: Images, keys: 'G D' },
  { id: 'runs', label: 'Runs', path: '/runs', icon: Activity, keys: 'G R' },
  { id: 'models', label: 'Models', path: '/models', icon: Box, keys: 'G M' },
  { id: 'playground', label: 'Playground', path: '/playground', icon: ScanSearch, keys: 'G P' },
];

export const settingsNav: NavItem = { id: 'settings', label: 'Settings', path: '/settings', icon: Settings, keys: 'G S' };
