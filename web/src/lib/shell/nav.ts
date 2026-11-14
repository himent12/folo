import LayoutDashboard from '$icons/layout-dashboard';
import Images from '$icons/images';
import Activity from '$icons/activity';
import Box from '$icons/box';
import Target from '$icons/target';
import ScanEye from '$icons/scan-eye';
import Settings from '$icons/settings';
import type { IconType } from '$lib/ui/types';

export interface NavItem {
  id: string;
  label: string;
  path: string;
  icon: IconType;
  /** Two-key "go to" shortcut, e.g. "G R". */
  keys: string;
  /**
   * Not a tab of its own in the phone tab bar: five tabs is what 320 px holds with their names whole (a sixth cut
   * "Benchmarks" and "Playground"). The section lives under this tab instead, which is current while it is open, and
   * the tab's list leads to it with a strip of the two (PhoneSections), as labeling lives under Datasets.
   */
  tabUnder?: string;
}

export const mainNav: NavItem[] = [
  { id: 'overview', label: 'Overview', path: '/', icon: LayoutDashboard, keys: 'G O' },
  { id: 'datasets', label: 'Datasets', path: '/datasets', icon: Images, keys: 'G D' },
  { id: 'runs', label: 'Runs', path: '/runs', icon: Activity, keys: 'G R' },
  { id: 'models', label: 'Models', path: '/models', icon: Box, keys: 'G M' },
  { id: 'benchmarks', label: 'Benchmarks', path: '/benchmarks', icon: Target, keys: 'G B', tabUnder: 'models' },
  { id: 'playground', label: 'Playground', path: '/playground', icon: ScanEye, keys: 'G P' },
];

/** The phone tab bar's sections. */
export const tabNav: NavItem[] = mainNav.filter((n) => !n.tabUnder);

/** The phone tab a section is shown under: its own, or the one it lives under (`tabUnder`). */
export function tabOf(section: string): string {
  return mainNav.find((n) => n.id === section)?.tabUnder ?? section;
}

export const settingsNav: NavItem = { id: 'settings', label: 'Settings', path: '/settings', icon: Settings, keys: 'G S' };
