// The phone Model view's sections: one at a time behind a segmented control (wider screens show every part).

export type ModelSection = 'curves' | 'classes' | 'confusion' | 'details';

export const MODEL_SECTIONS: { value: ModelSection; label: string }[] = [
  { value: 'curves', label: 'Curves' },
  { value: 'classes', label: 'Classes' },
  { value: 'confusion', label: 'Confusion' },
  { value: 'details', label: 'Details' },
];

/** Whether a part shows: always on wider screens, only the picked section on a phone. */
export function sectionShown(phone: boolean, section: ModelSection, part: ModelSection): boolean {
  return !phone || section === part;
}

/** Which half of the PR panel to show: both on wider screens; on a phone the curves or the class table. */
export function prView(phone: boolean, section: ModelSection): 'all' | 'curves' | 'classes' {
  return phone && (section === 'curves' || section === 'classes') ? section : 'all';
}
