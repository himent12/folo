/**
 * The guard behind "visible guidance is also the accessible description" (DESIGN_DIRECTION §2.11): in every `.field`
 * of a rendered screen, each message the field shows (its hint or error, a `p` directly in it) has an id, and every
 * control in the field lists that id in `aria-describedby`. Returns what breaks it, one line each, empty when none.
 * Used by the screens' tests (New run, Add dataset, Settings) and `fields.svelte.test.ts`.
 */
const CONTROLS = 'input:not([type="hidden"]), select, textarea, [role="radiogroup"]';

export function undescribedFields(root: ParentNode): string[] {
  const out: string[] = [];
  for (const field of root.querySelectorAll<HTMLElement>('.field')) {
    const label = field.querySelector('label')?.textContent?.trim() || '(unlabelled field)';
    const messages = [...field.children].filter((el): el is HTMLParagraphElement => el.tagName === 'P');
    const controls = [...field.querySelectorAll<HTMLElement>(CONTROLS)];
    for (const m of messages) {
      if (!m.id) {
        out.push(`${label}: "${m.textContent?.trim()}" has no id`);
        continue;
      }
      for (const c of controls) {
        const ids = (c.getAttribute('aria-describedby') ?? '').split(/\s+/);
        if (!ids.includes(m.id)) out.push(`${label}: its ${c.tagName.toLowerCase()} is not described by #${m.id}`);
      }
    }
  }
  return out;
}
