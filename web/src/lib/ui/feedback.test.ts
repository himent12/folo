// The toaster is for page-level events; a dialog shows its own results (DESIGN_DIRECTION §2.9, "Feedback where the
// person is"). The toaster sits outside the modal, behind its scrim, so a toast fired while a dialog is open is easy to
// miss and cannot be reached from the dialog's focus trap. This scan fails on a toast in a component that owns a modal
// dialog unless the toast plainly fires once the dialog is closed (`if (!open)`, or a `…open = false` / `confirm… =
// false` just above it) or a comment within the 8 lines above says `page-level:` and why. Results inside the dialog go
// to its status region: `DialogStatus` (ui/feedback.svelte.ts), or `feedback()` in a component that may sit in either
// place.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { describe, expect, it } from 'vitest';

const src = join(__dirname, '..', '..');

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? files(p) : n.endsWith('.svelte') ? [p] : [];
  });
}

const CALL = /\btoast\.(success|info|error|show)\(/;
const CLOSED = /!open\b|\b\w*[oO]pen = false|\bconfirm\w* = false|page-level:/;

describe('dialog feedback', () => {
  it('a component that owns a modal dialog never toasts while it is open', () => {
    const bad: string[] = [];
    for (const f of files(src)) {
      const text = readFileSync(f, 'utf-8');
      if (!/<Dialog\b|\.showModal\(/.test(text)) continue;
      const lines = text.split('\n');
      lines.forEach((line, i) => {
        if (!CALL.test(line) || line.trim().startsWith('//') || line.trim().startsWith('*')) return;
        const near = lines.slice(Math.max(0, i - 8), i + 1).join('\n');
        if (!CLOSED.test(near)) bad.push(`${relative(src, f)}:${i + 1}: ${line.trim()}`);
      });
    }
    expect(bad).toEqual([]);
  });

  it('the shared copy controls report through feedback(), so inside a dialog they report in it', () => {
    for (const f of ['lib/ui/CommandLine.svelte', 'lib/ui/DriverReport.svelte']) {
      const text = readFileSync(join(src, f), 'utf-8');
      expect(text, f).toMatch(/feedback\(\)/);
      expect(text, f).not.toMatch(CALL);
    }
  });
});
