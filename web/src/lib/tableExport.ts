/**
 * Client-side CSV and JSON table export helpers: RFC-4180 CSV quoting, OWASP spreadsheet formula neutralization on
 * string cells (ASCII and full-width formula operators plus leading tab/CR/LF, while keeping finite numbers numeric),
 * safe slug filenames, and clean Blob downloads with `URL.revokeObjectURL` cleanup and error toast fallback.
 */
import { toast } from '$lib/ui/toast';

export type CsvCell = string | number | boolean | null | undefined;

// Neutralize ASCII formula triggers (=, +, -, @), their full-width Unicode equivalents (＝, ＋, －, ＠), and any
// leading tab/CR/LF control character (even standalone LF or after leading whitespace).
const FORMULA_LEAD = /^[\t\r\n ]*[=+\-@\uFF1D\uFF0B\uFF0D\uFF20]|^[\t\r\n]/;
const NEEDS_QUOTES = /[",\r\n]/;

/**
 * Formats one CSV cell per RFC-4180. String values that could be interpreted as spreadsheet formulas (`=`, `+`, `-`,
 * `@`, full-width `＝`/`＋`/`－`/`＠`, or leading tab/CR/LF) are prefixed with a single quote; finite `number` values
 * remain plain numeric literals so spreadsheets keep them as numbers.
 */
export function escapeCsvCell(value: CsvCell): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : '';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  const safe = FORMULA_LEAD.test(value) ? `'${value}` : value;
  return NEEDS_QUOTES.test(safe) ? `"${safe.replaceAll('"', '""')}"` : safe;
}

/** Builds an RFC-4180 CSV string with CRLF line endings. */
export function toCsv(headers: readonly string[], rows: readonly (readonly CsvCell[])[]): string {
  const lines = [headers.map(escapeCsvCell).join(',')];
  for (const row of rows) lines.push(row.map(escapeCsvCell).join(','));
  return `${lines.join('\r\n')}\r\n`;
}

/** Formats an export payload as pretty-printed JSON with a trailing newline. Throws if not JSON-serializable. */
export function toJson(data: unknown): string {
  const text = JSON.stringify(data, null, 2);
  if (text === undefined) throw new TypeError('Export payload is not JSON-serializable.');
  return `${text}\n`;
}

/** Normalises an arbitrary title into a clean, cross-platform lowercase filename. */
export function safeFilename(base: string, ext: 'csv' | 'json'): string {
  const slug = base
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80)
    .replace(/-+$/g, '');
  return `${slug || 'export'}.${ext}`;
}

/** Triggers a browser download for a text payload and revokes its object URL cleanly. */
export function downloadText(content: string, filename: string, mimeType: string, toastKey?: string): boolean {
  let url: string | undefined;
  let a: HTMLAnchorElement | undefined;
  try {
    const blob = new Blob([content], { type: mimeType });
    url = URL.createObjectURL(blob);
    a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    const revoke = url;
    setTimeout(() => URL.revokeObjectURL(revoke), 1000);
    return true;
  } catch (err) {
    if (url) URL.revokeObjectURL(url);
    toast.error(
      err instanceof Error ? err : new Error('Your browser blocked the file download.'),
      'Download failed',
      toastKey ? { key: toastKey } : undefined,
    );
    return false;
  } finally {
    a?.remove();
  }
}

export function exportTableCsv(
  filenameBase: string,
  headers: readonly string[],
  rows: readonly (readonly CsvCell[])[],
  toastKey?: string,
): boolean {
  return downloadText(toCsv(headers, rows), safeFilename(filenameBase, 'csv'), 'text/csv;charset=utf-8', toastKey);
}

export function exportTableJson(filenameBase: string, data: unknown, toastKey?: string): boolean {
  let content: string;
  try {
    content = toJson(data);
  } catch (err) {
    toast.error(
      err instanceof Error ? err : new Error('Could not serialize export data.'),
      'Download failed',
      toastKey ? { key: toastKey } : undefined,
    );
    return false;
  }
  return downloadText(content, safeFilename(filenameBase, 'json'), 'application/json;charset=utf-8', toastKey);
}
