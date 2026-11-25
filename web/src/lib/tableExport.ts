/**
 * Client-side CSV and JSON table export helpers: RFC-4180 CSV quoting, OWASP spreadsheet formula neutralization on
 * string cells (while keeping finite numbers numeric), safe slug filenames, and clean Blob downloads with
 * `URL.revokeObjectURL` cleanup and toast fallback.
 */
import { toast } from '$lib/ui/toast';

export type CsvCell = string | number | boolean | null | undefined;

const FORMULA_LEAD = /^[\t\r\n ]*[=+\-@]|^[\t\r]/;
const NEEDS_QUOTES = /[",\r\n]/;

/**
 * Formats one CSV cell per RFC-4180. String values that could be interpreted as spreadsheet formulas (`=`, `+`, `-`,
 * `@`, leading tab/CR, even after spaces) are prefixed with a single quote; finite `number` values remain plain
 * numeric literals so spreadsheets keep them as numbers.
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

/** Formats an export payload as pretty-printed JSON with a trailing newline. */
export function toJson(data: unknown): string {
  return `${JSON.stringify(data, null, 2)}\n`;
}

/** Normalises an arbitrary title into a clean, cross-platform lowercase filename. */
export function safeFilename(base: string, ext: 'csv' | 'json'): string {
  const slug = base
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80);
  return `${slug || 'export'}.${ext}`;
}

/** Triggers a browser download for a text payload and revokes its object URL cleanly. */
export function downloadText(content: string, filename: string, mimeType: string): boolean {
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
  } catch {
    if (url) URL.revokeObjectURL(url);
    toast.info('Download failed', 'Your browser blocked the file download.');
    return false;
  } finally {
    a?.remove();
  }
}

export function exportTableCsv(filenameBase: string, headers: readonly string[], rows: readonly (readonly CsvCell[])[]): boolean {
  return downloadText(toCsv(headers, rows), safeFilename(filenameBase, 'csv'), 'text/csv;charset=utf-8');
}

export function exportTableJson(filenameBase: string, data: unknown): boolean {
  return downloadText(toJson(data), safeFilename(filenameBase, 'json'), 'application/json;charset=utf-8');
}
