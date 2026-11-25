// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { toast } from '$lib/ui/toast';
import { downloadText, escapeCsvCell, exportTableCsv, exportTableJson, safeFilename, toCsv, toJson } from './tableExport';

describe('tableExport', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    toast.items = [];
  });

  it('escapes RFC-4180 cells and neutralizes spreadsheet formula injection on strings while keeping numbers numeric', () => {
    expect(escapeCsvCell(null)).toBe('');
    expect(escapeCsvCell(undefined)).toBe('');
    expect(escapeCsvCell(true)).toBe('true');
    expect(escapeCsvCell(false)).toBe('false');
    expect(escapeCsvCell(0.842)).toBe('0.842');
    expect(escapeCsvCell(-0.05)).toBe('-0.05');
    expect(escapeCsvCell(Number.NaN)).toBe('');
    expect(escapeCsvCell(Number.POSITIVE_INFINITY)).toBe('');

    expect(escapeCsvCell('plain')).toBe('plain');
    expect(escapeCsvCell('a, b')).toBe('"a, b"');
    expect(escapeCsvCell('say "hi"')).toBe('"say ""hi"""');
    expect(escapeCsvCell('line1\nline2')).toBe('"line1\nline2"');

    // OWASP CSV formula neutralization on string cells (including leading whitespace / tab / CR)
    expect(escapeCsvCell('=SUM(A1:A2)')).toBe("'=SUM(A1:A2)");
    expect(escapeCsvCell('+1+1')).toBe("'+1+1");
    expect(escapeCsvCell('-2+3')).toBe("'-2+3");
    expect(escapeCsvCell('@SUM(A1:A2)')).toBe("'@SUM(A1:A2)");
    expect(escapeCsvCell('@SUM(1,2)')).toBe(`"'@SUM(1,2)"`);
    expect(escapeCsvCell('  =CMD()')).toBe("'  =CMD()");
    expect(escapeCsvCell('\t=1')).toBe("'\t=1");
    expect(escapeCsvCell('=HYPERLINK("http://x","a,b")')).toBe(`"'=HYPERLINK(""http://x"",""a,b"")"`);
  });

  it('builds RFC-4180 CSV with CRLF endings, pretty JSON, and safe slug filenames', () => {
    const csv = toCsv(['Name', 'Score'], [['=evil,1', 91.5], ['ok', null]]);
    expect(csv).toBe('Name,Score\r\n"\'=evil,1",91.5\r\nok,\r\n');
    expect(toJson({ a: 1 })).toBe('{\n  "a": 1\n}\n');
    expect(safeFilename('Sorting tray, night shift (v2)!', 'csv')).toBe('sorting-tray-night-shift-v2.csv');
    expect(safeFilename('   ///   ', 'json')).toBe('export.json');
    expect(safeFilename('a'.repeat(120), 'csv')).toBe(`${'a'.repeat(80)}.csv`);
  });

  it('downloads Blobs cleanly and revokes object URLs, with error fallback toast', async () => {
    vi.useFakeTimers();
    const created: Array<{ blob: Blob; url: string }> = [];
    const revoked: string[] = [];
    const clicked: Array<{ href: string; download: string }> = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
      const url = `blob:test/${created.length}`;
      created.push({ blob: b as Blob, url });
      return url;
    });
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation((u) => revoked.push(u));
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      clicked.push({ href: this.href, download: this.download });
    });

    expect(exportTableCsv('Fasteners Val', ['col'], [['v']])).toBe(true);
    expect(exportTableJson('Fasteners Val', { ok: true })).toBe(true);
    expect(clicked).toEqual([
      { href: 'blob:test/0', download: 'fasteners-val.csv' },
      { href: 'blob:test/1', download: 'fasteners-val.json' },
    ]);
    expect(created[0]!.blob.type).toBe('text/csv;charset=utf-8');
    expect(created[1]!.blob.type).toBe('application/json;charset=utf-8');
    expect(await created[0]!.blob.text()).toBe('col\r\nv\r\n');
    expect(await created[1]!.blob.text()).toBe('{\n  "ok": true\n}\n');
    expect(document.querySelectorAll('a[download]')).toHaveLength(0);

    vi.advanceTimersByTime(1000);
    expect(revoked).toEqual(['blob:test/0', 'blob:test/1']);

    // Failure path revokes immediately and reports a toast.
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {
      throw new Error('blocked');
    });
    expect(downloadText('x', 'x.csv', 'text/csv')).toBe(false);
    expect(revoked).toContain('blob:test/2');
    expect(toast.items.some((t) => t.title === 'Download failed')).toBe(true);
  });
});
