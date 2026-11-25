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

  it('escapes RFC-4180 cells and neutralizes ASCII, full-width, and control-char formula prefixes while keeping numbers numeric', () => {
    expect(escapeCsvCell(null)).toBe('');
    expect(escapeCsvCell(undefined)).toBe('');
    expect(escapeCsvCell(true)).toBe('true');
    expect(escapeCsvCell(false)).toBe('false');
    expect(escapeCsvCell(0.842)).toBe('0.842');
    expect(escapeCsvCell(-0.05)).toBe('-0.05');
    expect(escapeCsvCell(-42)).toBe('-42');
    expect(escapeCsvCell(Number.NaN)).toBe('');
    expect(escapeCsvCell(Number.POSITIVE_INFINITY)).toBe('');

    expect(escapeCsvCell('plain')).toBe('plain');
    expect(escapeCsvCell('a, b')).toBe('"a, b"');
    expect(escapeCsvCell('say "hi"')).toBe('"say ""hi"""');
    expect(escapeCsvCell('line1\nline2')).toBe('"line1\nline2"');

    // OWASP CSV formula neutralization on string cells (ASCII, full-width Unicode variants, leading tab/CR/LF)
    expect(escapeCsvCell('=SUM(A1:A2)')).toBe("'=SUM(A1:A2)");
    expect(escapeCsvCell('+1+1')).toBe("'+1+1");
    expect(escapeCsvCell('-2+3')).toBe("'-2+3");
    expect(escapeCsvCell('@SUM(A1:A2)')).toBe("'@SUM(A1:A2)");
    expect(escapeCsvCell('@SUM(1,2)')).toBe(`"'@SUM(1,2)"`);
    expect(escapeCsvCell('\uFF1DSUM(A1)')).toBe("'\uFF1DSUM(A1)");
    expect(escapeCsvCell('\uFF0B1+1')).toBe("'\uFF0B1+1");
    expect(escapeCsvCell('\uFF0D2+3')).toBe("'\uFF0D2+3");
    expect(escapeCsvCell('\uFF20SUM(A1)')).toBe("'\uFF20SUM(A1)");
    expect(escapeCsvCell('  =CMD()')).toBe("'  =CMD()");
    expect(escapeCsvCell('  \uFF1DCMD()')).toBe("'  \uFF1DCMD()");
    expect(escapeCsvCell('\t=1')).toBe("'\t=1");
    expect(escapeCsvCell('\nfoo')).toBe(`"'\nfoo"`);
    expect(escapeCsvCell('\rfoo')).toBe(`"'\rfoo"`);
    expect(escapeCsvCell('=HYPERLINK("http://x","a,b")')).toBe(`"'=HYPERLINK(""http://x"",""a,b"")"`);
  });

  it('builds RFC-4180 CSV (neutralizing headers too), pretty JSON, and clean slug filenames', () => {
    const csv = toCsv(['=Header', 'Score'], [['=evil,1', -0.04], ['ok', null]]);
    expect(csv).toBe("'=Header,Score\r\n\"'=evil,1\",-0.04\r\nok,\r\n");
    expect(toJson({ a: 1 })).toBe('{\n  "a": 1\n}\n');
    expect(() => toJson(undefined)).toThrow();
    expect(safeFilename('Sorting tray, night shift (v2)!', 'csv')).toBe('sorting-tray-night-shift-v2.csv');
    expect(safeFilename('   ///   ', 'json')).toBe('export.json');
    expect(safeFilename('a'.repeat(120), 'csv')).toBe(`${'a'.repeat(80)}.csv`);
    // Truncating right at a separator must not leave a trailing hyphen.
    expect(safeFilename(`${'a'.repeat(79)} bbb`, 'csv')).toBe(`${'a'.repeat(79)}.csv`);
  });

  it('downloads Blobs cleanly, revokes object URLs, and handles serialization and browser failures with error toasts', async () => {
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

    // Unserializable JSON values return false and report an error toast instead of throwing.
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    expect(exportTableJson('bad', circular)).toBe(false);
    expect(exportTableJson('undef', undefined)).toBe(false);
    expect(toast.items.some((t) => t.title === 'Download failed' && t.tone === 'error')).toBe(true);

    // Download failure revokes immediately and reports an error toast.
    toast.items = [];
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {
      throw new Error('blocked');
    });
    expect(downloadText('x', 'x.csv', 'text/csv')).toBe(false);
    expect(revoked).toContain('blob:test/2');
    expect(toast.items.some((t) => t.title === 'Download failed' && t.tone === 'error')).toBe(true);
  });
});
