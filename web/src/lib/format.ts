// Number, size, time and duration formatting. Metric values render with tabular numerals at the call site.

const cache = new Map<number, Intl.NumberFormat>();
function nf(digits: number): Intl.NumberFormat {
  let f = cache.get(digits);
  if (!f) {
    f = new Intl.NumberFormat('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
    cache.set(digits, f);
  }
  return f;
}

const ok = (x: number | null | undefined): x is number => x !== null && x !== undefined && Number.isFinite(x);

export function num(x: number | null | undefined, digits = 0): string {
  return ok(x) ? nf(digits).format(x) : '—';
}

/** 0.7412 → "0.741" (metrics like mAP stay in 0..1 with three decimals, as researchers read them). */
export function metric(x: number | null | undefined): string {
  return num(x, 3);
}

/** 0.7412 → "74.1%". */
export function pct(x: number | null | undefined, digits = 1): string {
  return ok(x) ? `${nf(digits).format(x * 100)}%` : '—';
}

/** 3 significant digits, without trailing zeros: 2.63, 12.4, 20, 940. */
function trim(x: number): string {
  const a = Math.abs(x);
  return String(Number(nf(a >= 100 ? 0 : a >= 10 ? 1 : 2).format(x).replace(/,/g, '')));
}

/** 2_630_000 → "2.63M"; 12_400 → "12.4k"; 940 → "940". */
export function compact(x: number | null | undefined): string {
  if (!ok(x)) return '—';
  const a = Math.abs(x);
  if (a >= 1e9) return `${trim(x / 1e9)}B`;
  if (a >= 1e6) return `${trim(x / 1e6)}M`;
  if (a >= 1e4) return `${trim(x / 1e3)}k`;
  return nf(0).format(x);
}

/** Bytes with binary (GiB) or metric (GB) units. */
export function bytes(x: number | null | undefined, units: 'binary' | 'metric' = 'binary'): string {
  if (!ok(x)) return '—';
  const k = units === 'binary' ? 1024 : 1000;
  const names = units === 'binary' ? ['B', 'KiB', 'MiB', 'GiB', 'TiB'] : ['B', 'kB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = x;
  while (Math.abs(v) >= k && i < names.length - 1) {
    v /= k;
    i++;
  }
  return `${i === 0 ? nf(0).format(v) : trim(v)} ${names[i]}`;
}

/** Seconds → "1h 04m", "12m 30s", "42s". */
export function duration(s: number | null | undefined): string {
  if (!ok(s)) return '—';
  const t = Math.max(0, Math.round(s));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = t % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(sec).padStart(2, '0')}s`;
  return `${sec}s`;
}

/** Milliseconds with sensible precision: 3.14159 → "3.14 ms", 12.34 → "12.3 ms". */
export function ms(x: number | null | undefined): string {
  if (!ok(x)) return '—';
  return `${nf(x < 10 ? 2 : x < 100 ? 1 : 0).format(x)} ms`;
}

/** Relative time: "just now", "3m ago", "2h ago", "yesterday", "5d ago", then a date. */
export function ago(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return '—';
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return '—';
  const s = Math.round((now - t) / 1000);
  if (s < 45) return 'just now';
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  if (s < 2 * 86400) return 'yesterday';
  if (s < 14 * 86400) return `${Math.round(s / 86400)}d ago`;
  return date(iso);
}

/** "Nov 8, 14:30" in local time. */
export function date(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false });
}

/** "14:30:05" in local time. */
export function clock(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}

/** Learning rates: 0.0098 → "9.80e-3". */
export function lr(x: number | null | undefined): string {
  if (!ok(x)) return '—';
  if (x === 0) return '0';
  return x.toExponential(2).replace('e+', 'e');
}
