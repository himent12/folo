// Number, size, time and duration formatting. Metric values render with tabular numerals at the call site.
import { unitTop } from './charts/domain';

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

/**
 * A run's best score, as its outcome line and toast say it ("best mAP50-95 0.00042 at epoch 3"), and a class's in Per
 * class: as `metric`, but a score that is not zero never reads as zero (issue #382). Under 0.0005 it takes the decimals
 * a chart fitted to it gives its legend (`axisMetric` on `unitTop`: 0.00042 → "0.00042"), and under five decimals'
 * reach it reads "<0.00001". Only where the text has room for it: a KPI tile or a "±" keeps `metric`.
 */
export function score(x: number | null | undefined): string {
  if (!ok(x) || x === 0 || Math.abs(x) >= 0.0005) return num(x, 3);
  const a = Math.abs(x);
  const d = scoreDecimals(unitTop(a));
  return a >= 0.5 * 10 ** -d ? num(x, d) : `${x < 0 ? '>-' : '<'}${num(10 ** -d, d)}`;
}

/**
 * A 0–1 score in the legend of a chart whose y axis is fitted to `top` (lib/charts/domain `unitTop`), as Desktop's
 * `fmt_score_on` writes it (#338): three decimals down to a top of 0.01, then one more per power of ten, at most five.
 */
export function axisMetric(x: number | null | undefined, top?: number): string {
  return num(x, scoreDecimals(top));
}

/** `clamp(ceil(-log10(top) - 1e-6) + 1, 3, 5)`; no top: 3. The `1e-6` puts a power-of-ten top on Desktop's (f32) side. */
export function scoreDecimals(top?: number): number {
  return top !== undefined && top > 0 ? Math.min(5, Math.max(3, Math.ceil(-Math.log10(top) - 1e-6) + 1)) : 3;
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
  // From 999.5 of a unit `trim` reads 1000: that is one of the next unit.
  if (a >= 999.5e6) return `${trim(x / 1e9)}B`;
  if (a >= 999.5e3) return `${trim(x / 1e6)}M`;
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
  // From k - 0.05 the one decimal reads k ("1,024.0 KiB"): that is one of the next unit.
  while (Math.abs(v) >= k - 0.05 && i < names.length - 1) {
    v /= k;
    i++;
  }
  // One format for every size in the app: one decimal above bytes ("37.0 MiB", "16.0 GiB"), so columns read evenly.
  return `${i === 0 ? nf(0).format(v) : nf(1).format(v)} ${names[i]}`;
}

/**
 * A size as a number and its own unit, for a table cell that sets the unit apart (Models' Size: "37.0" then "MiB",
 * the column's heading "Size"). Desktop's steps and rounding (apps/desktop `format::bytes`): GiB and MiB to one
 * decimal, KiB whole, bytes as they are; `units` picks MiB or MB (prefs.units). 38_797_312 → ["37.0", "MiB"].
 */
export function sizeParts(x: number | null | undefined, units: 'binary' | 'metric' = 'binary'): [value: string, unit: string] {
  if (!ok(x)) return ['—', ''];
  const k = units === 'binary' ? 1024 : 1000;
  const [kb, mb, gb] = units === 'binary' ? ['KiB', 'MiB', 'GiB'] : ['kB', 'MB', 'GB'];
  if (x >= k ** 3) return [nf(1).format(x / k ** 3), gb];
  if (x >= k ** 2) return [nf(1).format(x / k ** 2), mb];
  if (x >= k) return [nf(0).format(x / k), kb];
  return [nf(0).format(x), 'B'];
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

/**
 * Time left (an estimate): to the second under two minutes, then whole minutes rounded up. Estimates taken a moment
 * apart (the sidebar's poll, the run's stream) then read the same, and the figure does not flicker every step.
 */
export function eta(s: number | null | undefined): string {
  if (!ok(s)) return '—';
  const t = Math.max(0, Math.round(s));
  if (t < 120) return duration(t);
  const m = Math.ceil(t / 60);
  return m < 60 ? `${m}m` : `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
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
  // Each unit is rounded, and one that rounds up to the next reads as the next ("1h ago", never
  // "60m ago"). Desktop's `ago` (apps/desktop/src/api/time.rs) uses the same rule.
  const m = Math.max(1, Math.round(s / 60));
  if (m < 60) return `${m}m ago`;
  const h = Math.round(s / 3600);
  if (h < 24) return `${h}h ago`;
  if (s < 2 * 86400) return 'yesterday';
  const d = Math.round(s / 86400);
  if (d < 14) return `${d}d ago`;
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
const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };

/** 10 to the power `e` with a superscript exponent: "10⁻³". */
export function pow10(e: number): string {
  return `10${[...String(e)].map((c) => SUP[c] ?? c).join('')}`;
}

/** Learning rates in scientific notation as people write it: 0.0098 → "9.80 × 10⁻³". */
export function lr(x: number | null | undefined): string {
  if (!ok(x)) return '—';
  if (x === 0) return '0';
  let e = Math.floor(Math.log10(Math.abs(x)));
  // A mantissa that rounds up to 10 (0.009996 → "10.00") belongs to the next power.
  if (Math.abs(Number((x / 10 ** e).toFixed(2))) >= 10) e++;
  return `${(x / 10 ** e).toFixed(2)} × ${pow10(e)}`;
}

/**
 * A device for display, the same on every screen: the GPU's model without the vendor's brand prefix or the graphics
 * backend ("NVIDIA GeForce RTX 4080 SUPER (vulkan)" → "RTX 4080 SUPER"); a CPU keeps saying so ("AMD Ryzen 9 7950X
 * (cpu)" → "Ryzen 9 7950X (CPU)", a bare "cpu" → "CPU"). The raw label and the config's device id ("auto", "gpu:0")
 * stay available as secondary text.
 */
export function deviceName(label: string): string {
  const cpu = /\s*\(cpu\)$/i.test(label);
  const s = label
    .trim()
    .replace(/\s*\((?:cpu|vulkan|dx12|metal|gl|cuda|webgpu)\)$/i, '')
    .replace(/^(?:NVIDIA GeForce|NVIDIA|AMD Radeon|AMD|Intel\(R\)|Intel|Apple)\s+/i, '');
  if (/^cpu\b/i.test(s)) return 'CPU';
  return cpu ? `${s} (CPU)` : s || label;
}

/**
 * A count with its noun, as every screen says it: "1 class", "2 classes", "2,400 images" (the number as `num` writes
 * it), the noun singular for one. `many` for a plural that is not the noun plus -s / -es.
 */
export function count(n: number, one: string, many = `${one}${/(s|x|sh|ch)$/.test(one) ? 'es' : 's'}`): string {
  return `${num(n)} ${n === 1 ? one : many}`;
}

/**
 * A server message or hint as a sentence: a capital first letter and a closing full stop ("device gpu:3 is not
 * available" → "Device gpu:3 is not available."). `plain` also drops the backticks around identifiers, for places
 * that cannot set them in mono (toasts, field messages). A message that starts with an identifier keeps its case
 * ("`claude` printed no result" → "claude printed no result.", never "Claude"): it is a name, not a word.
 */
export function sentence(s: string, plain = false): string {
  const raw = s.trim();
  const t = plain ? raw.replace(/`([^`]*)`/g, '$1') : raw;
  if (!t) return t;
  const cap = raw.startsWith('`') ? t : t[0]!.toUpperCase() + t.slice(1);
  return /[.!?]$/.test(cap) ? cap : `${cap}.`;
}

/** A confidence threshold on the 0.001 grid, as people set one: 0.312 → "0.31" (three decimals below 0.1: "0.045"). */
export function conf(x: number | null | undefined): string {
  return ok(x) ? num(x, x < 0.1 ? 3 : 2) : '—';
}

/**
 * Where precision and recall were read (docs/API.md, Operating point): "at conf 0.31 (best F1)" when the server says
 * which threshold (`best_f1_conf`), else the honest "at the best-F1 confidence" (older servers and files, or no class
 * with ground truth).
 */
export function atBestF1(c: number | null | undefined): string {
  return ok(c) ? `at conf ${conf(c)} (best F1)` : 'at the best-F1 confidence';
}

/** An hourly price in US dollars: 0.69 → "$0.69/h"; "—" for compute that is not billed (`null`). */
export function pricePerHour(x: number | null | undefined): string {
  return ok(x) ? `$${nf(2).format(x)}/h` : '—';
}

/** An amount in US dollars: "$0.42"; under a cent "<$0.01"; from $100 whole dollars ("$1,240"); "—" when unknown. */
export function usd(x: number | null | undefined): string {
  if (!ok(x)) return '—';
  if (x > 0 && x < 0.005) return '<$0.01';
  return `$${nf(Math.abs(x) >= 100 ? 0 : 2).format(x)}`;
}

/** A range of amounts: "$0.12–$0.93", one amount when both ends read the same. */
export function usdRange(lo: number, hi: number): string {
  const a = usd(Math.min(lo, hi));
  const b = usd(Math.max(lo, hi));
  return a === b ? a : `${a}–${b}`;
}

/** Hours as a time limit is set: 12 → "12 h", 0.5 → "0.5 h", 0.25 → "0.25 h". */
export function hoursText(h: number | null | undefined): string {
  if (!ok(h)) return '—';
  return `${new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(h)} h`;
}

/** A moment as a clock time in local time, "14:32", with the date when it is not today: "Nov 9, 14:32". */
export function timeOfDay(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  if (d.toDateString() !== new Date(now).toDateString()) return date(iso);
  return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
}
