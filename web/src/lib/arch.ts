// Architecture family and scale helpers shared across FOLO v1 (`folo-n..x`) and FOLO v2 (`folo2-n..x`).
import type { Arch, ArchFamily, ArchScale, SystemInfo } from '$lib/api/types';

/** Capability flag advertised by servers that accept `folo2-*` architectures (`crates/folo-server`). */
export const FOLO2_FEATURE = 'folo2_arch';

export const V1_ARCHES: readonly Arch[] = ['folo-n', 'folo-s', 'folo-m', 'folo-l', 'folo-x'];
export const V2_ARCHES: readonly Arch[] = ['folo2-n', 'folo2-s', 'folo2-m', 'folo2-l', 'folo2-x'];
export const ALL_ARCHES: readonly Arch[] = [...V1_ARCHES, ...V2_ARCHES];

const ARCH_SET: ReadonlySet<string> = new Set(ALL_ARCHES);
const SCALE_ORDER: readonly ArchScale[] = ['n', 's', 'm', 'l', 'x'];

/** Type guard for any valid FOLO v1 or FOLO v2 architecture string. */
export function isArch(v: string | null | undefined): v is Arch {
  return typeof v === 'string' && ARCH_SET.has(v);
}

/** True when `arch` belongs to the FOLO v2 (`folo2-*`) family. */
export function isFolo2Arch(arch: string): boolean {
  return arch.startsWith('folo2-');
}

/** Architecture family (`folo` for v1, `folo2` for v2). */
export function archFamily(arch: Arch): ArchFamily {
  return isFolo2Arch(arch) ? 'folo2' : 'folo';
}

/** Scale letter (`n`, `s`, `m`, `l`, or `x`). */
export function archScale(arch: Arch): ArchScale {
  return arch.slice(arch.lastIndexOf('-') + 1) as ArchScale;
}

/** Numeric scale rank `0..4` (`n=0`, `s=1`, `m=2`, `l=3`, `x=4`), or `-1` if unknown. */
export function archScaleRank(arch: string): number {
  const idx = arch.lastIndexOf('-');
  if (idx < 0) return -1;
  return SCALE_ORDER.indexOf(arch.slice(idx + 1) as ArchScale);
}

/** Switches an architecture between `folo` (v1) and `folo2` (v2) while keeping its scale (`n..x`). */
export function archWithFamily(arch: Arch, family: ArchFamily): Arch {
  return `${family}-${archScale(arch)}` as Arch;
}

/** Next smaller scale within the same architecture family (`folo2-m` → `folo2-s`), or undefined at `n`. */
export function smallerArchInFamily(arch: Arch): Arch | undefined {
  const list = isFolo2Arch(arch) ? V2_ARCHES : V1_ARCHES;
  const idx = list.indexOf(arch);
  return idx > 0 ? list[idx - 1] : undefined;
}

/** Next bigger scale within the same architecture family (`folo-s` → `folo-m`), or undefined at `x`. */
export function biggerArchInFamily(arch: Arch): Arch | undefined {
  const list = isFolo2Arch(arch) ? V2_ARCHES : V1_ARCHES;
  const idx = list.indexOf(arch);
  return idx >= 0 && idx + 1 < list.length ? list[idx + 1] : undefined;
}

/** Short run-name suffix matching `folo-server`'s `default_run_name_for_arch` (`folo-s` → `s`, `folo2-s` → `folo2-s`). */
export function archNameSuffix(arch: Arch): string {
  return arch.replace(/^folo-/, '');
}

/** Whether the server advertises `folo2_arch` (returns true while `system` is still loading to avoid premature lockouts). */
export function supportsFolo2(system: Pick<SystemInfo, 'features'> | null | undefined): boolean {
  return system === undefined || system === null || (system.features?.includes(FOLO2_FEATURE) ?? false);
}

/** Extracts a canonical FOLO v1/v2 architecture from a `folo train` CLI `argv` (`--model folo2-s` or `--model=folo2-s`). */
export function archFromCli(cmd: { argv?: readonly string[] } | null | undefined): Arch | null {
  const argv = cmd?.argv;
  if (!argv) return null;
  for (let i = 0; i < argv.length; i++) {
    const tok = argv[i]!;
    if (tok === '--model' && i + 1 < argv.length && isArch(argv[i + 1])) return argv[i + 1] as Arch;
    if (tok.startsWith('--model=')) {
      const val = tok.slice('--model='.length);
      if (isArch(val)) return val;
    }
  }
  return null;
}
