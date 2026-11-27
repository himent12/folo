// Shell-safe `folo dataset` curation & export CLI command builders paired with DatasetInsights and StatsTab:
// `check` (perceptual dHash duplicate/leakage audit + dry-run fixes), `split` (stratified train/val),
// `filter` (class/split/empty-image curation), `merge` (multi-dataset unification), `convert` (YOLO/COCO/VOC),
// and `export` (spec-compliant COCO instances JSON with split selection, copy/link/reference images, pretty JSON).
// Every subcommand and flag matches crates/folo-cli/src/cli/dataset.rs and docs/CLI.md verbatim.
import type { Dataset, DatasetStats, Split } from '$lib/api/types';
import { quote } from '$lib/trainCli';

export type CurateCliMode = 'check' | 'split' | 'filter' | 'merge' | 'convert' | 'export';
export type ConvertTargetFormat = 'yolo' | 'coco' | 'voc';
export type CocoImageMode = 'copy' | 'link' | 'reference';

export interface CurateCliCommand {
  mode: CurateCliMode;
  label: string;
  command: string;
  description: string;
  /** Non-null when a class name or path cannot be represented safely in PowerShell / clap flags. */
  warning: string | null;
  /** Exact placeholder tokens inserted by this builder (never literal quoted paths or class names). */
  placeholders: readonly string[];
}

export const CURATE_CLI_MODES: readonly { value: CurateCliMode; label: string }[] = [
  { value: 'check', label: 'Audit & dedup' },
  { value: 'split', label: 'Split' },
  { value: 'filter', label: 'Filter' },
  { value: 'merge', label: 'Merge' },
  { value: 'convert', label: 'Convert' },
  { value: 'export', label: 'Export COCO' },
];

export const CONVERT_TARGET_FORMATS: readonly { value: ConvertTargetFormat; label: string }[] = [
  { value: 'yolo', label: 'YOLO' },
  { value: 'coco', label: 'COCO' },
  { value: 'voc', label: 'VOC' },
];

export const COCO_IMAGE_MODES: readonly { value: CocoImageMode; label: string }[] = [
  { value: 'copy', label: 'Copy' },
  { value: 'link', label: 'Hard-link' },
  { value: 'reference', label: 'Reference' },
];

/** Picks the most useful default conversion target format for a dataset (YOLO for COCO datasets, else COCO). */
export function defaultConvertTarget(format: string): ConvertTargetFormat {
  return format.toLowerCase() === 'coco' ? 'yolo' : 'coco';
}

/**
 * Windows PowerShell 5.1 strips literal double quotes (`"`) when forwarding single-quoted strings to native
 * executables, and raw CR/LF breaks one-line commands.
 */
const UNSAFE_PS_NATIVE_ARG = /["\r\n]/;

/** Formats a flag + value safely even if `value` begins with `-`. */
function flagValue(flag: string, value: string): string {
  return value.startsWith('-') ? quote(`${flag}=${value}`) : `${flag} ${quote(value)}`;
}

/**
 * Normalizes a dataset path so Windows PowerShell 5.1 passes it verbatim to native executables:
 * - A literal `--%` folder name is prefixed with `./` because PowerShell 5.1 strips `--%` even inside single quotes.
 * - A trailing backslash on a path containing spaces (or any non-root path) is trimmed or converted to `/` so
 *   Windows CRT `CommandLineToArgvW` never interprets `\"` as an escaped quote absorbing subsequent flags.
 */
export function normalizeCliPath(raw: string): string {
  if (raw === '--%') return './--%';
  if (/^[A-Za-z]:\\+$/.test(raw)) return `${raw.slice(0, 2)}/`;
  if (raw.length > 1 && raw.endsWith('\\')) {
    const stripped = raw.replace(/\\+$/, '');
    return stripped.length > 0 ? stripped : raw;
  }
  return raw;
}

/** Quotes a positional dataset path; prefixes `--` when the path starts with `-` so clap never reads it as a flag. */
function positionalPath(path: string, trailingFlags: string[]): string[] {
  const safe = normalizeCliPath(path);
  if (safe.startsWith('-')) return [...trailingFlags, '--', quote(safe)];
  return [quote(safe), ...trailingFlags];
}

/** Suggests a sibling output folder name derived cleanly from the dataset name (never inside the input folder). */
function suggestedOut(dataset: Dataset, suffix: string): string {
  const slug =
    dataset.name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 48) || 'dataset';
  return `<out_dir>/${slug}-${suffix}`;
}

/** Builds a shell-safe `folo dataset` curation or format-conversion command for the loaded dataset and filter context. */
export function buildDatasetCurateCli(
  dataset: Dataset,
  mode: CurateCliMode,
  opts: {
    stats?: DatasetStats;
    activeClass?: number;
    activeSplit?: 'all' | Split;
    targetFormat?: ConvertTargetFormat;
    cocoImageMode?: CocoImageMode;
    cocoPretty?: boolean;
  } = {},
): CurateCliCommand {
  const pathUnsafe = !!dataset.path && UNSAFE_PS_NATIVE_ARG.test(dataset.path);
  const hasRealPath = !!dataset.path && !pathUnsafe;
  const path = hasRealPath ? dataset.path : '<dataset_path>';
  const quoted = (p: string) => (hasRealPath ? quote(p) : p);
  const pathWarning = pathUnsafe
    ? `Dataset "${dataset.name}" has a path with a quote or newline that Windows PowerShell cannot pass verbatim; replace <dataset_path> with your shell-quoted path.`
    : null;

  const pathPlaceholder = hasRealPath ? [] : ['<dataset_path>'];

  if (mode === 'check') {
    const flags = ['--perceptual', '--fix-dry-run'];
    const args = hasRealPath ? positionalPath(path, flags) : [path, ...flags];
    return {
      mode,
      label: 'Audit & dedup',
      command: `folo dataset check ${args.join(' ')}`,
      description:
        'Checks broken labels, corrupt images, exact byte duplicates, and 64-bit dHash perceptual near-duplicate / cross-split leakage (dry-run fix notes; modifies nothing).',
      warning: pathWarning,
      placeholders: pathPlaceholder,
    };
  }

  if (mode === 'split') {
    const out = suggestedOut(dataset, 'split');
    const flags = [`--out ${out}`, '--val 0.2', '--seed 0'];
    const args = hasRealPath ? positionalPath(path, flags) : [path, ...flags];
    return {
      mode,
      label: 'Split',
      command: `folo dataset split ${args.join(' ')}`,
      description:
        dataset.splits.val === 0
          ? 'This dataset has no validation split yet. Builds a deterministic multi-label stratified 80/20 train/val split in a new folder.'
          : 'Re-splits into a deterministic multi-label stratified 80/20 train/val split in a new folder (adjust --val / --test ratios as needed).',
      warning: pathWarning,
      placeholders: [...pathPlaceholder, '<out_dir>'],
    };
  }

  if (mode === 'filter') {
    const out = suggestedOut(dataset, 'filtered');
    const flags = [`--out ${out}`];
    const activeName = opts.activeClass !== undefined ? dataset.classes[opts.activeClass] : undefined;
    let classWarning: string | null = null;
    let includedClass = false;
    if (activeName !== undefined) {
      if (activeName.includes(',')) {
        classWarning = `Class "${activeName}" contains a comma; folo dataset filter --include-class splits on commas, so --include-class was omitted.`;
      } else if (UNSAFE_PS_NATIVE_ARG.test(activeName)) {
        classWarning = `Class "${activeName}" contains a double quote or newline that Windows PowerShell 5.1 alters when passing native arguments, so --include-class was omitted.`;
      } else {
        flags.push(flagValue('--include-class', activeName));
        includedClass = true;
      }
    }
    const splitFilter = opts.activeSplit && opts.activeSplit !== 'all' ? opts.activeSplit : null;
    if (splitFilter) {
      flags.push(`--split ${splitFilter}`);
    }
    if ((opts.stats?.empty_images ?? 0) > 0 || (!includedClass && !splitFilter)) {
      flags.push('--drop-empty-images');
    }
    const args = hasRealPath ? positionalPath(path, flags) : [path, ...flags];
    const scopeParts: string[] = [];
    if (includedClass) scopeParts.push(`boxes of class "${activeName}"`);
    if (splitFilter) scopeParts.push(`the ${splitFilter} split`);
    const desc = scopeParts.length
      ? `Exports only ${scopeParts.join(' from ')}${opts.stats?.empty_images ? ' and drops empty images' : ''} into a new dataset folder.`
      : 'Filters classes, splits, box dimensions, or empty background images into a new dataset folder (add --include-class or --exclude-class to narrow classes).';
    return {
      mode,
      label: 'Filter',
      command: `folo dataset filter ${args.join(' ')}`,
      description: desc,
      warning: [pathWarning, classWarning].filter(Boolean).join(' ') || null,
      placeholders: [...pathPlaceholder, '<out_dir>'],
    };
  }

  if (mode === 'convert') {
    const to = opts.targetFormat ?? defaultConvertTarget(dataset.format);
    const out = suggestedOut(dataset, to);
    const flags = [`--to ${to}`];
    const args = hasRealPath && path.startsWith('-') ? [...flags, '--', quoted(path), out] : [quoted(path), out, ...flags];
    const formatNames: Record<ConvertTargetFormat, string> = {
      yolo: 'YOLO (data.yaml + labels/*.txt)',
      coco: 'COCO (annotations/instances_<split>.json)',
      voc: 'Pascal VOC (Annotations/*.xml + ImageSets/Main/)',
    };
    return {
      mode,
      label: 'Convert',
      command: `folo dataset convert ${args.join(' ')}`,
      description: `Converts this ${dataset.format.toUpperCase()} dataset into ${formatNames[to]} in a new folder (images are copied unchanged).`,
      warning: pathWarning,
      placeholders: [...pathPlaceholder, '<out_dir>'],
    };
  }

  if (mode === 'export') {
    const imageMode: CocoImageMode = opts.cocoImageMode ?? 'copy';
    const splitFilter = opts.activeSplit && opts.activeSplit !== 'all' ? opts.activeSplit : null;
    const flags = ['--out <out_dir>', '--format coco'];
    if (splitFilter) flags.push(`--split ${splitFilter}`);
    flags.push(`--images ${imageMode}`);
    if (opts.cocoPretty) flags.push('--pretty');
    const args = hasRealPath ? positionalPath(path, flags) : [path, ...flags];
    const imageDesc: Record<CocoImageMode, string> = {
      copy: 'copies source images into <out_dir>/images/<split>/',
      link: 'hard-links source images into <out_dir>/images/<split>/ (falls back to copy across filesystems)',
      reference: 'references source images in place via relative paths without duplicating files',
    };
    const splitDesc = splitFilter ? `the ${splitFilter} split` : 'all non-empty splits';
    return {
      mode,
      label: 'Export COCO',
      command: `folo dataset export ${args.join(' ')}`,
      description: `Exports ${splitDesc} to spec-compliant COCO Detection JSON (annotations/instances_<split>.json) and ${imageDesc[imageMode]}${opts.cocoPretty ? ', pretty-printed' : ''}.`,
      warning: pathWarning,
      placeholders: [...pathPlaceholder, '<out_dir>'],
    };
  }

  // merge
  const out = suggestedOut(dataset, 'merged');
  const flags = [`--out ${out}`, '--splits stratified', '--val 0.2'];
  const inputs = hasRealPath && path.startsWith('-') ? [...flags, '--', quoted(path), '<other_dataset>'] : [quoted(path), '<other_dataset>', ...flags];
  return {
    mode,
    label: 'Merge',
    command: `folo dataset merge ${inputs.join(' ')}`,
    description:
      'Merges this dataset with another YOLO, COCO, or Pascal VOC folder using collision-safe filenames and stratified train/val splits.',
    warning: pathWarning,
    placeholders: [...pathPlaceholder, '<other_dataset>', '<out_dir>'],
  };
}

/** Formats copy-toast guidance: reminds users to replace placeholders before running so PowerShell never parses `<...>`. */
export function curateCliCopiedMessage(snippet: Pick<CurateCliCommand, 'label' | 'placeholders'>): string {
  const placeholders = snippet.placeholders;
  const name = snippet.label.toLowerCase();
  if (placeholders.length === 0) {
    return `Run this ${name} command from your terminal.`;
  }
  const list =
    placeholders.length === 1
      ? placeholders[0]!
      : `${placeholders.slice(0, -1).join(', ')} and ${placeholders[placeholders.length - 1]!}`;
  return `Replace ${list} before running this ${name} command.`;
}
