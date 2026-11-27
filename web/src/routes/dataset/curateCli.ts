// Shell-safe `folo dataset` curation & export CLI command builders paired with DatasetInsights and StatsTab:
// `check` (perceptual dHash duplicate/leakage audit + dry-run fixes), `split` (stratified train/val),
// `filter` (class/split/empty-image curation), `merge` (multi-dataset unification), and `convert` (YOLO/COCO/VOC export).
// Every subcommand and flag matches crates/folo-cli/src/cli/dataset.rs and docs/CLI.md verbatim.
import type { Dataset, DatasetStats, Split } from '$lib/api/types';
import { quote } from '$lib/trainCli';

export type CurateCliMode = 'check' | 'split' | 'filter' | 'merge' | 'convert';
export type ConvertTargetFormat = 'yolo' | 'coco' | 'voc';

export interface CurateCliCommand {
  mode: CurateCliMode;
  label: string;
  command: string;
  description: string;
  /** Non-null when a class name or path cannot be represented safely in PowerShell / clap flags. */
  warning: string | null;
}

export const CURATE_CLI_MODES: readonly { value: CurateCliMode; label: string }[] = [
  { value: 'check', label: 'Audit & dedup' },
  { value: 'split', label: 'Split' },
  { value: 'filter', label: 'Filter' },
  { value: 'merge', label: 'Merge' },
  { value: 'convert', label: 'Convert' },
];

export const CONVERT_TARGET_FORMATS: readonly { value: ConvertTargetFormat; label: string }[] = [
  { value: 'yolo', label: 'YOLO' },
  { value: 'coco', label: 'COCO' },
  { value: 'voc', label: 'VOC' },
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

/** Quotes a positional dataset path; prefixes `--` when the path starts with `-` so clap never reads it as a flag. */
function positionalPath(path: string, trailingFlags: string[]): string[] {
  if (path.startsWith('-')) return [...trailingFlags, '--', quote(path)];
  return [quote(path), ...trailingFlags];
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
  } = {},
): CurateCliCommand {
  const pathUnsafe = !!dataset.path && UNSAFE_PS_NATIVE_ARG.test(dataset.path);
  const hasRealPath = !!dataset.path && !pathUnsafe;
  const path = hasRealPath ? dataset.path : '<dataset_path>';
  const quoted = (p: string) => (hasRealPath ? quote(p) : p);
  const pathWarning = pathUnsafe
    ? `Dataset "${dataset.name}" has a path with a quote or newline that Windows PowerShell cannot pass verbatim; replace <dataset_path> with your shell-quoted path.`
    : null;

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
  };
}

/** Extracts unique angle-bracket placeholder tokens (e.g. `<out_dir>`, `<dataset_path>`) in appearance order. */
export function extractCliPlaceholders(command: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const match of command.matchAll(/<[a-z0-9_]+>/g)) {
    const token = match[0];
    if (!seen.has(token)) {
      seen.add(token);
      out.push(token);
    }
  }
  return out;
}

/** Formats copy-toast guidance: reminds users to replace placeholders before running so PowerShell never parses `<...>`. */
export function curateCliCopiedMessage(snippet: Pick<CurateCliCommand, 'label' | 'command'>): string {
  const placeholders = extractCliPlaceholders(snippet.command);
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
