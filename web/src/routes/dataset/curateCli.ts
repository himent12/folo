// Shell-safe `folo dataset` curation CLI command builders paired with DatasetInsights:
// `check` (perceptual dHash duplicate/leakage audit + dry-run fixes), `split` (stratified train/val),
// `filter` (class/empty-image curation), and `merge` (multi-dataset unification).
// Every subcommand and flag matches crates/folo-cli/src/cli/dataset.rs and docs/CLI.md verbatim.
import type { Dataset, DatasetStats } from '$lib/api/types';
import { quote } from '$lib/trainCli';

export type CurateCliMode = 'check' | 'split' | 'filter' | 'merge';

export interface CurateCliCommand {
  mode: CurateCliMode;
  label: string;
  command: string;
  description: string;
}

export const CURATE_CLI_MODES: readonly { value: CurateCliMode; label: string }[] = [
  { value: 'check', label: 'Audit & dedup' },
  { value: 'split', label: 'Split' },
  { value: 'filter', label: 'Filter' },
  { value: 'merge', label: 'Merge' },
];

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

/** Builds a shell-safe `folo dataset` curation command for the loaded dataset and current filter context. */
export function buildDatasetCurateCli(
  dataset: Dataset,
  mode: CurateCliMode,
  opts: { stats?: DatasetStats; activeClass?: number } = {},
): CurateCliCommand {
  const path = dataset.path || `<path to ${dataset.name}>`;
  const hasRealPath = !!dataset.path;
  const quoted = (p: string) => (hasRealPath ? quote(p) : p);

  if (mode === 'check') {
    const flags = ['--perceptual', '--fix-dry-run'];
    const args = hasRealPath ? positionalPath(path, flags) : [path, ...flags];
    return {
      mode,
      label: 'Audit & dedup',
      command: `folo dataset check ${args.join(' ')}`,
      description:
        'Checks broken labels, corrupt images, exact byte duplicates, and 64-bit dHash perceptual near-duplicate / cross-split leakage (dry-run fix notes; modifies nothing).',
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
          : 'Re-splits into a deterministic multi-label stratified 80/20 train/val split in a new folder (adjust `--val` / `--test` ratios as needed).',
    };
  }

  if (mode === 'filter') {
    const out = suggestedOut(dataset, 'filtered');
    const flags = [`--out ${out}`];
    const activeName = opts.activeClass !== undefined ? dataset.classes[opts.activeClass] : undefined;
    if (activeName) {
      flags.push(flagValue('--include-class', activeName));
    }
    if ((opts.stats?.empty_images ?? 0) > 0 || !activeName) {
      flags.push('--drop-empty-images');
    }
    const args = hasRealPath ? positionalPath(path, flags) : [path, ...flags];
    const desc = activeName
      ? `Exports only boxes of class ${activeName}${opts.stats?.empty_images ? ' and drops empty images' : ''} into a new dataset folder.`
      : 'Filters classes, box dimensions, or empty background images into a new dataset folder (add `--include-class` or `--exclude-class` to narrow classes).';
    return {
      mode,
      label: 'Filter',
      command: `folo dataset filter ${args.join(' ')}`,
      description: desc,
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
  };
}
