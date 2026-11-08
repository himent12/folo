// Reading `folo dataset check` issues of one severity from GET /api/datasets/{id}/issues.

import type { Api } from '$lib/api/client';
import type { DatasetIssue, IssueKind, IssueSeverity } from '$lib/api/types';

/** Every kind has one severity (docs/CLI.md), so a severity is a set of kinds the endpoint can filter by. */
export const KINDS: Record<IssueSeverity, IssueKind[]> = {
  error: ['broken_label', 'class_overflow', 'out_of_range_box', 'corrupt_image', 'missing_image'],
  warning: ['clipped_box', 'zero_size_box', 'duplicate_label', 'duplicate_image', 'other'],
  info: ['empty_image'],
};

/**
 * All issues of `severity`, each kind read page by page until exhausted or `cap` issues are held. `total` is the
 * server's count, so a caller can say when the list is incomplete. Sorted by file, then line (dataset order).
 */
export async function loadIssues(
  api: Pick<Api, 'datasetIssues'>,
  id: string,
  severity: IssueSeverity,
  cap = 2000,
  signal?: AbortSignal,
): Promise<{ total: number; list: DatasetIssue[] }> {
  let total = 0;
  const list: DatasetIssue[] = [];
  for (const kind of KINDS[severity]) {
    for (let offset = 0; ; ) {
      const r = await api.datasetIssues(id, { kind, offset, limit: 1000 }, signal);
      if (offset === 0) total += r.total;
      list.push(...r.issues.slice(0, Math.max(0, cap - list.length)));
      offset += r.issues.length;
      if (!r.issues.length || offset >= r.total || list.length >= cap) break;
    }
  }
  list.sort((a, b) => (a.file === b.file ? (a.line ?? 0) - (b.line ?? 0) : a.file < b.file ? -1 : 1));
  return { total, list };
}
