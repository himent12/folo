// Reading `folo dataset check` issues from GET /api/datasets/{id}/issues, by severity and optionally one kind.

import type { Api } from '$lib/api/client';
import type { DatasetIssue, DatasetIssues, DatasetIssuesQuery, IssueCounts, IssueKind, IssueSeverity } from '$lib/api/types';

/** Every kind has one severity (docs/CLI.md), so a severity is a set of kinds the endpoint can filter by. */
export const KINDS: Record<IssueSeverity, IssueKind[]> = {
  error: ['broken_label', 'class_overflow', 'out_of_range_box', 'corrupt_image', 'missing_image'],
  warning: ['clipped_box', 'zero_size_box', 'duplicate_label', 'duplicate_image', 'other'],
  info: ['empty_image'],
};

/** Kind names as a filter lists them. */
export const KIND_LABEL: Record<IssueKind, string> = {
  broken_label: 'Broken label',
  class_overflow: 'Class out of range',
  out_of_range_box: 'Box out of range',
  corrupt_image: 'Corrupt image',
  missing_image: 'Missing image',
  clipped_box: 'Clipped box',
  zero_size_box: 'Zero-size box',
  duplicate_label: 'Duplicate label',
  duplicate_image: 'Duplicate image',
  other: 'Other',
  empty_image: 'Empty image',
};

export interface LoadedIssues {
  /** The server's count of matches, so a caller can say when the list is incomplete. */
  total: number;
  list: DatasetIssue[];
  /** Issues per kind of the severity: from `counts`, or (older servers, unfiltered) each kind's total. */
  byKind?: Partial<Record<IssueKind, number>>;
  /** The whole dataset's counts, from servers that send them. */
  counts?: IssueCounts;
}

type Read = Pick<Api, 'datasetIssues'>;

/** Pages one query until it is exhausted or `cap` issues are held, starting from an answer already read. */
async function pages(api: Read, id: string, q: DatasetIssuesQuery, cap: number, list: DatasetIssue[], first: DatasetIssues, signal?: AbortSignal) {
  let r = first;
  for (let offset = 0; ; ) {
    list.push(...r.issues.slice(0, Math.max(0, cap - list.length)));
    offset += r.issues.length;
    if (!r.issues.length || offset >= r.total || list.length >= cap) return;
    r = await api.datasetIssues(id, { ...q, offset, limit: 1000 }, signal);
  }
}

/**
 * The issues of `severity` (of one `kind` when given), page by page until exhausted or `cap` issues are held.
 * Servers that send `counts` filter by `?severity=`; older ones ignore it, so their issues are read kind by kind
 * (the filter every server has). Sorted by file, then line (dataset order).
 */
export async function loadIssues(
  api: Read,
  id: string,
  severity: IssueSeverity,
  cap = 2000,
  signal?: AbortSignal,
  kind?: IssueKind,
): Promise<LoadedIssues> {
  const list: DatasetIssue[] = [];
  const pick = (c: IssueCounts) => Object.fromEntries(KINDS[severity].filter((k) => c.by_kind[k]).map((k) => [k, c.by_kind[k]!]));
  let out: LoadedIssues;
  if (kind) {
    const q = { kind };
    const first = await api.datasetIssues(id, { ...q, offset: 0, limit: 1000 }, signal);
    await pages(api, id, q, cap, list, first, signal);
    out = { total: first.total, list, counts: first.counts, byKind: first.counts ? pick(first.counts) : undefined };
  } else {
    const q = { severity };
    const first = await api.datasetIssues(id, { ...q, offset: 0, limit: 1000 }, signal);
    if (first.counts) {
      await pages(api, id, q, cap, list, first, signal);
      out = { total: first.total, list, counts: first.counts, byKind: pick(first.counts) };
    } else {
      let total = 0;
      const byKind: Partial<Record<IssueKind, number>> = {};
      for (const k of KINDS[severity]) {
        const kq = { kind: k };
        const r = await api.datasetIssues(id, { ...kq, offset: 0, limit: 1000 }, signal);
        total += r.total;
        if (r.total) byKind[k] = r.total;
        await pages(api, id, kq, cap, list, r, signal);
      }
      out = { total, list, byKind };
    }
  }
  list.sort((a, b) => (a.file === b.file ? (a.line ?? 0) - (b.line ?? 0) : a.file < b.file ? -1 : 1));
  return out;
}

const NOUN: Record<IssueSeverity, [string, string]> = { error: ['error', 'errors'], warning: ['warning', 'warnings'], info: ['note', 'notes'] };
const MEANS: Record<IssueSeverity, string> = {
  error: 'these files are left out of training until fixed',
  warning: 'training still uses these files',
  info: 'nothing to fix',
};

/**
 * The line above the issue list, saying exactly what its rows are: "3 warnings: training still uses these files",
 * "2 clipped box warnings: …" with a kind picked. Null when there are none (the list says so itself).
 */
export function issueCaption(severity: IssueSeverity, total: number, kind?: IssueKind, bare = false): string | null {
  if (total <= 0) return null;
  const noun = NOUN[severity][total === 1 ? 0 : 1];
  const what = kind ? `${KIND_LABEL[kind].toLowerCase()} ${noun}` : noun;
  // `bare`: under a finding that already says what they mean, only the count and what they are.
  return bare ? `${total.toLocaleString('en-US')} ${what}` : `${total.toLocaleString('en-US')} ${what}: ${MEANS[severity]}.`;
}
