/** How the inspector names an image's file: its name, its folders, and the folders shortened to fit a width. */
export const stem = (f: string) => f.split('/').pop() ?? f;
/** The folders of a path ("images/train"): the part a middle ellipsis shortens. */
export const dir = (f: string) => f.slice(0, Math.max(0, f.length - stem(f).length - 1));

/** The folders cut to whole mono cells (`chW` px each) so the path fits `pathW` px beside its copy button. */
export function shortDir(f: string, pathW: number, chW: number): { text: string; cut: boolean } {
  const d = dir(f);
  if (!d || !pathW || !chW) return { text: d, cut: false };
  // 24 px copy button and 4 px gap beside the path; the ellipsis takes about one cell.
  const fit = Math.floor((pathW - 28) / chW) - stem(f).length - 1;
  return d.length <= fit ? { text: d, cut: false } : { text: d.slice(0, Math.max(1, fit - 1)), cut: true };
}
