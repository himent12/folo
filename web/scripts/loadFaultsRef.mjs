// The reference of scripts/load-faults.mjs: what a clean cold load of a screen had loaded when it first showed. Every
// case is checked against it, so a reference missing a stylesheet (taken too early, a build that split differently)
// would let a case pass without it.

/**
 * What the reference `sheets` (stylesheet paths, `/assets/<name>-<hash>.css`) of the screen titled `title` lacks: the
 * screen's own stylesheet (`<title>-*.css`, which the bundler names after the screen's module) and the shared one
 * (`shared-*.css`, every screen's ui parts). Empty when it has both.
 * @param {string} title
 * @param {string[]} sheets
 */
export function referenceGaps(title, sheets) {
  const has = (/** @type {string} */ name) => sheets.some((s) => new RegExp(`/assets/${name}-[^/]+\\.css$`).test(s));
  return [`${title}-*.css`, 'shared-*.css'].filter((n) => !has(n.slice(0, -'-*.css'.length)));
}
