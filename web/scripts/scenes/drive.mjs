// Steps the scenes share to drive a page: waits that advance the page's fake clock (the mock's latency runs on it)
// until something shows, and keyboard focus reached with Tab.

/**
 * Pages shot with `shoot.mjs --exact`, each with its wait for real-time work (image loads, blobs, lazy chunks) to finish.
 * A wait below calls it before each look, so whether a look sees the page done never depends on how fast the machine
 * is, and a wait takes the same number of clock steps on every shoot (the live run's progress is a function of them).
 * @type {WeakMap<import('playwright-core').Page, () => Promise<void>>}
 */
export const quietWaits = new WeakMap();

/** Waits for the page's real-time work, when the page has such a wait (--exact). */
export async function quiet(/** @type {import('playwright-core').Page} */ page) {
  await quietWaits.get(page)?.();
}

/**
 * Advances the page's fake clock until `selector` is visible (the mock's latency runs on that clock).
 * @param {import('playwright-core').Page} page
 * @param {string} selector
 */
export async function until(page, selector, ms = 8000) {
  for (let t = 0; t < ms; t += 250) {
    await quiet(page);
    if (await page.locator(selector).first().isVisible().catch(() => false)) return;
    await page.clock.runFor(250);
  }
  throw new Error(`timed out waiting for ${selector}`);
}

/** Advances the page's clock until `select` holds an option of `value` (options are never "visible"). */
export async function optionIn(/** @type {import('playwright-core').Page} */ page, select, value, ms = 8000) {
  for (let t = 0; t < ms; t += 250) {
    await quiet(page);
    if (await page.locator(`${select} option[value="${value}"]`).count()) return;
    await page.clock.runFor(250);
  }
  throw new Error(`timed out waiting for ${value} in ${select}`);
}

/** As `until`, also letting real time pass (an image under a condition is drawn on a canvas, in real time). */
export async function untilDrawn(/** @type {import('playwright-core').Page} */ page, selector, rounds = 80) {
  for (let i = 0; i < rounds; i++) {
    await quiet(page);
    if (await page.locator(selector).first().isVisible().catch(() => false)) return;
    await page.clock.runFor(250);
    await page.waitForTimeout(100);
  }
  throw new Error(`timed out waiting for ${selector}`);
}

/** Waits in real time for the page's images to decode (their natural sizes are read). */
export async function imagesDecoded(/** @type {import('playwright-core').Page} */ page) {
  for (let i = 0; i < 50; i++) {
    if (!(await page.evaluate(() => [...document.images].some((im) => !im.complete)))) return;
    await page.waitForTimeout(100);
  }
}

/**
 * Presses Tab until `selector` holds focus, as a keyboard user reaches it (so `:focus-visible` is the keyboard's).
 * @param {import('playwright-core').Page} page
 * @param {string} selector
 */
export async function tabTo(page, selector, max = 60) {
  await until(page, selector);
  for (let i = 0; i < max; i++) {
    if (await page.evaluate((sel) => !!document.activeElement?.matches(sel), selector)) return;
    await page.keyboard.press('Tab');
  }
  throw new Error(`Tab never reached ${selector}`);
}
