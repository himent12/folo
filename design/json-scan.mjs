// The JSON a theme file may hold beyond what JSON.parse checks (design/THEMES.md, Custom themes): Desktop's serde_json
// refuses three things while parsing that JSON.parse accepts, so the JS readers (Studio's themes.ts, contrast.mjs)
// scan a file's decoded text for them first and refuse the file as not valid JSON, as Desktop does:
// - nesting deeper than serde_json's limit (JSON_MAX_DEPTH arrays and objects, the top-level value counted);
// - a lone surrogate escape (`\uD800` without its pair) in any string, keys included, checked on the raw escapes, so a
//   later duplicate member can't hide one (JSON.parse keeps only the last value of a repeated key);
// - a number outside a double's range (`1e400`), which JSON.parse reads as Infinity. Both sides round decimals
//   correctly (Desktop builds serde_json with `float_roundtrip`), so the limit and every value agree to the last bit:
//   `1.7976931348623158e308` is the largest double, `1.7976931348623159e308` is out of range.
// One pass over the characters with a depth counter, no recursion: a deeply nested file can't overflow the stack.
// Everything else (syntax) is JSON.parse's to report.

/** The deepest nesting serde_json accepts: 127 arrays and objects, the top-level object counted (128 is refused). */
export const JSON_MAX_DEPTH = 127;

const isHex = (c) => (c >= 0x30 && c <= 0x39) || (c >= 0x41 && c <= 0x46) || (c >= 0x61 && c <= 0x66);
const isHigh = (u) => u >= 0xd800 && u <= 0xdbff;
const isLow = (u) => u >= 0xdc00 && u <= 0xdfff;
/** The characters a JSON number is made of: a run of them is one number token (or not JSON, which JSON.parse says). */
const isNumberChar = (c) => (c >= 0x30 && c <= 0x39) || c === 0x2d || c === 0x2b || c === 0x2e || c === 0x65 || c === 0x45;

const DEEP = 'nested too deep';
const LONE = 'a lone surrogate in a string';
const RANGE = 'a number out of range';

/**
 * Why `text` is not JSON Desktop would accept, though JSON.parse might (the detail of `(file): not valid JSON (…)`), or
 * undefined. Text that is not JSON at all may pass: JSON.parse refuses it next.
 */
export function jsonProblem(text) {
  const n = text.length;
  let depth = 0;
  let i = 0;
  while (i < n) {
    const c = text.charCodeAt(i);
    if (c === 0x22) {
      // A string: every `\uXXXX` escape of a surrogate must be a high one followed at once by a low one, as serde_json
      // requires; a raw surrogate (text that never was UTF-8) must be paired too.
      i++;
      let high = false;
      for (;;) {
        if (i >= n) return undefined;
        const d = text.charCodeAt(i);
        if (d === 0x5c && text.charCodeAt(i + 1) === 0x75) {
          let u = 0;
          for (let k = i + 2; k < i + 6; k++) {
            const h = text.charCodeAt(k);
            if (!isHex(h)) return undefined;
            u = u * 16 + parseInt(text[k], 16);
          }
          if (high ? !isLow(u) : isLow(u)) return LONE;
          high = !high && isHigh(u);
          i += 6;
          continue;
        }
        if (high) return LONE;
        if (d === 0x22) break;
        if (d === 0x5c) i += 2;
        else if (isHigh(d) && isLow(text.charCodeAt(i + 1))) i += 2;
        else if (isHigh(d) || isLow(d)) return LONE;
        else i++;
      }
      i++;
    } else if (c === 0x5b || c === 0x7b) {
      if (++depth > JSON_MAX_DEPTH) return DEEP;
      i++;
    } else if (c === 0x5d || c === 0x7d) {
      depth--;
      i++;
    } else if (c === 0x2d || (c >= 0x30 && c <= 0x39)) {
      const start = i;
      while (i < n && isNumberChar(text.charCodeAt(i))) i++;
      const v = Number(text.slice(start, i));
      if (v === Infinity || v === -Infinity) return RANGE;
    } else i++;
  }
  return undefined;
}
