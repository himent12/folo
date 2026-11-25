//! YOLO label text parsing (boxes and segment polygons).

use crate::bbox::BBox;
use std::collections::HashSet;

/// Labels parsed from one YOLO `.txt` file.
#[derive(Debug, Default)]
pub(crate) struct ParsedLabels {
    pub boxes: Vec<BBox>,
    /// Non-fatal problems (duplicates removed, degenerate boxes dropped, coordinates clipped).
    pub warnings: Vec<String>,
}

/// Class ids at or above this are rejected, so a malformed label cannot make the reader allocate names for
/// billions of classes.
pub(crate) const MAX_CLASSES: usize = 65_536;

/// Coordinates may exceed `[0, 1]` by this much (rounding in exporters) before the file counts as broken.
const COORD_TOLERANCE: f32 = 0.01;
/// Overshoot below this is treated as print rounding: kept as written, no warning.
const ROUNDING_SLACK: f32 = 1e-5;
/// Box count at which duplicate detection switches from L1 linear scan to a hash set/map.
pub(crate) const DEDUP_HASH_THRESHOLD: usize = 32;

/// Exact hashable key for a finite [`BBox`] (`-0.0` canonicalised to `+0.0` so equality matches `BBox::eq`).
pub(crate) type BoxKey = (u32, u32, u32, u32, u32);

#[inline]
fn canon_f32(v: f32) -> u32 {
    if v == 0.0 { 0 } else { v.to_bits() }
}

/// Canonical hash key matching [`BBox`]'s `PartialEq` for finite boxes.
#[inline]
pub(crate) fn box_key(b: &BBox) -> BoxKey {
    (b.class, canon_f32(b.cx), canon_f32(b.cy), canon_f32(b.w), canon_f32(b.h))
}

/// Why a YOLO label line makes its file broken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LineError {
    /// Not numbers, or the wrong number of values.
    Malformed,
    /// A class id that is not a non-negative integer below the class count (or [`MAX_CLASSES`]).
    ClassId,
    /// Coordinates clearly outside `[0, 1]` (not normalised), or a negative size.
    OutOfRange,
}

/// What one non-empty YOLO label line holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Line {
    /// A usable box. `clipped` is true when it poked slightly out of the image and was clipped.
    Box { b: BBox, clipped: bool },
    /// A box with no area once clipped to the image (dropped with a warning).
    Degenerate { clipped: bool },
}

#[inline]
fn parse_finite(tok: &str) -> Result<f32, (LineError, String)> {
    match tok.parse::<f32>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err((LineError::Malformed, format!("not a finite number: {tok:?}"))),
    }
}

/// Parses the class token on a YOLO label line: fast-paths plain ASCII integer ids below [`MAX_CLASSES`], falling
/// back to `f32` for float/negative/oversized/malformed tokens so diagnostics stay exact.
#[inline]
fn parse_class_token(tok: &str, num_classes: Option<usize>) -> Result<u32, (LineError, String)> {
    let class = if !tok.is_empty() && tok.bytes().all(|b| b.is_ascii_digit()) {
        match tok.parse::<u32>() {
            Ok(v) if (v as usize) < MAX_CLASSES => v,
            _ => parse_class_f32(parse_finite(tok)?)?,
        }
    } else {
        parse_class_f32(parse_finite(tok)?)?
    };
    if let Some(nc) = num_classes
        && class as usize >= nc
    {
        return Err((LineError::ClassId, format!("class id {class} out of range for {nc} classes")));
    }
    Ok(class)
}

#[cold]
fn parse_class_f32(c: f32) -> Result<u32, (LineError, String)> {
    if c < 0.0 || c.fract() != 0.0 || c > u32::MAX as f32 {
        return Err((LineError::ClassId, format!("class id must be a non-negative integer, got {c}")));
    }
    if c as usize >= MAX_CLASSES {
        return Err((LineError::ClassId, format!("class id {c} is implausibly large (limit {MAX_CLASSES})")));
    }
    Ok(c as u32)
}

#[inline]
fn finalize_box(
    class: u32,
    xyxy: [f32; 4],
    exact_orig: Option<[f32; 4]>,
) -> Result<Line, (LineError, String)> {
    let [x1, y1, x2, y2] = xyxy;
    let lo = -COORD_TOLERANCE;
    let hi = 1.0 + COORD_TOLERANCE;
    if x1 < lo || y1 < lo || x2 > hi || y2 > hi {
        return Err((LineError::OutOfRange, "coordinates out of [0, 1]: labels must be normalised".into()));
    }
    // Overshoot from printing normalised values with a few decimals is noise, not a label problem.
    let exact = x1 >= -ROUNDING_SLACK
        && y1 >= -ROUNDING_SLACK
        && x2 <= 1.0 + ROUNDING_SLACK
        && y2 <= 1.0 + ROUNDING_SLACK;
    let (cx1, cy1, cx2, cy2) =
        (x1.clamp(0.0, 1.0), y1.clamp(0.0, 1.0), x2.clamp(0.0, 1.0), y2.clamp(0.0, 1.0));
    if cx2 - cx1 <= 0.0 || cy2 - cy1 <= 0.0 {
        return Ok(Line::Degenerate { clipped: !exact });
    }
    let b = match (exact, exact_orig) {
        (true, Some([cx, cy, w, h])) => BBox::new(class, cx, cy, w, h),
        _ => BBox::from_xyxy(class, [cx1, cy1, cx2, cy2]),
    };
    Ok(Line::Box { b, clipped: !exact })
}

/// Classifies one label line (`class cx cy w h` or a polygon `class x1 y1 x2 y2 x3 y3 ...`, normalised) without
/// heap allocations. A polygon becomes its bounding box.
pub(crate) fn parse_line(line: &str, num_classes: Option<usize>) -> Result<Line, (LineError, String)> {
    let mut it = line.split_ascii_whitespace();
    let Some(t0) = it.next() else {
        return Err((LineError::Malformed, "expected 5 values (class cx cy w h) or a polygon, got 0".into()));
    };
    // Parse every token's finiteness in order before checking token count so `"0 0.5 NaN"` reports `"NaN"`.
    let (t1, t2, t3, t4, t5) = (it.next(), it.next(), it.next(), it.next(), it.next());
    if let (Some(t1), Some(t2), Some(t3), Some(t4), None) = (t1, t2, t3, t4, t5) {
        // Fast path: standard 5-token `class cx cy w h` box on the stack.
        let fast_class = (!t0.is_empty() && t0.bytes().all(|b| b.is_ascii_digit()))
            .then(|| t0.parse::<u32>().ok().filter(|&v| (v as usize) < MAX_CLASSES))
            .flatten();
        if fast_class.is_none() {
            let _ = parse_finite(t0)?;
        }
        let cx = parse_finite(t1)?;
        let cy = parse_finite(t2)?;
        let w = parse_finite(t3)?;
        let h = parse_finite(t4)?;
        let class = match fast_class {
            Some(c) => {
                if let Some(nc) = num_classes
                    && c as usize >= nc
                {
                    return Err((LineError::ClassId, format!("class id {c} out of range for {nc} classes")));
                }
                c
            }
            None => parse_class_token(t0, num_classes)?,
        };
        if w < 0.0 || h < 0.0 {
            return Err((LineError::OutOfRange, format!("negative box size (w={w}, h={h})")));
        }
        let xyxy = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
        return finalize_box(class, xyxy, Some([cx, cy, w, h]));
    }

    // General path: polygons (`>= 7` odd values) or wrong-count diagnostics, streaming min/max with zero Vec.
    let c0 = parse_finite(t0)?;
    let mut n = 1usize;
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for tok in [t1, t2, t3, t4, t5].into_iter().flatten().chain(it) {
        let v = parse_finite(tok)?;
        if n % 2 == 1 {
            min_x = min_x.min(v);
            max_x = max_x.max(v);
        } else {
            min_y = min_y.min(v);
            max_y = max_y.max(v);
        }
        n += 1;
    }
    if n < 7 || n.is_multiple_of(2) {
        let detail = if n >= 6 && n.is_multiple_of(2) {
            format!(
                "expected 5 values (class cx cy w h) or a polygon (odd count >= 7: class x1 y1 ...), got {n}"
            )
        } else {
            format!("expected 5 values (class cx cy w h) or a polygon, got {n}")
        };
        return Err((LineError::Malformed, detail));
    }
    let class = parse_class_f32(c0)?;
    if let Some(nc) = num_classes
        && class as usize >= nc
    {
        return Err((LineError::ClassId, format!("class id {class} out of range for {nc} classes")));
    }
    finalize_box(class, [min_x, min_y, max_x, max_y], None)
}

/// `text` without a leading UTF-8 byte order mark (Notepad and PowerShell 5.1 write one; PyYAML and Python's
/// `utf-8-sig` skip it too).
pub(crate) fn strip_bom(text: &str) -> &str {
    text.strip_prefix('\u{feff}').unwrap_or(text)
}

/// Parses YOLO label text (see [`parse_line`]). Returns `Err((line, message))` for a broken file.
pub(crate) fn parse_yolo(text: &str, num_classes: Option<usize>) -> Result<ParsedLabels, (usize, String)> {
    let text = strip_bom(text);
    let est = if text.is_empty() {
        0
    } else {
        let lines = text.as_bytes().iter().filter(|&&b| b == b'\n').count() + 1;
        lines.min(text.len() / 9 + 1)
    };
    let mut out = ParsedLabels { boxes: Vec::with_capacity(est), warnings: Vec::new() };
    let mut seen: Option<HashSet<BoxKey>> = None;
    let mut clipped = 0usize;
    let mut degenerate = 0usize;
    for (ln, line) in text.lines().enumerate() {
        let line_no = ln + 1;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_line(line, num_classes).map_err(|(_, m)| (line_no, m))? {
            Line::Degenerate { clipped: c } => {
                clipped += usize::from(c);
                degenerate += 1;
            }
            Line::Box { b, clipped: c } => {
                clipped += usize::from(c);
                let dup = if let Some(set) = seen.as_mut() {
                    !set.insert(box_key(&b))
                } else if out.boxes.contains(&b) {
                    true
                } else {
                    if out.boxes.len() + 1 >= DEDUP_HASH_THRESHOLD {
                        let mut set = HashSet::with_capacity(est.max(DEDUP_HASH_THRESHOLD * 2));
                        for prev in &out.boxes {
                            set.insert(box_key(prev));
                        }
                        set.insert(box_key(&b));
                        seen = Some(set);
                    }
                    false
                };
                if dup {
                    out.warnings.push(format!("line {line_no}: duplicate label removed"));
                    continue;
                }
                out.boxes.push(b);
            }
        }
    }
    if clipped > 0 {
        out.warnings.push(format!("{clipped} box(es) slightly outside the image were clipped"));
    }
    if degenerate > 0 {
        out.warnings.push(format!("{degenerate} zero-size box(es) dropped"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxes_and_polygons() {
        let text = "\u{feff}0 0.5 0.5 0.2 0.4\r\n\n  1\t0.1 0.1 0.3 0.1 0.3 0.5 0.1 0.5\n";
        let p = parse_yolo(text, Some(2)).unwrap();
        assert_eq!(p.boxes.len(), 2);
        assert_eq!(p.boxes[0], BBox::new(0, 0.5, 0.5, 0.2, 0.4));
        let b = p.boxes[1];
        assert_eq!(b.class, 1);
        assert!((b.cx - 0.2).abs() < 1e-6 && (b.cy - 0.3).abs() < 1e-6);
        assert!((b.w - 0.2).abs() < 1e-6 && (b.h - 0.4).abs() < 1e-6);
        assert!(p.warnings.is_empty());
    }

    #[test]
    fn broken_and_warnings() {
        assert_eq!(parse_yolo("0 0.5 0.5 0.2\n", None).unwrap_err().0, 1);
        assert!(parse_yolo("0 0.5 0.5 0.2 x\n", None).is_err());
        assert!(parse_yolo("0 0.5 0.5 0.2 NaN\n", None).is_err());
        assert!(parse_yolo("0 0.5 0.5 0.2 inf\n", None).is_err());
        assert!(parse_yolo("-1 0.5 0.5 0.2 0.2\n", None).is_err());
        assert!(parse_yolo("1.5 0.5 0.5 0.2 0.2\n", None).is_err());
        assert!(parse_yolo("3 0.5 0.5 0.2 0.2\n", Some(3)).is_err());
        assert!(parse_yolo("1000000000 0.5 0.5 0.2 0.2", None).is_err());
        assert!(parse_yolo("0 50 60 20 20\n", None).is_err(), "pixel coordinates are rejected");
        let neg = parse_yolo("0 0.5 0.5 -0.1 0.2\n", None).unwrap_err();
        assert!(neg.1.contains("negative box size") && neg.1.contains("-0.1"), "{neg:?}");
        let even_poly = parse_yolo("0 0.1 0.1 0.2 0.2 0.3\n", None).unwrap_err();
        assert!(even_poly.1.contains("odd count >= 7") && even_poly.1.contains("got 6"), "{even_poly:?}");
        // Bad token on a short line still reports the non-finite token first.
        let bad_short = parse_yolo("0 0.5 NaN\n", None).unwrap_err();
        assert!(bad_short.1.contains("not a finite number"), "{bad_short:?}");
        // Integral float class id is accepted identically to integer.
        let float_cls = parse_yolo("2.0 0.5 0.5 0.2 0.2\n", Some(3)).unwrap();
        assert_eq!(float_cls.boxes, vec![BBox::new(2, 0.5, 0.5, 0.2, 0.2)]);
        let p =
            parse_yolo("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0 0.2\n0 0.995 0.5 0.02 0.1\n", None)
                .unwrap();
        assert_eq!(p.boxes.len(), 2);
        assert_eq!(p.warnings.len(), 3);
        assert!(parse_yolo("", None).unwrap().boxes.is_empty());
    }

    #[test]
    fn dense_file_dedup_handles_hash_promotion_and_negative_zero() {
        use std::fmt::Write;
        let mut text = String::new();
        for i in 0..80 {
            let cx = 0.05 + (i as f32) * 0.01;
            let _ = writeln!(text, "{} {cx:.4} 0.5 0.02 0.02", i % 4);
        }
        // Duplicate of line 1 and line 45, plus `-0.0` vs `0.0` centre edge case.
        text.push_str("0 0.0500 0.5 0.02 0.02\n");
        text.push_str("1 0.5000 0.5 0.02 0.02\n");
        text.push_str("0 0.0 0.5 0.02 0.02\n");
        text.push_str("0 -0.0 0.5 0.02 0.02\n");
        let p = parse_yolo(&text, Some(4)).unwrap();
        // 80 distinct + 1 clipped edge box (`cx=0.0`, `w=0.02` -> clipped to `[0, 0.01]`, same as `cx=-0.0`).
        assert_eq!(p.boxes.len(), 81);
        assert_eq!(p.warnings.iter().filter(|w| w.contains("duplicate")).count(), 3, "{:?}", p.warnings);
    }
}
