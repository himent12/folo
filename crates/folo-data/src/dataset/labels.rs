//! YOLO label text parsing (boxes and segment polygons).

use crate::bbox::BBox;

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

/// Classifies one label line (`class cx cy w h` or a polygon `class x1 y1 x2 y2 x3 y3 ...`, normalised). `vals` is
/// scratch space. A polygon becomes its bounding box.
pub(crate) fn parse_line(
    line: &str,
    num_classes: Option<usize>,
    vals: &mut Vec<f32>,
) -> Result<Line, (LineError, String)> {
    vals.clear();
    for tok in line.split_ascii_whitespace() {
        match tok.parse::<f32>() {
            Ok(v) if v.is_finite() => vals.push(v),
            _ => return Err((LineError::Malformed, format!("not a finite number: {tok:?}"))),
        }
    }
    let n = vals.len();
    if n != 5 && (n < 7 || n.is_multiple_of(2)) {
        return Err((
            LineError::Malformed,
            format!("expected 5 values (class cx cy w h) or a polygon, got {n}"),
        ));
    }
    let c = vals[0];
    if c < 0.0 || c.fract() != 0.0 || c > u32::MAX as f32 {
        return Err((LineError::ClassId, format!("class id must be a non-negative integer, got {c}")));
    }
    if c as usize >= MAX_CLASSES {
        return Err((LineError::ClassId, format!("class id {c} is implausibly large (limit {MAX_CLASSES})")));
    }
    let class = c as u32;
    if let Some(nc) = num_classes
        && class as usize >= nc
    {
        return Err((LineError::ClassId, format!("class id {class} out of range for {nc} classes")));
    }
    let lo = -COORD_TOLERANCE;
    let hi = 1.0 + COORD_TOLERANCE;
    let [x1, y1, x2, y2] = if n == 5 {
        let (cx, cy, w, h) = (vals[1], vals[2], vals[3], vals[4]);
        if w < 0.0 || h < 0.0 {
            return Err((LineError::OutOfRange, "negative box size".into()));
        }
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
    } else {
        let (mut a, mut b, mut c2, mut d) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in vals[1..].as_chunks::<2>().0 {
            a = a.min(p[0]);
            c2 = c2.max(p[0]);
            b = b.min(p[1]);
            d = d.max(p[1]);
        }
        [a, b, c2, d]
    };
    if x1 < lo || y1 < lo || x2 > hi || y2 > hi {
        return Err((LineError::OutOfRange, "coordinates out of [0, 1]: labels must be normalised".into()));
    }
    // Overshoot from printing normalised values with a few decimals is noise, not a label problem.
    let exact =
        [x1, y1].iter().all(|&v| v >= -ROUNDING_SLACK) && [x2, y2].iter().all(|&v| v <= 1.0 + ROUNDING_SLACK);
    let c = |v: f32| v.clamp(0.0, 1.0);
    let (cx1, cy1, cx2, cy2) = (c(x1), c(y1), c(x2), c(y2));
    if cx2 - cx1 <= 0.0 || cy2 - cy1 <= 0.0 {
        return Ok(Line::Degenerate { clipped: !exact });
    }
    let b = if n == 5 && exact {
        BBox::new(class, vals[1], vals[2], vals[3], vals[4])
    } else {
        BBox::from_xyxy(class, [cx1, cy1, cx2, cy2])
    };
    Ok(Line::Box { b, clipped: !exact })
}

/// Parses YOLO label text (see [`parse_line`]). Returns `Err((line, message))` for a broken file.
pub(crate) fn parse_yolo(text: &str, num_classes: Option<usize>) -> Result<ParsedLabels, (usize, String)> {
    let mut out = ParsedLabels::default();
    let mut vals: Vec<f32> = Vec::with_capacity(16);
    let mut clipped = 0usize;
    let mut degenerate = 0usize;
    for (ln, line) in text.lines().enumerate() {
        let line_no = ln + 1;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_line(line, num_classes, &mut vals).map_err(|(_, m)| (line_no, m))? {
            Line::Degenerate { clipped: c } => {
                clipped += usize::from(c);
                degenerate += 1;
            }
            Line::Box { b, clipped: c } => {
                clipped += usize::from(c);
                if out.boxes.contains(&b) {
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
        let text = "0 0.5 0.5 0.2 0.4\n\n  1 0.1 0.1 0.3 0.1 0.3 0.5 0.1 0.5\n";
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
        assert!(parse_yolo("-1 0.5 0.5 0.2 0.2\n", None).is_err());
        assert!(parse_yolo("1.5 0.5 0.5 0.2 0.2\n", None).is_err());
        assert!(parse_yolo("3 0.5 0.5 0.2 0.2\n", Some(3)).is_err());
        assert!(parse_yolo("1000000000 0.5 0.5 0.2 0.2", None).is_err());
        assert!(parse_yolo("0 50 60 20 20\n", None).is_err(), "pixel coordinates are rejected");
        assert!(parse_yolo("0 0.1 0.1 0.2 0.2 0.3\n", None).is_err());
        let p =
            parse_yolo("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0 0.2\n0 0.995 0.5 0.02 0.1\n", None)
                .unwrap();
        assert_eq!(p.boxes.len(), 2);
        assert_eq!(p.warnings.len(), 3);
        assert!(parse_yolo("", None).unwrap().boxes.is_empty());
    }
}
