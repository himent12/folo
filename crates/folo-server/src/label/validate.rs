//! Checks of what clients send: names, class lists, instructions and notes, model names and drawn boxes.

use std::collections::HashSet;

use folo_api::BBox;

use crate::error::Result;

use super::{MAX_CLASS_CHARS, MAX_CLASSES, MAX_MODEL_CHARS, MAX_NAME_CHARS, invalid, prompt};

/// The model an agent is asked for: trimmed, `None` when blank, else at most [`MAX_MODEL_CHARS`] of
/// `A-Z a-z 0-9 . _ : / - [ ]` (it travels as a command-line argument).
pub(crate) fn check_model(model: Option<&str>) -> Result<Option<String>> {
    let model = model.map(str::trim).filter(|m| !m.is_empty());
    if let Some(m) = model
        && (m.chars().count() > MAX_MODEL_CHARS
            || !m.chars().all(|c| c.is_ascii_alphanumeric() || "._:/-[]".contains(c)))
    {
        return Err(invalid(format!(
            "model must be at most {MAX_MODEL_CHARS} characters of A-Z a-z 0-9 . _ : / - [ ]"
        )));
    }
    Ok(model.map(str::to_string))
}

/// A display name: trimmed, not blank, one line of at most [`MAX_NAME_CHARS`].
pub(crate) fn check_name(name: &str) -> Result<String> {
    let name = name.trim();
    if name.is_empty() {
        return Err(invalid("name must not be empty"));
    }
    if name.chars().any(char::is_control) {
        return Err(invalid("name must not contain line breaks or other control characters"));
    }
    if name.chars().count() > MAX_NAME_CHARS {
        return Err(invalid(format!("name must be at most {MAX_NAME_CHARS} characters")));
    }
    Ok(name.to_string())
}

/// A class list: 1 to [`MAX_CLASSES`] trimmed names, distinct without case, each short and free of the characters
/// that separate classes in the agent's prompt.
pub(super) fn check_classes(classes: &[String]) -> Result<Vec<String>> {
    if classes.is_empty() || classes.len() > MAX_CLASSES {
        return Err(invalid(format!("classes must have 1 to {MAX_CLASSES} names")));
    }
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(classes.len());
    for class in classes {
        let class = class.trim();
        if class.is_empty() {
            return Err(invalid("classes must not have an empty name"));
        }
        if class.chars().count() > MAX_CLASS_CHARS {
            return Err(invalid(format!("class '{class}' is longer than {MAX_CLASS_CHARS} characters")));
        }
        if class.contains([',', ':']) || class.chars().any(char::is_control) {
            return Err(invalid(format!(
                "class '{}' must not contain a comma, a colon, a line break or another control character",
                class.escape_debug()
            )));
        }
        if !seen.insert(class.to_lowercase()) {
            return Err(invalid(format!(
                "class '{class}' is listed twice (names are compared without case)"
            )));
        }
        out.push(class.to_string());
    }
    Ok(out)
}

pub(crate) fn check_text(what: &str, text: &str, max: usize) -> Result<String> {
    if text.chars().count() > max {
        return Err(invalid(format!("{what} must be at most {max} characters")));
    }
    Ok(text.trim().to_string())
}

/// Hand-drawn boxes for a `width` x `height` image of a job with `classes` classes, clipped to the image.
pub(super) fn check_boxes(boxes: &[BBox], classes: usize, (width, height): (u32, u32)) -> Result<Vec<BBox>> {
    if boxes.len() > prompt::MAX_BOXES {
        return Err(invalid(format!("an image takes at most {} boxes", prompt::MAX_BOXES)));
    }
    let (w, h) = (width as f32, height as f32);
    boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            if b.class as usize >= classes {
                return Err(invalid(format!(
                    "boxes[{i}]: class {} is not one of the job's {classes}",
                    b.class
                )));
            }
            if ![b.x1, b.y1, b.x2, b.y2].iter().all(|v| v.is_finite()) {
                return Err(invalid(format!("boxes[{i}]: coordinates must be finite numbers")));
            }
            let (x1, x2) = (b.x1.min(b.x2).clamp(0.0, w), b.x1.max(b.x2).clamp(0.0, w));
            let (y1, y2) = (b.y1.min(b.y2).clamp(0.0, h), b.y1.max(b.y2).clamp(0.0, h));
            if x2 - x1 < 1.0 || y2 - y1 < 1.0 {
                return Err(invalid(format!(
                    "boxes[{i}]: less than a pixel of it is inside the {width}x{height} image"
                )));
            }
            Ok(BBox { class: b.class, x1, y1, x2, y2 })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_classes_and_boxes_are_checked() {
        assert_eq!(check_name("  street  ").unwrap(), "street");
        assert!(check_name(" ").is_err() && check_name(&"x".repeat(MAX_NAME_CHARS + 1)).is_err());
        assert!(check_name("street\npath: D:/other").is_err() && check_name("a\tb").is_err());

        let classes =
            |names: &[&str]| check_classes(&names.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(classes(&[" car ", "Traffic light"]).unwrap(), ["car", "Traffic light"]);
        for bad in [&[][..], &["car", "Car"], &["a,b"], &["a: b"], &[""], &["a\nb"], &["a\u{7}b"]] {
            assert_eq!(classes(bad).unwrap_err().body.code, "invalid_body", "{bad:?}");
        }
        assert!(check_classes(&vec!["c".to_string(); 1]).is_ok());
        let many: Vec<String> = (0..=MAX_CLASSES).map(|i| format!("c{i}")).collect();
        assert!(check_classes(&many).is_err() && check_classes(&many[..MAX_CLASSES]).is_ok());

        let b = |class, x1, y1, x2, y2| BBox { class, x1, y1, x2, y2 };
        // Clipped to the image, corners put in order.
        let ok =
            check_boxes(&[b(1, 90.0, 40.0, -5.0, 10.0), b(0, 0.0, 0.0, 500.0, 500.0)], 2, (100, 50)).unwrap();
        assert_eq!(ok, [b(1, 0.0, 10.0, 90.0, 40.0), b(0, 0.0, 0.0, 100.0, 50.0)]);
        for bad in [b(2, 0.0, 0.0, 9.0, 9.0), b(0, f32::NAN, 0.0, 9.0, 9.0), b(0, 100.0, 0.0, 150.0, 9.0)] {
            let e = check_boxes(&[bad], 2, (100, 50)).unwrap_err();
            assert!(e.body.code == "invalid_body" && e.body.message.starts_with("boxes[0]"), "{e}");
        }
    }
}
