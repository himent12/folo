//! Single-pass streaming JSON parser for COCO instances files.
//!
//! Avoids building `serde_json::Value` trees for `images` and `annotations` (and avoids double-scanning huge
//! `segmentation` arrays), while preserving exact JSON semantics: duplicate keys take the last value, non-object
//! entries are skipped/diagnosed rather than parsed positionally, integral float ids (`1.0`) match integers (`1`),
//! and string/integer ids follow the exact same equivalence classes as [`serde_json::Value`] stringification.

use serde::de::{
    self, Deserialize, DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor,
};
use serde_json::{Map, Number, Value};
use std::borrow::Cow;
use std::fmt;

/// Compact image id key: canonical `i64` decimal ids live inline with zero heap allocation, while non-canonical
/// strings (`"img_1"`, `"01"`, `"-0"`) and numbers outside `i64` fall back to `String`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum IdKey {
    Int(i64),
    Str(String),
}

impl IdKey {
    /// Normalises a string id so canonical `i64` decimals (`"0"`, `"42"`, `"-7"`) map to [`IdKey::Int`] and match
    /// JSON numeric ids identically to `Value::to_string`.
    #[inline]
    pub(super) fn from_str_id(s: &str) -> Self {
        if is_canonical_i64(s)
            && let Ok(n) = s.parse::<i64>()
        {
            return IdKey::Int(n);
        }
        IdKey::Str(s.to_string())
    }

    #[inline]
    pub(super) fn from_u64(n: u64) -> Self {
        match i64::try_from(n) {
            Ok(i) => IdKey::Int(i),
            Err(_) => IdKey::Str(n.to_string()),
        }
    }

    #[inline]
    pub(super) fn from_f64(f: f64) -> Self {
        if f.fract() == 0.0 && f.abs() < (1u64 << 53) as f64 {
            IdKey::Int(f as i64)
        } else {
            IdKey::Str(Number::from_f64(f).map_or_else(|| f.to_string(), |n| n.to_string()))
        }
    }

    #[cfg(test)]
    pub(super) fn from_value(v: &Value) -> Option<Self> {
        match v {
            Value::Number(n) => Some(if let Some(i) = n.as_i64() {
                IdKey::Int(i)
            } else if let Some(u) = n.as_u64() {
                IdKey::from_u64(u)
            } else if let Some(f) = n.as_f64() {
                IdKey::from_f64(f)
            } else {
                IdKey::Str(n.to_string())
            }),
            Value::String(s) => Some(IdKey::from_str_id(s)),
            _ => None,
        }
    }
}

impl fmt::Display for IdKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdKey::Int(n) => write!(f, "{n}"),
            IdKey::Str(s) => f.write_str(s),
        }
    }
}

/// Whether `s` is the exact canonical decimal representation of an `i64` (`"0"`, or optional `'-'` + `[1-9][0-9]*`).
#[inline]
fn is_canonical_i64(s: &str) -> bool {
    let b = s.as_bytes();
    if b.is_empty() || b.len() > 20 {
        return false;
    }
    if b == b"0" {
        return true;
    }
    let digits = if b[0] == b'-' { &b[1..] } else { b };
    !digits.is_empty() && matches!(digits[0], b'1'..=b'9') && digits[1..].iter().all(|c| c.is_ascii_digit())
}

/// Non-negative whole number from a JSON value (`1`, `1.0`).
#[inline]
pub(super) fn whole_number_value(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| {
        let f = v.as_f64()?;
        (f.fract() == 0.0 && (0.0..(1u64 << 53) as f64).contains(&f)).then_some(f as u64)
    })
}

/// Parsed `bbox` of one COCO annotation: stack-allocated `[f64; 4]` on the common valid path, exact `Value` only
/// when malformed.
#[derive(Debug, Default, PartialEq)]
pub(super) enum ParsedBbox {
    #[default]
    Missing,
    Valid([f64; 4]),
    Invalid(Value),
}

impl ParsedBbox {
    #[cfg(test)]
    pub(super) fn from_value(v: Option<&Value>) -> Self {
        let Some(v) = v.filter(|v| !v.is_null()) else {
            return ParsedBbox::Missing;
        };
        if let Some(arr) = v.as_array()
            && arr.len() == 4
            && let (Some(x), Some(y), Some(w), Some(h)) =
                (arr[0].as_f64(), arr[1].as_f64(), arr[2].as_f64(), arr[3].as_f64())
        {
            return ParsedBbox::Valid([x, y, w, h]);
        }
        ParsedBbox::Invalid(v.clone())
    }
}

/// One parsed entry of the COCO `images` array.
#[derive(Debug, PartialEq)]
pub(super) enum ParsedImage {
    Valid { id: IdKey, file_name: String, declared: Option<(u32, u32)> },
    Bad(String),
}

/// The fields of one `annotations` entry the loader reads.
#[derive(Debug, Default, PartialEq)]
pub(super) struct AnnFields {
    pub(super) image_id: Option<IdKey>,
    pub(super) category_id: Option<u64>,
    pub(super) bbox: ParsedBbox,
    pub(super) iscrowd: bool,
    /// Diagnostic summary (only populated for entries without a valid `image_id`).
    pub(super) text: String,
}

/// What the loader reads from one COCO file.
#[derive(Debug, PartialEq)]
pub(super) struct Sections {
    /// `None` when the file has no `images` array; `Some(vec![])` from [`Sections::parse_categories`].
    pub(super) images: Option<Vec<ParsedImage>>,
    pub(super) categories: Option<Value>,
    pub(super) annotations: Vec<AnnFields>,
}

impl Sections {
    /// Parses a wanted COCO file in a single streaming pass.
    pub(super) fn parse(bytes: &[u8]) -> serde_json::Result<Self> {
        let mut de = serde_json::Deserializer::from_slice(bytes);
        let out = TopSeed { read_full: true }.deserialize(&mut de)?;
        de.end()?;
        Ok(out)
    }

    /// Parses a COCO file of an unwanted split: checks whether `images` is an array and reads `categories`,
    /// skipping every image entry and all `annotations` without building them.
    pub(super) fn parse_categories(bytes: &[u8]) -> serde_json::Result<Self> {
        let mut de = serde_json::Deserializer::from_slice(bytes);
        let out = TopSeed { read_full: false }.deserialize(&mut de)?;
        de.end()?;
        Ok(out)
    }

    #[cfg(test)]
    pub(super) fn parse_value(v: Value) -> serde_json::Result<Self> {
        let images = v.get("images").and_then(Value::as_array).map(|arr| {
            arr.iter()
                .map(|im| {
                    let id = im.get("id").and_then(IdKey::from_value);
                    let file = im.get("file_name").and_then(Value::as_str);
                    if let (Some(id), Some(file)) = (id, file) {
                        let dim =
                            |k| im.get(k).and_then(Value::as_u64).filter(|&d| d > 0 && d <= u32::MAX as u64);
                        let declared = dim("width").zip(dim("height")).map(|(w, h)| (w as u32, h as u32));
                        ParsedImage::Valid { id, file_name: file.to_string(), declared }
                    } else {
                        ParsedImage::Bad(im.to_string())
                    }
                })
                .collect()
        });
        let annotations = v
            .get("annotations")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|a| AnnFields {
                image_id: a.get("image_id").and_then(IdKey::from_value),
                category_id: a.get("category_id").and_then(whole_number_value),
                bbox: ParsedBbox::from_value(a.get("bbox")),
                iscrowd: a
                    .get("iscrowd")
                    .is_some_and(|v| v.as_bool() == Some(true) || whole_number_value(v) == Some(1)),
                text: String::new(),
            })
            .collect();
        Ok(Sections { images, categories: v.get("categories").cloned(), annotations })
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Serde streaming visitors
// ---------------------------------------------------------------------------------------------------------------

fn drain_seq<'de, A: SeqAccess<'de>>(mut seq: A) -> Result<(), A::Error> {
    while seq.next_element::<IgnoredAny>()?.is_some() {}
    Ok(())
}

fn drain_map<'de, A: MapAccess<'de>>(mut map: A) -> Result<(), A::Error> {
    while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
    Ok(())
}

struct TopSeed {
    read_full: bool,
}

impl<'de> DeserializeSeed<'de> for TopSeed {
    type Value = Sections;

    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Sections, D::Error> {
        d.deserialize_any(TopVisitor { read_full: self.read_full })
    }
}

struct TopVisitor {
    read_full: bool,
}

impl<'de> Visitor<'de> for TopVisitor {
    type Value = Sections;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Sections, A::Error> {
        let mut images: Option<Vec<ParsedImage>> = None;
        let mut categories: Option<Value> = None;
        let mut annotations: Vec<AnnFields> = Vec::new();
        while let Some(key) = map.next_key::<Cow<'de, str>>()? {
            match key.as_ref() {
                "images" => {
                    if self.read_full {
                        images = map.next_value::<OptImages>()?.0;
                    } else {
                        let is_arr = map.next_value::<IsArray>()?.0;
                        images = is_arr.then(Vec::new);
                    }
                }
                "categories" => {
                    categories = map.next_value::<Option<Value>>()?;
                }
                "annotations" if self.read_full => {
                    annotations = map.next_value::<OptAnnotations>()?.0;
                }
                _ => {
                    let _ = map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(Sections { images, categories, annotations })
    }

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Sections, A::Error> {
        drain_seq(seq)?;
        Ok(Sections { images: None, categories: None, annotations: Vec::new() })
    }
    fn visit_bool<E>(self, _: bool) -> Result<Sections, E> {
        Ok(Sections { images: None, categories: None, annotations: Vec::new() })
    }
    fn visit_i64<E>(self, _: i64) -> Result<Sections, E> {
        Ok(Sections { images: None, categories: None, annotations: Vec::new() })
    }
    fn visit_u64<E>(self, _: u64) -> Result<Sections, E> {
        Ok(Sections { images: None, categories: None, annotations: Vec::new() })
    }
    fn visit_f64<E>(self, _: f64) -> Result<Sections, E> {
        Ok(Sections { images: None, categories: None, annotations: Vec::new() })
    }
    fn visit_str<E>(self, _: &str) -> Result<Sections, E> {
        Ok(Sections { images: None, categories: None, annotations: Vec::new() })
    }
    fn visit_unit<E>(self) -> Result<Sections, E> {
        Ok(Sections { images: None, categories: None, annotations: Vec::new() })
    }
}

/// Whether a JSON value is an array, consuming it without building it.
struct IsArray(bool);

impl<'de> Deserialize<'de> for IsArray {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = IsArray;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<IsArray, A::Error> {
                drain_seq(seq)?;
                Ok(IsArray(true))
            }
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<IsArray, A::Error> {
                drain_map(map)?;
                Ok(IsArray(false))
            }
            fn visit_bool<E>(self, _: bool) -> Result<IsArray, E> {
                Ok(IsArray(false))
            }
            fn visit_i64<E>(self, _: i64) -> Result<IsArray, E> {
                Ok(IsArray(false))
            }
            fn visit_u64<E>(self, _: u64) -> Result<IsArray, E> {
                Ok(IsArray(false))
            }
            fn visit_f64<E>(self, _: f64) -> Result<IsArray, E> {
                Ok(IsArray(false))
            }
            fn visit_str<E>(self, _: &str) -> Result<IsArray, E> {
                Ok(IsArray(false))
            }
            fn visit_unit<E>(self) -> Result<IsArray, E> {
                Ok(IsArray(false))
            }
        }
        d.deserialize_any(V)
    }
}

struct OptImages(Option<Vec<ParsedImage>>);

impl<'de> Deserialize<'de> for OptImages {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = OptImages;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<OptImages, A::Error> {
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(item) = seq.next_element::<ParsedImageItem>()? {
                    out.push(item.0);
                }
                Ok(OptImages(Some(out)))
            }
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<OptImages, A::Error> {
                drain_map(map)?;
                Ok(OptImages(None))
            }
            fn visit_bool<E>(self, _: bool) -> Result<OptImages, E> {
                Ok(OptImages(None))
            }
            fn visit_i64<E>(self, _: i64) -> Result<OptImages, E> {
                Ok(OptImages(None))
            }
            fn visit_u64<E>(self, _: u64) -> Result<OptImages, E> {
                Ok(OptImages(None))
            }
            fn visit_f64<E>(self, _: f64) -> Result<OptImages, E> {
                Ok(OptImages(None))
            }
            fn visit_str<E>(self, _: &str) -> Result<OptImages, E> {
                Ok(OptImages(None))
            }
            fn visit_unit<E>(self) -> Result<OptImages, E> {
                Ok(OptImages(None))
            }
        }
        d.deserialize_any(V)
    }
}

struct ParsedImageItem(ParsedImage);

impl<'de> Deserialize<'de> for ParsedImageItem {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ParsedImageItem;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<ParsedImageItem, A::Error> {
                let mut id: Option<IdKey> = None;
                let mut file_name: Option<String> = None;
                let mut width: Option<u32> = None;
                let mut height: Option<u32> = None;
                while let Some(key) = map.next_key::<Cow<'de, str>>()? {
                    match key.as_ref() {
                        "id" => id = map.next_value::<OptIdKey>()?.0,
                        "file_name" => file_name = map.next_value::<OptString>()?.0,
                        "width" => width = map.next_value::<OptPosU32>()?.0,
                        "height" => height = map.next_value::<OptPosU32>()?.0,
                        _ => {
                            let _ = map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                let item = match (id, file_name) {
                    (Some(id), Some(file_name)) => {
                        ParsedImage::Valid { id, file_name, declared: width.zip(height) }
                    }
                    (Some(id), None) => ParsedImage::Bad(format!("missing string `file_name` (id={id})")),
                    (None, Some(f)) => ParsedImage::Bad(format!("missing valid `id` (file_name={f:?})")),
                    (None, None) => ParsedImage::Bad("missing `id` and `file_name`".into()),
                };
                Ok(ParsedImageItem(item))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<ParsedImageItem, A::Error> {
                drain_seq(seq)?;
                Ok(ParsedImageItem(ParsedImage::Bad("expected JSON object, got array".into())))
            }
            fn visit_bool<E>(self, b: bool) -> Result<ParsedImageItem, E> {
                Ok(ParsedImageItem(ParsedImage::Bad(format!("expected JSON object, got {b}"))))
            }
            fn visit_i64<E>(self, n: i64) -> Result<ParsedImageItem, E> {
                Ok(ParsedImageItem(ParsedImage::Bad(format!("expected JSON object, got {n}"))))
            }
            fn visit_u64<E>(self, n: u64) -> Result<ParsedImageItem, E> {
                Ok(ParsedImageItem(ParsedImage::Bad(format!("expected JSON object, got {n}"))))
            }
            fn visit_f64<E>(self, n: f64) -> Result<ParsedImageItem, E> {
                Ok(ParsedImageItem(ParsedImage::Bad(format!("expected JSON object, got {n}"))))
            }
            fn visit_str<E>(self, s: &str) -> Result<ParsedImageItem, E> {
                Ok(ParsedImageItem(ParsedImage::Bad(format!("expected JSON object, got {s:?}"))))
            }
            fn visit_unit<E>(self) -> Result<ParsedImageItem, E> {
                Ok(ParsedImageItem(ParsedImage::Bad("expected JSON object, got null".into())))
            }
        }
        d.deserialize_any(V)
    }
}

struct OptAnnotations(Vec<AnnFields>);

impl<'de> Deserialize<'de> for OptAnnotations {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = OptAnnotations;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<OptAnnotations, A::Error> {
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(item) = seq.next_element::<AnnItem>()? {
                    out.push(item.0);
                }
                Ok(OptAnnotations(out))
            }
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<OptAnnotations, A::Error> {
                drain_map(map)?;
                Ok(OptAnnotations(Vec::new()))
            }
            fn visit_bool<E>(self, _: bool) -> Result<OptAnnotations, E> {
                Ok(OptAnnotations(Vec::new()))
            }
            fn visit_i64<E>(self, _: i64) -> Result<OptAnnotations, E> {
                Ok(OptAnnotations(Vec::new()))
            }
            fn visit_u64<E>(self, _: u64) -> Result<OptAnnotations, E> {
                Ok(OptAnnotations(Vec::new()))
            }
            fn visit_f64<E>(self, _: f64) -> Result<OptAnnotations, E> {
                Ok(OptAnnotations(Vec::new()))
            }
            fn visit_str<E>(self, _: &str) -> Result<OptAnnotations, E> {
                Ok(OptAnnotations(Vec::new()))
            }
            fn visit_unit<E>(self) -> Result<OptAnnotations, E> {
                Ok(OptAnnotations(Vec::new()))
            }
        }
        d.deserialize_any(V)
    }
}

struct AnnItem(AnnFields);

impl<'de> Deserialize<'de> for AnnItem {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = AnnItem;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<AnnItem, A::Error> {
                let mut a = AnnFields::default();
                while let Some(key) = map.next_key::<Cow<'de, str>>()? {
                    match key.as_ref() {
                        "image_id" => a.image_id = map.next_value::<OptIdKey>()?.0,
                        "category_id" => a.category_id = map.next_value::<OptWholeNumber>()?.0,
                        "bbox" => a.bbox = map.next_value::<ParsedBbox>()?,
                        "iscrowd" => a.iscrowd = map.next_value::<IsCrowd>()?.0,
                        _ => {
                            let _ = map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                if a.image_id.is_none() {
                    a.text = match a.category_id {
                        Some(cat) => format!("missing or invalid `image_id` (category_id={cat})"),
                        None => "missing or invalid `image_id`".into(),
                    };
                }
                Ok(AnnItem(a))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<AnnItem, A::Error> {
                drain_seq(seq)?;
                Ok(AnnItem(AnnFields {
                    text: "expected JSON object, got array".into(),
                    ..Default::default()
                }))
            }
            fn visit_bool<E>(self, b: bool) -> Result<AnnItem, E> {
                Ok(AnnItem(AnnFields {
                    text: format!("expected JSON object, got {b}"),
                    ..Default::default()
                }))
            }
            fn visit_i64<E>(self, n: i64) -> Result<AnnItem, E> {
                Ok(AnnItem(AnnFields {
                    text: format!("expected JSON object, got {n}"),
                    ..Default::default()
                }))
            }
            fn visit_u64<E>(self, n: u64) -> Result<AnnItem, E> {
                Ok(AnnItem(AnnFields {
                    text: format!("expected JSON object, got {n}"),
                    ..Default::default()
                }))
            }
            fn visit_f64<E>(self, n: f64) -> Result<AnnItem, E> {
                Ok(AnnItem(AnnFields {
                    text: format!("expected JSON object, got {n}"),
                    ..Default::default()
                }))
            }
            fn visit_str<E>(self, s: &str) -> Result<AnnItem, E> {
                Ok(AnnItem(AnnFields {
                    text: format!("expected JSON object, got {s:?}"),
                    ..Default::default()
                }))
            }
            fn visit_unit<E>(self) -> Result<AnnItem, E> {
                Ok(AnnItem(AnnFields { text: "expected JSON object, got null".into(), ..Default::default() }))
            }
        }
        d.deserialize_any(V)
    }
}

struct OptIdKey(Option<IdKey>);

impl<'de> Deserialize<'de> for OptIdKey {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = OptIdKey;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_i64<E>(self, v: i64) -> Result<OptIdKey, E> {
                Ok(OptIdKey(Some(IdKey::Int(v))))
            }
            fn visit_u64<E>(self, v: u64) -> Result<OptIdKey, E> {
                Ok(OptIdKey(Some(IdKey::from_u64(v))))
            }
            fn visit_f64<E>(self, v: f64) -> Result<OptIdKey, E> {
                Ok(OptIdKey(Some(IdKey::from_f64(v))))
            }
            fn visit_str<E>(self, v: &str) -> Result<OptIdKey, E> {
                Ok(OptIdKey(Some(IdKey::from_str_id(v))))
            }
            fn visit_string<E>(self, v: String) -> Result<OptIdKey, E> {
                if is_canonical_i64(&v)
                    && let Ok(n) = v.parse::<i64>()
                {
                    return Ok(OptIdKey(Some(IdKey::Int(n))));
                }
                Ok(OptIdKey(Some(IdKey::Str(v))))
            }
            fn visit_bool<E>(self, _: bool) -> Result<OptIdKey, E> {
                Ok(OptIdKey(None))
            }
            fn visit_unit<E>(self) -> Result<OptIdKey, E> {
                Ok(OptIdKey(None))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<OptIdKey, A::Error> {
                drain_seq(seq)?;
                Ok(OptIdKey(None))
            }
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<OptIdKey, A::Error> {
                drain_map(map)?;
                Ok(OptIdKey(None))
            }
        }
        d.deserialize_any(V)
    }
}

struct OptString(Option<String>);

impl<'de> Deserialize<'de> for OptString {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = OptString;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_str<E>(self, v: &str) -> Result<OptString, E> {
                Ok(OptString(Some(v.to_string())))
            }
            fn visit_string<E>(self, v: String) -> Result<OptString, E> {
                Ok(OptString(Some(v)))
            }
            fn visit_bool<E>(self, _: bool) -> Result<OptString, E> {
                Ok(OptString(None))
            }
            fn visit_i64<E>(self, _: i64) -> Result<OptString, E> {
                Ok(OptString(None))
            }
            fn visit_u64<E>(self, _: u64) -> Result<OptString, E> {
                Ok(OptString(None))
            }
            fn visit_f64<E>(self, _: f64) -> Result<OptString, E> {
                Ok(OptString(None))
            }
            fn visit_unit<E>(self) -> Result<OptString, E> {
                Ok(OptString(None))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<OptString, A::Error> {
                drain_seq(seq)?;
                Ok(OptString(None))
            }
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<OptString, A::Error> {
                drain_map(map)?;
                Ok(OptString(None))
            }
        }
        d.deserialize_any(V)
    }
}

struct OptPosU32(Option<u32>);

impl<'de> Deserialize<'de> for OptPosU32 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = OptPosU32;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_u64<E>(self, v: u64) -> Result<OptPosU32, E> {
                Ok(OptPosU32((v > 0 && v <= u32::MAX as u64).then_some(v as u32)))
            }
            fn visit_i64<E>(self, v: i64) -> Result<OptPosU32, E> {
                Ok(OptPosU32((v > 0 && v <= u32::MAX as i64).then_some(v as u32)))
            }
            fn visit_f64<E>(self, _: f64) -> Result<OptPosU32, E> {
                Ok(OptPosU32(None))
            }
            fn visit_bool<E>(self, _: bool) -> Result<OptPosU32, E> {
                Ok(OptPosU32(None))
            }
            fn visit_str<E>(self, _: &str) -> Result<OptPosU32, E> {
                Ok(OptPosU32(None))
            }
            fn visit_unit<E>(self) -> Result<OptPosU32, E> {
                Ok(OptPosU32(None))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<OptPosU32, A::Error> {
                drain_seq(seq)?;
                Ok(OptPosU32(None))
            }
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<OptPosU32, A::Error> {
                drain_map(map)?;
                Ok(OptPosU32(None))
            }
        }
        d.deserialize_any(V)
    }
}

struct OptWholeNumber(Option<u64>);

impl<'de> Deserialize<'de> for OptWholeNumber {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = OptWholeNumber;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_u64<E>(self, v: u64) -> Result<OptWholeNumber, E> {
                Ok(OptWholeNumber(Some(v)))
            }
            fn visit_i64<E>(self, v: i64) -> Result<OptWholeNumber, E> {
                Ok(OptWholeNumber((v >= 0).then_some(v as u64)))
            }
            fn visit_f64<E>(self, f: f64) -> Result<OptWholeNumber, E> {
                let ok = f.fract() == 0.0 && (0.0..(1u64 << 53) as f64).contains(&f);
                Ok(OptWholeNumber(ok.then_some(f as u64)))
            }
            fn visit_bool<E>(self, _: bool) -> Result<OptWholeNumber, E> {
                Ok(OptWholeNumber(None))
            }
            fn visit_str<E>(self, _: &str) -> Result<OptWholeNumber, E> {
                Ok(OptWholeNumber(None))
            }
            fn visit_unit<E>(self) -> Result<OptWholeNumber, E> {
                Ok(OptWholeNumber(None))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<OptWholeNumber, A::Error> {
                drain_seq(seq)?;
                Ok(OptWholeNumber(None))
            }
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<OptWholeNumber, A::Error> {
                drain_map(map)?;
                Ok(OptWholeNumber(None))
            }
        }
        d.deserialize_any(V)
    }
}

struct IsCrowd(bool);

impl<'de> Deserialize<'de> for IsCrowd {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = IsCrowd;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_bool<E>(self, b: bool) -> Result<IsCrowd, E> {
                Ok(IsCrowd(b))
            }
            fn visit_u64<E>(self, v: u64) -> Result<IsCrowd, E> {
                Ok(IsCrowd(v == 1))
            }
            fn visit_i64<E>(self, v: i64) -> Result<IsCrowd, E> {
                Ok(IsCrowd(v == 1))
            }
            fn visit_f64<E>(self, f: f64) -> Result<IsCrowd, E> {
                Ok(IsCrowd(f == 1.0))
            }
            fn visit_str<E>(self, _: &str) -> Result<IsCrowd, E> {
                Ok(IsCrowd(false))
            }
            fn visit_unit<E>(self) -> Result<IsCrowd, E> {
                Ok(IsCrowd(false))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<IsCrowd, A::Error> {
                drain_seq(seq)?;
                Ok(IsCrowd(false))
            }
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<IsCrowd, A::Error> {
                drain_map(map)?;
                Ok(IsCrowd(false))
            }
        }
        d.deserialize_any(V)
    }
}

enum BboxElem {
    Int(i64),
    Uint(u64),
    Float(f64),
    Other(Value),
}

impl BboxElem {
    #[inline]
    fn as_f64(&self) -> Option<f64> {
        match *self {
            BboxElem::Int(n) => Some(n as f64),
            BboxElem::Uint(n) => Some(n as f64),
            BboxElem::Float(f) => Some(f),
            BboxElem::Other(_) => None,
        }
    }

    fn into_value(self) -> Value {
        match self {
            BboxElem::Int(n) => Value::Number(n.into()),
            BboxElem::Uint(n) => Value::Number(n.into()),
            BboxElem::Float(f) => Number::from_f64(f).map_or(Value::Null, Value::Number),
            BboxElem::Other(v) => v,
        }
    }
}

impl<'de> Deserialize<'de> for BboxElem {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = BboxElem;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_i64<E>(self, v: i64) -> Result<BboxElem, E> {
                Ok(BboxElem::Int(v))
            }
            fn visit_u64<E>(self, v: u64) -> Result<BboxElem, E> {
                Ok(BboxElem::Uint(v))
            }
            fn visit_f64<E>(self, v: f64) -> Result<BboxElem, E> {
                Ok(BboxElem::Float(v))
            }
            fn visit_bool<E>(self, v: bool) -> Result<BboxElem, E> {
                Ok(BboxElem::Other(Value::Bool(v)))
            }
            fn visit_str<E>(self, v: &str) -> Result<BboxElem, E> {
                Ok(BboxElem::Other(Value::String(v.to_string())))
            }
            fn visit_string<E>(self, v: String) -> Result<BboxElem, E> {
                Ok(BboxElem::Other(Value::String(v)))
            }
            fn visit_unit<E>(self) -> Result<BboxElem, E> {
                Ok(BboxElem::Other(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<BboxElem, A::Error> {
                let mut v = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(elem) = seq.next_element::<Value>()? {
                    v.push(elem);
                }
                Ok(BboxElem::Other(Value::Array(v)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<BboxElem, A::Error> {
                let mut m = Map::new();
                while let Some((k, val)) = map.next_entry::<String, Value>()? {
                    m.insert(k, val);
                }
                Ok(BboxElem::Other(Value::Object(m)))
            }
        }
        d.deserialize_any(V)
    }
}

impl<'de> Deserialize<'de> for ParsedBbox {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ParsedBbox;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_unit<E>(self) -> Result<ParsedBbox, E> {
                Ok(ParsedBbox::Missing)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<ParsedBbox, A::Error> {
                let mut head: [Option<BboxElem>; 4] = [None, None, None, None];
                let mut count = 0usize;
                let mut all_num = true;
                while count < 4 {
                    match seq.next_element::<BboxElem>()? {
                        Some(elem) => {
                            all_num &= elem.as_f64().is_some();
                            head[count] = Some(elem);
                            count += 1;
                        }
                        None => break,
                    }
                }
                if count == 4 && all_num {
                    if let Some(extra) = seq.next_element::<Value>()? {
                        let mut vals = Vec::with_capacity(5);
                        for elem in head.into_iter().flatten() {
                            vals.push(elem.into_value());
                        }
                        vals.push(extra);
                        while let Some(rest) = seq.next_element::<Value>()? {
                            vals.push(rest);
                        }
                        return Ok(ParsedBbox::Invalid(Value::Array(vals)));
                    }
                    let nums = [
                        head[0].as_ref().and_then(BboxElem::as_f64).unwrap(),
                        head[1].as_ref().and_then(BboxElem::as_f64).unwrap(),
                        head[2].as_ref().and_then(BboxElem::as_f64).unwrap(),
                        head[3].as_ref().and_then(BboxElem::as_f64).unwrap(),
                    ];
                    return Ok(ParsedBbox::Valid(nums));
                }
                let mut vals = Vec::with_capacity(count);
                for elem in head.into_iter().flatten() {
                    vals.push(elem.into_value());
                }
                while let Some(rest) = seq.next_element::<Value>()? {
                    vals.push(rest);
                }
                Ok(ParsedBbox::Invalid(Value::Array(vals)))
            }
            fn visit_bool<E>(self, v: bool) -> Result<ParsedBbox, E> {
                Ok(ParsedBbox::Invalid(Value::Bool(v)))
            }
            fn visit_i64<E>(self, v: i64) -> Result<ParsedBbox, E> {
                Ok(ParsedBbox::Invalid(Value::Number(v.into())))
            }
            fn visit_u64<E>(self, v: u64) -> Result<ParsedBbox, E> {
                Ok(ParsedBbox::Invalid(Value::Number(v.into())))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<ParsedBbox, E> {
                Ok(ParsedBbox::Invalid(Number::from_f64(v).map_or(Value::Null, Value::Number)))
            }
            fn visit_str<E>(self, v: &str) -> Result<ParsedBbox, E> {
                Ok(ParsedBbox::Invalid(Value::String(v.to_string())))
            }
            fn visit_string<E>(self, v: String) -> Result<ParsedBbox, E> {
                Ok(ParsedBbox::Invalid(Value::String(v)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<ParsedBbox, A::Error> {
                let mut m = Map::new();
                while let Some((k, val)) = map.next_entry::<String, Value>()? {
                    m.insert(k, val);
                }
                Ok(ParsedBbox::Invalid(Value::Object(m)))
            }
        }
        d.deserialize_any(V)
    }
}
