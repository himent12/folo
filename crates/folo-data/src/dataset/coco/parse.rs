//! Single-pass streaming JSON parser for COCO instances files.
//!
//! Avoids building `serde_json::Value` trees for `images` and `annotations` (and avoids double-scanning huge
//! `segmentation` arrays), while preserving exact JSON semantics: duplicate keys take the last value, non-object
//! entries are skipped/diagnosed rather than parsed positionally, integral float ids (`1.0`) match integers (`1`),
//! string/integer ids follow the exact same equivalence classes as [`serde_json::Value`] stringification, invalid
//! UTF-8 anywhere in the file (including skipped fields) is rejected, and a single overflowing number inside an
//! image or annotation entry falls back to per-entry parsing so the rest of the split is retained.

#[cfg(test)]
use super::fields::whole_number_value;
use super::fields::{
    IdKey, IsCrowd, OptIdKey, OptPosU32, OptString, OptWholeNumber, ParsedBbox, drain_map, drain_seq,
};
use serde::de::{
    self, Deserialize, DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor,
};
use serde_json::Value;
use serde_json::value::RawValue;
use std::borrow::Cow;
use std::fmt;

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
    ///
    /// Validates UTF-8 across the entire input up front (including skipped fields like `segmentation`). If a single
    /// image or annotation contains an overflowing JSON number (`1e400`) that aborts the fast streaming pass,
    /// retries with per-entry `RawValue` isolation so only the malformed row is reported and the rest of the split
    /// is kept.
    pub(super) fn parse(bytes: &[u8]) -> serde_json::Result<Self> {
        let text = std::str::from_utf8(bytes).map_err(de::Error::custom)?;
        let mut de = serde_json::Deserializer::from_str(text);
        let res = TopSeed { read_full: true }.deserialize(&mut de).and_then(|out| de.end().map(|()| out));
        match res {
            Ok(out) => Ok(out),
            Err(err) => parse_tolerant(text).or(Err(err)),
        }
    }

    /// Parses a COCO file of an unwanted split: validates UTF-8, checks whether `images` is an array, and reads
    /// `categories`, skipping every image entry and all `annotations` without building them.
    pub(super) fn parse_categories(bytes: &[u8]) -> serde_json::Result<Self> {
        let text = std::str::from_utf8(bytes).map_err(de::Error::custom)?;
        let mut de = serde_json::Deserializer::from_str(text);
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

/// Fault-tolerant fallback invoked only when the single-pass streaming parser fails (e.g. a single annotation
/// contains an overflowing number like `1e400`). Isolates each `images` and `annotations` element via `&RawValue`
/// so one bad entry reports a row issue instead of discarding the whole split.
fn parse_tolerant(text: &str) -> serde_json::Result<Sections> {
    let mut de = serde_json::Deserializer::from_str(text);
    let raw = TolerantTop.deserialize(&mut de)?;
    de.end()?;
    let images = raw.images.map(|arr| {
        arr.into_iter()
            .map(|r| {
                serde_json::from_str::<ParsedImageItem>(r.get())
                    .map(|p| p.0)
                    .unwrap_or_else(|_| ParsedImage::Bad(r.get().to_string()))
            })
            .collect()
    });
    let annotations = raw
        .annotations
        .into_iter()
        .map(|r| {
            serde_json::from_str::<AnnItem>(r.get())
                .map(|a| a.0)
                .unwrap_or_else(|_| AnnFields { text: r.get().to_string(), ..Default::default() })
        })
        .collect();
    Ok(Sections { images, categories: raw.categories, annotations })
}

struct RawSections<'a> {
    images: Option<Vec<&'a RawValue>>,
    categories: Option<Value>,
    annotations: Vec<&'a RawValue>,
}

struct TolerantTop;

impl<'de> DeserializeSeed<'de> for TolerantTop {
    type Value = RawSections<'de>;

    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<RawSections<'de>, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = RawSections<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<RawSections<'de>, A::Error> {
                let mut images: Option<Vec<&'de RawValue>> = None;
                let mut categories: Option<Value> = None;
                let mut annotations: Vec<&'de RawValue> = Vec::new();
                while let Some(key) = map.next_key::<Cow<'de, str>>()? {
                    match key.as_ref() {
                        "images" => images = map.next_value::<OptRawSeq<'de>>()?.0,
                        "categories" => categories = map.next_value::<Option<Value>>()?,
                        "annotations" => {
                            annotations = map.next_value::<OptRawSeq<'de>>()?.0.unwrap_or_default();
                        }
                        _ => {
                            let _ = map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                Ok(RawSections { images, categories, annotations })
            }
            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<RawSections<'de>, A::Error> {
                drain_seq(seq)?;
                Ok(RawSections { images: None, categories: None, annotations: Vec::new() })
            }
            fn visit_bool<E>(self, _: bool) -> Result<RawSections<'de>, E> {
                Ok(RawSections { images: None, categories: None, annotations: Vec::new() })
            }
            fn visit_i64<E>(self, _: i64) -> Result<RawSections<'de>, E> {
                Ok(RawSections { images: None, categories: None, annotations: Vec::new() })
            }
            fn visit_u64<E>(self, _: u64) -> Result<RawSections<'de>, E> {
                Ok(RawSections { images: None, categories: None, annotations: Vec::new() })
            }
            fn visit_f64<E>(self, _: f64) -> Result<RawSections<'de>, E> {
                Ok(RawSections { images: None, categories: None, annotations: Vec::new() })
            }
            fn visit_str<E>(self, _: &str) -> Result<RawSections<'de>, E> {
                Ok(RawSections { images: None, categories: None, annotations: Vec::new() })
            }
            fn visit_unit<E>(self) -> Result<RawSections<'de>, E> {
                Ok(RawSections { images: None, categories: None, annotations: Vec::new() })
            }
        }
        d.deserialize_any(V)
    }
}

struct OptRawSeq<'a>(Option<Vec<&'a RawValue>>);

impl<'de> Deserialize<'de> for OptRawSeq<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = OptRawSeq<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<OptRawSeq<'de>, A::Error> {
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(item) = seq.next_element::<&'de RawValue>()? {
                    out.push(item);
                }
                Ok(OptRawSeq(Some(out)))
            }
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<OptRawSeq<'de>, A::Error> {
                drain_map(map)?;
                Ok(OptRawSeq(None))
            }
            fn visit_bool<E>(self, _: bool) -> Result<OptRawSeq<'de>, E> {
                Ok(OptRawSeq(None))
            }
            fn visit_i64<E>(self, _: i64) -> Result<OptRawSeq<'de>, E> {
                Ok(OptRawSeq(None))
            }
            fn visit_u64<E>(self, _: u64) -> Result<OptRawSeq<'de>, E> {
                Ok(OptRawSeq(None))
            }
            fn visit_f64<E>(self, _: f64) -> Result<OptRawSeq<'de>, E> {
                Ok(OptRawSeq(None))
            }
            fn visit_str<E>(self, _: &str) -> Result<OptRawSeq<'de>, E> {
                Ok(OptRawSeq(None))
            }
            fn visit_unit<E>(self) -> Result<OptRawSeq<'de>, E> {
                Ok(OptRawSeq(None))
            }
        }
        d.deserialize_any(V)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Serde streaming visitors
// ---------------------------------------------------------------------------------------------------------------

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
                    (Some(id), None) => ParsedImage::Bad(format!("id={id}, missing or invalid `file_name`")),
                    (None, Some(file_name)) => {
                        ParsedImage::Bad(format!("file_name={file_name:?}, missing or invalid `id`"))
                    }
                    (None, None) => ParsedImage::Bad("missing or invalid `id` and `file_name`".into()),
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
