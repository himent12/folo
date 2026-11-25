//! Field-level streaming deserializers for COCO `images` and `annotations` entries.

use serde::de::{self, Deserialize, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
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
pub(super) fn is_canonical_i64(s: &str) -> bool {
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

pub(super) fn drain_seq<'de, A: SeqAccess<'de>>(mut seq: A) -> Result<(), A::Error> {
    while seq.next_element::<IgnoredAny>()?.is_some() {}
    Ok(())
}

pub(super) fn drain_map<'de, A: MapAccess<'de>>(mut map: A) -> Result<(), A::Error> {
    while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
    Ok(())
}

pub(super) struct OptIdKey(pub(super) Option<IdKey>);

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

pub(super) struct OptString(pub(super) Option<String>);

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

pub(super) struct OptPosU32(pub(super) Option<u32>);

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

pub(super) struct OptWholeNumber(pub(super) Option<u64>);

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

pub(super) struct IsCrowd(pub(super) bool);

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
