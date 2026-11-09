//! The safetensors container: parsing with docs/FORMAT.md's reader acceptance rules, and the FOLO writer
//! layout (descending dtype size, then name).

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::Path;

use folo_core::{DType, Tensor};
use serde::Deserialize;
use serde::de::{self, Deserializer, MapAccess, Visitor};

use crate::{Error, Result};

/// Largest accepted header (`N`), in bytes.
pub const MAX_HEADER: u64 = 100_000_000;

/// A safetensors element type. Every type safetensors defines parses; [`FileDType::to_core`] says which ones
/// FOLO can turn into tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum FileDType {
    Bool,
    U8,
    I8,
    F8E5M2,
    F8E4M3,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
}

impl FileDType {
    const ALL: [FileDType; 15] = [
        FileDType::Bool,
        FileDType::U8,
        FileDType::I8,
        FileDType::F8E5M2,
        FileDType::F8E4M3,
        FileDType::I16,
        FileDType::U16,
        FileDType::F16,
        FileDType::BF16,
        FileDType::I32,
        FileDType::U32,
        FileDType::F32,
        FileDType::F64,
        FileDType::I64,
        FileDType::U64,
    ];

    /// The header spelling (`"F32"`, `"BF16"`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            FileDType::Bool => "BOOL",
            FileDType::U8 => "U8",
            FileDType::I8 => "I8",
            FileDType::F8E5M2 => "F8_E5M2",
            FileDType::F8E4M3 => "F8_E4M3",
            FileDType::I16 => "I16",
            FileDType::U16 => "U16",
            FileDType::F16 => "F16",
            FileDType::BF16 => "BF16",
            FileDType::I32 => "I32",
            FileDType::U32 => "U32",
            FileDType::F32 => "F32",
            FileDType::F64 => "F64",
            FileDType::I64 => "I64",
            FileDType::U64 => "U64",
        }
    }

    /// Parse a header spelling.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|d| d.as_str() == s)
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            FileDType::Bool | FileDType::U8 | FileDType::I8 | FileDType::F8E5M2 | FileDType::F8E4M3 => 1,
            FileDType::I16 | FileDType::U16 | FileDType::F16 | FileDType::BF16 => 2,
            FileDType::I32 | FileDType::U32 | FileDType::F32 => 4,
            FileDType::F64 | FileDType::I64 | FileDType::U64 => 8,
        }
    }

    /// The folo-core dtype, for the five FOLO uses (`F32`, `F16`, `I64`, `U8`, `U32`).
    pub fn to_core(self) -> Option<DType> {
        Some(match self {
            FileDType::F32 => DType::F32,
            FileDType::F16 => DType::F16,
            FileDType::I64 => DType::I64,
            FileDType::U8 => DType::U8,
            FileDType::U32 => DType::U32,
            _ => return None,
        })
    }

    /// The file dtype of a folo-core dtype.
    pub fn from_core(d: DType) -> Self {
        match d {
            DType::F32 => FileDType::F32,
            DType::F16 => FileDType::F16,
            DType::I64 => FileDType::I64,
            DType::U8 => FileDType::U8,
            DType::U32 => FileDType::U32,
        }
    }
}

/// One tensor entry of a parsed header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// The key.
    pub name: String,
    /// Element type.
    pub dtype: FileDType,
    /// Dimensions (`[]` for a scalar).
    pub shape: Vec<usize>,
    /// `[begin, end)` relative to the start of the data section.
    pub offsets: (usize, usize),
}

impl TensorInfo {
    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A parsed container: metadata and tensor entries (in header order). The data stays in the caller's
/// buffer; [`Container::data`] slices it.
#[derive(Debug, Clone)]
pub struct Container {
    /// `__metadata__`, if present.
    pub metadata: Option<BTreeMap<String, String>>,
    /// Tensor entries, in header order.
    pub tensors: Vec<TensorInfo>,
    data_start: usize,
}

impl Container {
    /// Parse and validate `bytes` (the whole file) under docs/FORMAT.md's reader acceptance rules.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Self::parse_header(bytes, bytes.len() as u64)
    }

    /// Read and validate only the header of the file at `path` (its first `8 + N` bytes), under the same rules
    /// as [`Container::parse`]: the tensors must cover the rest of the file exactly, but the tensor data itself is
    /// never read. For listing a file's metadata without loading its payload; [`Container::data`] needs the whole
    /// file.
    pub fn read_header(path: &Path) -> Result<Self> {
        use std::io::Read;
        let f = std::fs::File::open(path).map_err(|e| Error::io(path, e))?;
        let file_len = f.metadata().map_err(|e| Error::io(path, e))?.len();
        let mut prefix = Vec::with_capacity(8);
        let mut f = f.take(8);
        f.read_to_end(&mut prefix).map_err(|e| Error::io(path, e))?;
        if let Ok(len) = <[u8; 8]>::try_from(prefix.as_slice()) {
            // An N over the limit or past the end of the file is refused by `parse_header` without reading or
            // allocating for it.
            let n = u64::from_le_bytes(len);
            if n <= MAX_HEADER && n <= file_len.saturating_sub(8) {
                f.set_limit(n);
                prefix.reserve(n as usize);
                f.read_to_end(&mut prefix).map_err(|e| Error::io(path, e))?;
            }
        }
        Self::parse_header(&prefix, file_len)
    }

    /// [`Container::parse`] of a `file_len`-byte file of which `prefix` holds the start: at least the 8-byte
    /// length and the header (the data may be missing).
    fn parse_header(prefix: &[u8], file_len: u64) -> Result<Self> {
        let bad = |m: String| Error::Format(m);
        if file_len < 8 || prefix.len() < 8 {
            return Err(bad(format!("{file_len} bytes is too short for the 8-byte header length")));
        }
        let n = u64::from_le_bytes(prefix[..8].try_into().expect("8 bytes"));
        if n > MAX_HEADER {
            return Err(bad(format!("header length {n} exceeds the {MAX_HEADER}-byte limit")));
        }
        if 8 + n > file_len {
            return Err(bad(format!("header length {n} runs past the end of the {file_len}-byte file")));
        }
        let n = n as usize;
        let data_start = 8 + n;
        if data_start > prefix.len() {
            // The file was shorter when it was read than when its length was taken.
            return Err(bad(format!(
                "header length {n} runs past the end of the {}-byte file",
                prefix.len()
            )));
        }
        let header = &prefix[8..data_start];
        if header.first() != Some(&b'{') {
            return Err(bad("the header must start with `{` (no leading whitespace or BOM)".into()));
        }
        let end = header.iter().rposition(|&b| b != b' ').expect("starts with `{`") + 1;
        if header[end - 1] != b'}' {
            return Err(bad("the header JSON object may only be followed by spaces".into()));
        }
        let raw: RawHeader =
            serde_json::from_slice(&header[..end]).map_err(|e| bad(format!("header JSON: {e}")))?;

        let data_len = file_len - data_start as u64;
        let mut tensors = Vec::with_capacity(raw.entries.len());
        for (name, r) in raw.entries {
            let dtype = FileDType::parse(&r.dtype)
                .ok_or_else(|| bad(format!("tensor `{name}`: unknown dtype `{}`", r.dtype)))?;
            let mut numel = 1usize;
            let mut shape = Vec::with_capacity(r.shape.len());
            for &d in &r.shape {
                let d = usize::try_from(d)
                    .map_err(|_| bad(format!("tensor `{name}`: dimension {d} is too large")))?;
                numel =
                    numel.checked_mul(d).ok_or_else(|| bad(format!("tensor `{name}`: shape overflows")))?;
                shape.push(d);
            }
            let nbytes = numel
                .checked_mul(dtype.size())
                .ok_or_else(|| bad(format!("tensor `{name}`: size overflows")))?;
            let [b, e] = r.data_offsets;
            let (b, e) = match (usize::try_from(b), usize::try_from(e)) {
                (Ok(b), Ok(e)) if b <= e => (b, e),
                _ => return Err(bad(format!("tensor `{name}`: invalid data_offsets [{b}, {e}]"))),
            };
            if e - b != nbytes {
                return Err(bad(format!(
                    "tensor `{name}`: data_offsets [{b}, {e}] hold {} bytes, but {} {:?} needs {nbytes}",
                    e - b,
                    dtype.as_str(),
                    shape
                )));
            }
            tensors.push(TensorInfo { name, dtype, shape, offsets: (b, e) });
        }
        let mut order: Vec<&TensorInfo> = tensors.iter().collect();
        order.sort_by_key(|t| t.offsets);
        let mut cursor = 0usize;
        for t in order {
            if t.offsets.0 != cursor {
                return Err(bad(format!(
                    "tensor `{}` starts at {} but the previous tensor ends at {cursor} (overlap or hole)",
                    t.name, t.offsets.0
                )));
            }
            cursor = t.offsets.1;
        }
        if cursor as u64 != data_len {
            return Err(bad(format!("the tensors cover {cursor} bytes but the data section has {data_len}")));
        }
        Ok(Container { metadata: raw.metadata, tensors, data_start })
    }

    /// The raw little-endian bytes of `info` within `bytes` (the buffer given to [`Container::parse`]).
    pub fn data<'a>(&self, bytes: &'a [u8], info: &TensorInfo) -> &'a [u8] {
        &bytes[self.data_start + info.offsets.0..self.data_start + info.offsets.1]
    }
}

/// Decode a tensor's bytes into a CPU tensor. Dtypes outside FOLO's five are an error naming the tensor.
pub fn decode(info: &TensorInfo, data: &[u8]) -> Result<Tensor> {
    let dtype = info.dtype.to_core().ok_or_else(|| {
        Error::Load(format!(
            "tensor `{}` has dtype {}, which FOLO does not support",
            info.name,
            info.dtype.as_str()
        ))
    })?;
    let native = le_to_native(data, info.dtype.size());
    Ok(Tensor::from_bytes(&native, dtype, info.shape.as_slice())?)
}

#[cfg(target_endian = "little")]
fn le_to_native(b: &[u8], _size: usize) -> std::borrow::Cow<'_, [u8]> {
    std::borrow::Cow::Borrowed(b)
}

#[cfg(target_endian = "big")]
fn le_to_native(b: &[u8], size: usize) -> std::borrow::Cow<'_, [u8]> {
    let mut v = b.to_vec();
    for c in v.chunks_exact_mut(size) {
        c.reverse();
    }
    std::borrow::Cow::Owned(v)
}

/// The FOLO layout of `tensors`: indices sorted by descending dtype size, then by name.
fn layout_order(tensors: &[(String, Tensor)]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..tensors.len()).collect();
    idx.sort_by(|&a, &b| {
        let (na, ta) = &tensors[a];
        let (nb, tb) = &tensors[b];
        tb.dtype().size_in_bytes().cmp(&ta.dtype().size_in_bytes()).then_with(|| na.cmp(nb))
    });
    idx
}

/// Write a safetensors file: the header (metadata first, then tensors in layout order, padded with spaces
/// so the data section starts 8-aligned) followed by the tensor bytes in layout order. Names must be unique
/// and must not be `__metadata__`.
pub fn write(
    w: &mut dyn Write,
    metadata: Option<&BTreeMap<String, String>>,
    tensors: &[(String, Tensor)],
) -> Result<()> {
    let mut seen = HashSet::new();
    for (n, _) in tensors {
        if n == "__metadata__" || !seen.insert(n.as_str()) {
            return Err(Error::Invalid(format!(
                "safetensors writer: duplicate or reserved tensor name `{n}`"
            )));
        }
    }
    let order = layout_order(tensors);
    let mut header = String::from("{");
    let mut first = true;
    if let Some(m) = metadata {
        header.push_str("\"__metadata__\":");
        header.push_str(&json(m));
        first = false;
    }
    let mut offset = 0usize;
    for &i in &order {
        let (name, t) = &tensors[i];
        let nbytes = t.numel() * t.dtype().size_in_bytes();
        if !first {
            header.push(',');
        }
        first = false;
        header.push_str(&json(name));
        header.push_str(&format!(
            ":{{\"dtype\":\"{}\",\"shape\":{},\"data_offsets\":[{},{}]}}",
            FileDType::from_core(t.dtype()).as_str(),
            json(t.dims()),
            offset,
            offset + nbytes
        ));
        offset += nbytes;
    }
    header.push('}');
    while !(8 + header.len()).is_multiple_of(8) {
        header.push(' ');
    }
    let io = |e| Error::Io { path: "<writer>".into(), source: e };
    w.write_all(&(header.len() as u64).to_le_bytes()).map_err(io)?;
    w.write_all(header.as_bytes()).map_err(io)?;
    for &i in &order {
        let t = &tensors[i].1;
        let host = t.to_cpu_storage()?;
        w.write_all(&native_to_le(host.as_bytes(), t.dtype().size_in_bytes())).map_err(io)?;
    }
    Ok(())
}

#[cfg(target_endian = "little")]
fn native_to_le(b: &[u8], _size: usize) -> std::borrow::Cow<'_, [u8]> {
    std::borrow::Cow::Borrowed(b)
}

#[cfg(target_endian = "big")]
fn native_to_le(b: &[u8], size: usize) -> std::borrow::Cow<'_, [u8]> {
    le_to_native(b, size)
}

/// A new, uniquely named file next to `path` (`<path>.<pid>.<n>.tmp`, opened with `create_new`) for atomic
/// writes. It never truncates an existing file, even one whose name looks temporary.
pub fn create_temp_beside(path: &Path) -> Result<(std::path::PathBuf, std::fs::File)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    loop {
        let mut name = path.as_os_str().to_owned();
        name.push(format!(".{}.{}.tmp", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        let tmp = std::path::PathBuf::from(name);
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp) {
            Ok(f) => return Ok((tmp, f)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(Error::io(&tmp, e)),
        }
    }
}

/// [`write`] to `path` atomically: the bytes go to a fresh temporary file next to it ([`create_temp_beside`]),
/// which is then renamed over `path`.
pub fn write_file(
    path: &Path,
    metadata: Option<&BTreeMap<String, String>>,
    tensors: &[(String, Tensor)],
) -> Result<()> {
    let (tmp, f) = create_temp_beside(path)?;
    let res = (|| {
        let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
        write(&mut w, metadata, tensors)?;
        // The file is closed at the end of this block, before the rename.
        {
            let f = w.into_inner().map_err(|e| Error::io(&tmp, e.into_error()))?;
            f.sync_all().map_err(|e| Error::io(&tmp, e))?;
        }
        std::fs::rename(&tmp, path).map_err(|e| Error::io(path, e))
    })();
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// JSON-encode a header value (strings are escaped as JSON requires).
fn json<T: serde::Serialize + ?Sized>(v: &T) -> String {
    serde_json::to_string(v).expect("header values serialise")
}

// ----- header deserialisation with duplicate-key detection -----

struct RawHeader {
    metadata: Option<BTreeMap<String, String>>,
    entries: Vec<(String, RawInfo)>,
}

#[derive(Deserialize)]
struct RawInfo {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

impl<'de> Deserialize<'de> for RawHeader {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = RawHeader;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<RawHeader, A::Error> {
                let mut seen = HashSet::new();
                let mut out = RawHeader { metadata: None, entries: Vec::new() };
                while let Some(key) = map.next_key::<String>()? {
                    if !seen.insert(key.clone()) {
                        return Err(de::Error::custom(format!("duplicate key `{key}`")));
                    }
                    if key == "__metadata__" {
                        out.metadata = Some(map.next_value::<StringMap>()?.0);
                    } else {
                        out.entries.push((key, map.next_value()?));
                    }
                }
                Ok(out)
            }
        }
        d.deserialize_map(V)
    }
}

/// A JSON object of string values with no duplicate keys.
struct StringMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for StringMap {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = StringMap;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an object mapping strings to strings")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<StringMap, A::Error> {
                let mut out = BTreeMap::new();
                while let Some((k, v)) = map.next_entry::<String, String>()? {
                    if out.contains_key(&k) {
                        return Err(de::Error::custom(format!("duplicate __metadata__ key `{k}`")));
                    }
                    out.insert(k, v);
                }
                Ok(StringMap(out))
            }
        }
        d.deserialize_map(V)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut v = (header.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(header.as_bytes());
        v.extend_from_slice(data);
        v
    }

    #[test]
    fn acceptance_rules() {
        let ok = file(r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}  "#, &[0; 4]);
        assert!(Container::parse(&ok).is_ok());
        let cases: &[(&str, &[u8])] = &[
            (r#" {"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, &[0; 4]), // leading space
            (r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, &[0; 5]),  // trailing data
            (r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#, &[0; 4]),  // size mismatch
            (
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                &[0; 4],
            ),
            (
                r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#,
                &[0; 3],
            ), // overlap
            (r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#, &[0; 2]), // hole at 0
            (r#"{"__metadata__":{"k":1}}"#, &[]),                                  // non-string metadata
            (r#"{"__metadata__":{"k":"a","k":"b"}}"#, &[]),                        // duplicate metadata key
            (r#"{"a":{"dtype":"Q8","shape":[1],"data_offsets":[0,1]}}"#, &[0; 1]), // unknown dtype
            ("{}\n", &[]),                                                         // trailing newline
        ];
        for (h, d) in cases {
            assert!(Container::parse(&file(h, d)).is_err(), "accepted {h}");
        }
        let mut huge = (MAX_HEADER + 1).to_le_bytes().to_vec();
        huge.push(b'{');
        assert!(Container::parse(&huge).is_err());
        // Zero-sized and scalar tensors, any key order.
        let z = file(
            r#"{"s":{"dtype":"I64","shape":[],"data_offsets":[0,8]},"z":{"dtype":"F32","shape":[0,3],"data_offsets":[8,8]}}"#,
            &[1, 0, 0, 0, 0, 0, 0, 0],
        );
        let c = Container::parse(&z).unwrap();
        let s = c.tensors.iter().find(|t| t.name == "s").unwrap();
        assert_eq!(decode(s, c.data(&z, s)).unwrap().item::<i64>().unwrap(), 1);
        let bf = file(r#"{"b":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#, &[0; 2]);
        let c = Container::parse(&bf).unwrap();
        let e = decode(&c.tensors[0], c.data(&bf, &c.tensors[0])).unwrap_err().to_string();
        assert!(e.contains("BF16") && e.contains("not support"), "{e}");
    }

    #[test]
    fn read_header_applies_the_same_rules_from_disk() {
        let dir = std::env::temp_dir().join(format!("folo-nn-header-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.safetensors");
        let read = |bytes: &[u8]| {
            std::fs::write(&path, bytes).unwrap();
            Container::read_header(&path)
        };
        let ok = file(
            r#"{"__metadata__":{"folo.dtype":"F16"},"b":{"dtype":"F16","shape":[2],"data_offsets":[4,8]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}  "#,
            &[0; 8],
        );
        let (whole, header) = (Container::parse(&ok).unwrap(), read(&ok).unwrap());
        assert_eq!((header.metadata, header.tensors), (whole.metadata, whole.tensors));
        // The tensors must still cover the data on disk: a byte too many or too few is refused.
        let mut long = ok.clone();
        long.push(0);
        for bad in [&long[..], &ok[..ok.len() - 1]] {
            assert!(read(bad).unwrap_err().to_string().contains("data section"));
        }
        // Too short for the length, a length past the end of the file, an oversized length.
        assert!(read(&[1, 0, 0]).is_err());
        assert!(read(&file("{}", &[])[..9]).unwrap_err().to_string().contains("runs past the end"));
        let mut huge = (MAX_HEADER + 1).to_le_bytes().to_vec();
        huge.push(b'{');
        assert!(read(&huge).unwrap_err().to_string().contains("limit"));
        assert!(Container::read_header(&dir.join("missing")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
