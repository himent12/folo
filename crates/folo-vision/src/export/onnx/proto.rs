//! A minimal protobuf encoder for the ONNX messages FOLO writes (`onnx.proto`, IR version 8).
//!
//! Only encoding, and only the fields used: `ModelProto`, `OperatorSetIdProto`, `StringStringEntryProto`,
//! `GraphProto`, `NodeProto`, `AttributeProto`, `TensorProto` (with `raw_data`) and `ValueInfoProto` with its
//! `TypeProto.Tensor` / `TensorShapeProto`. Field numbers are onnx.proto's. Repeated scalars are written
//! unpacked (one tag per element), proto2's default encoding, which every protobuf parser accepts.

/// `TensorProto.DataType` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    /// `FLOAT = 1`.
    Float = 1,
    /// `INT64 = 7`.
    Int64 = 7,
    /// `FLOAT16 = 10`.
    Float16 = 10,
}

impl DataType {
    /// The protobuf enum value.
    pub fn code(self) -> i64 {
        self as i64
    }

    /// The ONNX type name, e.g. `tensor(float)`.
    pub fn name(self) -> &'static str {
        match self {
            DataType::Float => "float",
            DataType::Int64 => "int64",
            DataType::Float16 => "float16",
        }
    }
}

/// A growing protobuf message body.
#[derive(Default)]
pub struct Msg(pub Vec<u8>);

const VARINT: u64 = 0;
const I32: u64 = 5;
const LEN: u64 = 2;

impl Msg {
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.0.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }

    fn tag(&mut self, field: u64, wire: u64) {
        self.varint((field << 3) | wire);
    }

    /// An `int64` / `int32` / enum field (negative values take ten bytes, as protobuf specifies).
    pub fn int(&mut self, field: u64, v: i64) -> &mut Self {
        self.tag(field, VARINT);
        self.varint(v as u64);
        self
    }

    /// A `float` field.
    pub fn float(&mut self, field: u64, v: f32) -> &mut Self {
        self.tag(field, I32);
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    /// A `bytes` field.
    pub fn bytes(&mut self, field: u64, v: &[u8]) -> &mut Self {
        self.tag(field, LEN);
        self.varint(v.len() as u64);
        self.0.extend_from_slice(v);
        self
    }

    /// A `string` field.
    pub fn string(&mut self, field: u64, v: &str) -> &mut Self {
        self.bytes(field, v.as_bytes())
    }

    /// An embedded message field.
    pub fn msg(&mut self, field: u64, m: &Msg) -> &mut Self {
        self.bytes(field, &m.0)
    }

    /// The length prefix of an embedded message of `len` bytes, for writing its body separately.
    pub fn msg_header(&mut self, field: u64, len: usize) -> &mut Self {
        self.tag(field, LEN);
        self.varint(len as u64);
        self
    }
}

/// One dimension of a `TensorShapeProto`: a fixed size or a symbolic name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    /// `dim_value`.
    Value(usize),
    /// `dim_param`, e.g. `batch`.
    Param(String),
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dim::Value(v) => write!(f, "{v}"),
            Dim::Param(p) => f.write_str(p),
        }
    }
}

/// `ValueInfoProto { name, type: tensor(elem, shape) }`.
pub fn value_info(name: &str, elem: DataType, dims: &[Dim]) -> Msg {
    let mut shape = Msg::default();
    for d in dims {
        let mut dm = Msg::default();
        match d {
            Dim::Value(v) => dm.int(1, *v as i64),
            Dim::Param(p) => dm.string(2, p),
        };
        shape.msg(1, &dm);
    }
    let mut tensor = Msg::default();
    tensor.int(1, elem.code()).msg(2, &shape);
    let mut ty = Msg::default();
    ty.msg(1, &tensor);
    let mut vi = Msg::default();
    vi.string(1, name).msg(2, &ty);
    vi
}

/// `TensorProto { dims, data_type, name, raw_data }` (little-endian raw data).
pub fn tensor(name: &str, elem: DataType, dims: &[usize], raw: &[u8]) -> Msg {
    let mut t = Msg::default();
    for &d in dims {
        t.int(1, d as i64);
    }
    t.int(2, elem.code()).string(8, name).bytes(9, raw);
    t
}

/// An attribute value (`AttributeProto` with its `type`).
#[derive(Clone, Debug, PartialEq)]
pub enum AttrValue {
    /// `FLOAT = 1`.
    Float(f32),
    /// `INT = 2`.
    Int(i64),
    /// `STRING = 3`.
    Str(String),
    /// `INTS = 7`.
    Ints(Vec<i64>),
}

/// `AttributeProto`.
pub fn attribute(name: &str, v: &AttrValue) -> Msg {
    let mut a = Msg::default();
    a.string(1, name);
    match v {
        AttrValue::Float(f) => {
            a.float(2, *f).int(20, 1);
        }
        AttrValue::Int(i) => {
            a.int(3, *i).int(20, 2);
        }
        AttrValue::Str(s) => {
            a.string(4, s).int(20, 3);
        }
        AttrValue::Ints(v) => {
            for &i in v {
                a.int(8, i);
            }
            a.int(20, 7);
        }
    }
    a
}

/// `NodeProto { input, output, name, op_type, attribute }` in the default domain.
pub fn node(
    op: &str,
    name: &str,
    inputs: &[String],
    outputs: &[String],
    attrs: &[(String, AttrValue)],
) -> Msg {
    let mut n = Msg::default();
    for i in inputs {
        n.string(1, i);
    }
    for o in outputs {
        n.string(2, o);
    }
    n.string(3, name).string(4, op);
    for (k, v) in attrs {
        n.msg(5, &attribute(k, v));
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_and_tags() {
        let mut m = Msg::default();
        m.int(1, 1).int(2, 300).int(3, -1);
        assert_eq!(&m.0[..5], &[0x08, 0x01, 0x10, 0xac, 0x02]);
        // -1 as int64 is ten bytes: nine 0xff and 0x01.
        assert_eq!(m.0[5], 0x18);
        assert_eq!(&m.0[6..], &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]);
    }

    #[test]
    fn strings_floats_and_messages() {
        let mut inner = Msg::default();
        inner.string(1, "ab");
        let mut m = Msg::default();
        m.msg(7, &inner).float(2, 1.0);
        assert_eq!(m.0, [0x3a, 4, 0x0a, 2, b'a', b'b', 0x15, 0, 0, 0x80, 0x3f]);
    }

    #[test]
    fn attribute_types() {
        let a = attribute("axis", &AttrValue::Int(1));
        assert_eq!(a.0, [0x0a, 4, b'a', b'x', b'i', b's', 0x18, 1, 0xa0, 0x01, 2]);
        let a = attribute("p", &AttrValue::Ints(vec![1, 2]));
        assert_eq!(a.0, [0x0a, 1, b'p', 0x40, 1, 0x40, 2, 0xa0, 0x01, 7]);
    }
}
