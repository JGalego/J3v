//! `.j3a` container: a JSON header plus 64-byte-aligned little-endian tensors.
//!
//! ```text
//! b"J3VA" | u32 format | u32 header_len | header JSON | pad to 64 | tensor blob
//! ```
//! The header holds `kind`, free-form `meta` (schema, calibration, conformance report) and a tensor index.
//! The same layout is used for shared encoders (`kind = "encoder"`), pi heads and mcu models, and is simple
//! enough to parse on a microcontroller straight out of flash.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const MAGIC: &[u8; 4] = b"J3VA";
pub const FORMAT: u32 = 1;
const ALIGN: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    I8,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::I8 => 1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub offset: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Header {
    pub kind: String,
    pub meta: Value,
    pub tensors: Vec<TensorInfo>,
}

/// Owned artifact: header plus tensor data (kept as raw bytes, decoded on access).
pub struct Artifact {
    pub header: Header,
    pub data: Vec<u8>,
    index: BTreeMap<String, usize>,
}

pub enum Tensor {
    F32(Vec<f32>),
    I8(Vec<i8>),
}

impl Artifact {
    pub fn new(kind: &str, meta: Value) -> Self {
        Artifact { header: Header { kind: kind.into(), meta, tensors: vec![] }, data: vec![], index: BTreeMap::new() }
    }

    fn push(&mut self, name: &str, dtype: DType, shape: &[usize], bytes: &[u8]) {
        while self.data.len() % ALIGN != 0 {
            self.data.push(0);
        }
        assert_eq!(bytes.len(), shape.iter().product::<usize>() * dtype.size(), "tensor {} size mismatch", name);
        self.index.insert(name.into(), self.header.tensors.len());
        self.header.tensors.push(TensorInfo { name: name.into(), dtype, shape: shape.to_vec(), offset: self.data.len() });
        self.data.extend_from_slice(bytes);
    }

    pub fn add_f32(&mut self, name: &str, shape: &[usize], v: &[f32]) {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.push(name, DType::F32, shape, &b);
    }

    pub fn add_i8(&mut self, name: &str, shape: &[usize], v: &[i8]) {
        let b: Vec<u8> = v.iter().map(|&x| x as u8).collect();
        self.push(name, DType::I8, shape, &b);
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.header.tensors[i])
    }

    pub fn bytes(&self, name: &str) -> Option<&[u8]> {
        let t = self.info(name)?;
        Some(&self.data[t.offset..t.offset + t.numel() * t.dtype.size()])
    }

    pub fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.info(name).ok_or_else(|| format!("missing tensor `{}`", name))?;
        if t.dtype != DType::F32 {
            return Err(format!("tensor `{}` is {:?}, expected f32", name, t.dtype));
        }
        Ok(self.bytes(name).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn i8(&self, name: &str) -> Result<Vec<i8>, String> {
        let t = self.info(name).ok_or_else(|| format!("missing tensor `{}`", name))?;
        if t.dtype != DType::I8 {
            return Err(format!("tensor `{}` is {:?}, expected i8", name, t.dtype));
        }
        Ok(self.bytes(name).unwrap().iter().map(|&b| b as i8).collect())
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], String> {
        self.info(name).map(|t| t.shape.as_slice()).ok_or_else(|| format!("missing tensor `{}`", name))
    }

    /// Bytes of tensor payload (what lands in flash on the mcu target).
    pub fn payload_bytes(&self) -> usize {
        self.header.tensors.iter().map(|t| t.numel() * t.dtype.size()).sum()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let h = serde_json::to_vec(&self.header).expect("header serializes");
        let mut out = Vec::with_capacity(12 + h.len() + ALIGN + self.data.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT.to_le_bytes());
        out.extend_from_slice(&(h.len() as u32).to_le_bytes());
        out.extend_from_slice(&h);
        while out.len() % ALIGN != 0 {
            out.push(0);
        }
        out.extend_from_slice(&self.data);
        out
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        if b.len() < 12 || &b[..4] != MAGIC {
            return Err("not a J3v artifact (bad magic)".into());
        }
        let fmt = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
        if fmt != FORMAT {
            return Err(format!("artifact format {} is not supported by this runtime (expects {})", fmt, FORMAT));
        }
        let hl = u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize;
        if 12 + hl > b.len() {
            return Err("truncated artifact header".into());
        }
        let header: Header = serde_json::from_slice(&b[12..12 + hl]).map_err(|e| format!("bad artifact header: {}", e))?;
        let start = (12 + hl + ALIGN - 1) / ALIGN * ALIGN;
        let data = b.get(start..).ok_or("truncated artifact")?.to_vec();
        let mut index = BTreeMap::new();
        for (i, t) in header.tensors.iter().enumerate() {
            if t.offset + t.numel() * t.dtype.size() > data.len() {
                return Err(format!("tensor `{}` runs past the end of the artifact", t.name));
            }
            index.insert(t.name.clone(), i);
        }
        Ok(Artifact { header, data, index })
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
        Self::from_bytes(&b).map_err(|e| format!("{}: {}", path, e))
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, self.to_bytes()).map_err(|e| format!("{}: {}", path, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trip() {
        let mut a = Artifact::new("test", serde_json::json!({"x": 1}));
        a.add_f32("w", &[2, 2], &[1.0, -2.0, 3.5, 0.0]);
        a.add_i8("q", &[3], &[-128, 0, 127]);
        let b = Artifact::from_bytes(&a.to_bytes()).unwrap();
        assert_eq!(b.f32("w").unwrap(), vec![1.0, -2.0, 3.5, 0.0]);
        assert_eq!(b.i8("q").unwrap(), vec![-128, 0, 127]);
        assert_eq!(b.header.meta["x"], 1);
        assert_eq!(b.info("q").unwrap().offset % 64, 0);
    }
}
