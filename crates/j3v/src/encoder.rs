//! Shared int8 BERT encoder for the `pi` target.
//!
//! Weights: symmetric per-output-channel int8. Activations: dynamic per-token int8 in front of every
//! linear layer, i32 accumulation. Attention, LayerNorm and GELU stay in f32 (they are <5% of FLOPs
//! at the sequence lengths J3v serves).

use crate::tokenizer::WordPiece;
use j3v_core::artifact::Artifact;
use serde_json::{json, Value};
use std::sync::OnceLock;

pub struct QLinear {
    w: Vec<i8>,
    scale: Vec<f32>,
    bias: Vec<f32>,
    pub out: usize,
    pub inp: usize,
}

pub struct Layer {
    qkv: QLinear,
    o: QLinear,
    ln1: (Vec<f32>, Vec<f32>),
    f1: QLinear,
    f2: QLinear,
    ln2: (Vec<f32>, Vec<f32>),
}

pub struct Encoder {
    pub tok: WordPiece,
    word: Vec<i8>,
    word_scale: Vec<f32>,
    pos: Vec<f32>,
    type0: Vec<f32>,
    ln: (Vec<f32>, Vec<f32>),
    layers: Vec<Layer>,
    pub d: usize,
    heads: usize,
    pub max_pos: usize,
    eps: f32,
    pub id: String,
}

// ----------------------------------------------------------------------------- quantization

pub fn quantize_rows(w: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols];
    let mut s = vec![0f32; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let m = row.iter().fold(0f32, |a, &b| a.max(b.abs()));
        let sc = if m > 0.0 { m / 127.0 } else { 1.0 };
        s[r] = sc;
        for c in 0..cols {
            q[r * cols + c] = (row[c] / sc).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, s)
}

// ----------------------------------------------------------------------------- kernels

#[inline(always)]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let mut s = 0i32;
    for i in 0..a.len() {
        s += a[i] as i16 as i32 * b[i] as i16 as i32;
    }
    s
}

#[inline(always)]
fn gemm_body(x: &[i8], t: usize, w: &[i8], o0: usize, o1: usize, inp: usize, acc: &mut [i32], stride: usize) {
    for o in o0..o1 {
        let wr = &w[o * inp..(o + 1) * inp];
        for r in 0..t {
            acc[r * stride + (o - o0)] = dot_i8(&x[r * inp..(r + 1) * inp], wr);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_avx2(x: &[i8], t: usize, w: &[i8], o0: usize, o1: usize, inp: usize, acc: &mut [i32], stride: usize) {
    gemm_body(x, t, w, o0, o1, inp, acc, stride)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn gemm_dotprod(x: &[i8], t: usize, w: &[i8], o0: usize, o1: usize, inp: usize, acc: &mut [i32], stride: usize) {
    gemm_body(x, t, w, o0, o1, inp, acc, stride)
}

type Gemm = fn(&[i8], usize, &[i8], usize, usize, usize, &mut [i32], usize);

fn gemm_impl() -> (Gemm, &'static str) {
    static G: OnceLock<(Gemm, &'static str)> = OnceLock::new();
    *G.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            return (|x, t, w, a, b, i, acc, s| unsafe { gemm_avx2(x, t, w, a, b, i, acc, s) }, "avx2");
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return (|x, t, w, a, b, i, acc, s| unsafe { gemm_dotprod(x, t, w, a, b, i, acc, s) }, "neon+dotprod");
        }
        (gemm_body, "portable")
    })
}

pub fn kernel_name() -> &'static str {
    gemm_impl().1
}

static THREADS: OnceLock<usize> = OnceLock::new();

/// Worker threads for large matmuls (default 1: J3v inputs are short and thread fan-out costs more than it saves).
pub fn set_threads(n: usize) {
    let _ = THREADS.set(n.max(1));
}

fn threads() -> usize {
    *THREADS.get_or_init(|| {
        std::env::var("J3V_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
            1
        })
    })
}

impl QLinear {
    fn from(a: &Artifact, p: &str) -> Result<Self, String> {
        let sh = a.shape(&format!("{}.w", p))?.to_vec();
        Ok(QLinear {
            w: a.i8(&format!("{}.w", p))?,
            scale: a.f32(&format!("{}.s", p))?,
            bias: a.f32(&format!("{}.b", p))?,
            out: sh[0],
            inp: sh[1],
        })
    }

    /// y[t, out] = x[t, inp] W^T + b, with x quantized per row.
    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let (inp, out) = (self.inp, self.out);
        let mut xq = vec![0i8; t * inp];
        let mut xs = vec![0f32; t];
        for r in 0..t {
            let row = &x[r * inp..(r + 1) * inp];
            let m = row.iter().fold(0f32, |a, &b| a.max(b.abs()));
            let s = if m > 0.0 { m / 127.0 } else { 1.0 };
            xs[r] = s;
            let inv = 1.0 / s;
            for c in 0..inp {
                xq[r * inp + c] = (row[c] * inv).round() as i8;
            }
        }
        let gemm = gemm_impl().0;
        let nt = if t * out * inp >= 1 << 20 { threads() } else { 1 };
        let mut acc = vec![0i32; t * out];
        if nt <= 1 {
            gemm(&xq, t, &self.w, 0, out, inp, &mut acc, out);
        } else {
            // each worker owns a contiguous block of output columns, written to its own buffer
            let chunk = (out + nt - 1) / nt;
            let parts: Vec<(usize, usize, Vec<i32>)> = std::thread::scope(|sc| {
                let hs: Vec<_> = (0..nt)
                    .map(|k| {
                        let (o0, o1) = (k * chunk, ((k + 1) * chunk).min(out));
                        let (xq, w) = (&xq, &self.w);
                        sc.spawn(move || {
                            let mut a = vec![0i32; t * (o1 - o0)];
                            gemm(xq, t, w, o0, o1, inp, &mut a, o1 - o0);
                            (o0, o1, a)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for (o0, o1, a) in parts {
                let n = o1 - o0;
                for r in 0..t {
                    acc[r * out + o0..r * out + o1].copy_from_slice(&a[r * n..(r + 1) * n]);
                }
            }
        }
        let mut y = vec![0f32; t * out];
        for r in 0..t {
            for o in 0..out {
                y[r * out + o] = acc[r * out + o] as f32 * xs[r] * self.scale[o] + self.bias[o];
            }
        }
        y
    }
}

fn layer_norm(x: &mut [f32], d: usize, g: &[f32], b: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(d) {
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            row[i] = (row[i] - mean) * inv * g[i] + b[i];
        }
    }
}

/// erf, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
pub fn erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs() as f64;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    s * y as f32
}

fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

impl Encoder {
    pub fn load(a: &Artifact) -> Result<Self, String> {
        if a.header.kind != "encoder" {
            return Err(format!("expected an encoder artifact, got `{}`", a.header.kind));
        }
        let m = &a.header.meta;
        let g = |k: &str| m[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("encoder meta lacks {}", k));
        let vocab: Vec<String> =
            m["vocab"].as_array().ok_or("encoder meta lacks vocab")?.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
        let tok = WordPiece::new(&vocab, m["lowercase"].as_bool().unwrap_or(true))?;
        let n = g("layers")?;
        let ln = |p: &str| -> Result<(Vec<f32>, Vec<f32>), String> { Ok((a.f32(&format!("{}.g", p))?, a.f32(&format!("{}.b", p))?)) };
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            layers.push(Layer {
                qkv: QLinear::from(a, &format!("l{}.qkv", i))?,
                o: QLinear::from(a, &format!("l{}.o", i))?,
                ln1: ln(&format!("l{}.ln1", i))?,
                f1: QLinear::from(a, &format!("l{}.f1", i))?,
                f2: QLinear::from(a, &format!("l{}.f2", i))?,
                ln2: ln(&format!("l{}.ln2", i))?,
            });
        }
        Ok(Encoder {
            tok,
            word: a.i8("word.w")?,
            word_scale: a.f32("word.s")?,
            pos: a.f32("pos")?,
            type0: a.f32("type0")?,
            ln: ln("emb.ln")?,
            layers,
            d: g("hidden")?,
            heads: g("heads")?,
            max_pos: g("max_pos")?,
            eps: m["eps"].as_f64().unwrap_or(1e-12) as f32,
            id: m["id"].as_str().unwrap_or("").to_string(),
        })
    }

    /// Last-layer hidden states, row-major [T, d].
    pub fn forward(&self, ids: &[u32]) -> Vec<f32> {
        let (d, t) = (self.d, ids.len().min(self.max_pos));
        let mut x = vec![0f32; t * d];
        for (r, &id) in ids[..t].iter().enumerate() {
            let s = self.word_scale[id as usize];
            let w = &self.word[id as usize * d..(id as usize + 1) * d];
            let p = &self.pos[r * d..(r + 1) * d];
            for c in 0..d {
                x[r * d + c] = w[c] as f32 * s + p[c] + self.type0[c];
            }
        }
        layer_norm(&mut x, d, &self.ln.0, &self.ln.1, self.eps);
        let (h, dh) = (self.heads, d / self.heads);
        let scale = 1.0 / (dh as f32).sqrt();
        let mut ctx = vec![0f32; t * d];
        let mut sc = vec![0f32; t];
        for l in &self.layers {
            let qkv = l.qkv.forward(&x, t);
            for hh in 0..h {
                let (qo, ko, vo) = (hh * dh, d + hh * dh, 2 * d + hh * dh);
                for i in 0..t {
                    let q = &qkv[i * 3 * d + qo..i * 3 * d + qo + dh];
                    let mut m = f32::NEG_INFINITY;
                    for j in 0..t {
                        let k = &qkv[j * 3 * d + ko..j * 3 * d + ko + dh];
                        let s = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                        sc[j] = s;
                        m = m.max(s);
                    }
                    let mut z = 0.0;
                    for s in sc[..t].iter_mut() {
                        *s = (*s - m).exp();
                        z += *s;
                    }
                    let c = &mut ctx[i * d + qo..i * d + qo + dh];
                    c.iter_mut().for_each(|v| *v = 0.0);
                    for j in 0..t {
                        let p = sc[j] / z;
                        let v = &qkv[j * 3 * d + vo..j * 3 * d + vo + dh];
                        for e in 0..dh {
                            c[e] += p * v[e];
                        }
                    }
                }
            }
            let a = l.o.forward(&ctx, t);
            for (xv, av) in x.iter_mut().zip(&a) {
                *xv += av;
            }
            layer_norm(&mut x, d, &l.ln1.0, &l.ln1.1, self.eps);
            let mut f = l.f1.forward(&x, t);
            f.iter_mut().for_each(|v| *v = gelu_erf(*v));
            let f = l.f2.forward(&f, t);
            for (xv, fv) in x.iter_mut().zip(&f) {
                *xv += fv;
            }
            layer_norm(&mut x, d, &l.ln2.0, &l.ln2.1, self.eps);
        }
        x
    }

    pub fn params(a: &Artifact) -> usize {
        a.header.tensors.iter().filter(|t| !t.name.ends_with(".s")).map(|t| t.numel()).sum()
    }
}

// ----------------------------------------------------------------------------- import from Hugging Face

fn read_safetensors(path: &str) -> Result<std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: Value = serde_json::from_slice(&b[8..8 + n]).map_err(|e| e.to_string())?;
    let data = &b[8 + n..];
    let mut out = std::collections::HashMap::new();
    for (k, v) in h.as_object().unwrap() {
        if k == "__metadata__" || v["dtype"] != "F32" {
            continue;
        }
        let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let o = v["data_offsets"].as_array().unwrap();
        let (s, e) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
        let f = data[s..e].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        out.insert(k.strip_prefix("bert.").unwrap_or(k).to_string(), (shape, f));
    }
    Ok(out)
}

/// Convert a Hugging Face BERT-family checkpoint directory (config.json, vocab.txt, model.safetensors)
/// into an int8 J3v encoder artifact.
pub fn import_hf(dir: &str, source: &str) -> Result<Artifact, String> {
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(format!("{}/config.json", dir)).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if cfg["model_type"] != "bert" {
        return Err(format!("only BERT-family encoders are supported, got {}", cfg["model_type"]));
    }
    let vocab: Vec<String> =
        std::fs::read_to_string(format!("{}/vocab.txt", dir)).map_err(|e| e.to_string())?.lines().map(str::to_string).collect();
    let lower = std::fs::read_to_string(format!("{}/tokenizer_config.json", dir))
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v["do_lower_case"].as_bool())
        .unwrap_or(true);
    let st = read_safetensors(&format!("{}/model.safetensors", dir))?;
    let get = |k: &str| st.get(k).ok_or_else(|| format!("checkpoint lacks {}", k));
    let (d, n, heads, ffn) = (
        cfg["hidden_size"].as_u64().unwrap() as usize,
        cfg["num_hidden_layers"].as_u64().unwrap() as usize,
        cfg["num_attention_heads"].as_u64().unwrap() as usize,
        cfg["intermediate_size"].as_u64().unwrap() as usize,
    );
    let max_pos = cfg["max_position_embeddings"].as_u64().unwrap() as usize;
    let meta = json!({
        "arch": "bert", "hidden": d, "layers": n, "heads": heads, "ffn": ffn, "max_pos": max_pos,
        "eps": cfg["layer_norm_eps"].as_f64().unwrap_or(1e-12), "lowercase": lower, "vocab": vocab,
        "source": source, "quant": "int8 weights per-channel, int8 activations per-token",
        "id": format!("{}#{:016x}", source, j3v_core::schema::fnv1a64(&std::fs::read(format!("{}/model.safetensors", dir)).unwrap_or_default())),
    });
    let mut a = Artifact::new("encoder", meta);
    let (ws, w) = get("embeddings.word_embeddings.weight")?;
    let (q, s) = quantize_rows(w, ws[0], ws[1]);
    a.add_i8("word.w", ws, &q);
    a.add_f32("word.s", &[ws[0]], &s);
    let (ps, p) = get("embeddings.position_embeddings.weight")?;
    a.add_f32("pos", ps, p);
    a.add_f32("type0", &[d], &get("embeddings.token_type_embeddings.weight")?.1[..d]);
    a.add_f32("emb.ln.g", &[d], &get("embeddings.LayerNorm.weight")?.1);
    a.add_f32("emb.ln.b", &[d], &get("embeddings.LayerNorm.bias")?.1);
    let lin = |a: &mut Artifact, name: &str, srcs: &[&str]| -> Result<(), String> {
        let mut w = Vec::new();
        let mut b = Vec::new();
        let mut rows = 0;
        let mut cols = 0;
        for s in srcs {
            let (sh, v) = get(&format!("{}.weight", s))?;
            rows += sh[0];
            cols = sh[1];
            w.extend_from_slice(v);
            b.extend_from_slice(&get(&format!("{}.bias", s))?.1);
        }
        let (q, sc) = quantize_rows(&w, rows, cols);
        a.add_i8(&format!("{}.w", name), &[rows, cols], &q);
        a.add_f32(&format!("{}.s", name), &[rows], &sc);
        a.add_f32(&format!("{}.b", name), &[rows], &b);
        Ok(())
    };
    for i in 0..n {
        let p = format!("encoder.layer.{}", i);
        lin(&mut a, &format!("l{}.qkv", i), &[
            &format!("{}.attention.self.query", p),
            &format!("{}.attention.self.key", p),
            &format!("{}.attention.self.value", p),
        ])?;
        lin(&mut a, &format!("l{}.o", i), &[&format!("{}.attention.output.dense", p)])?;
        a.add_f32(&format!("l{}.ln1.g", i), &[d], &get(&format!("{}.attention.output.LayerNorm.weight", p))?.1);
        a.add_f32(&format!("l{}.ln1.b", i), &[d], &get(&format!("{}.attention.output.LayerNorm.bias", p))?.1);
        lin(&mut a, &format!("l{}.f1", i), &[&format!("{}.intermediate.dense", p)])?;
        lin(&mut a, &format!("l{}.f2", i), &[&format!("{}.output.dense", p)])?;
        a.add_f32(&format!("l{}.ln2.g", i), &[d], &get(&format!("{}.output.LayerNorm.weight", p))?.1);
        a.add_f32(&format!("l{}.ln2.b", i), &[d], &get(&format!("{}.output.LayerNorm.bias", p))?.1);
    }
    Ok(a)
}
