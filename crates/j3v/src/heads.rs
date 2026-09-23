//! Per-schema decision heads over the shared encoder's token states.
//!
//! For each question q: attention pooling with a learned query, concatenated with the mean-pooled
//! state, then a 2-layer MLP to K logits. Options are baked into the weights at compile time, so one
//! encoder pass answers every question (Laya needs one cross-encoder pass per question).

use j3v_core::artifact::{Artifact, DType};

pub struct Head {
    a: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    pub k: usize,
    hid: usize,
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}

impl Head {
    pub fn load(a: &Artifact, j: usize) -> Result<Self, String> {
        let p = |n: &str| format!("q{}.{}", j, n);
        // weights are stored int8 with per-row scales (`.s`) in shipped artifacts, f32 in compile-time drafts
        let w = |n: &str| -> Result<Vec<f32>, String> {
            match a.info(&p(n)).map(|t| t.dtype) {
                Some(DType::I8) => {
                    let (q, s) = (a.i8(&p(n))?, a.f32(&p(&format!("{}.s", n)))?);
                    let cols = q.len() / s.len();
                    Ok(q.iter().enumerate().map(|(i, &v)| v as f32 * s[i / cols]).collect())
                }
                _ => a.f32(&p(n)),
            }
        };
        let sh = a.shape(&p("w2"))?.to_vec();
        Ok(Head { a: a.f32(&p("a"))?, w1: w("w1")?, b1: a.f32(&p("b1"))?, w2: w("w2")?, b2: a.f32(&p("b2"))?, k: sh[0], hid: sh[1] })
    }

    /// h: encoder output [t, d] -> K raw logits.
    pub fn forward(&self, h: &[f32], d: usize) -> Vec<f32> {
        let t = h.len() / d;
        let scale = 1.0 / (d as f32).sqrt();
        let s: Vec<f32> = h.chunks_exact(d).map(|r| r.iter().zip(&self.a).map(|(x, y)| x * y).sum::<f32>() * scale).collect();
        let m = s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let e: Vec<f32> = s.iter().map(|v| (v - m).exp()).collect();
        let z: f32 = e.iter().sum();
        let mut f = vec![0f32; 2 * d];
        for (r, row) in h.chunks_exact(d).enumerate() {
            let w = e[r] / z;
            for c in 0..d {
                f[c] += w * row[c];
                f[d + c] += row[c] / t as f32;
            }
        }
        let mut x = vec![0f32; self.hid];
        for o in 0..self.hid {
            let w = &self.w1[o * 2 * d..(o + 1) * 2 * d];
            x[o] = gelu_tanh(w.iter().zip(&f).map(|(a, b)| a * b).sum::<f32>() + self.b1[o]);
        }
        (0..self.k)
            .map(|o| self.w2[o * self.hid..(o + 1) * self.hid].iter().zip(&x).map(|(a, b)| a * b).sum::<f32>() + self.b2[o])
            .collect()
    }
}

/// Draft (f32) heads -> shipped heads: int8 per-row weights for the two matrices, f32 for vectors.
pub fn quantize(draft: &Artifact, nq: usize) -> Result<Artifact, String> {
    let mut out = Artifact::new("pi-heads", draft.header.meta.clone());
    for j in 0..nq {
        let p = |n: &str| format!("q{}.{}", j, n);
        out.add_f32(&p("a"), draft.shape(&p("a"))?, &draft.f32(&p("a"))?);
        for (m, b) in [("w1", "b1"), ("w2", "b2")] {
            let sh = draft.shape(&p(m))?.to_vec();
            let (q, s) = crate::encoder::quantize_rows(&draft.f32(&p(m))?, sh[0], sh[1]);
            out.add_i8(&p(m), &sh, &q);
            out.add_f32(&p(&format!("{}.s", m)), &[sh[0]], &s);
            out.add_f32(&p(b), draft.shape(&p(b))?, &draft.f32(&p(b))?);
        }
    }
    Ok(out)
}
