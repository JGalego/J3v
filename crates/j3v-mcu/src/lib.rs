//! J3v `mcu` tier: one tiny distilled model per schema.
//!
//! * Front end: hashed byte-level word unigrams, word bigrams and character 3/4-grams (fastText-style).
//!   No vocabulary table: a feature is `fnv1a32(kind, bytes) & (buckets - 1)`.
//! * Model: int8 embedding bag (mean) -> int8 linear + ReLU -> one int8 linear head per question.
//!   Math is f32 on dequantized weights (Cortex-M4F/M7 have single-precision FPUs).
//! * `no_std`, no allocator, fixed stack buffers. The same code runs in the firmware and in `j3v compile`,
//!   which calibrates and conformance-checks the model through it.
#![no_std]

pub const MAX_FEATS: usize = 256;
pub const MAX_K: usize = 20;
pub const MAX_DIM: usize = 128;
pub const MAX_HIDDEN: usize = 128;

pub struct Head<'a> {
    pub w: &'a [i8],
    pub s: &'a [f32],
    pub b: &'a [f32],
    pub k: usize,
    pub temperature: f32,
}

pub struct Model<'a> {
    pub buckets: u32,
    pub dim: usize,
    pub hidden: usize,
    pub max_bytes: usize,
    pub emb: &'a [i8],
    pub emb_s: &'a [f32],
    pub w1: &'a [i8],
    pub s1: &'a [f32],
    pub b1: &'a [f32],
    pub heads: &'a [Head<'a>],
    pub threshold: f32,
}

#[inline]
fn fnv(h: u32, b: u8) -> u32 {
    (h ^ b as u32).wrapping_mul(0x0100_0193)
}

const SEED: u32 = 0x811c_9dc5;

#[inline]
fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b >= 0x80
}

#[inline]
fn low(b: u8) -> u8 {
    b.to_ascii_lowercase()
}

/// Hashed feature ids for `text` (at most `max_bytes` bytes, at most MAX_FEATS features). Returns the count.
pub fn features(text: &[u8], max_bytes: usize, buckets: u32, out: &mut [u32; MAX_FEATS]) -> usize {
    let t = &text[..text.len().min(max_bytes)];
    let mask = buckets.wrapping_sub(1); // buckets = 0: raw 32-bit hashes (used by the compiler)
    let mut n = 0;
    let mut push = |h: u32, n: &mut usize| {
        if *n < MAX_FEATS {
            out[*n] = h & mask;
            *n += 1;
        }
    };
    let mut i = 0;
    let mut prev: Option<(usize, usize)> = None;
    while i < t.len() {
        while i < t.len() && !is_word(t[i]) {
            i += 1;
        }
        let st = i;
        while i < t.len() && is_word(t[i]) {
            i += 1;
        }
        if st == i {
            break;
        }
        let w = &t[st..i];
        // word unigram
        let mut h = fnv(SEED, 1);
        for &b in w {
            h = fnv(h, low(b));
        }
        push(h, &mut n);
        // word bigram
        if let Some((ps, pe)) = prev {
            let mut h = fnv(SEED, 2);
            for &b in &t[ps..pe] {
                h = fnv(h, low(b));
            }
            h = fnv(h, b' ');
            for &b in w {
                h = fnv(h, low(b));
            }
            push(h, &mut n);
        }
        prev = Some((st, i));
        // char 3- and 4-grams of "<word>"
        let len = w.len() + 2;
        let at = |j: usize| -> u8 {
            if j == 0 {
                b'<'
            } else if j == len - 1 {
                b'>'
            } else {
                low(w[j - 1])
            }
        };
        for g in [3usize, 4] {
            if len < g {
                continue;
            }
            for s in 0..=len - g {
                let mut h = fnv(SEED, g as u8);
                for j in s..s + g {
                    h = fnv(h, at(j));
                }
                push(h, &mut n);
            }
        }
    }
    n
}

/// Raw logits for every question. `out[j][..k_j]` is filled. Returns false for empty input (no features).
pub fn logits(m: &Model, text: &[u8], out: &mut [[f32; MAX_K]]) -> bool {
    let mut f = [0u32; MAX_FEATS];
    let n = features(text, m.max_bytes, m.buckets, &mut f);
    let d = m.dim;
    let mut x = [0f32; MAX_DIM];
    for &id in &f[..n] {
        let r = id as usize;
        let s = m.emb_s[r];
        for (xv, &e) in x[..d].iter_mut().zip(&m.emb[r * d..(r + 1) * d]) {
            *xv += e as f32 * s;
        }
    }
    let inv = if n > 0 { 1.0 / n as f32 } else { 0.0 };
    let mut h = [0f32; MAX_HIDDEN];
    for o in 0..m.hidden {
        let mut acc = 0f32;
        for (&w, &xv) in m.w1[o * d..(o + 1) * d].iter().zip(&x[..d]) {
            acc += w as f32 * xv;
        }
        let v = acc * inv * m.s1[o] + m.b1[o];
        h[o] = if v > 0.0 { v } else { 0.0 };
    }
    for (hd, z) in m.heads.iter().zip(out.iter_mut()) {
        for o in 0..hd.k {
            let mut acc = 0f32;
            for (&w, &hv) in hd.w[o * m.hidden..(o + 1) * m.hidden].iter().zip(&h[..m.hidden]) {
                acc += w as f32 * hv;
            }
            z[o] = acc * hd.s[o] + hd.b[o];
        }
    }
    n > 0
}

fn exp(x: f32) -> f32 {
    // no libm in core: exp(x) = 2^i * 2^f with i = round(x log2 e), |f| <= 0.5, Taylor to degree 6 (rel. error < 1e-6)
    let x = x.clamp(-87.0, 88.0) * core::f32::consts::LOG2_E;
    let xi = if x < 0.0 { (x - 0.5) as i32 } else { (x + 0.5) as i32 };
    let f = x - xi as f32;
    let p =
        1.0 + f * (0.693_147_2 + f * (0.240_226_5 + f * (0.055_504_11 + f * (0.009_618_129 + f * (0.001_333_355 + f * 0.000_154_035_3)))));
    p * f32::from_bits(((xi + 127) as u32) << 23)
}

/// Calibrated probabilities (temperature-scaled softmax) in place; returns (argmax, p_top).
pub fn calibrate(z: &mut [f32], temperature: f32) -> (usize, f32) {
    let m = z.iter().fold(f32::NEG_INFINITY, |a, &b| if b > a { b } else { a });
    let mut s = 0.0;
    for v in z.iter_mut() {
        *v = exp((*v - m) / temperature);
        s += *v;
    }
    let mut best = 0;
    for i in 0..z.len() {
        z[i] /= s;
        if z[i] > z[best] {
            best = i;
        }
    }
    (best, z[best])
}

/// Bytes of flash the model's weights occupy (what `--budget` constrains).
pub fn flash_bytes(buckets: usize, dim: usize, hidden: usize, ks: &[usize]) -> usize {
    buckets * dim + buckets * 4 + hidden * dim + hidden * 8 + ks.iter().map(|k| k * hidden + k * 8 + 4).sum::<usize>()
}

/// Static RAM for one inference (stack buffers above).
pub fn ram_bytes(dim: usize, hidden: usize, nq: usize) -> usize {
    MAX_FEATS * 4 + dim.max(1) * 4 + hidden * 4 + nq * MAX_K * 4 + 64
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exp_close() {
        for i in -200..200 {
            let x = i as f32 * 0.1;
            let (a, b) = (exp(x), (x as f64).exp() as f32);
            assert!(((a - b) / b).abs() < 1e-5, "{} {} {}", x, a, b);
        }
    }
    #[test]
    fn feats_stable() {
        let mut f = [0u32; MAX_FEATS];
        let n = features(b"Refund me!", 256, 1024, &mut f);
        // "refund": 1 uni + 6 tri + 5 quad; "me": 1 uni + 1 bi + 2 tri + 1 quad
        assert_eq!(n, 1 + 6 + 5 + 1 + 1 + 2 + 1);
        let mut g = [0u32; MAX_FEATS];
        assert_eq!(features(b"REFUND me", 256, 1024, &mut g), n);
        assert_eq!(f[..n], g[..n]);
    }
}
