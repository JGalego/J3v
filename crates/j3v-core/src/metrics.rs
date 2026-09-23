//! Calibration and conformance metrics. Pure functions, no allocation-heavy deps; shared by every tier.

/// Softmax of `z / t` into `out`.
pub fn softmax_t(z: &[f32], t: f32, out: &mut [f32]) {
    let m = z.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut s = 0.0;
    for (o, &x) in out.iter_mut().zip(z) {
        *o = ((x - m) / t).exp();
        s += *o;
    }
    for o in out.iter_mut() {
        *o /= s;
    }
}

pub fn argmax(p: &[f32]) -> usize {
    let mut b = 0;
    for i in 1..p.len() {
        if p[i] > p[b] {
            b = i;
        }
    }
    b
}

/// 15-bin equal-width ECE on top-1 confidence, bins `(lo, hi]` (matches Laya's `ece_score`).
pub fn ece(conf: &[f64], correct: &[bool]) -> f64 {
    ece_idx(conf, correct, &(0..conf.len()).collect::<Vec<_>>())
}

fn ece_idx(conf: &[f64], correct: &[bool], idx: &[usize]) -> f64 {
    const BINS: usize = 15;
    let mut n = [0usize; BINS];
    let mut sc = [0f64; BINS];
    let mut sa = [0f64; BINS];
    for &i in idx {
        let b = ((conf[i] * BINS as f64).ceil() as usize).clamp(1, BINS) - 1;
        n[b] += 1;
        sc[b] += conf[i];
        sa[b] += correct[i] as u8 as f64;
    }
    let tot = idx.len().max(1) as f64;
    (0..BINS).filter(|&b| n[b] > 0).map(|b| (n[b] as f64 / tot) * ((sc[b] - sa[b]) / n[b] as f64).abs()).sum()
}

fn nll(logits: &[Vec<f32>], labels: &[usize], beta: f64) -> f64 {
    let mut s = 0.0;
    for (z, &y) in logits.iter().zip(labels) {
        let m = z.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(beta * b as f64));
        let lse = m + z.iter().map(|&x| (beta * x as f64 - m).exp()).sum::<f64>().ln();
        s += lse - beta * z[y] as f64;
    }
    s / labels.len().max(1) as f64
}

/// Post-hoc temperature scaling: minimise NLL over T. NLL is convex in beta = 1/T, so golden-section is exact.
pub fn fit_temperature(logits: &[Vec<f32>], labels: &[usize]) -> f32 {
    let (mut a, mut b) = (0.01f64, 20.0f64);
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (b - g * (b - a), a + g * (b - a));
    let (mut fc, mut fd) = (nll(logits, labels, c), nll(logits, labels, d));
    for _ in 0..80 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = nll(logits, labels, c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = nll(logits, labels, d);
        }
    }
    (2.0 / (a + b)) as f32
}

/// Deterministic xorshift for bootstrap resampling.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Point estimate and one-sided 95% bootstrap bounds (5th and 95th percentile).
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Est {
    pub value: f64,
    pub lo: f64,
    pub hi: f64,
    pub n: usize,
}

pub fn bootstrap(n: usize, reps: usize, seed: u64, f: impl Fn(&[usize]) -> f64) -> Est {
    let all: Vec<usize> = (0..n).collect();
    let value = f(&all);
    if n == 0 {
        return Est { value: f64::NAN, lo: f64::NAN, hi: f64::NAN, n };
    }
    let mut rng = Rng::new(seed);
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let idx: Vec<usize> = (0..n).map(|_| rng.below(n)).collect();
            f(&idx)
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Est { value, lo: v[reps * 5 / 100], hi: v[(reps * 95 / 100).min(reps - 1)], n }
}

pub fn accuracy_est(correct: &[bool], seed: u64) -> Est {
    bootstrap(correct.len(), 1000, seed, |ix| ix.iter().filter(|&&i| correct[i]).count() as f64 / ix.len() as f64)
}

pub fn ece_est(conf: &[f64], correct: &[bool], seed: u64) -> Est {
    bootstrap(conf.len(), 1000, seed, |ix| ece_idx(conf, correct, ix))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn temperature_recovers_scale() {
        // labels drawn so that softmax(z / 2) is the true distribution
        let mut rng = Rng::new(7);
        let (mut zs, mut ys) = (vec![], vec![]);
        for _ in 0..4000 {
            let z: Vec<f32> = (0..3).map(|_| (rng.below(1000) as f32 / 100.0) - 5.0).collect();
            let mut p = [0f32; 3];
            softmax_t(&z, 2.0, &mut p);
            let u = rng.below(1_000_000) as f32 / 1e6;
            let y = if u < p[0] {
                0
            } else if u < p[0] + p[1] {
                1
            } else {
                2
            };
            zs.push(z);
            ys.push(y);
        }
        let t = fit_temperature(&zs, &ys);
        assert!((t - 2.0).abs() < 0.2, "{}", t);
    }
    #[test]
    fn ece_perfect_and_bad() {
        assert!(ece(&[1.0, 1.0], &[true, true]) < 1e-9);
        assert!((ece(&[0.9, 0.9], &[false, false]) - 0.9).abs() < 1e-9);
    }
}
