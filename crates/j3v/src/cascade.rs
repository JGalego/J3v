//! Cascade: mcu -> pi -> upstream (Laya / Jev over `POST /v1/systemone`).
//!
//! Escalation is per question: a tier keeps an answer when its calibrated `p_top` clears the tier's threshold,
//! and forwards only the remaining questions. The last tier answers everything it receives.
//! Escalation is only meaningful when every tier is calibrated, so the report measures ECE for every tier on
//! the same states, then the accuracy of what each tier kept.

use crate::engine::Engine;
use crate::mcu::Owned;
use j3v_core::metrics::{accuracy_est, argmax, ece_est, softmax_t, Est};
use j3v_core::schema::{state_text, QType, Schema};
use serde_json::{json, Map, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

/// Minimal HTTP/1.1 JSON POST (plain http; put TLS in a sidecar or reverse proxy).
pub fn http_post(url: &str, body: &Value, bearer: Option<&str>) -> Result<Value, String> {
    let rest = url.strip_prefix("http://").ok_or("upstream must be an http:// URL")?;
    let (hostport, path) = rest.split_once('/').map(|(h, p)| (h, format!("/{}", p))).unwrap_or((rest, "/".into()));
    let mut s = TcpStream::connect(hostport).map_err(|e| format!("{}: {}", hostport, e))?;
    s.set_read_timeout(Some(Duration::from_secs(120))).ok();
    let b = body.to_string();
    let auth = bearer.map(|k| format!("Authorization: Bearer {}\r\n", k)).unwrap_or_default();
    write!(
        s,
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n{}",
        path,
        hostport,
        b.len(),
        auth,
        b
    )
    .map_err(|e| e.to_string())?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).map_err(|e| e.to_string())?;
    let txt = String::from_utf8_lossy(&raw);
    let (head, body) = txt.split_once("\r\n\r\n").ok_or("malformed HTTP response")?;
    let status = head.split_whitespace().nth(1).unwrap_or("0");
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") { dechunk(body) } else { body.to_string() };
    if status != "200" {
        return Err(format!("upstream returned {}: {}", status, body.chars().take(300).collect::<String>()));
    }
    serde_json::from_str(&body).map_err(|e| format!("upstream JSON: {}", e))
}

fn dechunk(s: &str) -> String {
    let (mut out, mut rest) = (String::new(), s);
    while let Some((len, tail)) = rest.split_once("\r\n") {
        let n = usize::from_str_radix(len.trim(), 16).unwrap_or(0);
        if n == 0 || tail.len() < n {
            break;
        }
        out.push_str(&tail[..n]);
        rest = tail[n..].trim_start_matches("\r\n");
    }
    out
}

pub enum Tier {
    Mcu(Owned),
    Pi(Engine),
    /// Laya/Jev endpoint. `recal[j]` = shipped / refit teacher temperature: the upstream's probabilities are
    /// re-tempered with the calibration fitted at compile time (Laya ships over-confident).
    Upstream {
        url: String,
        recal: Vec<f32>,
        key: Option<String>,
    },
    /// Offline stand-in for the upstream: the compile-time teacher labels (already recalibrated), keyed by state id.
    Cached {
        name: String,
        probs: std::collections::HashMap<String, Vec<Vec<f32>>>,
    },
}

impl Tier {
    pub fn name(&self) -> String {
        match self {
            Tier::Mcu(_) => "mcu".into(),
            Tier::Pi(_) => "pi".into(),
            Tier::Upstream { url, .. } => format!("laya@{}", url),
            Tier::Cached { name, .. } => name.clone(),
        }
    }

    pub fn threshold(&self) -> Option<f64> {
        match self {
            Tier::Mcu(m) => Some(m.threshold as f64),
            Tier::Pi(e) => Some(e.threshold),
            _ => None,
        }
    }

    /// Calibrated probabilities for the questions `qs` (schema indices).
    pub fn probs(&self, schema: &Schema, id: &str, state: &Value, qs: &[usize]) -> Result<Vec<Vec<f32>>, String> {
        match self {
            Tier::Mcu(m) => {
                let z = m.logits(&state_text(state));
                Ok(qs
                    .iter()
                    .map(|&j| {
                        let mut p = vec![0f32; z[j].len()];
                        softmax_t(&z[j], m.temps[j], &mut p);
                        p
                    })
                    .collect())
            }
            Tier::Pi(e) => {
                let s = e.score(state, &e.temps);
                Ok(qs.iter().map(|&j| s.probs[j].clone()).collect())
            }
            Tier::Cached { probs, .. } => {
                let p = probs.get(id).ok_or_else(|| format!("no cached teacher answer for {}", id))?;
                Ok(qs.iter().map(|&j| p[j].clone()).collect())
            }
            Tier::Upstream { url, recal, key } => {
                let all = schema.to_laya_questions();
                let mut sub = Map::new();
                for &j in qs {
                    let q = &schema.questions[j];
                    sub.insert(q.id.clone(), all[&q.id].clone());
                }
                let v = http_post(
                    &format!("{}/v1/systemone", url.trim_end_matches('/')),
                    &json!({"state": state, "questions": sub}),
                    key.as_deref(),
                )?;
                qs.iter()
                    .map(|&j| {
                        let q = &schema.questions[j];
                        let a = &v["answers"][&q.id];
                        let p: Vec<f32> = match q.qtype {
                            QType::Noul => {
                                let t = a["noul"].as_f64().ok_or_else(|| format!("upstream gave no noul for {}", q.id))? as f32;
                                vec![1.0 - t, t]
                            }
                            _ => q
                                .options
                                .iter()
                                .map(|o| a["probabilities"][&o.key].as_f64().map(|x| x as f32))
                                .collect::<Option<Vec<_>>>()
                                .ok_or_else(|| format!("upstream gave no probabilities for {}", q.id))?,
                        };
                        // re-temper: p ∝ exp(z / T_ship) -> exp(z / T_refit) == softmax(log p * T_ship / T_refit)
                        let z: Vec<f32> = p.iter().map(|&x| x.max(1e-6).ln() * recal[j]).collect();
                        let mut out = vec![0f32; z.len()];
                        softmax_t(&z, 1.0, &mut out);
                        Ok(out)
                    })
                    .collect()
            }
        }
    }
}

pub struct Case {
    pub id: String,
    pub state: Value,
    pub gt: Vec<Option<usize>>,
}

/// Run every tier on every case (for per-tier calibration), then route per question through the cascade.
pub fn run(schema: &Schema, tiers: &[Tier], cases: &[Case]) -> Result<Value, String> {
    let nq = schema.questions.len();
    let all: Vec<usize> = (0..nq).collect();
    // full[t][c][j] = calibrated probs of tier t on case c, question j; lat[t] = per-call latency (ms)
    let mut full = Vec::new();
    let mut lat = Vec::new();
    for t in tiers {
        let mut rows = Vec::new();
        let mut l = Vec::new();
        for (i, c) in cases.iter().enumerate() {
            let t0 = Instant::now();
            rows.push(t.probs(schema, &c.id, &c.state, &all)?);
            l.push(t0.elapsed().as_secs_f64() * 1e3);
            if matches!(t, Tier::Upstream { .. }) && (i + 1) % 50 == 0 {
                eprintln!("  {}: {}/{}", t.name(), i + 1, cases.len());
            }
        }
        full.push(rows);
        lat.push(l);
    }
    let mut per_tier = Vec::new();
    for (ti, t) in tiers.iter().enumerate() {
        let mut qs = Vec::new();
        for (j, q) in schema.questions.iter().enumerate() {
            let lab: Vec<(f64, bool)> = cases
                .iter()
                .enumerate()
                .filter_map(|(ci, c)| c.gt[j].map(|g| (full[ti][ci][j][argmax(&full[ti][ci][j])] as f64, argmax(&full[ti][ci][j]) == g)))
                .collect();
            let conf: Vec<f64> = lab.iter().map(|x| x.0).collect();
            let cor: Vec<bool> = lab.iter().map(|x| x.1).collect();
            let seed = j as u64 + 17 * ti as u64 + 1;
            qs.push(json!({"id": q.id, "n_labelled": lab.len(),
                "accuracy": (!lab.is_empty()).then(|| accuracy_est(&cor, seed)),
                "ece": (!lab.is_empty()).then(|| ece_est(&conf, &cor, seed))}));
        }
        let mut l = lat[ti].clone();
        l.sort_by(|a, b| a.partial_cmp(b).unwrap());
        per_tier.push(json!({"tier": t.name(), "threshold": t.threshold(), "questions": qs,
                             "latency_ms_p50": l[l.len() / 2], "latency_ms_p95": l[(l.len() * 95 / 100).min(l.len() - 1)]}));
    }
    // routing
    let nt = tiers.len();
    let mut kept = vec![vec![(0usize, 0usize, 0usize); nq]; nt]; // (answered, labelled, correct)
    let mut correct_total = (0usize, 0usize);
    let mut calls = vec![0usize; nt];
    for (ci, c) in cases.iter().enumerate() {
        let mut pending: Vec<usize> = all.clone();
        for ti in 0..nt {
            if pending.is_empty() {
                break;
            }
            calls[ti] += 1;
            let last = ti == nt - 1;
            let thr = tiers[ti].threshold().unwrap_or(0.0);
            let mut next = Vec::new();
            for &j in &pending {
                let p = &full[ti][ci][j];
                let top = argmax(p);
                if last || p[top] as f64 >= thr {
                    kept[ti][j].0 += 1;
                    if let Some(g) = c.gt[j] {
                        kept[ti][j].1 += 1;
                        kept[ti][j].2 += (top == g) as usize;
                        correct_total.0 += (top == g) as usize;
                        correct_total.1 += 1;
                    }
                } else {
                    next.push(j);
                }
            }
            pending = next;
        }
    }
    let mut routing = Vec::new();
    let mut failures = Vec::new();
    for ti in 0..nt {
        let mut qs = Vec::new();
        for (j, q) in schema.questions.iter().enumerate() {
            let (a, l, k) = kept[ti][j];
            let acc = if l > 0 { Some(k as f64 / l as f64) } else { None };
            if let (Some(acc), Some(thr)) = (acc, tiers[ti].threshold()) {
                if l >= 20 && acc < thr {
                    failures.push(format!(
                        "tier `{}` question `{}`: kept answers are {:.3} accurate, below its threshold {:.2}",
                        tiers[ti].name(),
                        q.id,
                        acc,
                        thr
                    ));
                }
            }
            qs.push(json!({"id": q.id, "answered": a, "share": a as f64 / cases.len() as f64, "accuracy_on_kept": acc}));
        }
        routing.push(json!({"tier": tiers[ti].name(), "requests_reaching_tier": calls[ti], "questions": qs}));
    }
    // cost model: mean latency = sum over tiers of (fraction of requests reaching the tier) x (tier p50)
    let mean_latency: f64 = (0..nt)
        .map(|ti| {
            let mut l = lat[ti].clone();
            l.sort_by(|a, b| a.partial_cmp(b).unwrap());
            calls[ti] as f64 / cases.len() as f64 * l[l.len() / 2]
        })
        .sum();
    Ok(json!({
        "n": cases.len(), "tiers": per_tier, "routing": routing,
        "cascade_accuracy": if correct_total.1 > 0 { Some(correct_total.0 as f64 / correct_total.1 as f64) } else { None },
        "expected_latency_ms": mean_latency, "passed": failures.is_empty(), "failures": failures,
    }))
}

pub fn est_str(v: &Value) -> String {
    if v.is_null() {
        return "-".into();
    }
    let e: Est = Est {
        value: v["value"].as_f64().unwrap_or(f64::NAN),
        lo: v["lo"].as_f64().unwrap_or(f64::NAN),
        hi: v["hi"].as_f64().unwrap_or(f64::NAN),
        n: 0,
    };
    format!("{:.3} [{:.3},{:.3}]", e.value, e.lo, e.hi)
}
