//! `pi` inference engine: shared encoder + compiled schema heads -> Laya-shaped answers.

use crate::encoder::Encoder;
use crate::heads::Head;
use j3v_core::artifact::Artifact;
use j3v_core::metrics::{argmax, softmax_t};
use j3v_core::schema::{question_from_laya, state_text, QType, Schema};
use serde_json::{json, Map, Value};

pub struct Engine {
    pub enc: Encoder,
    pub heads: Vec<Head>,
    pub schema: Schema,
    pub temps: Vec<f32>,
    pub threshold: f64,
    pub model: String,
}

pub struct Scored {
    pub probs: Vec<Vec<f32>>,
    pub n_tokens: usize,
}

/// Jev/Laya `confidence`: 1 - normalized entropy. Kept for wire compatibility; J3v escalates on `p_top`.
pub fn entropy_confidence(p: &[f32]) -> f64 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let h: f64 = p.iter().map(|&x| x.max(1e-12) as f64).map(|x| -x * x.ln()).sum();
    1.0 - h / (k as f64).ln()
}

fn r4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

impl Engine {
    pub fn new(enc: Encoder, a: &Artifact) -> Result<Self, String> {
        if a.header.kind != "pi" && a.header.kind != "pi-heads" {
            return Err(format!("expected a pi artifact, got `{}`", a.header.kind));
        }
        let m = &a.header.meta;
        let schema: Schema = serde_json::from_value(m["schema"].clone()).map_err(|e| format!("artifact schema: {}", e))?;
        let want = m["encoder"].as_str().unwrap_or("");
        if !want.is_empty() && want != enc.id {
            return Err(format!("artifact was compiled against encoder `{}`, but `{}` is loaded", want, enc.id));
        }
        let heads = (0..schema.questions.len()).map(|j| Head::load(a, j)).collect::<Result<Vec<_>, _>>()?;
        for (h, q) in heads.iter().zip(&schema.questions) {
            if h.k != q.options.len() {
                return Err(format!("head for `{}` has {} outputs, schema has {} options", q.id, h.k, q.options.len()));
            }
        }
        let temps = match m["calibration"]["temperature"].as_array() {
            Some(t) => t.iter().map(|v| v.as_f64().unwrap_or(1.0) as f32).collect(),
            None => vec![1.0; heads.len()],
        };
        let threshold = m["threshold"].as_f64().unwrap_or(schema.threshold);
        let h = m["schema_hash"].as_str().unwrap_or("dev");
        let model = format!("j3v-pi/{}@{}", schema.name, &h[..h.len().min(8)]);
        Ok(Engine { enc, heads, schema, temps, threshold, model })
    }

    /// Raw head logits for one state (one encoder pass for all questions).
    pub fn logits(&self, state: &Value) -> (Vec<Vec<f32>>, usize) {
        let ids = self.enc.tok.encode(&state_text(state), self.schema.max_state_tokens);
        let h = self.enc.forward(&ids);
        (self.heads.iter().map(|hd| hd.forward(&h, self.enc.d)).collect(), ids.len())
    }

    pub fn score(&self, state: &Value, temps: &[f32]) -> Scored {
        let (z, n) = self.logits(state);
        let probs = z
            .iter()
            .zip(temps)
            .map(|(z, &t)| {
                let mut p = vec![0f32; z.len()];
                softmax_t(z, t, &mut p);
                p
            })
            .collect();
        Scored { probs, n_tokens: n }
    }

    /// One answer in Laya's shape plus `j3v: {p_top, escalate, tier}`. `local` = answered by this tier (may escalate).
    pub fn render(&self, j: usize, p: &[f32], tier: &str, local: bool) -> Value {
        let q = &self.schema.questions[j];
        let top = argmax(p);
        let p_top = p[top] as f64;
        let ext = json!({"p_top": r4(p_top), "escalate": local && p_top < self.threshold, "tier": tier});
        let probs: Map<String, Value> = q.options.iter().zip(p).map(|(o, &v)| (o.key.clone(), json!(r4(v as f64)))).collect();
        match q.qtype {
            QType::Choice => json!({"type": "choice", "choice": q.options[top].key, "probabilities": probs,
                                    "confidence": r4(entropy_confidence(p)), "j3v": ext}),
            QType::Score => {
                let e: f64 = p.iter().enumerate().map(|(i, &v)| i as f64 * v as f64).sum();
                let legend: Map<String, Value> =
                    q.options.iter().map(|o| (o.key.clone(), json!(o.description.clone().unwrap_or_default()))).collect();
                json!({"type": "score", "score": r4(e), "legend": legend, "probabilities": probs,
                       "confidence": r4(entropy_confidence(p)), "j3v": ext})
            }
            QType::Noul => json!({"type": "noul", "noul": r4(p[1] as f64), "j3v": ext}),
        }
    }

    /// Answer a Laya/Jev request body. `questions` may be omitted (answer the whole schema) or must be a
    /// subset of the compiled schema with identical definitions: J3v cannot answer questions it was not
    /// compiled for, and says so instead of guessing.
    pub fn answer(&self, req: &Value) -> Result<Value, (u16, String)> {
        let state = req.get("state").ok_or((422, "missing `state`".to_string()))?;
        let wanted: Vec<usize> = match req.get("questions") {
            None | Some(Value::Null) => (0..self.schema.questions.len()).collect(),
            Some(Value::Object(qs)) => {
                let mut v = Vec::new();
                for (id, qd) in qs {
                    let q = question_from_laya(id, qd).map_err(|e| (422, e))?;
                    let j = self.schema.questions.iter().position(|c| c.id == *id).ok_or((
                        422,
                        format!(
                            "question {:?} is not compiled into this artifact (schema `{}`); route it to the next tier",
                            id, self.schema.name
                        ),
                    ))?;
                    let c = &self.schema.questions[j];
                    if c.qtype != q.qtype
                        || c.instructions != q.instructions
                        || c.options.iter().map(|o| &o.key).ne(q.options.iter().map(|o| &o.key))
                    {
                        return Err((
                            422,
                            format!("question {:?} differs from the compiled definition; recompile or route it to the next tier", id),
                        ));
                    }
                    v.push(j);
                }
                v
            }
            Some(_) => return Err((422, "`questions` must be an object".into())),
        };
        let s = self.score(state, &self.temps);
        let mut answers = Map::new();
        let mut escalate = false;
        for j in wanted {
            let a = self.render(j, &s.probs[j], "pi", true);
            escalate |= a["j3v"]["escalate"].as_bool().unwrap_or(false);
            answers.insert(self.schema.questions[j].id.clone(), a);
        }
        Ok(json!({"model": self.model, "answers": answers, "escalate": escalate,
                  "usage": {"input_tokens": s.n_tokens, "output_tokens": 0}}))
    }
}
