//! Host side of the `mcu` target: size search under a flash/RAM budget, certification through the exact
//! `no_std` j3v-mcu code the firmware runs, and code generation of the model as Rust statics.

use crate::compile::{abs_path, certify, prepare, q8, report_base, run_python, Budget, Opts, Outcome, Prepared};
use j3v_core::artifact::Artifact;
use j3v_core::schema::{state_text, Schema};
use j3v_mcu as m;
use serde_json::{json, Value};
use std::fmt::Write as _;
use std::time::Instant;

pub const MAX_BYTES: usize = 512;

#[derive(Clone, Copy, Debug)]
pub struct Cand {
    pub buckets: usize,
    pub dim: usize,
    pub hidden: usize,
}

pub fn candidates(ks: &[usize]) -> Vec<(Cand, usize, usize)> {
    let mut v = Vec::new();
    for &buckets in &[512usize, 1024, 2048, 4096, 8192, 16384, 32768] {
        for &dim in &[16usize, 32, 64] {
            for &hidden in &[32usize, 64] {
                let c = Cand { buckets, dim, hidden };
                v.push((c, m::flash_bytes(buckets, dim, hidden, ks), m::ram_bytes(dim, hidden, ks.len())));
            }
        }
    }
    v.sort_by_key(|x| x.1);
    v
}

/// Owned int8 model; `with` lends it out as the borrowed `j3v_mcu::Model` the firmware uses.
pub struct Owned {
    pub c: Cand,
    emb: Vec<i8>,
    emb_s: Vec<f32>,
    w1: Vec<i8>,
    s1: Vec<f32>,
    b1: Vec<f32>,
    heads: Vec<(Vec<i8>, Vec<f32>, Vec<f32>, usize)>,
    pub temps: Vec<f32>,
    pub threshold: f32,
}

impl Owned {
    pub fn from_draft(a: &Artifact, c: Cand, nq: usize) -> Result<Self, String> {
        let (emb, emb_s, _) = q8(a, "emb")?;
        let (w1, s1, _) = q8(a, "w1")?;
        let mut heads = Vec::new();
        for j in 0..nq {
            let (w, s, sh) = q8(a, &format!("q{}.w", j))?;
            heads.push((w, s, a.f32(&format!("q{}.b", j))?, sh[0]));
        }
        Ok(Owned { c, emb, emb_s, w1, s1, b1: a.f32("b1")?, heads, temps: vec![1.0; nq], threshold: 0.8 })
    }

    pub fn load(a: &Artifact) -> Result<Self, String> {
        if a.header.kind != "mcu" {
            return Err(format!("expected an mcu artifact, got `{}`", a.header.kind));
        }
        let mt = &a.header.meta;
        let g = |k: &str| mt[k].as_u64().unwrap_or(0) as usize;
        let nq = mt["schema"]["questions"].as_array().map_or(0, |v| v.len());
        let mut heads = Vec::new();
        for j in 0..nq {
            let p = |n: &str| format!("q{}.{}", j, n);
            heads.push((a.i8(&p("w"))?, a.f32(&p("s"))?, a.f32(&p("b"))?, a.shape(&p("w"))?[0]));
        }
        Ok(Owned {
            c: Cand { buckets: g("buckets"), dim: g("dim"), hidden: g("hidden") },
            emb: a.i8("emb.w")?,
            emb_s: a.f32("emb.s")?,
            w1: a.i8("w1.w")?,
            s1: a.f32("w1.s")?,
            b1: a.f32("w1.b")?,
            heads,
            temps: mt["calibration"]["temperature"]
                .as_array()
                .map_or(vec![1.0; nq], |t| t.iter().map(|v| v.as_f64().unwrap() as f32).collect()),
            threshold: mt["threshold"].as_f64().unwrap_or(0.8) as f32,
        })
    }

    pub fn with<R>(&self, f: impl FnOnce(&m::Model) -> R) -> R {
        let heads: Vec<m::Head> =
            self.heads.iter().zip(&self.temps).map(|((w, s, b, k), &t)| m::Head { w, s, b, k: *k, temperature: t }).collect();
        let model = m::Model {
            buckets: self.c.buckets as u32,
            dim: self.c.dim,
            hidden: self.c.hidden,
            max_bytes: MAX_BYTES,
            emb: &self.emb,
            emb_s: &self.emb_s,
            w1: &self.w1,
            s1: &self.s1,
            b1: &self.b1,
            heads: &heads,
            threshold: self.threshold,
        };
        f(&model)
    }

    pub fn logits(&self, text: &str) -> Vec<Vec<f32>> {
        let mut z = vec![[0f32; m::MAX_K]; self.heads.len()];
        self.with(|md| m::logits(md, text.as_bytes(), &mut z));
        z.iter().zip(&self.heads).map(|(r, h)| r[..h.3].to_vec()).collect()
    }

    pub fn flash_bytes(&self) -> usize {
        m::flash_bytes(self.c.buckets, self.c.dim, self.c.hidden, &self.heads.iter().map(|h| h.3).collect::<Vec<_>>())
    }

    pub fn to_artifact(&self, meta: Value) -> Artifact {
        let mut a = Artifact::new("mcu", meta);
        a.add_i8("emb.w", &[self.c.buckets, self.c.dim], &self.emb);
        a.add_f32("emb.s", &[self.c.buckets], &self.emb_s);
        a.add_i8("w1.w", &[self.c.hidden, self.c.dim], &self.w1);
        a.add_f32("w1.s", &[self.c.hidden], &self.s1);
        a.add_f32("w1.b", &[self.c.hidden], &self.b1);
        for (j, (w, s, b, k)) in self.heads.iter().enumerate() {
            a.add_i8(&format!("q{}.w", j), &[*k, self.c.hidden], w);
            a.add_f32(&format!("q{}.s", j), &[*k], s);
            a.add_f32(&format!("q{}.b", j), &[*k], b);
        }
        a
    }

    /// The model as `static` Rust items for the firmware crate (weights land in flash, not RAM).
    pub fn to_rust(&self, schema: &Schema) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "// @generated by `j3v compile --target mcu` for schema `{}` ({}). Do not edit.", schema.name, schema.hash());
        let _ = writeln!(s, "use j3v_mcu::{{Head, Model}};\n");
        let arr_i8 = |s: &mut String, n: &str, v: &[i8]| {
            let _ = write!(s, "static {}: [i8; {}] = [", n, v.len());
            for (i, x) in v.iter().enumerate() {
                if i % 32 == 0 {
                    s.push_str("\n    ");
                }
                let _ = write!(s, "{},", x);
            }
            s.push_str("\n];\n");
        };
        let arr_f32 = |s: &mut String, n: &str, v: &[f32]| {
            let _ = write!(s, "static {}: [f32; {}] = [", n, v.len());
            for (i, x) in v.iter().enumerate() {
                if i % 8 == 0 {
                    s.push_str("\n    ");
                }
                let _ = write!(s, "{:?},", x);
            }
            s.push_str("\n];\n");
        };
        arr_i8(&mut s, "EMB", &self.emb);
        arr_f32(&mut s, "EMB_S", &self.emb_s);
        arr_i8(&mut s, "W1", &self.w1);
        arr_f32(&mut s, "S1", &self.s1);
        arr_f32(&mut s, "B1", &self.b1);
        for (j, (w, sc, b, _)) in self.heads.iter().enumerate() {
            arr_i8(&mut s, &format!("Q{}_W", j), w);
            arr_f32(&mut s, &format!("Q{}_S", j), sc);
            arr_f32(&mut s, &format!("Q{}_B", j), b);
        }
        let _ = writeln!(s, "\n/// Question ids and option keys, in head order.");
        let _ = writeln!(s, "pub static QUESTIONS: [(&str, &[&str]); {}] = [", schema.questions.len());
        for q in &schema.questions {
            let keys: Vec<String> = q.options.iter().map(|o| format!("{:?}", o.key)).collect();
            let _ = writeln!(s, "    ({:?}, &[{}]),", q.id, keys.join(", "));
        }
        let _ = writeln!(s, "];\n\nstatic HEADS: [Head<'static>; {}] = [", self.heads.len());
        for (j, ((_, _, _, k), t)) in self.heads.iter().zip(&self.temps).enumerate() {
            let _ = writeln!(s, "    Head {{ w: &Q{j}_W, s: &Q{j}_S, b: &Q{j}_B, k: {k}, temperature: {t:?} }},", j = j, k = k, t = t);
        }
        let _ = writeln!(s, "];\n");
        let _ = writeln!(
            s,
            "pub static MODEL: Model<'static> = Model {{\n    buckets: {}, dim: {}, hidden: {}, max_bytes: {},\n    emb: &EMB, emb_s: &EMB_S, w1: &W1, s1: &S1, b1: &B1,\n    heads: &HEADS, threshold: {:?},\n}};",
            self.c.buckets, self.c.dim, self.c.hidden, MAX_BYTES, self.threshold
        );
        s
    }
}

fn kb(b: usize) -> String {
    format!("{:.1} KB", b as f64 / 1024.0)
}

pub fn compile(o: &Opts, schema: Schema) -> Result<Outcome, String> {
    let t0 = Instant::now();
    let budget: Budget = o.budget.ok_or("--budget is required for --target mcu (e.g. --budget 512KB or flash=512KB,ram=32KB)")?;
    let ks: Vec<usize> = schema.questions.iter().map(|q| q.options.len()).collect();
    let all = candidates(&ks);
    let fits: Vec<(Cand, usize, usize)> =
        all.iter().copied().filter(|(_, f, r)| *f <= budget.flash && budget.ram.map_or(true, |rb| *r <= rb)).collect();
    // static check, before any teacher call or training
    if fits.is_empty() {
        let (c, f, r) = all[0];
        return Err(format!(
            "error[E0302]: schema `{}` cannot fit target `mcu` within the budget\n  | smallest model (buckets={}, dim={}, hidden={}) needs {} flash and {} RAM\n  | budget: {} flash{}\n  = help: raise --budget, drop questions, or target `pi`",
            schema.name, c.buckets, c.dim, c.hidden, kb(f), kb(r), kb(budget.flash),
            budget.ram.map_or(String::new(), |r| format!(", {} RAM", kb(r)))
        ));
    }
    // train the largest few that fit (spread over sizes), certify each, ship the smallest that passes
    let mut pick: Vec<(Cand, usize, usize)> = Vec::new();
    for x in fits.iter().rev() {
        if pick.len() >= 4 {
            break;
        }
        if pick.iter().all(|p| p.1 as f64 > x.1 as f64 * 1.8) {
            pick.push(*x);
        }
    }
    let p: Prepared = prepare(o, schema)?;
    let schema = &p.schema;
    let fpath = p.bdir.join(format!("mcu-feats-{}.json", &p.states_hash[..8]));
    if !fpath.exists() {
        let hashes: Vec<Vec<u32>> = p
            .rows
            .iter()
            .map(|r| {
                let mut f = [0u32; m::MAX_FEATS];
                let n = m::features(state_text(&r.state).as_bytes(), MAX_BYTES, 0, &mut f);
                f[..n].to_vec()
            })
            .collect();
        std::fs::write(&fpath, json!({"hashes": hashes}).to_string()).map_err(|e| e.to_string())?;
    }
    let cands: Vec<Value> = pick
        .iter()
        .map(|(c, _, _)| {
            json!({"buckets": c.buckets, "dim": c.dim, "hidden": c.hidden,
                   "out": abs_path(&p.bdir).to_string() + &format!("/mcu-{}-{}-{}.draft.j3a", c.buckets, c.dim, c.hidden)})
        })
        .collect();
    eprintln!("[j3v] mcu: {} of {} model sizes fit {}; training {}", fits.len(), all.len(), kb(budget.flash), pick.len());
    run_python(
        o,
        "j3vc.distill_mcu",
        &["--feats", &abs_path(&fpath), "--targets", &abs_path(&p.targets), "--candidates", &Value::Array(cands.clone()).to_string()],
    )?;
    let texts: Vec<String> = p.rows.iter().map(|r| state_text(&r.state)).collect();
    let nq = schema.questions.len();
    let mut tried = Vec::new();
    let mut best: Option<(Owned, Value, Vec<String>)> = None;
    for ((c, flash, ram), cj) in pick.iter().zip(&cands).rev() {
        let draft = Artifact::load(cj["out"].as_str().unwrap())?;
        let mut om = Owned::from_draft(&draft, *c, nq)?;
        let run = |ix: &[usize]| -> Vec<Vec<Vec<f32>>> {
            let mut out = vec![Vec::with_capacity(ix.len()); nq];
            for &i in ix {
                for (j, z) in om.logits(&texts[i]).into_iter().enumerate() {
                    out[j].push(z);
                }
            }
            out
        };
        let (zc, zt) = (run(&p.calib), run(&p.test));
        let (reports, failures, temps) = certify(&p, &zc, &zt);
        om.temps = temps;
        om.threshold = schema.threshold as f32;
        let summary = json!({"buckets": c.buckets, "dim": c.dim, "hidden": c.hidden, "flash_bytes": flash, "ram_bytes": ram,
                             "passed": failures.is_empty(), "failures": failures,
                             "agreement": reports.iter().map(|r| (r.id.clone(), r.agreement.value)).collect::<Vec<_>>(),
                             "ece": reports.iter().map(|r| (r.id.clone(), r.ece.value)).collect::<Vec<_>>()});
        eprintln!(
            "  mcu {:>5}x{:<2} h{:<2} {:>9} flash: {}",
            c.buckets,
            c.dim,
            c.hidden,
            kb(*flash),
            if failures.is_empty() { "PASS".to_string() } else { format!("fail ({} bound(s))", failures.len()) }
        );
        tried.push(summary);
        let passed = failures.is_empty();
        let better = match &best {
            None => true,
            Some((_, _, bf)) => !bf.is_empty() && (passed || failures.len() < bf.len()),
        };
        if better {
            best = Some((om, json!(reports), failures));
        }
        if passed {
            break; // smallest passing size (candidates are visited small -> large)
        }
    }
    let (om, reports, failures) = best.unwrap();
    let mut report = report_base(o, &p, t0);
    report["model"] = json!({"buckets": om.c.buckets, "dim": om.c.dim, "hidden": om.c.hidden, "max_bytes": MAX_BYTES,
                             "flash_bytes": om.flash_bytes(), "ram_bytes": m::ram_bytes(om.c.dim, om.c.hidden, nq)});
    report["budget"] = json!(budget);
    report["search"] = json!(tried);
    report["questions"] = reports;
    report["passed"] = json!(failures.is_empty());
    report["failures"] = json!(failures);
    std::fs::write(p.bdir.join("conformance.mcu.json"), serde_json::to_string_pretty(&report).unwrap()).ok();
    if failures.is_empty() {
        let meta = json!({
            "schema": schema.to_canonical_json(), "schema_hash": schema.hash(), "buckets": om.c.buckets, "dim": om.c.dim,
            "hidden": om.c.hidden, "max_bytes": MAX_BYTES, "threshold": schema.threshold,
            "calibration": {"method": "temperature", "temperature": om.temps}, "conformance": report,
            "compiler": env!("CARGO_PKG_VERSION"),
        });
        om.to_artifact(meta).save(&o.out)?;
        let rs = o.out.trim_end_matches(".j3a").to_string() + ".rs";
        std::fs::write(&rs, om.to_rust(schema)).map_err(|e| e.to_string())?;
        eprintln!("[j3v] firmware source: {}", rs);
    }
    Ok(Outcome { report, failures })
}
