//! `j3v`: the J3v compiler driver and `pi` runtime in one static binary.

mod cascade;
mod compile;
mod encoder;
mod engine;
mod heads;
mod mcu;
mod serve;
mod tokenizer;

use j3v_core::artifact::Artifact;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

const USAGE: &str = "j3v - compile typed decision schemas into calibrated edge artifacts

USAGE:
  j3v check <schema.j3v|.json>                       parse + validate a schema, print canonical JSON
  j3v compile <schema> --target pi --encoder <enc.j3a> --states <states.jsonl> -o <out.j3a> [--budget 64MB]
  j3v compile <schema> --target mcu --budget 512KB --states <states.jsonl> -o <out.j3a>   (also writes <out>.rs)
              [--build build] [--python python3] [--compiler-dir compiler] [--hidden 128] [--noul-mode native|choice]
  j3v encoder import <hf_dir> -o <enc.j3a> [source-id]
  j3v predict --encoder <enc.j3a> <artifact.j3a> '<request json>'
  j3v serve   --encoder <enc.j3a> <artifact.j3a> [--addr 0.0.0.0:8000] [--threads 1]
  j3v bench   --encoder <enc.j3a> <artifact.j3a> <states.jsonl> [--n 500]
  j3v mcu-codegen <mcu.j3a> -o <model.rs>            firmware source for an mcu artifact
  j3v mcu-predict <mcu.j3a> <inputs.txt>            host run of the mcu model (same output as the firmware)
  j3v cascade --mcu <m.j3a> --pi <p.j3a> --encoder <enc.j3a> --states <s.jsonl> [--upstream http://host:8000 | --teacher <teacher.json>]
              [--split test|all] [--n 400]           run + certify the mcu -> pi -> laya cascade
  j3v inspect <artifact.j3a>
";

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("{}", msg);
    std::process::exit(1)
}

/// Positional args plus `--flag value` pairs.
fn parse(args: &[String]) -> (Vec<String>, HashMap<String, String>) {
    let (mut pos, mut fl) = (Vec::new(), HashMap::new());
    let mut i = 0;
    while i < args.len() {
        if let Some(k) = args[i].strip_prefix("--").or_else(|| if args[i] == "-o" { Some("out") } else { None }) {
            fl.insert(k.to_string(), args.get(i + 1).cloned().unwrap_or_default());
            i += 2;
        } else {
            pos.push(args[i].clone());
            i += 1;
        }
    }
    (pos, fl)
}

fn load_engine(fl: &HashMap<String, String>, art: &str) -> engine::Engine {
    let enc_path = fl.get("encoder").unwrap_or_else(|| die("error: --encoder <enc.j3a> is required"));
    let enc = encoder::Encoder::load(&Artifact::load(enc_path).unwrap_or_else(|e| die(e))).unwrap_or_else(|e| die(e));
    engine::Engine::new(enc, &Artifact::load(art).unwrap_or_else(|e| die(e))).unwrap_or_else(|e| die(e))
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (pos, fl) = parse(&args);
    let get = |k: &str, d: &str| fl.get(k).cloned().unwrap_or_else(|| d.to_string());
    if let Some(t) = fl.get("threads") {
        encoder::set_threads(t.parse().unwrap_or_else(|_| die("error: --threads must be an integer")));
    }
    match pos.first().map(String::as_str) {
        Some("check") => {
            let s = compile::load_schema(pos.get(1).unwrap_or_else(|| die(USAGE))).unwrap_or_else(|e| die(e));
            println!("{}", serde_json::to_string_pretty(&s.to_canonical_json()).unwrap());
        }
        Some("compile") => {
            let o = compile::Opts {
                schema: pos.get(1).cloned().unwrap_or_else(|| die(USAGE)),
                target: get("target", "pi"),
                encoder: fl.get("encoder").cloned(),
                states: fl.get("states").cloned().unwrap_or_else(|| die("error: --states <states.jsonl> is required")),
                out: fl.get("out").cloned().unwrap_or_else(|| die("error: -o <out.j3a> is required")),
                build: get("build", "build"),
                python: get("python", "python3"),
                compiler_dir: get("compiler-dir", "compiler"),
                hidden: get("hidden", "128").parse().unwrap_or_else(|_| die("error: --hidden must be an integer")),
                noul_mode: get("noul-mode", "native"),
                budget: fl.get("budget").map(|b| compile::parse_budget(b).unwrap_or_else(|e| die(format!("error: {}", e)))),
            };
            let out = compile::compile(&o).unwrap_or_else(|e| die(format!("error: {}", e)));
            print_report(&out.report);
            if let Some(mm) = out.report.get("model") {
                eprintln!("mcu model: {}", mm);
            }
            if !out.failures.is_empty() {
                let mut m = format!(
                    "\nerror[E0301]: `{}` cannot meet its conformance bounds on target `{}`\n --> {}\n",
                    out.report["schema"].as_str().unwrap_or(""),
                    o.target,
                    o.schema
                );
                for f in &out.failures {
                    m += &format!("  | {}\n", f);
                }
                m += "  = help: add states, raise --hidden, pick a larger encoder, or relax the `require` bounds in the schema\n";
                m += "  = note: bounds are checked against one-sided 95% bootstrap limits on the held-out split\n";
                die(m);
            }
            let sz = std::fs::metadata(&o.out).map(|m| m.len()).unwrap_or(0);
            eprintln!("\n[j3v] wrote {} ({:.1} KB); conformance passed", o.out, sz as f64 / 1024.0);
        }
        Some("encoder") if pos.get(1).map(String::as_str) == Some("import") => {
            let dir = pos.get(2).unwrap_or_else(|| die(USAGE));
            let out = fl.get("out").unwrap_or_else(|| die(USAGE));
            let src = pos.get(3).cloned().unwrap_or_else(|| dir.clone());
            let a = encoder::import_hf(dir, &src).unwrap_or_else(|e| die(e));
            a.save(out).unwrap_or_else(|e| die(e));
            println!("wrote {} ({} params, {:.1} MB)", out, encoder::Encoder::params(&a), a.to_bytes().len() as f64 / 1e6);
        }
        Some("predict") => {
            let eng = load_engine(&fl, pos.get(1).unwrap_or_else(|| die(USAGE)));
            let req: Value = serde_json::from_str(pos.get(2).unwrap_or_else(|| die(USAGE))).unwrap_or_else(|e| die(e));
            let req = if req.get("state").is_some() { req } else { json!({"state": req}) };
            match eng.answer(&req) {
                Ok(v) => println!("{}", serde_json::to_string_pretty(&v).unwrap()),
                Err((_, m)) => die(format!("error: {}", m)),
            }
        }
        Some("serve") => {
            let eng = load_engine(&fl, pos.get(1).unwrap_or_else(|| die(USAGE)));
            serve::serve(eng, &get("addr", "0.0.0.0:8000")).unwrap_or_else(|e| die(e));
        }
        Some("bench") => {
            let t = Instant::now();
            let eng = load_engine(&fl, pos.get(1).unwrap_or_else(|| die(USAGE)));
            let load_ms = t.elapsed().as_secs_f64() * 1e3;
            let rows = compile::read_rows(pos.get(2).unwrap_or_else(|| die(USAGE))).unwrap_or_else(|e| die(e));
            let n: usize = get("n", "500").parse().unwrap();
            for r in rows.iter().take(20) {
                let _ = eng.answer(&json!({"state": r.state}));
            }
            let (mut lat, mut toks) = (Vec::new(), 0usize);
            for r in rows.iter().cycle().take(n) {
                let t = Instant::now();
                let v = eng.answer(&json!({"state": r.state})).unwrap();
                lat.push(t.elapsed().as_secs_f64() * 1e3);
                toks += v["usage"]["input_tokens"].as_u64().unwrap() as usize;
            }
            let mean = lat.iter().sum::<f64>() / lat.len() as f64;
            println!(
                "{}",
                json!({"kernel": encoder::kernel_name(), "load_ms": load_ms, "n": n, "questions": eng.schema.questions.len(),
                       "mean_tokens": toks as f64 / n as f64, "p50_ms": pct(&mut lat.clone(), 0.5), "p95_ms": pct(&mut lat.clone(), 0.95),
                       "p99_ms": pct(&mut lat, 0.99), "mean_ms": mean})
            );
        }
        Some("cascade") => cascade_cmd(&fl),
        Some("mcu-codegen") => {
            let a = Artifact::load(pos.get(1).unwrap_or_else(|| die(USAGE))).unwrap_or_else(|e| die(e));
            let om = mcu::Owned::load(&a).unwrap_or_else(|e| die(e));
            let schema: j3v_core::schema::Schema = serde_json::from_value(a.header.meta["schema"].clone()).unwrap_or_else(|e| die(e));
            let out = fl.get("out").unwrap_or_else(|| die("error: -o <model.rs> is required"));
            std::fs::write(out, om.to_rust(&schema)).unwrap_or_else(|e| die(e));
        }
        Some("mcu-predict") => {
            // same computation and output format as firmware/cortex-m7, for bit-exact comparison
            let a = Artifact::load(pos.get(1).unwrap_or_else(|| die(USAGE))).unwrap_or_else(|e| die(e));
            let om = mcu::Owned::load(&a).unwrap_or_else(|e| die(e));
            let qs = a.header.meta["schema"]["questions"].as_array().cloned().unwrap_or_default();
            let inputs = std::fs::read_to_string(pos.get(2).unwrap_or_else(|| die(USAGE))).unwrap_or_else(|e| die(e));
            om.with(|m| {
                println!("j3v mcu: {} questions, {} buckets x {} dim, hidden {}", qs.len(), m.buckets, m.dim, m.hidden);
                for line in inputs.lines().filter(|l| !l.is_empty()) {
                    let mut z = vec![[0f32; j3v_mcu::MAX_K]; m.heads.len()];
                    j3v_mcu::logits(m, line.as_bytes(), &mut z);
                    println!("> {}", line);
                    for (j, hd) in m.heads.iter().enumerate() {
                        let (top, p) = j3v_mcu::calibrate(&mut z[j][..hd.k], hd.temperature);
                        let key = qs[j]["options"][top]["key"].as_str().unwrap_or("?");
                        println!("  {} = {} p_top={:.4} escalate={}", qs[j]["id"].as_str().unwrap_or("?"), key, p, p < m.threshold);
                    }
                }
            });
        }
        Some("inspect") => {
            let a = Artifact::load(pos.get(1).unwrap_or_else(|| die(USAGE))).unwrap_or_else(|e| die(e));
            let mut m = a.header.meta.clone();
            if m.get("vocab").is_some() {
                m["vocab"] = json!(format!("<{} tokens>", m["vocab"].as_array().map_or(0, |v| v.len())));
            }
            println!("kind: {}\npayload: {} bytes in {} tensors", a.header.kind, a.payload_bytes(), a.header.tensors.len());
            println!("{}", serde_json::to_string_pretty(&m).unwrap());
        }
        _ => die(USAGE),
    }
}

fn print_report(r: &Value) {
    eprintln!(
        "\nconformance: {} -> {} (test n={})",
        r["schema"].as_str().unwrap_or(""),
        r["target"].as_str().unwrap_or(""),
        r["states"]["test"]
    );
    eprintln!(
        "{:<18} {:>6} {:>17} {:>17} {:>17} {:>9} {:>14}",
        "question", "target", "agreement [lo]", "ECE [hi]", "acc(gt) [lo]", "teach.acc", "kept@thr acc"
    );
    for q in r["questions"].as_array().unwrap() {
        let est = |v: &Value, lo: &str| {
            if v.is_null() {
                "-".to_string()
            } else {
                format!("{:.3} [{:.3}]", v["value"].as_f64().unwrap(), v[lo].as_f64().unwrap())
            }
        };
        let s = &q["selective"];
        eprintln!(
            "{:<18} {:>6} {:>17} {:>17} {:>17} {:>9} {:>14}",
            q["id"].as_str().unwrap(),
            if q["calibration_target"] == "ground-truth" { "gt" } else { "teach" },
            est(&q["agreement"], "lo"),
            est(&q["ece"], "hi"),
            est(&q["accuracy_gt"], "lo"),
            q["teacher_accuracy_gt"]["value"].as_f64().map_or("-".into(), |v| format!("{:.3}", v)),
            format!("{:.0}% @ {:.3}", s["coverage"].as_f64().unwrap_or(0.0) * 100.0, s["accuracy"].as_f64().unwrap_or(f64::NAN)),
        );
    }
}

fn cascade_cmd(fl: &HashMap<String, String>) {
    use cascade::{Case, Tier};
    let mut tiers = Vec::new();
    let mut schema: Option<j3v_core::schema::Schema> = None;
    let mut hashes = Vec::new();
    let mut teacher_meta = Value::Null;
    if let Some(mp) = fl.get("mcu") {
        let a = Artifact::load(mp).unwrap_or_else(|e| die(e));
        hashes.push(a.header.meta["schema_hash"].clone());
        schema = serde_json::from_value(a.header.meta["schema"].clone()).ok();
        tiers.push(Tier::Mcu(mcu::Owned::load(&a).unwrap_or_else(|e| die(e))));
    }
    if let Some(pp) = fl.get("pi") {
        let a = Artifact::load(pp).unwrap_or_else(|e| die(e));
        hashes.push(a.header.meta["schema_hash"].clone());
        teacher_meta = a.header.meta["teacher"].clone();
        let eng = load_engine(fl, pp);
        schema = Some(eng.schema.clone());
        tiers.push(Tier::Pi(eng));
    }
    let schema = schema.unwrap_or_else(|| die("error: give at least one of --mcu / --pi"));
    if hashes.windows(2).any(|w| w[0] != w[1]) {
        die("error: --mcu and --pi artifacts were compiled from different schemas");
    }
    let nq = schema.questions.len();
    let refit: Vec<f32> = (0..nq).map(|j| teacher_meta["temperature"][j].as_f64().unwrap_or(1.0) as f32).collect();
    if let Some(u) = fl.get("upstream") {
        let ship: Vec<f32> = (0..nq).map(|j| teacher_meta["shipped_temperature"][j].as_f64().unwrap_or(1.0) as f32).collect();
        let recal = ship.iter().zip(&refit).map(|(s, r)| s / r).collect();
        tiers.push(Tier::Upstream { url: u.clone(), recal, key: std::env::var("J3V_UPSTREAM_KEY").ok() });
    } else if let Some(tp) = fl.get("teacher") {
        let t: Value = serde_json::from_str(&std::fs::read_to_string(tp).unwrap_or_else(|e| die(e))).unwrap_or_else(|e| die(e));
        let mut probs = HashMap::new();
        for (i, id) in t["ids"].as_array().unwrap().iter().enumerate() {
            let p: Vec<Vec<f32>> = schema
                .questions
                .iter()
                .enumerate()
                .map(|(j, q)| {
                    let z: Vec<f32> = t["logits"][&q.id][i].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
                    let mut p = vec![0f32; z.len()];
                    j3v_core::metrics::softmax_t(&z, refit[j], &mut p);
                    p
                })
                .collect();
            probs.insert(id.as_str().unwrap().to_string(), p);
        }
        tiers.push(Tier::Cached { name: "laya(cached)".into(), probs });
    }
    let rows = compile::read_rows(fl.get("states").unwrap_or_else(|| die("error: --states is required"))).unwrap_or_else(|e| die(e));
    let split = fl.get("split").map(String::as_str).unwrap_or("test");
    let n: usize = fl.get("n").map(|v| v.parse().unwrap()).unwrap_or(usize::MAX);
    let to_case = |r: &compile::Row| Case {
        id: r.id.clone(),
        state: r.state.clone(),
        gt: schema
            .questions
            .iter()
            .map(|q| r.labels.get(&q.id).and_then(Value::as_str).and_then(|k| q.options.iter().position(|o| o.key == k)))
            .collect(),
    };
    let calib: Vec<Case> = rows.iter().filter(|r| compile::split_of(&r.id) == 1).take(n).map(to_case).collect();
    let cases: Vec<Case> = rows.iter().filter(|r| split == "all" || compile::split_of(&r.id) == 2).take(n).map(to_case).collect();
    eprintln!("[j3v] cascade {} on {} states", tiers.iter().map(|t| t.name()).collect::<Vec<_>>().join(" -> "), cases.len());
    let rep = cascade::run(&schema, &tiers, &calib, &cases).unwrap_or_else(|e| die(format!("error: {}", e)));
    if let Some(out) = fl.get("out") {
        std::fs::write(out, serde_json::to_string_pretty(&rep).unwrap()).unwrap();
    }
    eprintln!("\nper-tier calibration on all {} states (accuracy / ECE vs ground truth, [95% bounds]):", cases.len());
    for t in rep["tiers"].as_array().unwrap() {
        eprintln!("  {} (threshold {}, p50 {:.2} ms)", t["tier"].as_str().unwrap(), t["threshold"], t["latency_ms_p50"].as_f64().unwrap());
        for q in t["questions"].as_array().unwrap() {
            eprintln!(
                "    {:<18} acc {:<24} ece {}",
                q["id"].as_str().unwrap(),
                cascade::est_str(&q["accuracy"]),
                cascade::est_str(&q["ece"])
            );
        }
    }
    for (key, title) in
        [("routing_uncorrected", "routing with marginal calibration only"), ("routing", "routing with conditional calibration")]
    {
        eprintln!("\n{}:", title);
        for t in rep[key].as_array().unwrap() {
            eprintln!("  {} (reached by {} requests)", t["tier"].as_str().unwrap(), t["requests_reaching_tier"]);
            for q in t["questions"].as_array().unwrap() {
                eprintln!(
                    "    {:<18} kept {:>5.1}%  accuracy on kept {} (n={})",
                    q["id"].as_str().unwrap(),
                    q["share"].as_f64().unwrap() * 100.0,
                    q["accuracy_on_kept"].as_f64().map_or("-".into(), |v| format!("{:.3}", v)),
                    q["labelled"]
                );
            }
        }
    }
    eprintln!("\nconditional temperatures: {}", rep["conditional_calibration"]);
    eprintln!("cascade accuracy with marginal calibration only: {:.3}", rep["cascade_accuracy_uncorrected"].as_f64().unwrap_or(f64::NAN));
    eprintln!(
        "\ncascade accuracy {:.3}, expected latency {:.2} ms/request",
        rep["cascade_accuracy"].as_f64().unwrap_or(f64::NAN),
        rep["expected_latency_ms"].as_f64().unwrap()
    );
    if !rep["failures"].as_array().unwrap().is_empty() {
        let f: Vec<String> = rep["failures"].as_array().unwrap().iter().map(|v| format!("  | {}", v.as_str().unwrap())).collect();
        die(format!("error[E0401]: cascade contract violated\n{}", f.join("\n")));
    }
}
