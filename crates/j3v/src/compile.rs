//! `j3v compile`: schema -> teacher labels -> features -> distilled model -> calibration -> conformance gate.
//!
//! Python is invoked only for the two training-time steps (teacher inference, model fitting). Calibration and
//! the conformance gate run here, on the artifact as it will ship, through the same kernels the runtime uses
//! (the int8 encoder for `pi`, the `no_std` j3v-mcu crate for `mcu`).

use crate::encoder::{quantize_rows, Encoder};
use crate::engine::Engine;
use crate::mcu;
use j3v_core::artifact::Artifact;
use j3v_core::metrics::{accuracy_est, argmax, bootstrap, ece_est, fit_temperature, softmax_t, Est};
use j3v_core::schema::{fnv1a64, from_laya_json, parse_dsl, state_text, QType, Schema};
use serde::Serialize;
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

pub struct Opts {
    pub schema: String,
    pub target: String,
    pub encoder: Option<String>,
    pub states: String,
    pub out: String,
    pub build: String,
    pub python: String,
    pub compiler_dir: String,
    pub hidden: usize,
    pub noul_mode: String,
    pub budget: Option<Budget>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Budget {
    pub flash: usize,
    pub ram: Option<usize>,
}

fn parse_size(s: &str) -> Result<usize, String> {
    let t = s.trim().to_ascii_uppercase();
    let (num, mul) = if let Some(n) = t.strip_suffix("KIB").or_else(|| t.strip_suffix("KB")).or_else(|| t.strip_suffix('K')) {
        (n, 1024)
    } else if let Some(n) = t.strip_suffix("MIB").or_else(|| t.strip_suffix("MB")).or_else(|| t.strip_suffix('M')) {
        (n, 1024 * 1024)
    } else if let Some(n) = t.strip_suffix('B') {
        (n, 1)
    } else {
        (t.as_str(), 1)
    };
    num.trim().parse::<f64>().map(|v| (v * mul as f64) as usize).map_err(|_| format!("bad size `{}` (e.g. 512KB, 2MB)", s))
}

/// `512KB` (flash) or `flash=512KB,ram=64KB`.
pub fn parse_budget(s: &str) -> Result<Budget, String> {
    let mut b = Budget { flash: 0, ram: None };
    for part in s.split(',') {
        match part.split_once('=') {
            Some(("flash", v)) => b.flash = parse_size(v)?,
            Some(("ram", v)) => b.ram = Some(parse_size(v)?),
            Some((k, _)) => return Err(format!("unknown budget key `{}` (flash, ram)", k)),
            None => b.flash = parse_size(part)?,
        }
    }
    if b.flash == 0 {
        return Err("budget needs a flash size".into());
    }
    Ok(b)
}

pub struct Row {
    pub id: String,
    pub state: Value,
    pub labels: Value,
}

#[derive(Serialize, Clone)]
pub struct Selective {
    pub threshold: f64,
    pub coverage: f64,
    pub accuracy: f64,
    pub n: usize,
}

#[derive(Serialize, Clone)]
pub struct QReport {
    pub id: String,
    #[serde(rename = "type")]
    pub qtype: String,
    pub k: usize,
    pub calibration_target: String,
    pub teacher_temperature: f32,
    pub teacher_temperature_source: String,
    pub student_temperature: f32,
    pub agreement: Est,
    pub ece: Est,
    pub ece_uncalibrated: f64,
    pub accuracy_gt: Option<Est>,
    pub teacher_accuracy_gt: Option<Est>,
    pub teacher_ece_gt: Option<Est>,
    pub mean_tv_to_teacher: f64,
    pub score_mae_to_teacher: Option<Est>,
    pub selective: Selective,
}

pub fn load_schema(path: &str) -> Result<Schema, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    if path.ends_with(".json") {
        let v: Value = serde_json::from_str(&src).map_err(|e| format!("{}: {}", path, e))?;
        from_laya_json(&v).map_err(|e| format!("{}: {}", path, e))
    } else {
        parse_dsl(&src).map_err(|e| e.render(path, &src))
    }
}

pub fn read_rows(path: &str) -> Result<Vec<Row>, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    src.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            let v: Value = serde_json::from_str(l).map_err(|e| format!("{}:{}: {}", path, i + 1, e))?;
            Ok(Row {
                id: v["id"].as_str().map(str::to_string).unwrap_or_else(|| format!("row-{}", i)),
                state: v.get("state").cloned().ok_or_else(|| format!("{}:{}: missing `state`", path, i + 1))?,
                labels: v.get("labels").cloned().unwrap_or(Value::Null),
            })
        })
        .collect()
}

/// 70 / 10 / 20 train / calib / test, by hash of the state id (stable across rebuilds and corpus growth).
pub fn split_of(id: &str) -> u8 {
    match fnv1a64(id.as_bytes()) % 10 {
        0..=6 => 0,
        7 => 1,
        _ => 2,
    }
}

pub fn softmax_rows(z: &[Vec<f32>], t: f32) -> Vec<Vec<f32>> {
    z.iter()
        .map(|r| {
            let mut p = vec![0f32; r.len()];
            softmax_t(r, t, &mut p);
            p
        })
        .collect()
}

fn run_py(o: &Opts, module: &str, args: &[&str]) -> Result<(), String> {
    let st = Command::new(&o.python)
        .arg("-m")
        .arg(module)
        .args(args)
        .current_dir(&o.compiler_dir)
        .status()
        .map_err(|e| format!("could not run {} ({}); set --python", o.python, e))?;
    if !st.success() {
        return Err(format!("{} failed ({})", module, st));
    }
    Ok(())
}

fn abs(p: &Path) -> String {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf()).to_string_lossy().into_owned()
}

pub struct Outcome {
    pub report: Value,
    pub failures: Vec<String>,
}

/// Everything both backends share: the schema, the states, the teacher's (recalibrated) answers and the split.
pub struct Prepared {
    pub schema: Schema,
    pub rows: Vec<Row>,
    pub states_hash: String,
    pub bdir: PathBuf,
    pub train: Vec<usize>,
    pub calib: Vec<usize>,
    pub test: Vec<usize>,
    pub t_probs: Vec<Vec<Vec<f32>>>,
    pub t_temp: Vec<f32>,
    pub t_src: Vec<String>,
    pub gt: Vec<Vec<Option<usize>>>,
    pub teacher_meta: Value,
    pub targets: PathBuf,
}

pub fn prepare(o: &Opts, schema: Schema) -> Result<Prepared, String> {
    let bdir = PathBuf::from(&o.build).join(&schema.name);
    std::fs::create_dir_all(&bdir).map_err(|e| e.to_string())?;
    let schema_json = bdir.join("schema.json");
    std::fs::write(&schema_json, serde_json::to_string_pretty(&schema.to_canonical_json()).unwrap()).map_err(|e| e.to_string())?;
    let rows = read_rows(&o.states)?;
    if rows.len() < 200 {
        return Err(format!("{} states is too few to distill and certify a schema; provide at least 200", rows.len()));
    }
    let states_hash = format!("{:016x}", fnv1a64(&std::fs::read(&o.states).unwrap()));
    eprintln!("[j3v] schema `{}`: {} questions, {} states", schema.name, schema.questions.len(), rows.len());

    // teacher labels, cached by what the teacher sees
    let tpath = bdir.join(format!("teacher-{}-{}-{}.json", schema.teacher_hash(), &states_hash[..8], o.noul_mode));
    if !tpath.exists() {
        eprintln!(
            "[j3v] labelling with teacher `{}` (cross-encoder, {} passes) -> {}",
            schema.teacher,
            rows.len() * schema.questions.len(),
            tpath.display()
        );
        let cache = abs(&PathBuf::from(&o.build).join("teacher-cache"));
        std::fs::create_dir_all(&cache).ok();
        let tout = format!("{}/{}", abs(&bdir), tpath.file_name().unwrap().to_string_lossy());
        run_py(
            o,
            "j3vc.teacher",
            &[
                "--schema",
                &abs(&schema_json),
                "--states",
                &abs(Path::new(&o.states)),
                "--cache",
                &cache,
                "--out",
                &tout,
                "--noul-mode",
                &o.noul_mode,
            ],
        )?;
    }
    let teacher: Value = serde_json::from_str(&std::fs::read_to_string(&tpath).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let tids: Vec<&str> = teacher["ids"].as_array().ok_or("teacher file lacks ids")?.iter().map(|v| v.as_str().unwrap_or("")).collect();
    if tids.len() != rows.len() || tids.iter().zip(&rows).any(|(a, r)| *a != r.id) {
        return Err(format!("teacher labels {} do not match the states file; delete it to relabel", tpath.display()));
    }
    let split: Vec<u8> = rows.iter().map(|r| split_of(&r.id)).collect();
    let idx = |s: u8| -> Vec<usize> { (0..rows.len()).filter(|&i| split[i] == s).collect() };
    let (train, calib, test) = (idx(0), idx(1), idx(2));

    // teacher temperature: refit on ground truth where the calib split has it, else Laya's shipped value
    let (mut t_probs, mut t_temp, mut t_src, mut gt, mut t_ship) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (j, q) in schema.questions.iter().enumerate() {
        let z: Vec<Vec<f32>> = teacher["logits"][&q.id]
            .as_array()
            .ok_or_else(|| format!("teacher file lacks `{}`", q.id))?
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect())
            .collect();
        let g: Vec<Option<usize>> = rows
            .iter()
            .map(|r| r.labels.get(&q.id).and_then(Value::as_str).and_then(|k| q.options.iter().position(|o| o.key == k)))
            .collect();
        let cal_gt: Vec<usize> = calib.iter().filter(|&&i| g[i].is_some()).copied().collect();
        let shipped = teacher["shipped_temperature"][j].as_f64().unwrap_or(1.0) as f32;
        t_ship.push(shipped);
        let t = if cal_gt.len() >= 100 {
            let zz: Vec<Vec<f32>> = cal_gt.iter().map(|&i| z[i].clone()).collect();
            let yy: Vec<usize> = cal_gt.iter().map(|&i| g[i].unwrap()).collect();
            t_src.push(format!("refit on {} labelled calib states (shipped {:.3})", cal_gt.len(), shipped));
            fit_temperature(&zz, &yy)
        } else {
            t_src.push("laya shipped".to_string());
            shipped
        };
        t_temp.push(t);
        t_probs.push(softmax_rows(&z, t));
        gt.push(g);
    }
    let targets = bdir.join("targets.json");
    let tq: Vec<Value> = schema.questions.iter().map(|q| json!({"id": q.id, "k": q.options.len()})).collect();
    let probs: serde_json::Map<String, Value> = schema.questions.iter().zip(&t_probs).map(|(q, p)| (q.id.clone(), json!(p))).collect();
    std::fs::write(&targets, json!({"questions": tq, "train": train, "calib": calib, "probs": probs}).to_string())
        .map_err(|e| e.to_string())?;
    let teacher_meta = json!({"name": schema.teacher, "revision": teacher["revision"], "noul_mode": teacher["noul_mode"],
                              "temperature": t_temp, "temperature_source": t_src, "shipped_temperature": t_ship});
    Ok(Prepared { schema, rows, states_hash, bdir, train, calib, test, t_probs, t_temp, t_src, gt, teacher_meta, targets })
}

/// Fit per-question temperatures on calib, measure on test, and apply the gate.
/// `zc[j][r]` / `zt[j][r]`: raw logits of question j on the r-th calib / test state.
pub fn certify(p: &Prepared, zc: &[Vec<Vec<f32>>], zt: &[Vec<Vec<f32>>]) -> (Vec<QReport>, Vec<String>, Vec<f32>) {
    let (schema, calib, test) = (&p.schema, &p.calib, &p.test);
    let (mut reports, mut failures, mut temps) = (Vec::new(), Vec::new(), Vec::new());
    for (j, q) in schema.questions.iter().enumerate() {
        let gt = &p.gt[j];
        let tp = &p.t_probs[j];
        let use_gt = calib.iter().all(|&i| gt[i].is_some()) && test.iter().all(|&i| gt[i].is_some());
        let tgt = |i: usize| -> usize {
            if use_gt {
                gt[i].unwrap()
            } else {
                argmax(&tp[i])
            }
        };
        let yc: Vec<usize> = calib.iter().map(|&i| tgt(i)).collect();
        let ts = fit_temperature(&zc[j], &yc);
        temps.push(ts);
        let ps = softmax_rows(&zt[j], ts);
        let ps_raw = softmax_rows(&zt[j], 1.0);
        let seed = fnv1a64(q.id.as_bytes());
        let agree: Vec<bool> = test.iter().zip(&ps).map(|(&i, p)| argmax(p) == argmax(&tp[i])).collect();
        let conf: Vec<f64> = ps.iter().map(|p| p[argmax(p)] as f64).collect();
        let correct: Vec<bool> = test.iter().zip(&ps).map(|(&i, p)| argmax(p) == tgt(i)).collect();
        let conf_raw: Vec<f64> = ps_raw.iter().map(|p| p[argmax(p)] as f64).collect();
        let correct_raw: Vec<bool> = test.iter().zip(&ps_raw).map(|(&i, p)| argmax(p) == tgt(i)).collect();
        let has_gt_t = test.iter().all(|&i| gt[i].is_some());
        let (accuracy_gt, teacher_accuracy_gt, teacher_ece_gt) = if has_gt_t {
            let tc: Vec<bool> = test.iter().map(|&i| argmax(&tp[i]) == gt[i].unwrap()).collect();
            let tconf: Vec<f64> = test.iter().map(|&i| tp[i][argmax(&tp[i])] as f64).collect();
            let sc: Vec<bool> = test.iter().zip(&ps).map(|(&i, p)| argmax(p) == gt[i].unwrap()).collect();
            (Some(accuracy_est(&sc, seed)), Some(accuracy_est(&tc, seed)), Some(ece_est(&tconf, &tc, seed)))
        } else {
            (None, None, None)
        };
        let tv: f64 =
            test.iter().zip(&ps).map(|(&i, p)| 0.5 * p.iter().zip(&tp[i]).map(|(a, b)| (a - b).abs() as f64).sum::<f64>()).sum::<f64>()
                / test.len() as f64;
        let score_mae = (q.qtype == QType::Score).then(|| {
            let ev = |p: &[f32]| p.iter().enumerate().map(|(i, &v)| i as f64 * v as f64).sum::<f64>();
            let err: Vec<f64> = test.iter().zip(&ps).map(|(&i, p)| (ev(p) - ev(&tp[i])).abs()).collect();
            bootstrap(err.len(), 1000, seed, |ix| ix.iter().map(|&i| err[i]).sum::<f64>() / ix.len() as f64)
        });
        let kept: Vec<usize> = (0..test.len()).filter(|&r| conf[r] >= schema.threshold).collect();
        let selective = Selective {
            threshold: schema.threshold,
            coverage: kept.len() as f64 / test.len() as f64,
            accuracy: if kept.is_empty() { f64::NAN } else { kept.iter().filter(|&&r| correct[r]).count() as f64 / kept.len() as f64 },
            n: kept.len(),
        };
        let r = QReport {
            id: q.id.clone(),
            qtype: q.qtype.name().into(),
            k: q.options.len(),
            calibration_target: if use_gt { "ground-truth".into() } else { "teacher".into() },
            teacher_temperature: p.t_temp[j],
            teacher_temperature_source: p.t_src[j].clone(),
            student_temperature: ts,
            agreement: accuracy_est(&agree, seed),
            ece: ece_est(&conf, &correct, seed),
            ece_uncalibrated: j3v_core::metrics::ece(&conf_raw, &correct_raw),
            accuracy_gt,
            teacher_accuracy_gt,
            teacher_ece_gt,
            mean_tv_to_teacher: tv,
            score_mae_to_teacher: score_mae,
            selective,
        };
        // the gate: bounds are checked against one-sided 95% bootstrap limits
        let rq = schema.require.for_question(&q.id);
        if let Some(min) = rq.agreement {
            if r.agreement.lo < min {
                failures.push(format!(
                    "question `{}`: agreement with teacher {:.3} (95% lower bound {:.3}) < required {:.3}",
                    q.id, r.agreement.value, r.agreement.lo, min
                ));
            }
        }
        if let Some(max) = rq.ece {
            if r.ece.hi > max {
                failures.push(format!(
                    "question `{}`: ECE vs {} {:.4} (95% upper bound {:.4}) > allowed {:.4}",
                    q.id, r.calibration_target, r.ece.value, r.ece.hi, max
                ));
            }
        }
        if let (Some(min), Some(a)) = (rq.accuracy, &r.accuracy_gt) {
            if a.lo < min {
                failures.push(format!("question `{}`: accuracy {:.3} (95% lower bound {:.3}) < required {:.3}", q.id, a.value, a.lo, min));
            }
        }
        if let (Some(max), Some(m)) = (rq.mae, &r.score_mae_to_teacher) {
            if m.hi > max {
                failures.push(format!(
                    "question `{}`: expected-score MAE vs teacher {:.3} (95% upper bound {:.3}) > allowed {:.3}",
                    q.id, m.value, m.hi, max
                ));
            }
        }
        if r.selective.n > 0 && r.selective.accuracy < schema.threshold {
            failures.push(format!(
                "question `{}`: cascade contract broken: answers kept at p_top >= {:.2} are only {:.3} accurate",
                q.id, schema.threshold, r.selective.accuracy
            ));
        }
        reports.push(r);
    }
    (reports, failures, temps)
}

fn base_report(o: &Opts, p: &Prepared, t0: Instant) -> Value {
    json!({
        "schema": p.schema.name, "schema_hash": p.schema.hash(), "target": o.target, "teacher": p.teacher_meta,
        "states": {"file": o.states, "hash": p.states_hash, "train": p.train.len(), "calib": p.calib.len(), "test": p.test.len()},
        "require": p.schema.require, "threshold": p.schema.threshold, "compile_seconds": t0.elapsed().as_secs_f64(),
    })
}

pub fn compile(o: &Opts) -> Result<Outcome, String> {
    let schema = load_schema(&o.schema)?;
    match o.target.as_str() {
        "pi" => compile_pi(o, schema),
        "mcu" => mcu::compile(o, schema),
        t => Err(format!("unknown target `{}` (supported: pi, mcu)", t)),
    }
}

fn compile_pi(o: &Opts, schema: Schema) -> Result<Outcome, String> {
    let t0 = Instant::now();
    let enc_path = o.encoder.as_ref().ok_or("--encoder <enc.j3a> is required for --target pi")?;
    let enc_art = Artifact::load(enc_path)?;
    let enc = Encoder::load(&enc_art)?;
    let enc_bytes = std::fs::metadata(enc_path).map(|m| m.len() as usize).unwrap_or(0);
    // static budget check (before any training): the shared encoder alone must fit
    if let Some(b) = o.budget {
        if enc_bytes > b.flash {
            return Err(format!(
                "error[E0302]: target `pi` cannot fit the budget before training\n  | shared encoder `{}` is {:.1} MB; storage budget is {:.1} MB\n  = help: import a smaller encoder or raise --budget",
                enc.id, enc_bytes as f64 / 1e6, b.flash as f64 / 1e6
            ));
        }
    }
    let p = prepare(o, schema)?;
    let schema = &p.schema;
    let fprefix = p.bdir.join(format!("feats-{:016x}-{}-{}", fnv1a64(enc.id.as_bytes()), &p.states_hash[..8], schema.max_state_tokens));
    if !PathBuf::from(format!("{}.json", fprefix.display())).exists() {
        eprintln!("[j3v] encoding {} states with `{}`...", p.rows.len(), enc.id);
        let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{}.f32", fprefix.display())).map_err(|e| e.to_string())?);
        let mut lens = Vec::new();
        for r in &p.rows {
            let h = enc.forward(&enc.tok.encode(&state_text(&r.state), schema.max_state_tokens));
            lens.push(h.len() / enc.d);
            for x in h {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        }
        f.flush().unwrap();
        std::fs::write(format!("{}.json", fprefix.display()), json!({"d": enc.d, "lens": lens}).to_string()).unwrap();
    }
    let draft = p.bdir.join("heads.draft.j3a");
    eprintln!("[j3v] distilling heads ({} train / {} calib / {} test)...", p.train.len(), p.calib.len(), p.test.len());
    run_py(
        o,
        "j3vc.distill",
        &["--feats", &abs(&fprefix), "--targets", &abs(&p.targets), "--out", &abs(&draft), "--hidden", &o.hidden.to_string()],
    )?;

    // run the draft artifact end-to-end (text -> tokenizer -> int8 encoder -> heads) on calib + test
    let mut art = Artifact::load(&draft.to_string_lossy())?;
    art.header.meta = json!({"schema": schema.to_canonical_json(), "encoder": enc.id});
    let eng = Engine::new(enc, &art)?;
    let nq = schema.questions.len();
    let run = |ix: &[usize]| -> Vec<Vec<Vec<f32>>> {
        let mut out = vec![Vec::with_capacity(ix.len()); nq];
        for &i in ix {
            for (j, zz) in eng.logits(&p.rows[i].state).0.into_iter().enumerate() {
                out[j].push(zz);
            }
        }
        out
    };
    let (zc, zt) = (run(&p.calib), run(&p.test));
    let (reports, mut failures, temps) = certify(&p, &zc, &zt);
    let heads_bytes = art.to_bytes().len();
    let total = heads_bytes + enc_bytes;
    if let Some(b) = o.budget {
        if total > b.flash {
            failures.push(format!("storage: encoder {} + heads {} = {} bytes > budget {} bytes", enc_bytes, heads_bytes, total, b.flash));
        }
    }
    let mut report = base_report(o, &p, t0);
    report["encoder"] = json!(eng.enc.id);
    report["size"] = json!({"heads_bytes": heads_bytes, "encoder_bytes": enc_bytes, "total_bytes": total, "budget": o.budget});
    report["questions"] = json!(reports);
    report["passed"] = json!(failures.is_empty());
    report["failures"] = json!(failures);
    std::fs::write(p.bdir.join("conformance.pi.json"), serde_json::to_string_pretty(&report).unwrap()).ok();
    if failures.is_empty() {
        art.header.kind = "pi".into();
        art.header.meta = json!({
            "schema": schema.to_canonical_json(), "schema_hash": schema.hash(), "encoder": eng.enc.id,
            "threshold": schema.threshold, "calibration": {"method": "temperature", "temperature": temps},
            "teacher": p.teacher_meta, "conformance": report, "compiler": env!("CARGO_PKG_VERSION"),
        });
        art.save(&o.out)?;
    }
    Ok(Outcome { report, failures })
}

/// Per-row int8 quantization of an f32 tensor from a draft artifact.
pub fn q8(a: &Artifact, name: &str) -> Result<(Vec<i8>, Vec<f32>, Vec<usize>), String> {
    let sh = a.shape(name)?.to_vec();
    let (q, s) = quantize_rows(&a.f32(name)?, sh[0], sh[1]);
    Ok((q, s, sh))
}

pub fn run_python(o: &Opts, module: &str, args: &[&str]) -> Result<(), String> {
    run_py(o, module, args)
}

pub fn abs_path(p: &Path) -> String {
    abs(p)
}

pub fn report_base(o: &Opts, p: &Prepared, t0: Instant) -> Value {
    base_report(o, p, t0)
}
