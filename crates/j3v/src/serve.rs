//! Minimal blocking HTTP/1.1 server exposing the Laya/Jev `POST /v1/systemone` shape.
//! No framework: the pi binary stays small and static.

use crate::cascade::Tier;
use crate::engine::Engine;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Instant;

fn respond(s: &mut TcpStream, code: u16, body: &Value) {
    let reason = match code {
        200 => "OK",
        401 => "Unauthorized",
        404 => "Not Found",
        413 => "Payload Too Large",
        _ => "Unprocessable Entity",
    };
    let b = body.to_string();
    let _ = write!(
        s,
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        code,
        reason,
        b.len(),
        b
    );
}

fn handle(eng: &Engine, up: &Option<Tier>, key: &Option<String>, mut s: TcpStream) {
    let mut r = BufReader::new(s.try_clone().unwrap());
    let mut line = String::new();
    if r.read_line(&mut line).is_err() {
        return;
    }
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string());
    let (mut len, mut auth) = (0usize, None);
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).is_err() || h == "\r\n" || h.is_empty() {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        } else if lower.starts_with("authorization:") {
            auth = Some(h["authorization:".len()..].trim().to_string());
        }
    }
    if len > 1 << 20 {
        return respond(&mut s, 413, &json!({"error": "body larger than 1 MiB"}));
    }
    let mut body = vec![0u8; len];
    if r.read_exact(&mut body).is_err() {
        return;
    }
    if let Some(k) = key {
        if auth.as_deref() != Some(&format!("Bearer {}", k)) {
            return respond(&mut s, 401, &json!({"error": "missing or wrong bearer token"}));
        }
    }
    match (method.as_str(), path.as_str()) {
        ("GET", "/healthz") => respond(&mut s, 200, &json!({"ok": true, "model": eng.model})),
        ("GET", "/v1/schema") => respond(
            &mut s,
            200,
            &json!({"model": eng.model, "thresholds": eng.schema.questions.iter().zip(&eng.thresholds).map(|(q, t)| (q.id.clone(), json!(t))).collect::<serde_json::Map<_, _>>(),
            "questions": eng.schema.to_laya_questions()}),
        ),
        ("POST", "/v1/systemone") => {
            let t = Instant::now();
            let req: Value = match serde_json::from_slice(&body) {
                Ok(v) => v,
                Err(e) => return respond(&mut s, 422, &json!({"error": format!("invalid JSON: {}", e)})),
            };
            match eng.answer(&req) {
                Ok(mut v) => {
                    if let (Some(tier), true) = (up, v["escalate"].as_bool() == Some(true)) {
                        escalate(eng, tier, &req["state"], &mut v);
                    }
                    v["latency_ms"] = json!((t.elapsed().as_secs_f64() * 1e4).round() / 10.0);
                    respond(&mut s, 200, &v)
                }
                Err((c, m)) => respond(&mut s, c, &json!({"error": m})),
            }
        }
        _ => respond(&mut s, 404, &json!({"error": "not found; POST /v1/systemone"})),
    }
}

/// Forward the questions this tier is unsure about to the upstream, and splice its (re-tempered) answers in.
/// If the upstream fails, the local answers stay, still flagged `escalate: true`, and the error is reported.
fn escalate(eng: &Engine, tier: &Tier, state: &Value, v: &mut Value) {
    let js: Vec<usize> = eng
        .schema
        .questions
        .iter()
        .enumerate()
        .filter(|(_, q)| v["answers"][&q.id]["j3v"]["escalate"].as_bool() == Some(true))
        .map(|(j, _)| j)
        .collect();
    match tier.probs(&eng.schema, "", state, &js) {
        Ok(ps) => {
            for (&j, p) in js.iter().zip(&ps) {
                v["answers"][&eng.schema.questions[j].id] = eng.render(j, p, "upstream", false);
            }
            v["escalate"] = json!(false);
            v["escalated"] = json!(js.iter().map(|&j| eng.schema.questions[j].id.clone()).collect::<Vec<_>>());
        }
        Err(e) => v["upstream_error"] = json!(e),
    }
}

pub fn serve(eng: Engine, up: Option<Tier>, addr: &str) -> std::io::Result<()> {
    let key = std::env::var("J3V_API_KEY").ok().filter(|k| !k.is_empty());
    let l = TcpListener::bind(addr)?;
    eprintln!("[j3v] serving {} on http://{} (POST /v1/systemone){}", eng.model, addr, if key.is_some() { ", bearer auth on" } else { "" });
    if let Some(t) = &up {
        eprintln!("[j3v] escalating answers below their p_top threshold to {}", t.name());
    }
    let (eng, up) = (Arc::new(eng), Arc::new(up));
    for s in l.incoming().flatten() {
        let (eng, up, key) = (eng.clone(), up.clone(), key.clone());
        std::thread::spawn(move || handle(&eng, &up, &key, s));
    }
    Ok(())
}
