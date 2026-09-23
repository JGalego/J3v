//! The J3v schema: typed questions with answer spaces fixed at compile time.
//!
//! Two front ends produce the same [`Schema`]:
//! * the `.j3v` DSL ([`parse_dsl`]), and
//! * a Laya/Jev request body `{"questions": {...}}` ([`from_laya_json`]).
//!
//! The canonical JSON form ([`Schema::to_canonical_json`]) is what the compile-time Python stage reads
//! and what gets embedded in every artifact.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QType {
    Choice,
    Score,
    Noul,
}

impl QType {
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Opt {
    /// choice: user key; score: "0".."K-1"; noul: "false" / "true".
    pub key: String,
    pub description: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Question {
    pub id: String,
    #[serde(rename = "type")]
    pub qtype: QType,
    pub instructions: String,
    pub options: Vec<Opt>,
}

/// Conformance bounds. They are checked against one-sided 95% bootstrap bounds, not point estimates,
/// so a schema does not flip between passing and failing across rebuilds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Requirements {
    /// Lower bound on argmax agreement with the (recalibrated) teacher.
    pub agreement: f64,
    /// Upper bound on 15-bin top-1 ECE against the calibration target.
    pub ece: f64,
    /// Optional lower bound on accuracy against ground-truth labels (only for questions that have them).
    pub accuracy: Option<f64>,
}

impl Default for Requirements {
    fn default() -> Self {
        Requirements { agreement: 0.85, ece: 0.05, accuracy: None }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Schema {
    pub name: String,
    pub version: u32,
    pub teacher: String,
    pub max_state_tokens: usize,
    /// Calibrated top-1 probability below which the artifact escalates to the next tier.
    pub threshold: f64,
    pub require: Requirements,
    pub questions: Vec<Question>,
}

pub const TEACHERS: &[&str] = &["laya", "laya-multilingual"];
pub const MAX_OPTIONS: usize = 20;

impl Schema {
    /// Canonical JSON: stable key order, used for hashing and for the Python stage.
    pub fn to_canonical_json(&self) -> Value {
        json!({
            "name": self.name,
            "version": self.version,
            "teacher": self.teacher,
            "max_state_tokens": self.max_state_tokens,
            "threshold": self.threshold,
            "require": {"agreement": self.require.agreement, "ece": self.require.ece, "accuracy": self.require.accuracy},
            "questions": self.questions.iter().map(|q| json!({
                "id": q.id, "type": q.qtype.name(), "instructions": q.instructions,
                "options": q.options.iter().map(|o| json!({"key": o.key, "description": o.description})).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        })
    }

    /// Hash of what the teacher sees (questions + teacher). Thresholds and bounds are excluded so tightening
    /// a bound does not invalidate cached teacher labels.
    pub fn teacher_hash(&self) -> String {
        let v = json!({"teacher": self.teacher, "questions": self.to_canonical_json()["questions"]});
        format!("{:016x}", fnv1a64(v.to_string().as_bytes()))
    }

    pub fn hash(&self) -> String {
        format!("{:016x}", fnv1a64(self.to_canonical_json().to_string().as_bytes()))
    }

    /// Laya / Jev `questions` object for this schema.
    pub fn to_laya_questions(&self) -> Value {
        let mut m = serde_json::Map::new();
        for q in &self.questions {
            let v = match q.qtype {
                QType::Choice => {
                    let mut c = serde_json::Map::new();
                    for o in &q.options {
                        c.insert(o.key.clone(), o.description.clone().map(Value::String).unwrap_or(Value::Null));
                    }
                    json!({"type": "choice", "instructions": q.instructions, "criteria": c})
                }
                QType::Score => json!({"type": "score", "instructions": q.instructions,
                    "criteria": q.options.iter().map(|o| o.description.clone().unwrap_or_default()).collect::<Vec<_>>()}),
                QType::Noul => {
                    let (f, t) = (&q.options[0].description, &q.options[1].description);
                    if f.is_none() && t.is_none() {
                        json!({"type": "noul", "instructions": q.instructions})
                    } else {
                        json!({"type": "noul", "instructions": q.instructions, "criteria": {"false": f, "true": t}})
                    }
                }
            };
            m.insert(q.id.clone(), v);
        }
        Value::Object(m)
    }

    pub fn question(&self, id: &str) -> Option<&Question> {
        self.questions.iter().find(|q| q.id == id)
    }
}

pub fn fnv1a64(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ----------------------------------------------------------------------------- errors

#[derive(Debug, Clone, PartialEq)]
pub struct SchemaError {
    pub line: usize,
    pub col: usize,
    pub msg: String,
    pub help: Option<String>,
}

impl SchemaError {
    fn new(line: usize, col: usize, msg: impl Into<String>) -> Self {
        SchemaError { line, col, msg: msg.into(), help: None }
    }
    fn help(mut self, h: impl Into<String>) -> Self {
        self.help = Some(h.into());
        self
    }

    /// rustc-style rendering with the offending source line.
    pub fn render(&self, path: &str, src: &str) -> String {
        let mut s = format!("error: {}\n --> {}:{}:{}\n", self.msg, path, self.line, self.col);
        if let Some(l) = src.lines().nth(self.line.saturating_sub(1)) {
            let n = self.line.to_string();
            let pad = " ".repeat(n.len());
            s += &format!("{} |\n{} | {}\n{} | {}^\n", pad, n, l, pad, " ".repeat(self.col.saturating_sub(1)));
        }
        if let Some(h) = &self.help {
            s += &format!("  = help: {}\n", h);
        }
        s
    }
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.col, self.msg)
    }
}

// ----------------------------------------------------------------------------- DSL lexer

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Word(String),
    Str(String),
    Num(f64),
    Op(&'static str),
}

#[derive(Debug, Clone)]
struct Spanned {
    tok: Tok,
    col: usize,
}

fn lex_line(line: &str, ln: usize) -> Result<Vec<Spanned>, SchemaError> {
    let cs: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        let col = i + 1;
        if c == '#' {
            break;
        } else if c.is_whitespace() {
            i += 1;
        } else if c == '"' {
            let mut s = String::new();
            i += 1;
            loop {
                match cs.get(i) {
                    None => return Err(SchemaError::new(ln, col, "unterminated string").help("close the string with `\"`")),
                    Some('"') => {
                        i += 1;
                        break;
                    }
                    Some('\\') => {
                        match cs.get(i + 1) {
                            Some('n') => s.push('\n'),
                            Some(&e) => s.push(e),
                            None => return Err(SchemaError::new(ln, i + 1, "dangling escape at end of line")),
                        }
                        i += 2;
                    }
                    Some(&e) => {
                        s.push(e);
                        i += 1;
                    }
                }
            }
            out.push(Spanned { tok: Tok::Str(s), col });
        } else if c == '>' || c == '<' {
            if cs.get(i + 1) == Some(&'=') {
                out.push(Spanned { tok: Tok::Op(if c == '>' { ">=" } else { "<=" }), col });
                i += 2;
            } else {
                return Err(SchemaError::new(ln, col, format!("unexpected `{}`", c)).help("bounds are written `>=` or `<=`"));
            }
        } else if c.is_ascii_digit() || (c == '.' && cs.get(i + 1).map_or(false, |d| d.is_ascii_digit())) {
            let st = i;
            while i < cs.len() && (cs[i].is_ascii_digit() || cs[i] == '.' || cs[i] == '_') {
                i += 1;
            }
            let t: String = cs[st..i].iter().filter(|&&c| c != '_').collect();
            let v = t.parse::<f64>().map_err(|_| SchemaError::new(ln, col, format!("invalid number `{}`", t)))?;
            out.push(Spanned { tok: Tok::Num(v), col });
        } else if c.is_alphabetic() || c == '_' {
            let st = i;
            while i < cs.len() && (cs[i].is_alphanumeric() || cs[i] == '_' || cs[i] == '-') {
                i += 1;
            }
            out.push(Spanned { tok: Tok::Word(cs[st..i].iter().collect()), col });
        } else {
            return Err(SchemaError::new(ln, col, format!("unexpected character `{}`", c)));
        }
    }
    Ok(out)
}

fn is_ident(s: &str) -> bool {
    let mut cs = s.chars();
    matches!(cs.next(), Some(c) if c.is_ascii_alphabetic() || c == '_') && cs.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

// ----------------------------------------------------------------------------- DSL parser

/// Parse the `.j3v` DSL.
///
/// ```text
/// schema support_triage
/// teacher laya
/// choice department "Which department?"
///   billing "invoices, payments"
///   other
/// score urgency "How urgent?"
///   "not urgent"
///   "critical"
/// noul churn "Does the user threaten to leave?"
/// threshold 0.8
/// require agreement >= 0.9
/// ```
pub fn parse_dsl(src: &str) -> Result<Schema, SchemaError> {
    let mut name: Option<String> = None;
    let mut version = 1u32;
    let mut teacher = "laya".to_string();
    let mut max_state_tokens = 128usize;
    let mut threshold = 0.8f64;
    let mut require = Requirements::default();
    let mut questions: Vec<Question> = Vec::new();
    let mut qlines: Vec<usize> = Vec::new();
    let mut in_question = false;

    for (idx, raw) in src.lines().enumerate() {
        let ln = idx + 1;
        let toks = lex_line(raw, ln)?;
        if toks.is_empty() {
            continue;
        }
        let indented = raw.starts_with(' ') || raw.starts_with('\t');
        if indented {
            if !in_question {
                return Err(SchemaError::new(ln, toks[0].col, "indented line outside a question")
                    .help("options are indented lines directly below a `choice`, `score` or `noul` line"));
            }
            let q = questions.last_mut().unwrap();
            parse_option(q, &toks, ln)?;
            continue;
        }
        in_question = false;
        let kw = match &toks[0].tok {
            Tok::Word(w) => w.clone(),
            _ => return Err(SchemaError::new(ln, toks[0].col, "expected a keyword")),
        };
        let arg = |i: usize, what: &str| -> Result<&Spanned, SchemaError> {
            toks.get(i).ok_or_else(|| SchemaError::new(ln, raw.trim_end().len() + 1, format!("`{}` needs {}", kw, what)))
        };
        let end = |n: usize| -> Result<(), SchemaError> {
            match toks.get(n) {
                Some(t) => Err(SchemaError::new(ln, t.col, format!("unexpected extra token after `{}`", kw))),
                None => Ok(()),
            }
        };
        let num = |t: &Spanned, lo: f64, hi: f64| -> Result<f64, SchemaError> {
            match t.tok {
                Tok::Num(v) if v >= lo && v <= hi => Ok(v),
                Tok::Num(v) => Err(SchemaError::new(ln, t.col, format!("{} is out of range [{}, {}]", v, lo, hi))),
                _ => Err(SchemaError::new(ln, t.col, "expected a number")),
            }
        };
        match kw.as_str() {
            "schema" => {
                let t = arg(1, "a name")?;
                match &t.tok {
                    Tok::Word(w) if is_ident(w) => name = Some(w.clone()),
                    _ => return Err(SchemaError::new(ln, t.col, "schema name must be an identifier")),
                }
                end(2)?;
            }
            "version" => {
                version = num(arg(1, "a number")?, 1.0, 1e9)? as u32;
                end(2)?;
            }
            "teacher" => {
                let t = arg(1, "a teacher name")?;
                match &t.tok {
                    Tok::Word(w) if TEACHERS.contains(&w.as_str()) => teacher = w.clone(),
                    _ => {
                        return Err(
                            SchemaError::new(ln, t.col, "unknown teacher").help(format!("supported teachers: {}", TEACHERS.join(", ")))
                        )
                    }
                }
                end(2)?;
            }
            "max_state_tokens" => {
                max_state_tokens = num(arg(1, "a number")?, 8.0, 512.0)? as usize;
                end(2)?;
            }
            "threshold" => {
                threshold = num(arg(1, "a probability")?, 0.0, 1.0)?;
                end(2)?;
            }
            "require" => {
                let m = arg(1, "a metric")?;
                let op = arg(2, "`>=` or `<=`")?;
                let v = num(arg(3, "a bound")?, 0.0, 1.0)?;
                end(4)?;
                let metric = match &m.tok {
                    Tok::Word(w) => w.as_str(),
                    _ => return Err(SchemaError::new(ln, m.col, "expected a metric name")),
                };
                let (want, slot): (&str, &mut f64) = match metric {
                    "agreement" => (">=", &mut require.agreement),
                    "ece" => ("<=", &mut require.ece),
                    "accuracy" => {
                        require.accuracy = Some(0.0);
                        (">=", require.accuracy.as_mut().unwrap())
                    }
                    _ => {
                        return Err(SchemaError::new(ln, m.col, format!("unknown metric `{}`", metric))
                            .help("metrics: agreement (>=), ece (<=), accuracy (>=)"))
                    }
                };
                if op.tok != Tok::Op(if want == ">=" { ">=" } else { "<=" }) {
                    return Err(SchemaError::new(ln, op.col, format!("`{}` takes `{}`", metric, want)));
                }
                *slot = v;
            }
            "choice" | "score" | "noul" => {
                let qtype = match kw.as_str() {
                    "choice" => QType::Choice,
                    "score" => QType::Score,
                    _ => QType::Noul,
                };
                let idt = arg(1, "a question id")?;
                let id = match &idt.tok {
                    Tok::Word(w) if is_ident(w) => w.clone(),
                    _ => return Err(SchemaError::new(ln, idt.col, "question id must be an identifier")),
                };
                if let Some(p) = questions.iter().position(|q| q.id == id) {
                    return Err(SchemaError::new(ln, idt.col, format!("duplicate question `{}`", id))
                        .help(format!("first declared on line {}", qlines[p])));
                }
                let it = arg(2, "instructions in quotes")?;
                let instructions = match &it.tok {
                    Tok::Str(s) if !s.trim().is_empty() => s.clone(),
                    _ => return Err(SchemaError::new(ln, it.col, "instructions must be a non-empty quoted string")),
                };
                end(3)?;
                let options = if qtype == QType::Noul {
                    vec![Opt { key: "false".into(), description: None }, Opt { key: "true".into(), description: None }]
                } else {
                    Vec::new()
                };
                questions.push(Question { id, qtype, instructions, options });
                qlines.push(ln);
                in_question = true;
            }
            _ => {
                return Err(SchemaError::new(ln, toks[0].col, format!("unknown keyword `{}`", kw))
                    .help("keywords: schema, version, teacher, max_state_tokens, choice, score, noul, threshold, require"))
            }
        }
    }
    let name = name.ok_or_else(|| SchemaError::new(1, 1, "missing `schema <name>` line"))?;
    if questions.is_empty() {
        return Err(SchemaError::new(1, 1, "schema declares no questions"));
    }
    for (q, &ln) in questions.iter().zip(&qlines) {
        let k = q.options.len();
        match q.qtype {
            QType::Choice if k < 2 => {
                return Err(SchemaError::new(ln, 1, format!("choice `{}` has {} option(s), needs at least 2", q.id, k))
                    .help("list options as indented lines below the question"))
            }
            QType::Score if k < 2 => return Err(SchemaError::new(ln, 1, format!("score `{}` has {} level(s), needs at least 2", q.id, k))),
            _ if k > MAX_OPTIONS => {
                return Err(SchemaError::new(ln, 1, format!("`{}` has {} options; J3v supports at most {}", q.id, k, MAX_OPTIONS))
                    .help("the Laya teacher degrades past ~20 options (Banking77: 0.425); split into a coarse-to-fine pair of questions"))
            }
            _ => {}
        }
    }
    Ok(Schema { name, version, teacher, max_state_tokens, threshold, require, questions })
}

fn parse_option(q: &mut Question, toks: &[Spanned], ln: usize) -> Result<(), SchemaError> {
    match q.qtype {
        QType::Choice => {
            let key = match &toks[0].tok {
                Tok::Word(w) if is_ident(w) => w.clone(),
                _ => {
                    return Err(SchemaError::new(ln, toks[0].col, "choice option must start with a key")
                        .help("write `key \"optional description\"`"))
                }
            };
            if q.options.iter().any(|o| o.key == key) {
                return Err(SchemaError::new(ln, toks[0].col, format!("duplicate option `{}` in `{}`", key, q.id)));
            }
            let description = match toks.get(1).map(|t| &t.tok) {
                None => None,
                Some(Tok::Str(s)) => Some(s.clone()),
                Some(_) => return Err(SchemaError::new(ln, toks[1].col, "option description must be a quoted string")),
            };
            if let Some(t) = toks.get(2) {
                return Err(SchemaError::new(ln, t.col, "unexpected extra token after option"));
            }
            q.options.push(Opt { key, description });
        }
        QType::Score => {
            let d = match &toks[0].tok {
                Tok::Str(s) => s.clone(),
                _ => {
                    return Err(SchemaError::new(ln, toks[0].col, "score level must be a quoted string")
                        .help("levels are ordered, lowest first; their keys are 0, 1, 2, ..."))
                }
            };
            if let Some(t) = toks.get(1) {
                return Err(SchemaError::new(ln, t.col, "unexpected extra token after score level"));
            }
            let k = q.options.len();
            q.options.push(Opt { key: k.to_string(), description: Some(d) });
        }
        QType::Noul => {
            let slot = match &toks[0].tok {
                Tok::Word(w) if w == "false" => 0,
                Tok::Word(w) if w == "true" => 1,
                _ => {
                    return Err(SchemaError::new(ln, toks[0].col, "noul options can only be `true` or `false`")
                        .help("noul is a calibrated yes/no; use `choice` for anything else"))
                }
            };
            match toks.get(1).map(|t| &t.tok) {
                Some(Tok::Str(s)) => q.options[slot].description = Some(s.clone()),
                _ => return Err(SchemaError::new(ln, toks[0].col, "noul option needs a quoted description")),
            }
        }
    }
    Ok(())
}

// ----------------------------------------------------------------------------- Laya JSON front end

/// Build a schema from a Laya/Jev request-shaped JSON: `{"name": "...", "questions": {id: {type, instructions, criteria}}}`.
pub fn from_laya_json(v: &Value) -> Result<Schema, String> {
    let name = v.get("name").and_then(Value::as_str).unwrap_or("schema").to_string();
    let qs = v.get("questions").and_then(Value::as_object).ok_or("missing `questions` object")?;
    let mut questions = Vec::new();
    for (id, q) in qs {
        questions.push(question_from_laya(id, q)?);
    }
    if questions.is_empty() {
        return Err("no questions".into());
    }
    let mut s = Schema {
        name,
        version: 1,
        teacher: v.get("teacher").and_then(Value::as_str).unwrap_or("laya").to_string(),
        max_state_tokens: 128,
        threshold: v.get("threshold").and_then(Value::as_f64).unwrap_or(0.8),
        require: Requirements::default(),
        questions,
    };
    if let Some(t) = v.get("max_state_tokens").and_then(Value::as_u64) {
        s.max_state_tokens = t as usize;
    }
    Ok(s)
}

/// One Laya question definition -> [`Question`]. Accepts every criteria shape Laya accepts.
pub fn question_from_laya(id: &str, q: &Value) -> Result<Question, String> {
    let t = q.get("type").and_then(Value::as_str).ok_or_else(|| format!("question {:?}: missing `type`", id))?;
    let instructions = match q.get("instructions") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => return Err(format!("question {:?}: missing `instructions`", id)),
    };
    let crit = q.get("criteria");
    let desc = |v: &Value| v.as_str().map(str::to_string);
    let (qtype, options) = match t {
        "choice" => {
            let opts: Vec<Opt> = match crit {
                Some(Value::Object(m)) => m.iter().map(|(k, v)| Opt { key: k.clone(), description: desc(v) }).collect(),
                Some(Value::Array(a)) => a
                    .iter()
                    .map(|v| v.as_str().map(|k| Opt { key: k.to_string(), description: None }))
                    .collect::<Option<_>>()
                    .ok_or_else(|| format!("question {:?}: choice criteria list must be strings", id))?,
                _ => return Err(format!("question {:?}: choice needs `criteria` (object or list)", id)),
            };
            (QType::Choice, opts)
        }
        "score" => {
            let a = crit.and_then(Value::as_array).ok_or_else(|| format!("question {:?}: score needs a `criteria` list", id))?;
            (QType::Score, a.iter().enumerate().map(|(i, v)| Opt { key: i.to_string(), description: desc(v) }).collect())
        }
        "noul" => {
            let f = crit.and_then(|c| c.get("false")).and_then(desc);
            let tr = crit.and_then(|c| c.get("true")).and_then(desc);
            (QType::Noul, vec![Opt { key: "false".into(), description: f }, Opt { key: "true".into(), description: tr }])
        }
        other => return Err(format!("question {:?}: unknown type {:?} (choice, score, noul)", id, other)),
    };
    if options.len() < 2 {
        return Err(format!("question {:?}: needs at least 2 options", id));
    }
    Ok(Question { id: id.to_string(), qtype, instructions, options })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = r#"
schema triage
teacher laya
choice dept "Which department?"
  billing "invoices"
  other
score urgency "How urgent?"
  "low"
  "high"
noul churn "Leaving?"
threshold 0.7
require agreement >= 0.9
require accuracy >= 0.8
"#;

    #[test]
    fn parses() {
        let s = parse_dsl(SRC).unwrap();
        assert_eq!(s.questions.len(), 3);
        assert_eq!(s.questions[0].options[1], Opt { key: "other".into(), description: None });
        assert_eq!(s.questions[1].options[1].key, "1");
        assert_eq!(s.threshold, 0.7);
        assert_eq!(s.require.accuracy, Some(0.8));
    }

    #[test]
    fn laya_round_trip() {
        let s = parse_dsl(SRC).unwrap();
        let v = json!({"name": "triage", "threshold": 0.7, "questions": s.to_laya_questions()});
        let s2 = from_laya_json(&v).unwrap();
        assert_eq!(s.teacher_hash(), s2.teacher_hash());
    }

    #[test]
    fn errors_point_at_the_problem() {
        let e = parse_dsl("schema x\nchoice a \"q\"\n  only\n").unwrap_err();
        assert!(e.msg.contains("needs at least 2"), "{}", e.msg);
        let e = parse_dsl("schema x\nrequire ece >= 0.1\n").unwrap_err();
        assert_eq!((e.line, e.col), (2, 13));
        let e = parse_dsl("schema x\nnoul a \"q\"\n  maybe \"x\"\n").unwrap_err();
        assert!(e.help.unwrap().contains("choice"));
        let e = parse_dsl("schema x\nchoice a \"q\n").unwrap_err();
        assert!(e.msg.contains("unterminated"));
    }
}

/// The text a student model sees for a state. Laya accepts a string or any JSON; J3v renders objects as
/// `key: value` lines (strings unquoted) so short states are not padded with JSON punctuation.
pub fn state_text(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        Value::Object(m) => m
            .iter()
            .map(|(k, v)| match v {
                Value::String(s) => format!("{}: {}", k, s),
                other => format!("{}: {}", k, other),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    }
}
