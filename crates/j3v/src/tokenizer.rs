//! BERT uncased WordPiece, byte-for-byte compatible with HF `BertTokenizer` on the inputs we test
//! (see `j3v tokenize` + compiler/tests/test_tokenizer.py).

use std::collections::HashMap;
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

pub struct WordPiece {
    vocab: HashMap<String, u32>,
    pub cls: u32,
    pub sep: u32,
    pub unk: u32,
    lowercase: bool,
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F | 0x2B820..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

fn is_punct(c: char) -> bool {
    let u = c as u32;
    (33..=47).contains(&u)
        || (58..=64).contains(&u)
        || (91..=96).contains(&u)
        || (123..=126).contains(&u)
        || matches!(u, 0xA1 | 0xA7 | 0xAB | 0xB6 | 0xB7 | 0xBB | 0xBF | 0x2010..=0x2027 | 0x2030..=0x205E
            | 0x3001..=0x3003 | 0x3008..=0x3011 | 0x3014..=0x301F | 0xFF01..=0xFF0F | 0xFF1A..=0xFF20
            | 0xFF3B..=0xFF3D | 0xFF5B..=0xFF65 | 0x055A..=0x055F | 0x0589 | 0x05BE | 0x060C | 0x061B | 0x061F
            | 0x066A..=0x066D | 0x06D4 | 0x0964 | 0x0965 | 0x0E4F | 0x0E5A | 0x0E5B)
}

fn is_control(c: char) -> bool {
    if c == '\t' || c == '\n' || c == '\r' {
        return false;
    }
    c.is_control() || matches!(c as u32, 0x200B..=0x200F | 0xFEFF | 0x00AD)
}

impl WordPiece {
    pub fn new(vocab: &[String], lowercase: bool) -> Result<Self, String> {
        let map: HashMap<String, u32> = vocab.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let id = |t: &str| map.get(t).copied().ok_or_else(|| format!("vocab lacks {}", t));
        Ok(WordPiece { cls: id("[CLS]")?, sep: id("[SEP]")?, unk: id("[UNK]")?, vocab: map, lowercase })
    }

    fn basic(&self, text: &str) -> Vec<String> {
        let mut clean = String::with_capacity(text.len());
        for c in text.chars() {
            if c == '\0' || c == '\u{FFFD}' || is_control(c) {
                continue;
            }
            if c.is_whitespace() {
                clean.push(' ');
            } else if is_cjk(c) {
                clean.push(' ');
                clean.push(c);
                clean.push(' ');
            } else {
                clean.push(c);
            }
        }
        let mut out = Vec::new();
        for w in clean.split_whitespace() {
            let w: String =
                if self.lowercase { w.to_lowercase().nfd().filter(|&c| !is_combining_mark(c)).collect() } else { w.to_string() };
            let mut cur = String::new();
            for c in w.chars() {
                if is_punct(c) {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                    out.push(c.to_string());
                } else {
                    cur.push(c);
                }
            }
            if !cur.is_empty() {
                out.push(cur);
            }
        }
        out
    }

    fn wordpiece(&self, w: &str, out: &mut Vec<u32>) {
        let cs: Vec<char> = w.chars().collect();
        if cs.len() > 100 {
            out.push(self.unk);
            return;
        }
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < cs.len() {
            let mut end = cs.len();
            let mut found = None;
            while start < end {
                let mut s: String = cs[start..end].iter().collect();
                if start > 0 {
                    s.insert_str(0, "##");
                }
                if let Some(&id) = self.vocab.get(&s) {
                    found = Some(id);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(id) => pieces.push(id),
                None => {
                    out.push(self.unk);
                    return;
                }
            }
            start = end;
        }
        out.extend(pieces);
    }

    /// Token ids without special tokens.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for w in self.basic(text) {
            self.wordpiece(&w, &mut out);
        }
        out
    }

    /// `[CLS] tokens[..max-2] [SEP]`.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<u32> {
        let mut t = self.tokenize(text);
        t.truncate(max_len.saturating_sub(2));
        let mut ids = Vec::with_capacity(t.len() + 2);
        ids.push(self.cls);
        ids.extend(t);
        ids.push(self.sep);
        ids
    }
}
