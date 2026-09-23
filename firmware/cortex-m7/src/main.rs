//! J3v mcu-tier demo: answer the compiled schema for a few fixed inputs and print Laya-style answers
//! (argmax key, calibrated p_top, escalate flag) over semihosting. Output format matches `j3v mcu-predict`.
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use panic_semihosting as _;

mod model {
    include!(concat!(env!("OUT_DIR"), "/model.rs"));
}

static INPUTS: &str = include_str!(concat!(env!("OUT_DIR"), "/inputs.txt"));

#[entry]
fn main() -> ! {
    let m = &model::MODEL;
    hprintln!("j3v mcu: {} questions, {} buckets x {} dim, hidden {}", model::QUESTIONS.len(), m.buckets, m.dim, m.hidden);
    for line in INPUTS.lines().filter(|l| !l.is_empty()) {
        let mut z = [[0f32; j3v_mcu::MAX_K]; 8];
        j3v_mcu::logits(m, line.as_bytes(), &mut z[..m.heads.len()]);
        hprintln!("> {}", line);
        for (j, hd) in m.heads.iter().enumerate() {
            let (top, p) = j3v_mcu::calibrate(&mut z[j][..hd.k], hd.temperature);
            let (qid, keys) = model::QUESTIONS[j];
            hprintln!("  {} = {} p_top={:.4} escalate={}", qid, keys[top], p, p < hd.threshold);
        }
    }
    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}
