//! J3v mcu-tier demo: answer the compiled schema for a few fixed inputs and print Laya-style answers
//! (argmax key, calibrated p_top, escalate flag) over semihosting. Output format matches `j3v mcu-predict`.
#![no_std]
#![no_main]

use cortex_m::peripheral::{syst::SystClkSource, SYST};
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
    bench(m);
    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}

/// SysTick ticks elapsed while running `f` (24-bit down-counter, so modulo 2^24; callers keep each run under one
/// full period).
fn ticks(syst: &mut SYST, f: impl FnOnce()) -> u32 {
    let _ = syst;
    let t0 = SYST::get_current();
    f();
    t0.wrapping_sub(SYST::get_current()) & 0x00FF_FFFF
}

/// Instruction-count proxy for latency, printed as `#` lines (the CI diff against the host ignores them).
/// Under `qemu -icount shift=0` every instruction advances virtual time by 1 ns, so SysTick ticks are
/// proportional to instructions executed. A loop of known length (2 instructions per iteration) calibrates
/// instructions per tick. On real silicon the same code measures cycles instead.
fn bench(m: &j3v_mcu::Model) {
    let mut p = cortex_m::Peripherals::take().unwrap();
    p.SYST.set_clock_source(SystClkSource::Core);
    p.SYST.set_reload(0x00FF_FFFF);
    p.SYST.clear_current();
    p.SYST.enable_counter();
    const CAL: u32 = 1_000_000;
    let cal = ticks(&mut p.SYST, || unsafe {
        core::arch::asm!("1:", "subs {0}, {0}, #1", "bne 1b", inout(reg) CAL => _, options(nomem, nostack));
    });
    let per_tick = (2 * CAL) as f32 / cal as f32;
    let lines: usize = INPUTS.lines().filter(|l| !l.is_empty()).count();
    const REPS: usize = 20;
    let t = ticks(&mut p.SYST, || {
        for _ in 0..REPS {
            for line in INPUTS.lines().filter(|l| !l.is_empty()) {
                let mut z = [[0f32; j3v_mcu::MAX_K]; 8];
                j3v_mcu::logits(m, line.as_bytes(), &mut z[..m.heads.len()]);
                for (j, hd) in m.heads.iter().enumerate() {
                    core::hint::black_box(j3v_mcu::calibrate(&mut z[j][..hd.k], hd.temperature));
                }
            }
        }
    });
    let insns = t as f32 * per_tick / (REPS * lines) as f32;
    hprintln!("# bench: {:.0} instructions per inference ({} questions), {:.1} per SysTick tick", insns, m.heads.len(), per_tick);
    hprintln!("# bench: ~{:.0} us at 480 MHz if 1 cycle/instruction (STM32H743; real CPI varies)", insns / 480.0);
}
