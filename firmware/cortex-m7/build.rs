//! Picks the memory map and copies the compiler-generated model (J3V_MODEL=path/to/model.rs) into OUT_DIR.
use std::{env, fs, path::PathBuf};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let memory = if env::var("CARGO_FEATURE_STM32H743").is_ok() {
        // STM32H743: 2 MB flash, 512 KB AXI SRAM
        "MEMORY { FLASH : ORIGIN = 0x08000000, LENGTH = 2048K\n RAM : ORIGIN = 0x24000000, LENGTH = 512K }\n"
    } else {
        // QEMU mps2-an500 (Cortex-M7): 4 MB code SRAM at 0, 4 MB data SRAM
        "MEMORY { FLASH : ORIGIN = 0x00000000, LENGTH = 4096K\n RAM : ORIGIN = 0x20000000, LENGTH = 4096K }\n"
    };
    fs::write(out.join("memory.x"), memory).unwrap();
    let model = env::var("J3V_MODEL").unwrap_or_else(|_| "src/model.rs".into());
    fs::copy(&model, out.join("model.rs")).unwrap_or_else(|e| panic!("J3V_MODEL={}: {} (run j3v compile --target mcu first)", model, e));
    let inputs = env::var("J3V_INPUTS").unwrap_or_else(|_| "src/inputs.txt".into());
    fs::copy(&inputs, out.join("inputs.txt")).unwrap_or_else(|e| panic!("J3V_INPUTS={}: {}", inputs, e));
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-env-changed=J3V_MODEL");
    println!("cargo:rerun-if-env-changed=J3V_INPUTS");
    println!("cargo:rerun-if-changed={}", model);
    println!("cargo:rerun-if-changed={}", inputs);
}
