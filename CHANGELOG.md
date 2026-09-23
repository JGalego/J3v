# Changelog

## [Unreleased]

### Added
- `install.sh`: one-line installer (`curl … | sh`) that picks the right static binary, verifies checksums, and
  can fetch the encoder and example models.
- `j3v --version`.

## [0.1.1] - 2026-09-23

### Added
- Static 32-bit ARM build (`armv7-unknown-linux-musleabihf`) for Raspberry Pis running a 32-bit OS. It uses the
  portable int8 kernel, which is slower than the aarch64 dot-product kernel.

## [0.1.0] - 2026-09-23

First release. J3v compiles typed decision schemas into calibrated, edge-sized artifacts distilled from
[Laya](https://huggingface.co/convaiinnovations/laya), and refuses to emit an artifact that misses its conformance
bounds or its budget.

### Added
- **Schema DSL** (`.j3v`): `choice`, `score`, `noul` questions; escalation `threshold` (schema-wide or per
  question); `require` bounds on agreement, ECE, ground-truth accuracy and expected-score MAE (schema-wide or per
  question). Laya/Jev request JSON is also accepted as a schema.
- **`j3v compile --target pi`**: a shared int8 BERT encoder (all-MiniLM-L6-v2) with distilled per-schema heads (int8)
  and per-question temperature scaling. Calibration and the gate run in Rust on the shipped artifact.
- **`j3v compile --target mcu --budget`**: one hashed-n-gram int8 model per schema, with no vocabulary table and no
  allocator. The compiler searches model sizes under the flash/RAM budget and generates firmware source
  (`j3v mcu-codegen`). Errors: `E0302` when nothing can fit (checked before training), `E0301` when no fitting
  size meets the bounds.
- **`pi` runtime**: one static binary (~1 MB, aarch64/x86_64 musl), `j3v serve` on Laya/Jev's `POST /v1/systemone`,
  `j3v predict`, `j3v bench`.
- **Cascade**: `j3v serve --upstream` escalates each question below its threshold to Laya/Jev, re-tempered with
  the compile-time calibration. `j3v cascade` certifies mcu → pi → Laya with per-(tier, question) conditional
  calibration and a Wilson-bound gate (`E0401`).
- **Cortex-M7 firmware** (STM32H743 / QEMU `mps2-an500`): output is byte-identical to the host runtime, and an
  instruction-count latency proxy runs under QEMU `-icount`.
- **CI**: fmt/clippy/tests, static builds, a native-ARM64 latency bench, firmware under QEMU, and an offline
  cascade certification.
- **Example**: `support_triage` (4 questions), certified on 1614 held-out Bitext states. See `docs/results.md`.

### Known limitations
- The shared encoder is off-the-shelf MiniLM, not distilled from Laya. Only the heads are distilled.
- mcu latency is an instruction count under emulation, not measured on silicon.
- `j3v serve --upstream` speaks plain `http://`. Put TLS in a reverse proxy.
- Students are only as good as the teacher on the schema's data. Laya zero-shot is weak on some schemas
  (see `docs/step0-laya-report.md`), so check `accuracy_gt` in the conformance report.
