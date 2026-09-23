# Results: `support_triage`

8000 Bitext customer-support states, labelled by Laya (`convaiinnovations/laya` @ `5e7b2b1b`, English checkpoint,
native `noul`). Split by state-id hash: 5588 train / 798 calib / 1614 test. Bounds are checked against one-sided 95%
bootstrap limits. Full reports: [`examples/support_triage/`](../examples/support_triage).

## M2: `pi` target

Shared encoder: all-MiniLM-L6-v2, int8 (22.6M params, 23.8 MB). Heads: 794 KB (int8, hidden 256).

| question | agreement w/ Laya | ECE | accuracy vs ground truth | Laya's accuracy |
|---|---|---|---|---|
| department (5-way) | 0.971 | 0.020 | 0.931 | 0.923 |
| urgency (3-level score) | 0.860 | 0.032 | n/a (no labels) | n/a |
| refund_requested (noul) | 0.980 | 0.009 | 0.971 | 0.961 |
| wants_human (noul) | 0.970 | 0.011 | 0.950 | 0.940 |

Binary: static musl, 1.0 MB (aarch64), 1.2 MB (x86_64). Latency for all 4 questions on 1 core, 1 thread, mean
13.8 tokens: **p50 6.7 ms / p95 9.4 ms** on an ARM64 Neoverse-N2 CI runner. A Pi 5 (Cortex-A76) will be somewhat slower.
Laya's model card gives 193–464 ms on CPU; we measured 705 ms p50 for 4 questions via `laya-serve` on the build host.

## M3: budget as a type error

- `--budget 4KB` → `error[E0302]` before any training (the smallest mcu model needs 11.1 KB).
- Missing a bound → `error[E0301]`, which names the question, the metric, its bootstrap bound and the requirement.
  `urgency` failed agreement ≥ 0.85 on both targets. Argmax agreement is a poor target for an ordinal question
  whose teacher mass is split across adjacent levels, so the schema holds it to expected-score MAE ≤ 0.25 and
  agreement ≥ 0.80. Both bounds were set after the first build, and the schema says so.

## M4: `mcu` target (Cortex-M7)

Budget 512 KB. The compiler trained 6 sizes and shipped the smallest that passes: 1024 buckets × 64 dim, hidden 64,
**73 KB of weights, ~2 KB RAM**. Firmware: 18 KB of code. The output under QEMU `mps2-an500` is byte-identical to
the host runtime, which CI checks on every push. Agreement 0.966 / 0.820 / 0.975 / 0.954, ECE ≤ 0.029, accuracy
vs ground truth 0.928 / – / 0.976 / 0.940. Latency proxy: under QEMU `-icount`, one inference (all 4 questions)
executes **~100k instructions**, which is ~210 µs on an STM32H743 at 480 MHz if it runs at 1 cycle per
instruction. That is an estimate, not a silicon measurement: real CPI on the M7 depends on dual issue and flash
wait states. The same firmware reports cycles when flashed to a board.

## M5: cascade mcu → pi → Laya

- **Marginal calibration is not enough.** Every tier had ECE ≤ 0.03 on the full test set. But on the states the mcu
  escalates, `pi` kept answers only 0.794 / 0.667 accurate against a 0.80 threshold. Both students learned from the
  same teacher on the same data, so they are wrong together.
- **Fix:** refit one temperature per (tier, question) on the calib states that actually reach that tier
  (conditional calibration). Gate: each non-final tier's kept answers must meet its threshold (Wilson 95% bound).
- **Per-question thresholds:** `urgency` is advisory, so it escalates below 0.60 instead of 0.80. Unlabelled
  questions are calibrated and gated against the final tier's answer.
- **Offline (1614 test states, cached Laya answers; runs in CI):** the mcu tier keeps 89–97% of each question.
  Laya is reached by 25% of requests (36% before per-question thresholds). Cascade accuracy 0.950. Laya is *less* accurate than the
  students on what reaches it (0.44–0.64), because those states are the ambiguous ones. On this data the cascade
  buys compute, not accuracy.
- **Live (300 test states, `laya-serve` on CPU):** contract holds. After re-tempering, Laya's ECE is 0.014–0.030.
  27% of requests reach Laya (37% before per-question thresholds). Cascade accuracy 0.951, expected latency
  **202 ms/request** (269 ms before), dominated by Laya's 734 ms p50. `j3v serve --upstream` does the same escalation
  at runtime, per question.
