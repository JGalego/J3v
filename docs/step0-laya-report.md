# Step 0: what Laya actually is

Source: `convaiinnovations/laya` @ `5e7b2b1b` (HF, 2026-09-23): model card, `rl_common.py`,
`rl_agent_api.py`, `rl_agent_config.json`, `eval/results.md`. `evaluate.py` and the training
loop are not in the HF repo (GitHub `NandhaKishorM/laya` only), so claims about them are
from the card and are marked as such.

## Primitives (from `rl_agent_api.py`)

All three are "softmax over K option markers, then divide by a per-bucket temperature".

| type | options rendered as | response fields |
|---|---|---|
| `choice` | `key` or `key: description` (dict, or list → keys only) | `choice` (argmax key), `probabilities{key:p}`, `confidence` |
| `score` | `level i: text` for ordered list `criteria` | `score` = E[i] = Σ i·pᵢ (a real number in [0, K−1], not an argmax), `legend`, `probabilities{"i":p}`, `confidence` |
| `noul` | always `[false: …, true: …]` | `noul` = P(true). **No `confidence` field.** |

`confidence` = 1 − H(p)/log K (normalized entropy). It is **not a probability** and is not what
ECE is measured on. Every answer also carries `rl_agent.act_probability`, which the card says is
useless (AUROC 0.30, issue #185).

Wire shape: `POST /v1/systemone {state, questions}` → `{model, answers, usage:{input_tokens, output_tokens:0}}`,
claimed Jev-compatible. `state` is a string or JSON (serialized with `json.dumps`).

## Cross-encoded or independent? Cross-encoded.

One sequence **per question**:
`[CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]`
through full bidirectional ModernBERT-large (395M) + 2 extra transformer layers; each option's
logit is read from its `[MASK]` position. Option tokens attend to state tokens, so **no
option-side computation can be precomputed** in the teacher, and N questions cost N encoder passes.

This does not hurt J3v. Our options are fixed at compile time, so the student does not need to
see option text at all: it is `state → encoder → per-question K-way head`. Options are folded
into head weights by distillation. One encoder pass per state, all questions for free.

## Calibration: how trained, how measured

- **Trained:** "RLCD" = REINFORCE with a group-mean baseline (GRPO-style), Gaussian noise on
  the option logits for exploration, reward = log score + 0.5·spherical (+ RPS for `score`).
  All strictly proper. **Not PPO, and not on sequence embeddings**: the policy is the per-option
  marker logits. (Reward code is in `rl_common.py`; the update loop is card-only.)
- **Post-hoc:** one temperature per `(type, option-count bucket)`: `2 / 3-5 / 6-10 / 11+`,
  stored in `rl_agent_config.json`. Raw checkpoint is over-confident (card: mean ECE 0.466 → 0.081
  after refit).
- **Measured:** 15-bin equal-width ECE, plus NLL, Brier, AURC, accuracy at 50% coverage.
  In-task ECE 0.030; zero-shot task families ECE 0.204. Which confidence goes into `ece_score` is
  in `evaluate.py`, which isn't published on HF. I assume top-1 probability; this needs checking on GitHub.

## Where the spec is wrong or risky

1. **"RLCD / PPO on sequence embeddings"** is wrong. See above. It doesn't matter to us:
   we distill outputs and don't reproduce RLCD.
2. **The teacher is weak zero-shot.** On typed-decisions the base checkpoint scores 0.362,
   below the 0.461 majority-class baseline. Only the benchmark-fine-tuned checkpoint scores 0.766.
   "Accuracy vs teacher" therefore measures **agreement**, and a student can hit 99% agreement
   on a useless teacher. Proposal: conformance reports (a) agreement with the teacher, and (b) accuracy
   and ECE against **ground-truth labels** whenever the schema ships a labelled held-out set.
   The budget gate uses (b) when it's available and (a) otherwise, and it says which one it used.
3. **The cascade threshold must be calibrated P(correct)**, not the API's `confidence`
   (an entropy score that isn't calibrated) and not `act_probability` (broken). J3v keeps the
   `confidence` field for compatibility and adds `p_top` (calibrated), which is what the threshold
   compares against. ECE is only meaningful per tier if every tier reports the same quantity.
4. **Teacher `noul` is unreliable on the English checkpoint** (it follows the `false:`/`true:`
   label text; issue #156). When labelling synthetic data, the compiler should query every `noul`
   as a 2-option `choice` with neutral keys, and then map the result back.
5. **Teacher temperatures look stale.** `choice:11+` has T = 0.10 (a 10× sharpening). That fits
   the card's own finding that 11+ option questions collapse (Banking77 0.425). The
   `typed-decisions` config carries the same `temperature_by_options` as the base checkpoint,
   so it was copied, not refit. J3v should refit teacher temperature per schema before distilling,
   and should refuse (or warn on) schemas with more than ~20 options unless the teacher is configured
   with a larger `head_max_len`.
6. **"Synthetic states per schema" needs a text generator** (an LLM) at compile time. That's fine
   offline, but it's an unstated dependency. Held-out ECE on synthetic states also says little about
   real traffic. The compiler should accept a sample of real unlabelled states and prefer it.
7. **A compile error that depends on training is statistical, not a type check.** The gate needs
   (i) a cheap static pre-check (param bytes + activation RAM vs budget, before any training) and
   (ii) a bound on the **upper confidence limit** of ECE and the lower limit of accuracy (bootstrap
   over the held-out set). Otherwise the same schema passes and fails on reruns.
8. **Teacher cost/context:** the English teacher has 512 tokens with ~320 for state; multilingual
   has 1024. The J3v state length should be capped at the teacher's, or labels for long states are
   produced from truncated input.
9. **Which teacher?** The repo holds three checkpoints. Default: the English root. Use
   `multilingual` when the schema declares non-English. Never use `typed-decisions` for general
   schemas, because it is fine-tuned on one benchmark.

## Language: Rust

Rust. The `pi` runtime must be one static binary. `cargo build --target aarch64-unknown-linux-musl`
does that without a hand-rolled toolchain, and the same core crate compiles `no_std` for the MCU
tier (`esp-hal` for ESP32-S3, `embassy`/`cortex-m` for STM32H7). The inference kernel and
tokenizer/hash front end are then shared bit-for-bit across tiers, which the cascade and
conformance tests rely on. The compiler, the server, the HTTP/JSON compatibility layer and the
int8 kernels are all in one language with memory safety at the network boundary. C would give
slightly smaller MCU images, but we would maintain two front ends. If a board vendor demands C,
emit a generated `.h` of const weights plus a ~300-line C kernel as a secondary MCU output.

## Unverified and needs checking before M2

- `evaluate.py`: which confidence goes into ECE, and whether the temperature is fit on the eval split.
- Laya CPU latency on a Pi 5 (the card gives 193–464 ms on an unnamed CPU). This is the baseline J3v-pi must beat.
- Whether the Jev API has fields Laya drops (e.g. multi-select, `null`/abstain). Laya "ignores unknown fields".
