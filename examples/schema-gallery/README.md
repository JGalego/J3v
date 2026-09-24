# Schema gallery

Five schemas that exist to be *read*, not to ship: each one exercises a part of the `.j3v` language that
[`schemas/support_triage.j3v`](../../schemas/support_triage.j3v) does not. None of them has a committed
artifact — compiling one needs states of your own. They are all checked by CI with `j3v check`, so they stay
valid as the language moves.

```bash
j3v check examples/schema-gallery/multilingual_feedback.j3v   # parse, validate, print canonical JSON
```

| file | what it shows |
|---|---|
| [`multilingual_feedback.j3v`](multilingual_feedback.j3v) | `teacher laya-multilingual`; a 5-level `score` held to `mae` instead of argmax agreement |
| [`sensor_alarm.j3v`](sensor_alarm.j3v) | a schema shaped for `--target mcu`: short `max_state_tokens`, tight budget, bounds an int8 hashed-n-gram model can reach |
| [`pr_triage.j3v`](pr_triage.j3v) | per-question `threshold`s that differ by an order of consequence, and `require accuracy` against ground truth |
| [`incident_summary.json`](incident_summary.json) | the **Laya/Jev JSON front end**: a request body is a schema, so you can compile the questions you already send to Laya |
| [`errors/`](errors) | four schemas that are *supposed* to fail, and the diagnostic each one produces |

## The answer space is the interface

Every schema here fixes its answer keys at compile time, and those keys are what your code branches on. In
`multilingual_feedback.j3v` the states may be in any language the multilingual teacher covers, but `topic` is
still one of `crash | usability | pricing | content | praise`. Nothing downstream parses text.

## Two kinds of question need two kinds of bound

`agreement` is argmax agreement with the teacher. It is the right bound for a `choice` or a `noul`, and a harsh
one for an ordinal `score`: when the teacher splits its mass between "unhappy" and "neutral", the argmax is a
coin flip and the student is punished for a disagreement nobody would defend. Ordinal questions are held to
`mae` — the mean error of the expected score, in levels — with a looser agreement floor:

```text
require sentiment.agreement >= 0.70
require sentiment.mae <= 0.40
```

The same argument, with numbers measured on a real build, is in `schemas/support_triage.j3v` and
[`docs/results.md`](../../docs/results.md).

## Thresholds follow consequence, not difficulty

`threshold` is the calibrated top-1 probability below which a question escalates to the next tier. It belongs
at the cost of being wrong, which is per question, not per schema. `pr_triage.j3v` sets three:

```text
threshold 0.80                   # the schema default
threshold area 0.70              # a wrong area routes to the wrong team; a human fixes it in seconds
threshold breaking_change 0.92   # a missed breaking change ships
```

Lowering a threshold keeps more answers locally and costs accuracy on what it keeps; raising it sends more
traffic to the tier above. `j3v cascade` measures that trade rather than assuming it — on `support_triage`,
moving one advisory question from 0.80 to 0.60 took the share of requests reaching Laya from 36% to 25%.

## Errors are the type system

The four schemas in [`errors/`](errors) do not compile, on purpose. Run them to see what the compiler says:

```bash
$ j3v check examples/schema-gallery/errors/mae_on_noul.j3v
error: `mae` only applies to score questions; `is_spam` is a noul
 --> examples/schema-gallery/errors/mae_on_noul.j3v:6:9
  |
6 | require is_spam.mae <= 0.2
  |         ^
```

```bash
$ j3v check examples/schema-gallery/errors/too_many_options.j3v
error: `intent` has 24 options; J3v supports at most 20
 --> examples/schema-gallery/errors/too_many_options.j3v:5:1
  |
5 | choice intent "Which banking intent is this?"
  | ^
  = help: the Laya teacher degrades past ~20 options (Banking77: 0.425); split into a coarse-to-fine pair of questions
```

| file | error |
|---|---|
| `errors/mae_on_noul.j3v` | `mae` asked of a question that has no ordinal scale |
| `errors/too_many_options.j3v` | 24-way choice; the teacher's own accuracy collapses past ~20 |
| `errors/unknown_question.j3v` | a `require` / `threshold` naming a question that was renamed away |
| `errors/one_option.j3v` | a `choice` with one option, which is not a decision |

These are all parse-time. The two that need a build are `error[E0302]` (nothing fits the flash budget, raised
*before* training) and `error[E0301]` (trained, but a bound is missed) — both in
[`docs/results.md`](../../docs/results.md).
