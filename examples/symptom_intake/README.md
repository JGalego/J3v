# `symptom_intake`

Routing a free-text symptom message to a queue, before a clinician reads it.
[`schemas/symptom_intake.j3v`](../../schemas/symptom_intake.j3v) asks which body system the symptoms point
to, how fast someone should look, whether a dangerous symptom is named, and whether the message says how
long it has been going on.

> **This decides a queue, not a diagnosis.** It is a demonstration of the compiler on a high-stakes routing
> problem, built on synthetic data. It is not a medical device, it has not been clinically validated, and
> nothing it answers is a clinical finding. The escalation thresholds exist precisely because a model like
> this should be handing most of its decisions to a human.

**The artifacts are not committed** — the schema, the data prep and the build are here, but nothing ships a
conformance report it has not earned.

## Why this corpus, and what its labels mean

[`gretelai/symptom_to_diagnosis`](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis)
(Apache-2.0) is 1065 **synthetic** patient-voice descriptions across 22 diagnoses. Synthetic is the right
trade here: the text reads like a real intake message, and no patient's words are in this repository.

The only ground truth this example claims is a **routing** label: `prepare.py` maps each diagnosis to the
queue a front desk would send it to. That mapping is coarse on purpose, and noisy where a diagnosis presents
across systems — a drug reaction is filed under `skin` because that is how it usually arrives, not because
that is what it is. The other three questions have no ground truth and are scored against the teacher, the
same way `urgency` is in `support_triage`.

## Build it

```bash
# 1. states (see data/README.md for the download)
python3 prepare.py states.jsonl sym_train.parquet sym_test.parquet
#    wrote 1060 states -- body_system: systemic=247, skin=246, respiratory=145, digestive=133,
#    cardiovascular=100, musculoskel=99, urinary=48, neuro=42

# 2. pi target
pip install -r ../../compiler/requirements.txt
j3v compile ../../schemas/symptom_intake.j3v --target pi \
  --encoder minilm.j3a --states states.jsonl -o symptom_intake.pi.j3a
```

## What to expect from the gate

1060 states is a small corpus — roughly 740 train / 105 calib / 215 test. That matters because the gate
checks **one-sided 95% bootstrap bounds, not point estimates**: on 215 test states, an agreement bound of
0.85 needs a point estimate near 0.89 to clear. Two of the eight `body_system` classes have fewer than 50
examples in total.

So this example is likely to *fail* its bounds on the first build, and the honest responses are to get more
states or to widen the classes — not to lower the bound until it passes. The schema is deliberately left at
the same `agreement >= 0.85` / `ece <= 0.05` as the others so that failure is visible rather than tuned away.

The one bound that should not be relaxed is on `red_flag`. It is the question that can shorten someone's
wait, its threshold is 0.97, and a false `false` is the error that matters.
