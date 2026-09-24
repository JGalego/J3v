# `content_moderation`

A second domain for the whole pipeline: moderating comments on a news site's discussion section.
[`schemas/content_moderation.j3v`](../../schemas/content_moderation.j3v) asks four questions —
which guideline a comment breaks, how severe it is, whether it contains obscenity, and whether it
threatens harm.

**The artifacts are not committed.** The schema, the data prep and the build are here; the `.j3a` files are
not, because an artifact in this repository carries a conformance report, and one that has not been built
has not earned it. Building it takes a Laya pass over the states — see below.

## Why this corpus

[Civil Comments](https://huggingface.co/datasets/google/civil_comments) (CC0-1.0) is a good fit for a
*calibrated* student, not just an accurate one: every comment carries the fraction of crowd raters who
marked it toxic, obscene, threatening and so on. A comment 3 of 10 raters called an insult is genuinely a
0.3. `prepare.py` keeps ground truth only where the raters were decisive (≥ 0.6 or ≤ 0.1) and leaves the
middle unlabelled, so `require accuracy` is measured against cases a human panel actually agreed on.

## Build it

```bash
# 1. states (see data/README.md for the download)
python3 prepare.py civil_comments.parquet states.jsonl 8000
#    wrote 8000 states -- violation: none=4000, insult=3088, profanity=491, identity_attack=248, threat=173

# 2. pi target: shared int8 encoder + distilled heads
pip install -r ../../compiler/requirements.txt
j3v compile ../../schemas/content_moderation.j3v --target pi \
  --encoder minilm.j3a --states states.jsonl -o content_moderation.pi.j3a

# 3. mcu target: one int8 hashed-n-gram model under a flash budget
j3v compile ../../schemas/content_moderation.j3v --target mcu --budget 512KB \
  --states states.jsonl -o content_moderation.mcu.j3a
```

## What to expect from the gate

The class balance above is the interesting part, and it is *real* — Civil Comments is ~92% benign, and
`prepare.py` already oversamples violations to half the file. Even so, `threat` lands around 2% of states.

- A rare class does not hurt **ECE** much — most of the probability mass is on the classes that are common,
  and that is exactly what a marginal calibration bound measures.
- It does hurt the **bootstrap lower bound on accuracy**, because the interval on a class with ~35 test
  examples is wide. If `violation` misses `require agreement >= 0.85`, read which class is dragging it
  before loosening the bound.
- `threat` is a `noul`, so it is scored over every state rather than over its own class — a much easier
  target, and the reason the schema sets `threshold threat 0.95`: a missed threat is the error that matters,
  so it escalates unless the artifact is nearly certain.

If a bound fails, the compiler refuses to emit the artifact and names the question, the metric, its bootstrap
bound and the requirement (`error[E0301]`). That is the design: the schema's numbers are a contract, and a
build that cannot meet them should not ship.
