"""Civil Comments -> J3v states JSONL with ground-truth labels.

Civil Comments (CC0-1.0) ships each comment with the *fraction of crowd raters* who marked it toxic,
obscene, threatening and so on. Those fractions are the ground truth here, and they are also why this
corpus is a good fit for a calibrated student: a comment 3 of 10 raters called an insult is genuinely a
0.3, not a mislabelled 0 or 1.

Labels are only emitted where the raters were decisive. A comment in the middle -- where raters split --
gets no label for that question rather than a coin-flip one, so `require accuracy` is measured against
cases a human panel agreed on. Those unlabelled states still train the student: the teacher labels every
state, and ground truth is only used for the accuracy gate.

    python3 prepare.py civil_comments.parquet states.jsonl 8000
"""
import json
import random
import re
import sys

# A rater fraction at or above this is a clear yes; at or below LO, a clear no. In between: no label.
HI, LO = 0.6, 0.1
# Civil Comments' sub-scores, in the order `violation` prefers them when more than one is decisive:
# a threat is the most consequential thing to get right, then identity attack, then obscenity, then insult.
SUBS = [("threat", "threat"), ("identity_attack", "identity_attack"), ("obscene", "profanity"), ("insult", "insult")]


def label_violation(r):
    """The `violation` key a human panel would agree on, or None if they were not decisive."""
    if r["toxicity"] <= LO and all(r[s] <= LO for s, _ in SUBS):
        return "none"
    for sub, key in SUBS:
        if r[sub] >= HI:
            return key
    return None  # toxic, but the panel did not agree on which rule it broke


def clean(t):
    t = re.sub(r"https?://\S+", "a link", t)       # URLs carry no signal here and eat the token budget
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main(src, dst, n=8000, seed=0):
    import pyarrow.parquet as pq

    rnd = random.Random(seed)
    cols = ["text", "toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
    tbl = pq.read_table(src, columns=cols)
    rows = tbl.to_pylist()

    # Civil Comments is ~92% benign. Sampling it raw would spend the whole budget on `none` and leave too
    # few threats to measure anything, so take every decisive violation and fill the rest with benign ones.
    viol, benign, seen = [], [], set()
    for r in rows:
        t = clean(r["text"] or "")
        if not (15 <= len(t) <= 1000) or t in seen:
            continue
        seen.add(t)
        v = label_violation(r)
        if v is None:
            continue
        (benign if v == "none" else viol).append((t, r, v))
        if len(viol) >= n // 2 and len(benign) >= n:
            break

    rnd.shuffle(viol)
    rnd.shuffle(benign)
    take_v = min(len(viol), n // 2)
    out = viol[:take_v] + benign[: n - take_v]
    rnd.shuffle(out)

    with open(dst, "w") as f:
        for i, (t, r, v) in enumerate(out):
            labels = {"violation": v}
            # Per-question ground truth, again only where the raters were decisive.
            for sub, key in (("obscene", "profanity"), ("threat", "threat")):
                if r[sub] >= HI:
                    labels[key] = "true"
                elif r[sub] <= LO:
                    labels[key] = "false"
            f.write(json.dumps({"id": "civil-%05d" % i, "state": {"comment": t}, "labels": labels}) + "\n")

    kinds = {}
    for _, _, v in out:
        kinds[v] = kinds.get(v, 0) + 1
    print("wrote %d states to %s" % (len(out), dst))
    print("violation: " + ", ".join("%s=%d" % kv for kv in sorted(kinds.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8000)
