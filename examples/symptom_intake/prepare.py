"""gretelai/symptom_to_diagnosis -> J3v states JSONL with ground-truth labels for `body_system`.

The corpus is *synthetic* patient-voice symptom descriptions (Apache-2.0), each tagged with one of 22
diagnoses. Synthetic is the right trade for a public example: the text reads like a real intake message,
and no patient's words are in this repository.

Ground truth here is a **routing** label, not a diagnosis. Each diagnosis maps to the body system whose
queue the message belongs in (below), and that mapping is the only ground truth this example claims.
It is deliberately coarse, and it is noisy for diagnoses that present across systems -- a drug reaction is
filed under `skin` because that is how it usually arrives at a front desk, not because that is what it is.
The other three questions in the schema have no ground truth at all and are scored against the teacher.

    python3 prepare.py states.jsonl train.parquet test.parquet
"""
import json
import random
import sys

# diagnosis -> the queue a front desk would route it to.
SYSTEM = {
    "bronchial asthma": "respiratory", "common cold": "respiratory", "pneumonia": "respiratory",
    "gastroesophageal reflux disease": "digestive", "peptic ulcer disease": "digestive", "jaundice": "digestive",
    "arthritis": "musculoskel", "cervical spondylosis": "musculoskel",
    "chicken pox": "skin", "fungal infection": "skin", "impetigo": "skin", "psoriasis": "skin",
    "drug reaction": "skin",
    "migraine": "neuro",
    "dengue": "systemic", "malaria": "systemic", "typhoid": "systemic", "diabetes": "systemic",
    "allergy": "systemic",
    "urinary tract infection": "urinary",
    "hypertension": "cardiovascular", "varicose veins": "cardiovascular",
}


def main(dst, *srcs, seed=0):
    import pyarrow.parquet as pq

    rnd = random.Random(seed)
    rows, seen = [], set()
    for src in srcs:
        for r in pq.read_table(src).to_pylist():
            t = " ".join((r["input_text"] or "").split())
            dx = (r["output_text"] or "").strip().lower()
            if not t or t in seen or dx not in SYSTEM:
                continue
            seen.add(t)
            # `diagnosis` rides along as provenance for the label; the schema never asks for it, and no
            # artifact is compiled to answer it. Keeping it makes the routing label auditable.
            rows.append({"state": {"message": t}, "labels": {"body_system": SYSTEM[dx]}, "diagnosis": dx})

    rnd.shuffle(rows)
    with open(dst, "w") as f:
        for i, o in enumerate(rows):
            o["id"] = "symptom-%05d" % i
            f.write(json.dumps(o) + "\n")

    kinds = {}
    for o in rows:
        k = o["labels"]["body_system"]
        kinds[k] = kinds.get(k, 0) + 1
    print("wrote %d states to %s" % (len(rows), dst))
    print("body_system: " + ", ".join("%s=%d" % kv for kv in sorted(kinds.items(), key=lambda kv: -kv[1])))
    if len(rows) < 3000:
        print("note: %d states is a small corpus. The conformance gate checks one-sided 95%% bootstrap\n"
              "      bounds, so a thin test split needs a clearly higher point estimate to pass." % len(rows))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__.strip().splitlines()[-1].strip())
    main(sys.argv[1], *sys.argv[2:])
