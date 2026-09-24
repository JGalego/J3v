"""Measure the accuracy/ECE cost of the shared encoder's int8 quantization, holding the trained
heads fixed.

`j3v compile --target pi` trains per-schema heads directly on the int8 Rust encoder's token states
(see distill.py's docstring: "no train/serve skew in the features"). This script re-runs those same
trained heads on token states from the original fp32 HuggingFace checkpoint instead, for the same
states, and compares accuracy/ECE against the int8 features already cached by `j3v compile` in its
build directory.

Caveat: the heads were *trained* on int8 features, so scoring them on fp32 features is a mild
train/eval mismatch, not a fully independent control -- a from-scratch fp32-feature head would be
the fair-fight version. Treat this as a bound on the encoder's contribution, not an exact isolation.
For the *heads'* own quantization cost (weights only, exact and free of that caveat), see the
"quantization" section `j3v compile` now writes to conformance.pi.json.

Usage (after `j3v compile --target pi ...` for the schema, using its build directory):

    python -m j3vc.encoder_quant_effect --schema build/support_triage/schema.json \\
        --states states.jsonl --build-dir build/support_triage
"""
import argparse
import glob
import json
import os

import numpy as np

from . import artifact
from .distill import Heads, load_feats


def fnv1a64(b):
    h = 0xCBF29CE484222325
    for x in b:
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def split_of(row_id):
    """Mirrors crates/j3v/src/compile.rs::split_of: 70/10/20 train/calib/test by id hash."""
    b = fnv1a64(row_id.encode("utf-8")) % 10
    return "train" if b <= 6 else "calib" if b == 7 else "test"


def state_text(state):
    """Mirrors crates/j3v-core/src/schema.rs::state_text."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(
            "%s: %s" % (k, v if isinstance(v, str) else json.dumps(v, separators=(",", ":"))) for k, v in state.items()
        )
    return json.dumps(state, separators=(",", ":"))


def ece(conf, correct, bins=15):
    """Mirrors crates/j3v-core/src/metrics.rs::ece (15-bin equal-width, on top-1 confidence)."""
    conf, correct = np.asarray(conf, np.float64), np.asarray(correct, bool)
    idx = np.clip(np.ceil(conf * bins).astype(int), 1, bins) - 1
    total, n = 0.0, len(conf)
    for b in range(bins):
        m = idx == b
        if m.any():
            total += (m.sum() / n) * abs(conf[m].mean() - correct[m].mean())
    return total


def fit_temperature(logits, labels):
    """Golden-section search minimising NLL over T (NLL convex in beta = 1/T). Mirrors
    crates/j3v-core/src/metrics.rs::fit_temperature so both sides calibrate the same way."""
    logits, labels = np.asarray(logits, np.float64), np.asarray(labels, np.int64)

    def nll(beta):
        z = beta * logits
        m = z.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(z - m).sum(1))
        return float(np.mean(lse - z[np.arange(len(labels)), labels]))

    a, b = 0.01, 20.0
    g = (5**0.5 - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(d)
    return 2.0 / (a + b)


def softmax(z, t):
    z = np.asarray(z, np.float64) / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def pad(hiddens):
    """Ragged per-row [t_i, d] arrays -> padded (H[n,T,d], M[n,T] bool). Mirrors distill.load_feats."""
    T = max(h.shape[0] for h in hiddens)
    d = hiddens[0].shape[1]
    H = np.zeros((len(hiddens), T, d), np.float32)
    M = np.zeros((len(hiddens), T), bool)
    for i, h in enumerate(hiddens):
        H[i, : h.shape[0]] = h
        M[i, : h.shape[0]] = True
    return H, M


def hf_features(source, texts, max_len, batch_size=32):
    """Per-row [t_i, d] fp32 hidden states from the original HuggingFace checkpoint, unpadded."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            h = model(**enc).last_hidden_state.numpy()
            mask = enc["attention_mask"].numpy().astype(bool)
            out.extend(h[r, mask[r]] for r in range(len(batch)))
    return out


def evaluate(net, H, M, calib_idx, test_idx, gt, teacher_probs):
    """Fit per-question temperature on calib, score on test. Mirrors compile.rs::certify, minus the
    bootstrap CIs and the mcu/pi-specific bookkeeping -- this is a lighter, encoder-focused cut."""
    import torch

    with torch.no_grad():
        z = [zz.numpy().astype(np.float64) for zz in net(torch.from_numpy(H), torch.from_numpy(M))]
    report = []
    for j, zj in enumerate(z):
        use_gt = all(gt[j][i] is not None for i in calib_idx) and all(gt[j][i] is not None for i in test_idx)
        target = (lambda i: gt[j][i]) if use_gt else (lambda i: int(np.argmax(teacher_probs[j][i])))
        yc = np.array([target(i) for i in calib_idx])
        t = fit_temperature(zj[calib_idx], yc)
        p = softmax(zj[test_idx], t)
        pred = p.argmax(1)
        conf = p[np.arange(len(test_idx)), pred]
        yt = np.array([target(i) for i in test_idx])
        teacher_pred = np.array([int(np.argmax(teacher_probs[j][i])) for i in test_idx])
        report.append({
            "temperature": t,
            "accuracy": float(np.mean(pred == yt)),
            "agreement_with_teacher": float(np.mean(pred == teacher_pred)),
            "ece": ece(conf, pred == yt),
            "pred": pred.tolist(),
        })
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", required=True, help="build/<name>/schema.json written by `j3v compile`")
    ap.add_argument("--states", required=True)
    ap.add_argument("--build-dir", required=True, help="build/<name> directory from `j3v compile --target pi`")
    ap.add_argument("--hf", default="sentence-transformers/all-MiniLM-L6-v2", help="fp32 HF source of the shared encoder")
    a = ap.parse_args()

    schema = json.load(open(a.schema))
    rows = [json.loads(l) for l in open(a.states) if l.strip()]
    ids = [r.get("id", "row-%d" % i) for i, r in enumerate(rows)]
    calib_idx = [i for i, rid in enumerate(ids) if split_of(rid) == "calib"]
    test_idx = [i for i, rid in enumerate(ids) if split_of(rid) == "test"]

    targets = json.load(open(os.path.join(a.build_dir, "targets.json")))
    teacher_probs = [targets["probs"][q["id"]] for q in targets["questions"]]
    gt = [[None] * len(rows) for _ in targets["questions"]]
    for j, q in enumerate(schema["questions"]):
        opts = [o["key"] for o in q["options"]]
        for i, r in enumerate(rows):
            key = r.get("labels", {}).get(q["id"])
            if key in opts:
                gt[j][i] = opts.index(key)

    feats_json = glob.glob(os.path.join(a.build_dir, "feats-*.json"))
    if len(feats_json) != 1:
        raise SystemExit(
            "expected exactly one feats-*.json in %s (found %d) -- re-run `j3v compile` for this schema first"
            % (a.build_dir, len(feats_json))
        )
    H_int8, M_int8 = load_feats(feats_json[0][: -len(".json")])

    kind, _meta, tensors = artifact.read(os.path.join(a.build_dir, "heads.draft.j3a"))
    if kind != "pi-heads":
        raise SystemExit("expected a pi-heads draft artifact, got %r" % kind)
    net = Heads.from_tensors(tensors)

    print("[j3v] scoring heads on int8 (cached) encoder features...")
    r_int8 = evaluate(net, H_int8, M_int8, calib_idx, test_idx, gt, teacher_probs)

    max_len = schema["max_state_tokens"]
    texts = [state_text(r["state"]) for r in rows]
    print("[j3v] computing fp32 features from `%s` for %d states..." % (a.hf, len(rows)))
    H_fp32, M_fp32 = pad(hf_features(a.hf, texts, max_len))
    print("[j3v] scoring heads on fp32 encoder features...")
    r_fp32 = evaluate(net, H_fp32, M_fp32, calib_idx, test_idx, gt, teacher_probs)

    out = {"schema": schema["name"], "n_test": len(test_idx), "hf_encoder": a.hf, "questions": []}
    for q, i8, f32 in zip(schema["questions"], r_int8, r_fp32):
        pred_agreement = float(np.mean(np.array(i8["pred"]) == np.array(f32["pred"])))
        out["questions"].append({
            "id": q["id"],
            "int8_encoder": {k: v for k, v in i8.items() if k != "pred"},
            "fp32_encoder": {k: v for k, v in f32.items() if k != "pred"},
            "accuracy_delta": i8["accuracy"] - f32["accuracy"],
            "ece_delta": i8["ece"] - f32["ece"],
            "pred_agreement_int8_vs_fp32": pred_agreement,
        })
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
