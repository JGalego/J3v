"""Label states with the Laya teacher. Outputs raw (uncalibrated) per-option logits.

Laya cross-encodes every (state, question) pair, so the cost is N_states x N_questions encoder
passes. We batch by length and run bf16 on CPU (AMX) or fp16/bf16 on GPU.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

LAYA_REPO = "convaiinnovations/laya"
LAYA_REVISION = "5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b"  # pinned: Step 0 read this revision
SUBFOLDER = {"laya": "", "laya-multilingual": "multilingual"}
FILES = ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json",
         "tokenizer/tokenizer_config.json", "rl_common.py", "rl_agent_api.py"]


def fetch_teacher(name, cache):
    from huggingface_hub import hf_hub_download
    sub = SUBFOLDER[name]
    d = os.path.join(cache, name)
    for f in FILES:
        src = f if f.startswith("rl_") and f.endswith(".py") else (sub + "/" + f if sub else f)
        dst = os.path.join(d, f)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            p = hf_hub_download(LAYA_REPO, src, revision=LAYA_REVISION)
            os.symlink(p, dst)
    return d


def teacher_question(q, noul_mode):
    """Schema question -> Laya internal question. `noul_mode="choice"` asks noul as a neutral 2-way choice
    (Laya issue #156: the English checkpoint follows the `false:`/`true:` label text). Index 1 is always "true"."""
    if q["type"] == "choice":
        return {"t": "choice", "ins": q["instructions"], "crit": {o["key"]: o.get("description") for o in q["options"]}}
    if q["type"] == "score":
        return {"t": "score", "ins": q["instructions"], "crit": [o["description"] for o in q["options"]]}
    desc = {o["key"]: o.get("description") for o in q["options"]}
    if noul_mode == "choice":
        return {"t": "choice", "ins": q["instructions"],
                "crit": {"B": desc.get("false") or "no, the statement does not hold",
                         "A": desc.get("true") or "yes, the statement holds"}}
    return {"t": "noul", "ins": q["instructions"], "crit": {"false": desc.get("false"), "true": desc.get("true")}}


def label(schema, states, teacher_dir, noul_mode="native", max_tokens=24000, log=True):
    import torch
    os.environ.setdefault("USE_TF", "0")
    sys.path.insert(0, teacher_dir)
    from rl_agent_api import RLAgent
    from rl_common import QTYPES, build_sequence, collate_items

    agent = RLAgent(teacher_dir, device="cuda" if torch.cuda.is_available() else "cpu")
    cfg, tok, model, dev = agent.cfg, agent.tok, agent.model, agent.device
    dtype = torch.bfloat16 if dev.type == "cpu" or torch.cuda.is_bf16_supported() else torch.float16
    items = []
    for qi, q in enumerate(schema["questions"]):
        tq = teacher_question(q, noul_mode)
        for si, s in enumerate(states):
            ids, markers = build_sequence(tok, s, tq, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(q["options"]):
                raise SystemExit("question %r: options do not fit the teacher's head_max_len" % q["id"])
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[tq["t"]], "target": [0.0] * len(markers),
                          "label": -1, "episode": 0, "ep_step": 0, "ep_len": 1, "src": "", "qi": qi, "si": si})
    out = [np.zeros((len(states), len(q["options"])), np.float32) for q in schema["questions"]]
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    t0, i = time.time(), 0
    with torch.no_grad():
        while i < len(order):
            j = i
            while j < len(order) and len(items[order[j]]["ids"]) * (j - i + 1) <= max_tokens:
                j += 1
            j = max(j, i + 1)
            sel = [items[order[t]] for t in range(i, j)]
            b = collate_items([sel], tok.pad_token_id)
            with torch.autocast(device_type=dev.type, dtype=dtype):
                logits, _ = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                                  b["marker_mask"].to(dev), b["qtype"].to(dev))
            logits = logits.float().cpu().numpy()
            for r, it in enumerate(sel):
                k = len(it["markers"])
                out[it["qi"]][it["si"]] = logits[r, :k]  # noul-as-choice is declared {B: false, A: true}: index 1 = true
            i = j
            if log:
                el = time.time() - t0
                sys.stderr.write("\r  teacher: %d/%d sequences, %.1f seq/s  " % (i, len(order), i / el))
    if log:
        sys.stderr.write("\n")
    return out, cfg


def shipped_temperature(cfg, qtype, k, noul_mode):
    t = {"choice": 0, "score": 1, "noul": 2}[qtype]
    if qtype == "noul" and noul_mode == "choice":
        qtype, t = "choice", 0
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return cfg.get("temperature_by_options", {}).get("%s:%s" % (qtype, size), cfg["temperature"][t])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--states", required=True, help="JSONL with {id, state, labels?}")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True, help="JSON with raw teacher logits per question")
    ap.add_argument("--noul-mode", default="native", choices=["choice", "native"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    schema = json.load(open(a.schema))
    rows = [json.loads(l) for l in open(a.states)]
    if a.limit:
        rows = rows[:a.limit]
    tdir = fetch_teacher(schema.get("teacher", "laya"), a.cache)
    out, cfg = label(schema, [r["state"] for r in rows], tdir, a.noul_mode)
    temps = [shipped_temperature(cfg, q["type"], len(q["options"]), a.noul_mode) for q in schema["questions"]]
    json.dump({"teacher": schema.get("teacher", "laya"), "revision": LAYA_REVISION, "noul_mode": a.noul_mode,
               "ids": [r["id"] for r in rows], "shipped_temperature": temps,
               "logits": {q["id"]: out[i].round(5).tolist() for i, q in enumerate(schema["questions"])}}, open(a.out, "w"))


if __name__ == "__main__":
    main()
