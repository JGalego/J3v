"""Distill one tiny per-schema model for the mcu tier: hashed n-gram embedding bag -> ReLU MLP -> heads.

Feature hashes come from the Rust j3v-mcu crate (`features()`, dumped by the compiler), so training sees
exactly the ids the firmware computes. Weights are exported as f32; the Rust side quantizes to int8 and
certifies the quantized model.
"""
import argparse
import json

import numpy as np

from . import artifact


def train(hashes, targets, ks, train_idx, calib_idx, buckets, dim, hidden, epochs=40, lr=3e-3, seed=0):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    mask = buckets - 1
    ids = [torch.tensor([h & mask for h in hs], dtype=torch.long) for hs in hashes]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.EmbeddingBag(buckets, dim, mode="mean")
            nn.init.normal_(self.emb.weight, 0, 0.1)
            self.l1 = nn.Linear(dim, hidden)
            self.heads = nn.ModuleList([nn.Linear(hidden, k) for k in ks])

        def forward(self, flat, offsets):
            x = self.emb(flat, offsets)
            h = F.relu(self.l1(x))
            return [hd(h) for hd in self.heads]

    def batch(ix):
        seqs = [ids[i] for i in ix]
        offs = torch.tensor([0] + [len(s) for s in seqs[:-1]]).cumsum(0)
        return torch.cat(seqs) if seqs else torch.zeros(0, dtype=torch.long), offs

    P = [torch.from_numpy(np.asarray(p, np.float32)) for p in targets]
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def loss_on(ix):
        z = net(*batch(ix))
        t = torch.tensor(ix)
        return sum(-(p[t] * F.log_softmax(zz, -1)).sum(-1).mean() for p, zz in zip(P, z))

    tr = np.array(train_idx)
    rng = np.random.RandomState(seed)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        net.train()
        rng.shuffle(tr)
        for b in range(0, len(tr), 64):
            opt.zero_grad()
            loss_on(list(tr[b:b + 64])).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            cl = float(loss_on(list(calib_idx)))
        if cl < best:
            best, best_state = cl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    t = [("emb", net.emb.weight.detach().numpy()), ("w1", net.l1.weight.detach().numpy()), ("b1", net.l1.bias.detach().numpy())]
    for j, hd in enumerate(net.heads):
        t += [("q%d.w" % j, hd.weight.detach().numpy()), ("q%d.b" % j, hd.bias.detach().numpy())]
    return t, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--candidates", required=True, help="JSON list of {buckets, dim, hidden, out}")
    a = ap.parse_args()
    hashes = json.load(open(a.feats))["hashes"]
    t = json.load(open(a.targets))
    ks = [q["k"] for q in t["questions"]]
    for c in json.loads(a.candidates):
        tensors, best = train(hashes, [t["probs"][q["id"]] for q in t["questions"]], ks, t["train"], t["calib"],
                              c["buckets"], c["dim"], c["hidden"])
        print("  mcu candidate buckets=%d dim=%d hidden=%d: calib soft-CE %.4f" % (c["buckets"], c["dim"], c["hidden"], best), flush=True)
        artifact.write(c["out"], "mcu-draft", {"calib_soft_ce": best, **{k: c[k] for k in ("buckets", "dim", "hidden")}}, tensors)


if __name__ == "__main__":
    main()
