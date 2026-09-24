"""Distill per-schema heads from teacher probabilities over the shared encoder's token states.

The token states are produced by the Rust runtime (`j3v embed`), i.e. by the exact int8 kernels the
artifact will run with, so there is no train/serve skew in the features.
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import artifact


class Heads(nn.Module):
    """Attention-pool + mean-pool + 2-layer MLP per question, over the shared encoder's token states.

    Mirrors `Head::forward` in crates/j3v/src/heads.rs exactly, so the same weights score identically
    whether run here (fp32, training/eval) or in the Rust runtime (int8, quantized).
    """

    def __init__(self, d, ks, hidden):
        super().__init__()
        self.d = d
        self.a = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in ks])
        self.l1 = nn.ModuleList([nn.Linear(2 * d, hidden) for _ in ks])
        self.l2 = nn.ModuleList([nn.Linear(hidden, k) for k in ks])
        self.drop = nn.Dropout(0.1)

    def forward(self, h, m):
        mean = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        out = []
        for a, l1, l2 in zip(self.a, self.l1, self.l2):
            s = (h @ a) / math.sqrt(self.d)
            w = torch.softmax(s.masked_fill(~m, -1e9), 1)
            f = torch.cat([(w[..., None] * h).sum(1), mean], -1)
            out.append(l2(F.gelu(l1(self.drop(f)), approximate="tanh")))
        return out

    def tensors(self):
        """Export as (name, np.ndarray) pairs in the `.j3a` layout `heads::quantize` expects."""
        out = []
        for j in range(len(self.a)):
            out += [("q%d.a" % j, self.a[j].detach().numpy()), ("q%d.w1" % j, self.l1[j].weight.detach().numpy()),
                    ("q%d.b1" % j, self.l1[j].bias.detach().numpy()), ("q%d.w2" % j, self.l2[j].weight.detach().numpy()),
                    ("q%d.b2" % j, self.l2[j].bias.detach().numpy())]
        return out

    @classmethod
    def from_tensors(cls, tensors):
        """Load weights previously written by `tensors()`, e.g. from a `pi-heads` draft artifact."""
        ks, j = [], 0
        while ("q%d.w2" % j) in tensors:
            ks.append(tensors["q%d.w2" % j].shape[0])
            j += 1
        d, hidden = tensors["q0.a"].shape[0], tensors["q0.w1"].shape[0]
        net = cls(d, ks, hidden)
        with torch.no_grad():
            for j in range(len(ks)):
                net.a[j].copy_(torch.from_numpy(tensors["q%d.a" % j]))
                net.l1[j].weight.copy_(torch.from_numpy(tensors["q%d.w1" % j]))
                net.l1[j].bias.copy_(torch.from_numpy(tensors["q%d.b1" % j]))
                net.l2[j].weight.copy_(torch.from_numpy(tensors["q%d.w2" % j]))
                net.l2[j].bias.copy_(torch.from_numpy(tensors["q%d.b2" % j]))
        net.eval()
        return net


def load_feats(prefix):
    meta = json.load(open(prefix + ".json"))
    d, lens = meta["d"], meta["lens"]
    flat = np.fromfile(prefix + ".f32", dtype=np.float32).reshape(-1, d)
    T = max(lens)
    H = np.zeros((len(lens), T, d), np.float32)
    M = np.zeros((len(lens), T), bool)
    o = 0
    for i, L in enumerate(lens):
        H[i, :L], M[i, :L] = flat[o:o + L], True
        o += L
    return H, M


def train(H, M, targets, ks, train_idx, calib_idx, hidden=128, epochs=60, lr=2e-3, seed=0, log=True):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    d = H.shape[-1]

    Ht, Mt = torch.from_numpy(H), torch.from_numpy(M)
    P = [torch.from_numpy(np.asarray(p, np.float32)) for p in targets]
    net = Heads(d, ks, hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def loss_on(idx):
        z = net(Ht[idx], Mt[idx])
        return sum(-(p[idx] * F.log_softmax(zz, -1)).sum(-1).mean() for p, zz in zip(P, z))

    best, best_state = float("inf"), None
    tr = np.array(train_idx)
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        net.train()
        rng.shuffle(tr)
        for b in range(0, len(tr), 128):
            opt.zero_grad()
            loss_on(torch.from_numpy(tr[b:b + 128])).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            cl = float(loss_on(torch.tensor(calib_idx)))
        if cl < best:
            best, best_state = cl, {k: v.clone() for k, v in net.state_dict().items()}
        if log and (ep % 10 == 0 or ep == epochs - 1):
            print("  epoch %d calib soft-CE %.4f" % (ep, cl), flush=True)
    net.load_state_dict(best_state)
    return net.tensors(), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    t = json.load(open(a.targets))
    H, M = load_feats(a.feats)
    ks = [q["k"] for q in t["questions"]]
    tensors, best = train(H, M, [t["probs"][q["id"]] for q in t["questions"]], ks, t["train"], t["calib"], a.hidden, a.epochs)
    artifact.write(a.out, "pi-heads", {"hidden": a.hidden, "calib_soft_ce": best}, tensors)


if __name__ == "__main__":
    main()
