"""Distill per-schema heads from teacher probabilities over the shared encoder's token states.

The token states are produced by the Rust runtime (`j3v embed`), i.e. by the exact int8 kernels the
artifact will run with, so there is no train/serve skew in the features.
"""
import argparse
import json
import math

import numpy as np

from . import artifact


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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    d = H.shape[-1]

    class Heads(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in ks])
            self.l1 = nn.ModuleList([nn.Linear(2 * d, hidden) for _ in ks])
            self.l2 = nn.ModuleList([nn.Linear(hidden, k) for k in ks])
            self.drop = nn.Dropout(0.1)

        def forward(self, h, m):
            mean = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
            out = []
            for a, l1, l2 in zip(self.a, self.l1, self.l2):
                s = (h @ a) / math.sqrt(d)
                w = torch.softmax(s.masked_fill(~m, -1e9), 1)
                f = torch.cat([(w[..., None] * h).sum(1), mean], -1)
                out.append(l2(F.gelu(l1(self.drop(f)), approximate="tanh")))
            return out

    Ht, Mt = torch.from_numpy(H), torch.from_numpy(M)
    P = [torch.from_numpy(np.asarray(p, np.float32)) for p in targets]
    net = Heads()
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
    tensors = []
    for j in range(len(ks)):
        tensors += [("q%d.a" % j, net.a[j].detach().numpy()), ("q%d.w1" % j, net.l1[j].weight.detach().numpy()),
                    ("q%d.b1" % j, net.l1[j].bias.detach().numpy()), ("q%d.w2" % j, net.l2[j].weight.detach().numpy()),
                    ("q%d.b2" % j, net.l2[j].bias.detach().numpy())]
    return tensors, best


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
