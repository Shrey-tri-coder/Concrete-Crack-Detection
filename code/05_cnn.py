"""Step 5 - Method 3: Convolutional Neural Network trained from scratch (PyTorch, CPU).

* Input: 128x128 grayscale tiles (cached by 01_preprocess.py), rescaled as (x/255 - 0.5)/0.25 (see cnn_model.py).
* Class imbalance: every epoch uses ALL cracked training tiles + a fresh random draw of
  2x as many uncracked tiles (balanced re-sampling), with pos_weight=2 in the BCE loss.
* Augmentation: random horizontal/vertical flips, 90-degree rotations, brightness/contrast jitter.
* Optimiser: AdamW + OneCycle learning-rate schedule; model selected on validation PR-AUC.
* Decision threshold tuned on the validation set (max F1), then applied once to the test set.
"""
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from cnn_model import CrackCNN, to_tensor
from config import PROC_DIR, CACHE_DIR, RES_DIR, IMG_SIZE, SEED

EPOCHS, BATCH, NEG_RATIO, LR = 12, 128, 2, 2e-3
torch.manual_seed(SEED)
torch.set_flush_denormal(True)      # avoid very slow denormal-float arithmetic on CPU
rng = np.random.default_rng(SEED)
CKPT = RES_DIR / "cnn_checkpoint.pt"


def augment(x):
    if rng.random() < 0.5:
        x = x.flip(3)
    if rng.random() < 0.5:
        x = x.flip(2)
    k = int(rng.integers(4))
    if k:
        x = torch.rot90(x, k, (2, 3))
    a = torch.empty(x.size(0), 1, 1, 1).uniform_(0.8, 1.2)
    b = torch.empty(x.size(0), 1, 1, 1).uniform_(-0.3, 0.3)
    return x * a + b


@torch.no_grad()
def predict(model, imgs, bs=512):
    model.eval()
    out = [torch.sigmoid(model(to_tensor(imgs[i:i + bs]))).numpy() for i in range(0, len(imgs), bs)]
    return np.concatenate(out)


def main():
    meta = pd.read_csv(PROC_DIR / "metadata.csv")
    imgs = np.load(CACHE_DIR / f"images_{IMG_SIZE}.npy")
    idx = {s: np.where(meta.split == s)[0] for s in ("train", "val", "test")}
    y = meta.label.values
    pos, neg = idx["train"][y[idx["train"]] == 1], idx["train"][y[idx["train"]] == 0]

    model = CrackCNN()
    steps_per_epoch = int(np.ceil(len(pos) * (1 + NEG_RATIO) / BATCH))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)
    pos_w = torch.tensor(float(NEG_RATIO))

    history, best_ap, start = [], -1, 1
    if CKPT.exists():                   # resume after an interruption
        ck = torch.load(CKPT, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        history, best_ap, start = ck["history"], ck["best_ap"], ck["epoch"] + 1
        rng.bit_generator.state = ck["rng"]
        print(f"resumed from epoch {ck['epoch']}", flush=True)
    for ep in range(start, EPOCHS + 1):
        t0 = time.time()
        model.train()
        ep_idx = np.concatenate([pos, rng.choice(neg, len(pos) * NEG_RATIO, replace=False)])
        rng.shuffle(ep_idx)
        losses = []
        for i in range(0, len(ep_idx), BATCH):
            b = np.sort(ep_idx[i:i + BATCH])
            xb = augment(to_tensor(imgs[b]))
            yb = torch.from_numpy(y[b]).float()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            if len(losses) % 40 == 0:
                print(f"  ep {ep} batch {len(losses)}/{steps_per_epoch} loss {np.mean(losses[-40:]):.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        pv = predict(model, imgs[idx["val"]])
        yv = y[idx["val"]]
        ap, auc = average_precision_score(yv, pv), roc_auc_score(yv, pv)
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_pr_auc": ap,
                        "val_roc_auc": auc, "seconds": time.time() - t0})
        print(f"epoch {ep:2d} loss {np.mean(losses):.4f} val PR-AUC {ap:.4f} ROC-AUC {auc:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if ap > best_ap:
            best_ap = ap
            torch.save(model.state_dict(), RES_DIR / "cnn_best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "history": history, "best_ap": best_ap, "epoch": ep, "rng": rng.bit_generator.state}, CKPT)

    model.load_state_dict(torch.load(RES_DIR / "cnn_best.pt"))
    pv, pt = predict(model, imgs[idx["val"]]), predict(model, imgs[idx["test"]])
    p, r, t = precision_recall_curve(y[idx["val"]], pv)
    f1 = 2 * p * r / (p + r + 1e-12)
    thr = float(t[np.argmax(f1[:-1])])
    pd.DataFrame({"idx": idx["val"], "prob": pv}).to_csv(RES_DIR / "cnn_val_probs.csv", index=False)
    pd.DataFrame({"idx": idx["test"], "prob": pt}).to_csv(RES_DIR / "cnn_test_probs.csv", index=False)
    json.dump({"threshold": thr, "history": history}, open(RES_DIR / "cnn_history.json", "w"), indent=2)
    print(f"best val PR-AUC {best_ap:.4f}; threshold {thr:.3f}; test PR-AUC "
          f"{average_precision_score(y[idx['test']], pt):.4f}")


if __name__ == "__main__":
    main()
