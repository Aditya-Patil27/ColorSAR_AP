"""
ColorSAR - colorizer training (Models 1 and 3 share this one script)

    Model 1 : L of the optical (s2) image  -> a/b of the same image
    Model 3 : the SAR (s1) tile (3 despeckling levels) -> L AND a/b of the s2 image

Network: ImageNet ResNet-50, fine-tuned at a lower learning rate than the U-Net
decoder (encoder BatchNorm statistics frozen), predicting at the full 256x256 input
resolution (model_architecture.py). a/b: class-rebalanced color-bin classification;
L (Model 3): L1. Mixed precision on CUDA. Train_model1.py / Train_model3.py call this
with each model's defaults; the Kaggle kernel runs them.

Splits: whole scenes (splits.scene_split). The test scenes are never touched here;
--n_images / --n_val only subsample the train / val portions.

Outputs (in --out):
    model_<k>.pth           best checkpoint (whole network + meta, incl. the split's scenes)
    split.json              the scenes of each part and the exact train / val pairs used
    history.json            per-epoch losses, val color error (a/b RMSE) and learning rate
    loss_curve.png          loss / val error progression
    sample_predictions.png  input / ground truth / prediction on val images

The best epoch is chosen by validation a/b RMSE in Lab units (plus L RMSE when L is
predicted) - a metric that means the same thing whatever the training loss is.
"""
import argparse
import json
import os
import time

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from color_utils import PairDataset, ab_to_lab_units, find_pairs, lab_to_rgb, rgb_to_L
from model_architecture import Colorizer, load_colorizer, out_channels, save_checkpoint
from pipeline import (INPUT_OF_MODEL, bin_loss, class_weights, compute_bins, decode_output,
                      model_input)
from splits import scene_split, split_record, subsample, write_split

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DEFAULTS = {
    1: dict(epochs=20, out="model1_run", predict_L=False),
    3: dict(epochs=30, out="model3_run", predict_L=True),
}


def build_parser(model):
    d = DEFAULTS[model]
    ap = argparse.ArgumentParser(description=f"Train ColorSAR Model {model}")
    ap.add_argument("--root", default="v_2")
    ap.add_argument("--n_images", type=int, default=0,
                    help="training pairs to use (0 = every pair of the training scenes)")
    ap.add_argument("--n_val", type=int, default=0,
                    help="validation pairs to use (0 = every pair of the validation scenes)")
    ap.add_argument("--epochs", type=int, default=d["epochs"])
    ap.add_argument("--batch_size", type=int, default=32,
                    help="total batch (split across GPUs with DataParallel)")
    ap.add_argument("--lr", type=float, default=5e-4, help="decoder learning rate")
    ap.add_argument("--encoder_lr_mult", type=float, default=0.1,
                    help="encoder learning rate = lr * this (0 = frozen encoder)")
    ap.add_argument("--bin_grid", type=float, default=10.0, help="color bin size in Lab units")
    ap.add_argument("--T", type=float, default=0.38, help="annealed-mean temperature")
    ap.add_argument("--L_weight", type=float, default=1.0, help="weight of the L1 loss on L")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=d["out"])
    for flag, default in (("predict_L", d["predict_L"]), ("augment", True), ("amp", True)):
        ap.add_argument(f"--{flag}", dest=flag, action="store_true")
        ap.add_argument(f"--no_{flag}", dest=flag, action="store_false")
        ap.set_defaults(**{flag: default})
    return ap


def augment(gray, rgb, ab):
    """Same random rot90 + flip for the whole batch (cheap, on device)."""
    k = int(torch.randint(4, (1,)))
    flip = bool(torch.randint(2, (1,)))
    out = []
    for x in (gray, rgb, ab):
        x = torch.rot90(x, k, dims=(2, 3))
        out.append(torch.flip(x, dims=(3,)) if flip else x)
    return out


class Objective:
    """Training loss + a loss-independent validation error, for one model config."""

    def __init__(self, meta, centers, weights, L_weight=1.0):
        self.meta, self.centers, self.weights, self.L_weight = meta, centers, weights, L_weight

    def loss(self, raw, ab, L_true):
        raw = raw.float()
        if self.meta["predict_L"]:
            loss_L = (torch.tanh(raw[:, :1]) - L_true).abs().mean()
            logits = raw[:, 1:]
        else:
            loss_L, logits = 0.0, raw
        return bin_loss(logits, ab, self.centers, self.weights) + self.L_weight * loss_L

    @torch.no_grad()
    def errors(self, raw, ab, L_true, chunk=8):
        """Per-batch sums of squared a/b and L errors in Lab units (decoded output)."""
        se_ab = se_L = 0.0
        for i in range(0, raw.shape[0], chunk):
            L_p, ab_p = decode_output(raw[i:i + chunk], self.meta, self.centers)
            d = ab_to_lab_units(ab_p) - ab_to_lab_units(ab[i:i + chunk])
            se_ab += d.pow(2).mean(dim=(1, 2, 3)).sum().item()
            if L_p is not None:
                se_L += ((L_p - L_true[i:i + chunk]) * 50).pow(2).mean(dim=(1, 2, 3)).sum().item()
        return se_ab, se_L


def run_epoch(loader, net, parallel, meta, obj, device, optimizer=None, scaler=None, aug=False,
              train_encoder=True, amp=False):
    training = optimizer is not None
    net.set_mode(training, train_encoder)
    tot, n_batches, se_ab, se_L, n = 0.0, 0, 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for gray, rgb, ab, _ in loader:
            gray, rgb, ab = (t.to(device, non_blocking=True) for t in (gray, rgb, ab))
            if training and aug:
                gray, rgb, ab = augment(gray, rgb, ab)
            x, _ = model_input(meta, gray=gray, rgb=rgb)
            L_true = rgb_to_L(rgb) if meta["predict_L"] else None
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                raw = parallel(x)
            loss = obj.loss(raw, ab, L_true)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                a, b = obj.errors(raw, ab, L_true)
                se_ab += a; se_L += b; n += raw.shape[0]
            tot += loss.item(); n_batches += 1
    loss = tot / max(n_batches, 1)
    if training:
        return loss, None, None
    return loss, (se_ab / max(n, 1)) ** 0.5, (se_L / max(n, 1)) ** 0.5


@torch.no_grad()
def save_samples(loader, net, meta, device, path, n=4):
    net.eval()
    gray, rgb, ab, _ = next(iter(loader))
    gray, rgb = gray[:n].to(device), rgb[:n].to(device)
    x, L_given = model_input(meta, gray=gray, rgb=rgb)
    L_pred, ab_pred = decode_output(net(x), meta)
    pred_rgb = lab_to_rgb(L_pred if L_pred is not None else L_given, ab_pred)
    gt = rgb.permute(0, 2, 3, 1).cpu().numpy()
    if meta["input"] == "sar3":
        shown, label = gray[:, 0].cpu().numpy(), "SAR (s1)"
    else:
        shown, label = ((L_given[:, 0] + 1) / 2).cpu().numpy(), "optical L"
    n = len(gt)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(shown[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Input ({label})")
        axes[1, i].imshow(gt[i]); axes[1, i].set_title("Ground truth (s2)")
        axes[2, i].imshow(pred_rgb[i]); axes[2, i].set_title(f"Model {meta['model']} output")
        for r in range(3):
            axes[r, i].axis("off")
    plt.suptitle(f"Model {meta['model']} - validation scenes (not used for training)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def main(model, argv=None):
    args = build_parser(model).parse_args(argv)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"
    print(f"Device: {device}  |  GPUs: {torch.cuda.device_count()}  |  AMP: {amp}")
    print("Config:", {k: v for k, v in vars(args).items() if k != "root"})

    pairs = find_pairs(args.root)
    assert pairs, f"No s1/s2 pairs found under {args.root}"
    split = scene_split(pairs)
    train_pairs = subsample(split["train"], args.n_images, args.seed)
    val_pairs = subsample(split["val"], args.n_val, args.seed)
    record = split_record(split)
    print(f"{len(pairs)} pairs  ->  train {len(train_pairs)}  val {len(val_pairs)}  "
          f"(test {len(split['test'])} held out, never touched here)")
    print(f"scenes: train {len(record['train'])}  val {record['val']}  test {record['test']}")
    write_split(os.path.join(args.out, "split.json"), split,
                used={"train": train_pairs, "val": val_pairs})

    input_kind = INPUT_OF_MODEL[model]
    load_sar = input_kind == "sar3"
    kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda",
              persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(PairDataset(train_pairs, load_sar=load_sar), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(PairDataset(val_pairs, load_sar=load_sar), batch_size=args.batch_size,
                            shuffle=False, **kw)

    # ---- color bins from the training scenes
    ds = PairDataset(subsample(train_pairs, 1000, seed=1), load_sar=False)
    ab_s = torch.cat([ds[i][2][:, ::4, ::4].reshape(2, -1).T for i in range(len(ds))])
    centers, prior = compute_bins(ab_s, grid=args.bin_grid)
    weights = class_weights(centers, prior)
    centers, weights = centers.to(device), weights.to(device)
    print(f"Color bins: {len(centers)} occupied {args.bin_grid:g}-unit cells "
          f"(weights {weights.min():.2f}..{weights.max():.2f})")

    train_encoder = args.encoder_lr_mult > 0
    meta = {"model": model, "input": input_kind, "predict_L": args.predict_L,
            "bins": centers.cpu().tolist(), "T": args.T, "split": record,
            "augment": args.augment, "epochs": args.epochs, "lr": args.lr,
            "encoder_lr_mult": args.encoder_lr_mult, "batch_size": args.batch_size,
            "amp": amp, "n_train": len(train_pairs)}
    obj = Objective(meta, centers, weights, args.L_weight)

    net = Colorizer(out_channels(meta), pretrained=True).to(device)
    for p in net.encoder.parameters():
        p.requires_grad = train_encoder
    # kept outside `net` (assigning it as an attribute would register a module cycle)
    parallel = nn.DataParallel(net) if torch.cuda.device_count() > 1 else net

    groups = [{"params": list(net.decoder.parameters()), "lr": args.lr}]
    if train_encoder:
        groups.append({"params": list(net.encoder.parameters()),
                       "lr": args.lr * args.encoder_lr_mult})
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"Trainable parameters: {n_train:,}  (encoder {'fine-tuned' if train_encoder else 'frozen'})")
    optimizer = optim.AdamW(groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                     eta_min=args.lr * 0.02)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    ckpt = os.path.join(args.out, f"model_{model}.pth")
    hist = {k: [] for k in ("train_loss", "val_loss", "val_ab_rmse", "val_L_rmse", "val_score", "lr")}
    best, t0 = float("inf"), time.time()
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tr, _, _ = run_epoch(train_loader, net, parallel, meta, obj, device, optimizer, scaler,
                             aug=args.augment, train_encoder=train_encoder, amp=amp)
        va, ab_rmse, L_rmse = run_epoch(val_loader, net, parallel, meta, obj, device, amp=amp)
        scheduler.step()
        score = ab_rmse + (L_rmse if args.predict_L else 0.0)
        for k, v in zip(hist, (tr, va, ab_rmse, L_rmse, score, lr)):
            hist[k].append(v)
        flag = ""
        if score < best:
            best = score
            save_checkpoint(ckpt, net, dict(meta, best_epoch=epoch + 1, best_val=score))
            flag = "  <- best, saved"
        print(f"Epoch {epoch+1}/{args.epochs}  train={tr:.4f}  val={va:.4f}  "
              f"val ab RMSE={ab_rmse:.2f}" + (f"  L RMSE={L_rmse:.2f}" if args.predict_L else "")
              + f"  lr={lr:.1e}  ({(time.time()-t0)/60:.1f} min){flag}", flush=True)

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    best_ep = hist["val_score"].index(best) + 1
    ep = range(1, args.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, hist["train_loss"], marker="o", ms=3, label="train")
    ax[0].plot(ep, hist["val_loss"], marker="s", ms=3, label="val")
    ax[0].set_ylabel(f"training objective (bins{' + L1(L)' if args.predict_L else ''})")
    ax[1].plot(ep, hist["val_ab_rmse"], marker="s", ms=3, label="val a/b RMSE")
    if args.predict_L:
        ax[1].plot(ep, hist["val_L_rmse"], marker="^", ms=3, label="val L RMSE")
    ax[1].set_ylabel("Lab units (lower = better)")
    for a in ax:
        a.axvline(best_ep, ls=":", c="gray"); a.set_xlabel("epoch"); a.grid(alpha=0.3); a.legend()
    fig.suptitle(f"Model {model} training - best epoch {best_ep}")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "loss_curve.png"), dpi=130); plt.close(fig)

    best_net, best_meta = load_colorizer(ckpt, device)
    save_samples(val_loader, best_net, best_meta, device,
                 os.path.join(args.out, "sample_predictions.png"))
    print(f"\nDone. Best epoch {best_ep} (val score {best:.3f}). Results in ./{args.out}/")
    print(f"Metrics next:  python evaluate.py --model {model} --checkpoint {ckpt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--model", type=int, choices=[1, 3], required=True)
    known, rest = ap.parse_known_args()
    main(known.model, rest)
