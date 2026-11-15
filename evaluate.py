"""
ColorSAR - Evaluation: PSNR / SSIM / color error for a trained colorizer

    Model 1 : L taken from the optical (s2) image  -> a/b predicted
    Model 3 : L and a/b both predicted from the SAR (s1) tile
RGB is rebuilt from (L, predicted a/b) and scored against the s2 image. "L RMSE" is
the brightness error vs the optical L (0 for Model 1 by design).

What gets scored
    --split test (default) or val: the whole held-out scenes of splits.scene_split.
    The script refuses to score a scene the checkpoint lists among its training scenes.

Rows - always read them together
    model         the checkpoint's colors
    no color      the same L with a = b = 0 (a grayscale image). PSNR/SSIM on RGB are
                  dominated by brightness; the gap between "model" and "no color" is
                  what the colorizer actually adds.
    mean color    the same L with one constant a/b: the average color of the training
                  scenes. A useful colorizer must beat this.
    oracle        the same L with the TRUE a/b. For Model 1 this is a Lab round trip
                  (not a meaningful ceiling); for Model 3 it is the best score any
                  colorizer could reach with that predicted brightness.
    ab RMSE       a/b error in Lab units (lower = better), independent of brightness.

--eval_size 224 resizes prediction and ground truth to 224x224 before PSNR/SSIM, the
resolution the pre-scene-split models were scored at (comparison only).

    python evaluate.py --model 3 --checkpoint model3_run/model_3.pth --per_terrain --tta \
        --out model3_run/metrics_test.json
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

from color_utils import AB_NEUTRAL, SIZE, PairDataset, ab_to_lab_units, find_pairs, lab_to_rgb, rgb_to_L
from model_architecture import load_colorizer
from pipeline import predict
from splits import scene_split, scenes, subsample

ROWS = ("model", "no_color", "mean_color", "oracle")
LABELS = {"model": "model (predicted color)", "no_color": "no-color baseline (a=b=0)",
          "mean_color": "mean-color baseline (train avg)", "oracle": "oracle (true ab, same L)"}


def psnr_ssim(pred, gt, size):
    if size != pred.shape[0]:
        pred = cv2.resize(pred.astype(np.float32), (size, size))
        gt = cv2.resize(gt.astype(np.float32), (size, size))
    p = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    s = structural_similarity(gt, pred, data_range=1.0, channel_axis=2)
    return p, s


def ab_rmse(ab_pred, ab_gt):
    """Per-image RMSE of a/b in Lab units. Tensors (N,2,H,W), normalized."""
    d = ab_to_lab_units(ab_pred) - ab_to_lab_units(ab_gt)
    return d.pow(2).mean(dim=(1, 2, 3)).sqrt().cpu().numpy()


def summarize(recs):
    """recs: per-image [psnr, ssim] for each row, ab RMSE for each row, then L RMSE."""
    a = np.array(recs)
    k = 2 * len(ROWS)
    out = {r: {"psnr": float(a[:, 2 * i].mean()), "ssim": float(a[:, 2 * i + 1].mean()),
               "ab_rmse": float(a[:, k + i].mean())} for i, r in enumerate(ROWS)}
    out["L_rmse"] = float(a[:, -1].mean())
    out["n"] = len(a)
    return out


def mean_ab(pairs, n=500):
    """Average normalized (a, b) over up to n images."""
    ds = PairDataset(subsample(pairs, n, seed=0), load_sar=False)
    return torch.stack([ds[i][2].mean(dim=(1, 2)) for i in range(len(ds))]).mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, choices=[1, 3], default=3)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--root", default="v_2")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--n_images", type=int, default=0, help="0 = every pair of the chosen split")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--tta", action="store_true", help="average over the 8 rotations/flips")
    ap.add_argument("--eval_size", type=int, default=SIZE,
                    help="resolution for PSNR/SSIM (224 = the old models' setting)")
    ap.add_argument("--per_terrain", action="store_true")
    ap.add_argument("--out", default=None, help="optional path for a JSON metrics file")
    args = ap.parse_args()
    if args.checkpoint is None:
        args.checkpoint = f"model{args.model}_run/model_{args.model}.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, meta = load_colorizer(args.checkpoint, device)
    if meta["model"] != args.model:
        raise SystemExit(f"{args.checkpoint} is a Model {meta['model']} checkpoint, not Model {args.model}")
    print(f"Device: {device}  |  Model {args.model}  |  checkpoint {args.checkpoint} "
          f"(best epoch {meta.get('best_epoch')})  |  TTA {args.tta}  |  eval size {args.eval_size}")

    pairs = find_pairs(args.root)
    assert pairs, f"No s1/s2 pairs found under {args.root}"
    split = scene_split(pairs)
    pairs_eval = subsample(split[args.split], args.n_images, seed=42)
    overlap = set(scenes(pairs_eval)) & set(meta["split"]["train"])
    if overlap:
        raise SystemExit(f"refusing to score training scenes: {sorted(overlap)}")
    desc = f"{args.split} scenes {scenes(pairs_eval)} (never trained on)"
    print(f"Evaluating on {len(pairs_eval)} pairs: {desc}")
    m_ab = mean_ab(split["train"]).to(device)

    loader = DataLoader(PairDataset(pairs_eval, load_sar=args.model == 3),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    stats, allrecs = {}, []
    with torch.no_grad():
        for gray, rgb_gt, ab_gt, terrain in loader:
            rgb_gt, ab_gt = rgb_gt.to(device), ab_gt.to(device)
            L, pred_ab = predict(net, meta, gray=gray.to(device), rgb=rgb_gt, tta=args.tta)
            L_rmse = ((L - rgb_to_L(rgb_gt)) * 50).pow(2).mean(dim=(1, 2, 3)).sqrt().cpu().numpy()
            neutral = torch.full_like(ab_gt, AB_NEUTRAL)
            const = m_ab.view(1, 2, 1, 1).expand_as(ab_gt)

            gt = rgb_gt.permute(0, 2, 3, 1).cpu().numpy()
            abs_ = [pred_ab, neutral, const, ab_gt]                  # same order as ROWS
            imgs = [lab_to_rgb(L, x) for x in abs_]
            errs = [ab_rmse(x, ab_gt) for x in abs_]
            for i in range(gt.shape[0]):
                rec = ([v for im in imgs for v in psnr_ssim(im[i], gt[i], args.eval_size)]
                       + [e[i] for e in errs] + [L_rmse[i]])
                stats.setdefault(terrain[i], []).append(rec)
                allrecs.append(rec)

    res = summarize(allrecs)
    print("\n" + "=" * 72)
    print(f"Model {args.model}  -  {res['n']} images  -  {args.split} scenes"
          f"{'  -  TTA' if args.tta else ''}  -  scored at {args.eval_size}px")
    print("=" * 72)
    print(f"  {'':38s}{'PSNR dB':>10s}{'SSIM':>9s}{'ab RMSE':>10s}")
    for r in ROWS:
        label = LABELS[r] + (" [round trip]" if r == "oracle" and args.model == 1 else "")
        print(f"  {label:38s}{res[r]['psnr']:10.4f}{res[r]['ssim']:9.4f}{res[r]['ab_rmse']:10.2f}")
    print(f"  brightness error (L RMSE vs optical L, Lab units): {res['L_rmse']:.2f}")
    m = res["model"]
    for name, key in (("no color  ", "no_color"), ("mean color", "mean_color")):
        b = res[key]
        print(f"  colorizer vs {name}: {m['psnr'] - b['psnr']:+.2f} dB PSNR, "
              f"{m['ssim'] - b['ssim']:+.4f} SSIM, ab error {b['ab_rmse']:.2f} -> {m['ab_rmse']:.2f}")

    per_terrain = {t: summarize(v) for t, v in sorted(stats.items())}
    if args.per_terrain:
        print(f"\n  {'terrain':<12}{'n':>6}{'PSNR':>9}{'no-color':>10}{'mean-col':>10}{'oracle':>9}"
              f"{'SSIM':>8}{'ab RMSE':>9}{'L RMSE':>8}")
        for t, s in per_terrain.items():
            print(f"  {t:<12}{s['n']:>6}{s['model']['psnr']:>9.3f}{s['no_color']['psnr']:>10.3f}"
                  f"{s['mean_color']['psnr']:>10.3f}{s['oracle']['psnr']:>9.3f}"
                  f"{s['model']['ssim']:>8.4f}{s['model']['ab_rmse']:>9.2f}{s['L_rmse']:>8.2f}")

    if args.out:
        result = {"model": args.model, "checkpoint": args.checkpoint, "eval_set": desc,
                  "split": args.split, "tta": args.tta, "eval_size": args.eval_size,
                  "best_epoch": meta.get("best_epoch"), **res, "per_terrain": per_terrain}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved metrics to {args.out}")


if __name__ == "__main__":
    main()
