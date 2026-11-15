"""
ColorSAR - regenerate a run's sample figure from its checkpoint.

    python make_figures.py --model 3            # model3_run/sample_predictions.png
    python make_figures.py --model 1 --run model1_run

Use this whenever a checkpoint is replaced (e.g. one trained on Kaggle) so the figure in
the run folder shows the checkpoint that is actually there. Samples are drawn from the
held-out test scenes (splits.scene_split). The loss curve is written by training.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from color_utils import PairDataset, find_pairs
from model_architecture import load_colorizer
from splits import scene_split, subsample
from train_colorizer import save_samples

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, choices=[1, 3], required=True)
    ap.add_argument("--run", default=None, help="run folder (default model<k>_run)")
    ap.add_argument("--root", default="v_2")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    run = args.run or f"model{args.model}_run"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, meta = load_colorizer(os.path.join(run, f"model_{args.model}.pth"), device)
    pairs = subsample(scene_split(find_pairs(args.root))["test"], args.n, seed=7)
    loader = DataLoader(PairDataset(pairs, load_sar=args.model == 3), batch_size=args.n)
    out = os.path.join(run, "sample_predictions.png")
    save_samples(loader, net, meta, device, out, n=args.n)
    print(f"[OK] {out}")
