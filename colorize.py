"""
ColorSAR - Unified inference: the end-to-end demo

The single entry point of the final system (the dashboard and the helper scripts
import it). It picks the right model for whatever you feed it:

    mode "sar"   Sentinel-1 SAR tile (e.g. v_2/*/s1/*.png)
        Model 3: s1 -> ResNet-50 U-Net -> predicted L + a/b -> RGB
        checkpoint: model3_run/model_3.pth

    mode "gray"  an optical image: RGB or grayscale (e.g. v_2/*/s2/*.png)
        L of the input -> Model 1 -> a/b -> RGB
        checkpoint: model1_run/model_1.pth

    mode "auto"  (default): "sar" when the file is a Sentinel-1 tile by name
        (`_s1_` in the filename or an `s1/` folder), otherwise "gray". The band count
        is NOT used: a grayscale photo is 1-band too. For other SAR files pass --mode sar.

Both models were trained only on 256x256 Sentinel-1/2 tiles: everyday photos
(sample_input.jpeg) are outside that domain and will get satellite-like colors.

Resolution: the network runs at 256x256. Model 1 keeps the input's own full-resolution
L and upsamples only the color; Model 3 predicts L as well, so its output is
upsampled from 256 to the input size.

If the input is a file inside v_2/, its pair (s1 <-> s2) is shown as a ground-truth panel.

Usage:
    python colorize.py --image v_2/agri/s1/<file>.png
    python colorize.py --image sample_input.jpeg
    python colorize.py --image v_2/urban/s1/<file>.png --out demo.png --tta
    python colorize.py --image v_2/agri/s1/<file>.png --sharp         # cosmetic sharpening

Outputs:
    <out>                 side-by-side figure (input / colorized / ground truth)
    <out stem>_raw.png    the colorized image alone, at the input's own resolution
"""
import argparse
import os

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from color_utils import SIZE, find_pairs, is_sentinel1_path, lab_to_rgb, rgb_to_L, unsharp
from model_architecture import load_colorizer
from pipeline import predict

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

CHECKPOINTS = {"sar": "model3_run/model_3.pth", "gray": "model1_run/model_1.pth"}


def detect_mode(path):
    return "sar" if is_sentinel1_path(path) else "gray"


def load_models(mode, device, checkpoint=None):
    """-> (network, meta) for the given mode."""
    path = checkpoint or CHECKPOINTS[mode]
    net, meta = load_colorizer(path, device)
    if mode == "sar" and meta["input"] != "sar3":
        raise ValueError(f"{path} is a Model {meta['model']} checkpoint; sar mode needs Model 3")
    meta["checkpoint"] = path
    return net, meta


def model_name(meta):
    return f"Model {meta['model']}"


@torch.no_grad()
def colorize(path, mode, net, meta, device, tta=False):
    """Return (prediction float RGB at the input's resolution, image shown as input, PIL)."""
    pil = Image.open(path)
    size = (pil.size[1], pil.size[0])
    if meta["input"] == "optical_L":
        rgb = np.array(pil.convert("RGB"), dtype=np.float32) / 255.0
        small = cv2.resize(rgb, (SIZE, SIZE)) if rgb.shape[:2] != (SIZE, SIZE) else rgb
        _, ab = predict(net, meta, rgb=torch.from_numpy(small).permute(2, 0, 1)[None].to(device),
                        tta=tta)
        L = rgb_to_L(torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device))  # full-res L
        shown = ((L[0, 0] + 1) / 2).cpu().numpy()
    else:
        gray = np.array(pil.convert("L"), dtype=np.float32) / 255.0
        L, ab = predict(net, meta, gray=torch.from_numpy(gray)[None, None].to(device), tta=tta)
        if L.shape[-2:] != size:
            L = F.interpolate(L, size=size, mode="bilinear", align_corners=False)
        shown = gray
    return lab_to_rgb(L, ab)[0], shown, pil       # a/b upsampled to L's resolution


def find_ground_truth(path, root):
    """If `path` is one of the s1/s2 files in the dataset, return its counterpart."""
    path = os.path.abspath(path)
    for sar, opt, _ in find_pairs(root):
        if os.path.abspath(sar) == path:
            return opt, "Ground truth (paired s2)"
        if os.path.abspath(opt) == path:
            return sar, "Paired input (s1)"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image (SAR s1 or RGB/grayscale photo)")
    ap.add_argument("--mode", choices=["auto", "sar", "gray"], default="auto",
                    help="auto (default): Sentinel-1 file name/folder -> sar, otherwise gray")
    ap.add_argument("--out", default="colorized_result.png", help="figure to write")
    ap.add_argument("--checkpoint", default=None, help="explicit checkpoint")
    ap.add_argument("--tta", action="store_true", help="average over the 8 rotations/flips")
    ap.add_argument("--sharp", action="store_true",
                    help="apply an unsharp mask to the output (cosmetic sharpening, "
                         "cannot invent detail the luminance does not have)")
    ap.add_argument("--root", default="v_2", help="dataset root used to find a ground truth pair")
    args = ap.parse_args()

    assert os.path.exists(args.image), f"Input image not found: {args.image}"
    mode = args.mode
    if mode == "auto":
        mode = detect_mode(args.image)
        print(f"Auto-detected mode: {mode} "
              f"({'Sentinel-1 file name' if mode == 'sar' else 'not a Sentinel-1 file; use --mode sar to override'})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, meta = load_models(mode, device, args.checkpoint)
    name = model_name(meta)
    print(f"Device: {device}  |  {name}  |  checkpoint: {meta['checkpoint']}")

    pred, shown, pil = colorize(args.image, mode, net, meta, device, tta=args.tta)
    if args.sharp:
        pred = unsharp(pred)
        print("Unsharp mask applied (--sharp)")

    raw_path = os.path.splitext(args.out)[0] + "_raw.png"
    Image.fromarray((pred * 255).round().astype(np.uint8)).save(raw_path)

    gt_path, gt_title = find_ground_truth(args.image, args.root)
    n_panels = 3 if gt_path else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes[0].imshow(shown, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Input ({'SAR' if meta['input'] == 'sar3' else 'L'})")
    axes[1].imshow(pred)
    axes[1].set_title(f"ColorSAR {name} output{' (+unsharp)' if args.sharp else ''}")
    if gt_path:
        axes[2].imshow(np.array(Image.open(gt_path).convert("RGB")))
        axes[2].set_title(gt_title)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"[OK] figure      -> {args.out}")
    print(f"[OK] colorized   -> {raw_path}  ({pil.size[0]}x{pil.size[1]})")


if __name__ == "__main__":
    main()
