"""
ColorSAR benchmark - comparison figure and results table from the Kaggle runs.

Put every downloaded results/ folder in one place (one sub-folder per set, any names):

    all_results/
        picma/results/picma/shown.npz, metrics_test.json, ...
        freq/results/transunet/...
        ddpm/results/ddpm/...

    python scripts/bench_figure.py all_results --out paper_figs --tiles 0 2 5 7

Writes to --out:
    comparison.png / .pdf   rows = test tiles, columns = SAR | each algorithm | real optical
    results.md              table for the chat / slides
    results.tex             IEEE table (booktabs), best value per column in bold
    ablation.tex            the ablation rows, if any ran
Every algorithm saves the same 8 test tiles (bench_common.SHOW), so --tiles picks from 0-7.
"""
import argparse
import glob
import json
import os

import numpy as np

ORDER = [  # (folder name, label in the paper)
    ("histmatch", "Hist. match"), ("welsh", "Welsh"), ("unet_r50", "U-Net R50"),
    ("pix2pix", "Pix2Pix"), ("cyclegan", "CycleGAN"), ("transunet", "TransUNet"),
    ("wavenet", "Wavelet U-Net"), ("ddpm", "DDPM"), ("picma", "PICMA (ours)"),
]
ABLATION = [("picma_short", "PICMA (2 h)"), ("picma_no_physics", "w/o physics"),
            ("picma_no_xattn", "w/o cross-attention"), ("picma_no_contrastive", "w/o contrastive")]
# (key in metrics_test.json, header, True if higher is better, format)
COLS = [("psnr", "PSNR$\\uparrow$", True, "{:.2f}"), ("ssim", "SSIM$\\uparrow$", True, "{:.3f}"),
        ("ab_rmse", "ab RMSE$\\downarrow$", False, "{:.2f}"), ("L_rmse", "L RMSE$\\downarrow$", False, "{:.2f}"),
        ("lpips", "LPIPS$\\downarrow$", False, "{:.3f}"), ("fid", "FID$\\downarrow$", False, "{:.1f}")]


def collect(root):
    """folder name -> directory holding its metrics_test.json (latest copy wins)."""
    found = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "metrics_test.json"), recursive=True)):
        found[os.path.basename(os.path.dirname(path))] = os.path.dirname(path)
    return found


def rows_for(names, found):
    rows = []
    for key, label in names:
        if key not in found:
            continue
        with open(os.path.join(found[key], "metrics_test.json")) as f:
            m = json.load(f)
        rows.append((label, m["sample"], m.get("train_hours", 0.0)))
        if m.get("mean"):
            rows.append((label.replace(" (ours)", "") + ", mean of %d" % m.get("k_samples", 4), m["mean"], m.get("train_hours", 0.0)))
    return rows


def best_per_col(rows):
    best = {}
    for key, _, higher, _ in COLS:
        vals = [r[1].get(key) for r in rows if r[1].get(key) is not None]
        if vals:
            best[key] = max(vals) if higher else min(vals)
    return best


def cell(s, key, fmt, best, bold):
    v = s.get(key)
    if v is None:
        return "-"
    t = fmt.format(v)
    return bold(t) if best.get(key) is not None and abs(v - best[key]) < 1e-9 else t


def write_tex(rows, path, caption, label):
    best = best_per_col(rows)
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{l" + "c" * (len(COLS) + 1) + "}", "\\toprule",
             "Method & " + " & ".join(c[1] for c in COLS) + " & Train (h) \\\\", "\\midrule"]
    for label_, s, h in rows:
        lines.append(label_.replace("&", "\\&") + " & " +
                     " & ".join(cell(s, k, f, best, lambda t: f"\\textbf{{{t}}}") for k, _, _, f in COLS) +
                     f" & {h:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_md(rows, title):
    best = best_per_col(rows)
    head = [c[1].replace("$\\uparrow$", " ↑").replace("$\\downarrow$", " ↓") for c in COLS]
    out = [f"## {title}", "", "| Method | " + " | ".join(head) + " | Train (h) |",
           "|---" * (len(COLS) + 2) + "|"]
    for label_, s, h in rows:
        out.append(f"| {label_} | " + " | ".join(cell(s, k, f, best, lambda t: f"**{t}**")
                                                 for k, _, _, f in COLS) + f" | {h:.1f} |")
    return "\n".join(out) + "\n\n"


def figure(found, tiles, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algos = [(k, lab) for k, lab in ORDER if k in found and os.path.isfile(os.path.join(found[k], "shown.npz"))]
    if not algos:
        print("no shown.npz found - figure skipped")
        return
    data = {k: np.load(os.path.join(found[k], "shown.npz")) for k, _ in algos}
    ref = data[algos[-1][0]]
    for k, _ in algos:   # every run must have scored the same test tiles
        if not np.array_equal(data[k]["idx"], ref["idx"]) or not np.array_equal(data[k]["gt"], ref["gt"]):
            raise SystemExit(f"{k}: different test tiles than {algos[-1][0]} - runs used different data")
    tiles = [t for t in tiles if t < len(ref["idx"])]
    cols = [("SAR input", lambda t: ref["sar"][t])] + \
           [(lab, lambda t, k=k: data[k]["pred"][t]) for k, lab in algos] + \
           [("Real optical", lambda t: ref["gt"][t])]
    fig, axes = plt.subplots(len(tiles), len(cols), figsize=(1.25 * len(cols), 1.25 * len(tiles)), squeeze=False)
    for r, t in enumerate(tiles):
        for c, (lab, get) in enumerate(cols):
            ax = axes[r][c]
            img = get(t)
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(lab, fontsize=7, fontweight="bold" if "ours" in lab else "normal")
    plt.subplots_adjust(wspace=0.03, hspace=0.03, left=0.01, right=0.99, top=0.93, bottom=0.01)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "comparison." + ext), dpi=300)
    print("figure:", os.path.join(out, "comparison.png"), "-", len(algos), "algorithms,", len(tiles), "tiles")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder holding all downloaded results/ folders")
    ap.add_argument("--out", default="paper_figs")
    ap.add_argument("--tiles", type=int, nargs="+", default=[0, 2, 5, 7], help="which of the 8 saved test tiles")
    a = ap.parse_args()
    if hasattr(os.sys.stdout, "reconfigure"):
        os.sys.stdout.reconfigure(encoding="utf-8")   # arrows in the table on a Windows console
    os.makedirs(a.out, exist_ok=True)
    found = collect(a.root)
    print("found:", ", ".join(sorted(found)) or "nothing")
    missing = [k for k, _ in ORDER if k not in found]
    if missing:
        print("missing:", ", ".join(missing))

    main_rows, abl_rows = rows_for(ORDER, found), rows_for(ABLATION, found)
    md = write_md(main_rows, "Comparison on the held-out test scenes")
    write_tex(main_rows, os.path.join(a.out, "results.tex"),
              "SAR-to-optical colorization on the held-out test scenes.", "tab:results")
    if abl_rows:
        md += write_md(abl_rows, "Ablation (same 2 h budget)")
        write_tex(abl_rows, os.path.join(a.out, "ablation.tex"),
                  "Ablation of PICMA, every variant trained for the same time.", "tab:ablation")
    with open(os.path.join(a.out, "results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    figure(found, a.tiles, a.out)
