"""
ColorSAR - interactive dashboard (Gradio)

    python app.py        ->  http://127.0.0.1:7860

Three tabs:
    Colorize  upload an image (or pick a dataset sample); Sentinel-1 files (`_s1_` in the
              name) go to Model 3, everything else to Model 1. Shows
              input / colorized / ground truth side by side
    Results   metrics on the held-out test scenes next to the no-color / mean-color
              baselines and the previous recipe retrained on the same scenes;
              per-terrain breakdown, training curves
    About     pipeline and how to reproduce everything

Uses colorize.py's functions directly, so with the same options the dashboard and
`python colorize.py` produce the same image.
"""
import glob
import json
import os
import time

import gradio as gr
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from color_utils import unsharp  # noqa: E402
from colorize import CHECKPOINTS, colorize, detect_mode, find_ground_truth, load_models  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "v_2"

print(f"[app] device: {DEVICE}   loading models ...")
MODELS = {mode: load_models(mode, DEVICE) for mode, path in CHECKPOINTS.items() if os.path.exists(path)}
print(f"[app] ready: {', '.join(MODELS) or 'no checkpoints found'}")


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- helpers
def _samples():
    out = []
    if os.path.isdir(DATA_ROOT):
        for terrain in sorted(os.listdir(DATA_ROOT)):
            files = sorted(glob.glob(os.path.join(DATA_ROOT, terrain, "s1", "*.png")))
            if files:
                out.append((f"{terrain} - SAR (s1)", files[0]))
        for terrain in sorted(os.listdir(DATA_ROOT)):
            files = sorted(glob.glob(os.path.join(DATA_ROOT, terrain, "s2", "*.png")))
            if files:
                out.append((f"{terrain} - optical (s2)", files[0]))
                break
    if os.path.exists("sample_input.jpeg"):
        out.append(("everyday photo (out of domain)", "sample_input.jpeg"))
    return out


SAMPLES = _samples()


def _gray3(a):
    a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    return (np.stack([a] * 3, axis=-1) * 255).round().astype(np.uint8)


# ---------------------------------------------------------------- tab 1
def run_colorize(upload_path, mode_choice, tta, sharpen, sample):
    t0 = time.time()
    try:
        if upload_path:
            path, gt_path = upload_path, None
            src = f"upload `{os.path.basename(upload_path)}`"
        elif sample:
            path = sample
            gt_path, _ = find_ground_truth(sample, DATA_ROOT)
            src = os.path.basename(sample)
        else:
            return [], "Pick a dataset sample or upload an image first."

        mode = detect_mode(path) if mode_choice == "auto" else mode_choice
        if mode not in MODELS:
            return [], f"No checkpoint for {mode} mode (`{CHECKPOINTS[mode]}` is missing)."
        net, meta = MODELS[mode]
        pred, shown, _ = colorize(path, mode, net, meta, DEVICE, tta=tta)
        notes = []
        if tta:
            notes.append("averaged over the 8 rotations/flips (TTA)")
        if sharpen:
            pred = unsharp(pred)
            notes.append("output sharpened (unsharp mask, cosmetic)")
        if mode == "gray" and not (gt_path or "_s2_" in os.path.basename(path)):
            notes.append("Model 1 was trained on satellite tiles only - colors on other "
                         "images are a guess in the 'satellite palette'")

        name = f"Model {meta['model']}"
        panels = [(_gray3(shown), f"input ({'SAR' if mode == 'sar' else 'L'})"),
                  ((pred * 255).round().astype(np.uint8), f"ColorSAR {name} output")]
        if gt_path:
            title = "ground truth (s2)" if mode == "sar" else "paired SAR (s1)"
            panels.append((np.array(Image.open(gt_path).convert("RGB")), title))
        else:
            notes.append("no ground truth (only dataset samples have one)")

        info = (f"source: {src}  |  mode: **{mode}**  |  model: **{name}**"
                + "".join(f"  \n- {n}" for n in notes)
                + f"  \ncheckpoint: `{meta['checkpoint']}`  |  device: {DEVICE}  |  "
                  f"{time.time() - t0:.1f}s")
        return panels, info
    except Exception as e:                       # surface errors in the UI, not the console
        return [], f"**Error:** `{type(e).__name__}: {e}`"


# ---------------------------------------------------------------- tab 2
RUNS = {  # key: (run folder, title)
    "3": ("model3_run", "Model 3 - SAR tile -> L + color"),
    "1": ("model1_run", "Model 1 - optical L -> color"),
}
METRICS = {k: _load_json(os.path.join(d, "metrics_test_tta.json")) for k, (d, _) in RUNS.items()}
HISTORY = {k: _load_json(os.path.join(d, "history.json")) for k, (d, _) in RUNS.items()}


def _loss_figure():
    keys = [k for k in RUNS if HISTORY[k] and "val_score" in HISTORY[k]]
    fig, axes = plt.subplots(1, max(len(keys), 1), figsize=(5.2 * max(len(keys), 1), 4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        h = HISTORY[k]
        ep = range(1, len(h["val_score"]) + 1)
        ax.plot(ep, h["val_ab_rmse"], marker="s", ms=3, label="val a/b RMSE")
        if any(h.get("val_L_rmse") or []):
            ax.plot(ep, h["val_L_rmse"], marker="^", ms=3, label="val L RMSE")
        ax.axvline(int(np.argmin(h["val_score"])) + 1, ls=":", c="gray")
        ax.set_title(RUNS[k][1], fontsize=11)
        ax.set_ylabel("Lab units")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _terrain_figure():
    keys = [k for k in RUNS if METRICS[k]]
    if not keys:
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.axis("off")
        ax.set_title("no metrics_test_tta.json yet - run evaluate.py --tta")
        return fig
    terrains = list(METRICS[keys[0]]["per_terrain"])
    bars = [(k, row, label, color) for k in keys
            for row, label, color in (("model", f"Model {k}", None),
                                      ("mean_color", f"mean color (Model {k}'s L)", "lightgray"))]
    x = np.arange(len(terrains))
    w = 0.8 / len(bars)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for j, (k, row, label, color) in enumerate(bars):
        pt = METRICS[k]["per_terrain"]
        b = ax.bar(x + (j - (len(bars) - 1) / 2) * w, [pt[t][row]["ab_rmse"] for t in terrains], w,
                   label=label, color=color)
        ax.bar_label(b, fmt="%.1f", fontsize=7)
    ax.set_xticks(x, terrains)
    ax.set_ylabel("a/b RMSE (Lab units, lower = better)")
    ax.set_title("Color error per terrain, held-out test scenes")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _row(name, x, L=None):
    L_txt = "" if L is None else f"{L:.2f}"
    return f"| {name} | {x['psnr']:.2f} | {x['ssim']:.4f} | {x['ab_rmse']:.2f} | {L_txt} |"


def _results_markdown():
    out = []
    for k, (d, title) in RUNS.items():
        m = METRICS[k]
        out.append(f"#### {title}")
        if not m:
            out += [f"*Metrics not available yet* - run `python evaluate.py --model {k} "
                    f"--tta --per_terrain --out {d}/metrics_test_tta.json`.", ""]
            continue
        has_L = k == "3"
        out += [f"*{m['n']:,} pairs from the held-out test scenes: {m['eval_set']}*", "",
                "| | PSNR (dB) | SSIM | a/b RMSE | L RMSE |", "|---|---|---|---|---|",
                _row("**model (TTA)**", m["model"], m["L_rmse"] if has_L else None)]
        plain = _load_json(os.path.join(d, "metrics_test.json"))
        if plain:
            out.append(_row("model, no TTA", plain["model"], plain["L_rmse"] if has_L else None))
        legacy = _load_json(os.path.join(d, "metrics_legacy_same_scenes.json"))
        if legacy:
            out.append(_row("previous recipe, retrained on the same scenes (scored at 224 px)",
                            legacy["model"], legacy["L_rmse"] if has_L else None))
        out += [_row("no-color baseline (a=b=0)", m["no_color"]),
                _row("mean-color baseline", m["mean_color"]),
                _row("oracle: true colors, same L" + (" (round trip)" if k == "1" else ""), m["oracle"]),
                ""]
    out += ["**How to read this.** PSNR/SSIM on RGB are dominated by brightness. Model 1 is "
            "*given* the optical L; Model 3 predicts its own, so its oracle row is the ceiling "
            "for any colorizer with that brightness. Color quality is best read from a/b RMSE "
            "and the gap to the mean-color baseline."]
    return "\n".join(out)


# ---------------------------------------------------------------- app
with gr.Blocks(title="ColorSAR") as demo:
    gr.Markdown("# ColorSAR — colorization from SAR and grayscale imagery")

    with gr.Tabs():
        with gr.Tab("Colorize"):
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(type="filepath", image_mode=None,
                                   label="Upload your own image (optional)")
                    gr.Markdown("*Wait for the preview to appear before clicking Colorize — "
                                "otherwise the dataset sample below is used instead.*")
                    sample = gr.Dropdown(choices=SAMPLES, value=SAMPLES[0][1] if SAMPLES else None,
                                         label="...or pick a sample")
                    mode_choice = gr.Radio(
                        ["auto", "sar", "gray"], value="auto", label="Input type",
                        info="auto: Sentinel-1 file name (`_s1_`) -> SAR model, anything else "
                             "-> optical model. Choose 'sar' for other SAR images.")
                    tta = gr.Checkbox(value=True, label="Average over 8 rotations/flips (TTA)")
                    sharpen = gr.Checkbox(value=False,
                                          label="Sharpen output (unsharp mask, cosmetic)")
                    btn = gr.Button("Colorize", variant="primary")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="input | ColorSAR output | ground truth (dataset samples only)",
                                         columns=3, height=420, object_fit="contain")
                    info = gr.Markdown()
            btn.click(run_colorize, [img, mode_choice, tta, sharpen, sample], [gallery, info])

        with gr.Tab("Results"):
            gr.Markdown(_results_markdown())
            gr.Plot(_terrain_figure())
            gr.Plot(_loss_figure())

        with gr.Tab("About"):
            gr.Markdown(f"""
**Pipeline**

| input | path | checkpoint |
|---|---|---|
| Sentinel-1 SAR tile, Model 3 | s1 (raw + 2 smoothing levels) -> ResNet-50 U-Net -> L + color bins | `{CHECKPOINTS['sar']}` |
| optical image, Model 1 | L of the input -> ResNet-50 U-Net -> color bins | `{CHECKPOINTS['gray']}` |

The ImageNet ResNet-50 encoder is fine-tuned at a tenth of the decoder's learning rate;
the decoder predicts at the full 256x256 resolution. Color is classified into ~150 Lab
bins (Zhang et al. 2016) and decoded with the annealed mean.

**Evaluation** holds out whole Sentinel scenes (`splits.scene_split`): patches cut from
one scene look alike, so scoring on patches of training scenes overstates accuracy.

**Reproduce**

```powershell
python Train_model3.py
python evaluate.py --model 3 --tta --per_terrain --out model3_run/metrics_test_tta.json
python colorize.py --image v_2/agri/s1/<file>.png --tta     # CLI demo
.\\run_demo.ps1                                              # full scripted demo
```
""")

if __name__ == "__main__":
    port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or 7860)
    demo.launch(server_name="127.0.0.1", server_port=port,
                inbrowser=not os.environ.get("COLORSAR_NO_BROWSER"),
                show_error=True, theme=gr.themes.Soft())
