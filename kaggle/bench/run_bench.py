"""
ColorSAR benchmark - Kaggle GPU job: trains and scores a set of the comparison algorithms
of bench_run.py on the held-out test scenes, one after the other, in one 12 h session.
The work is split over several Kaggle accounts, one SET each:

    picma     the proposed model, then histmatch, welsh, unet_r50 (test only; owner's
              account: unet_r50 needs the Model 3 checkpoint from colorsar-train)
    gans      pix2pix, cyclegan
    ddpm      the pixel-space diffusion baseline
    freq      transunet, wavenet
    ablation  the proposed model at a shorter budget, then with one component removed at
              a time (physics / contrastive pretraining / cross-attention)

Two ways to run it:
  1. Script kernel pushed with the Kaggle CLI (kernel-metadata.json): no arguments, so
     MODE and SET below apply; the code comes from the attached colorsar-code dataset.
  2. Any Kaggle notebook (GPU T4 x2, internet on, dataset
     requiemonk/sentinel12-image-pairs-segregated-by-terrain added), one cell:
         !git clone -q -b benchmark https://github.com/Aditya-Patil27/ColorSAR_AP
         !python ColorSAR_AP/kaggle/bench/run_bench.py --set freq
     then "Save Version" -> "Save & Run All" so it runs in the background.

Output: /kaggle/working/results/<job>/ (metrics_test.json, shown.npz, history.json,
weights), <job>.log, and summary.md - one table with every finished row.
--mode smoke runs every job on a few tiles for one epoch (catches bugs in minutes).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

MODE = "full"
SET = "picma_backup"
WORK = "/kaggle/working/ColorSAR"
RESULTS = "/kaggle/working/results"
CACHE = "/kaggle/working/cache"
BUDGET_H = 11.2          # Kaggle stops a session at 12 h
T0 = time.time()

# job name -> (bench_run.py args, training hours, hours needed incl. data + eval).
# Within a set the order is the priority: if the session runs short, the last jobs skip.
SETS = {
    # backup for the exam: the 9.3 h picma job was killed (exit -9) after ~1.1 h of diffusion
    # training, so this one trains 1 h only and finishes in ~2.3 h
    "picma_backup": [
        ("picma", ["--algo", "picma", "--pre_hours", "0.4"], 1.0, 2.5),
    ],
    "picma": [   # the proposed model first and longest; 1-3 only score on the test set
        ("picma", ["--algo", "picma", "--pre_hours", "0.4"], 9.3, 10.4),
        ("histmatch", ["--algo", "histmatch"], 0, 0.15),
        ("welsh", ["--algo", "welsh"], 0, 0.15),
        ("unet_r50", ["--algo", "unet_r50"], 0, 0.25),
    ],
    "gans": [
        ("pix2pix", ["--algo", "pix2pix"], 2.3, 2.5),
        ("cyclegan", ["--algo", "cyclegan"], 2.3, 2.5),
    ],
    "ddpm": [
        ("ddpm", ["--algo", "ddpm"], 9.0, 10.0),
    ],
    "freq": [
        ("transunet", ["--algo", "transunet"], 4.2, 4.5),
        ("wavenet", ["--algo", "wavenet"], 4.2, 4.5),
    ],
    "ablation": [  # every row gets the same budget, so they can be read against each other
        ("picma_short", ["--algo", "picma", "--pre_hours", "0.3"], 2.0, 2.6),
        ("picma_no_physics", ["--algo", "picma", "--pre_hours", "0.3", "--no_physics"], 2.0, 2.6),
        ("picma_no_xattn", ["--algo", "picma", "--pre_hours", "0.3", "--no_xattn"], 2.0, 2.6),
        ("picma_no_contrastive", ["--algo", "picma", "--no_contrastive"], 2.3, 2.6),
    ],
}


def hours():
    return (time.time() - T0) / 3600


def find_dir(predicate, root="/kaggle/input"):
    for d, dirs, files in os.walk(root):
        if predicate(d, dirs, files):
            return d
    return None


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=False)


def run(name, args, need_h):
    if hours() + need_h > BUDGET_H:
        print(f"\n===== {name}: SKIPPED, {hours():.1f} h used and it needs ~{need_h} h", flush=True)
        return
    t0 = time.time()
    print(f"\n===== {name}: {' '.join(args)}", flush=True)
    with open(os.path.join(RESULTS, name + ".log"), "w") as f:
        p = subprocess.Popen([sys.executable, "-u", "bench_run.py"] + args, cwd=WORK,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        killed = False
        for line in p.stdout:
            if "Warning" not in line and "%|" not in line:
                print(line, end="", flush=True)
            f.write(line)
            if hours() > BUDGET_H and not killed:   # never let Kaggle's 12 h limit kill the session
                print(f"===== {name}: STOPPED at {BUDGET_H} h (budget)", flush=True)
                p.kill()
                killed = True
        p.wait()
    print(f"===== {name}: exit {p.returncode} after {(time.time() - t0) / 60:.1f} min "
          f"(total {hours():.2f} h)", flush=True)
    src = os.path.join(WORK, "bench_out", name)
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(RESULTS, name), dirs_exist_ok=True)
    summary()


def summary():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*", "metrics_test.json"))):
        with open(path) as f:
            m = json.load(f)
        name = os.path.basename(os.path.dirname(path))
        for kind in ("sample", "mean"):
            s = m.get(kind)
            if s:
                fid = f"{s['fid']:.1f}" if s.get("fid") is not None else "-"
                rows.append(f"| {name}{' (mean of K)' if kind == 'mean' else ''} | {s['psnr']:.2f} | "
                            f"{s['ssim']:.4f} | {s['ab_rmse']:.2f} | {s['L_rmse']:.2f} | "
                            f"{s['lpips']:.4f} | {fid} | {m.get('train_hours', 0):.2f} | {s['n']} |")
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write(f"# ColorSAR benchmark ({MODE}, set {SET})\n\n"
                "| algorithm | PSNR | SSIM | ab RMSE | L RMSE | LPIPS | FID | train h | n |\n"
                "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")


def find_code():
    """bench_run.py next to this file (a git clone) or in an attached dataset."""
    try:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if os.path.isfile(os.path.join(here, "bench_run.py")):
            return here
    except NameError:
        pass
    return find_dir(lambda d, dirs, files: "bench_run.py" in files)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default=SET)
    ap.add_argument("--mode", choices=("full", "smoke"), default=MODE)
    cli, _ = ap.parse_known_args()
    SET, MODE = cli.set, cli.mode

    sh(f"{sys.executable} -m pip install -q --no-deps lpips torch-fidelity")
    sh(f"{sys.executable} -c \"import diffusers\" || {sys.executable} -m pip install -q diffusers")
    import torch
    print("mode", MODE, "| set", SET, "| torch", torch.__version__, "| GPUs:",
          [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
    assert torch.cuda.is_available(), "no GPU: set Accelerator to 'GPU T4 x2' in the notebook settings"

    s1 = find_dir(lambda d, dirs, f: "s1" in dirs and "s2" in dirs)
    assert s1, "dataset missing: add requiemonk/sentinel12-image-pairs-segregated-by-terrain"
    data_root = os.path.dirname(s1)
    code = find_code()
    ckpt = (glob.glob("/kaggle/input/**/m3_enc05/model_3.pth", recursive=True) or [None])[0]
    print("data:", data_root, "| code:", code, "| model 3 checkpoint:", ckpt, flush=True)
    assert code, "code missing: bench_run.py not found"
    os.makedirs(WORK, exist_ok=True)
    for src in glob.glob(os.path.join(code, "*.py")):
        if "__" not in os.path.basename(src):          # skip the legacy__* tree
            shutil.copy(src, WORK)
    os.makedirs(RESULTS, exist_ok=True)

    smoke = MODE == "smoke"
    common = ["--root", data_root, "--cache", CACHE, "--out", "bench_out"] + (["--smoke"] if smoke else [])
    for name, args, train_h, need_h in SETS[SET]:
        if name == "unet_r50":
            if not ckpt:
                print("===== unet_r50: SKIPPED, no Model 3 checkpoint attached", flush=True)
                continue
            args = args + ["--ckpt", ckpt]
        if smoke:
            args = args + ["--pre_hours", "0.01"]
        run(name, args + common + ["--name", name, "--hours", str(0.02 if smoke else train_h)],
            0.0 if smoke else need_h)

    shutil.rmtree(WORK, ignore_errors=True)
    shutil.rmtree(CACHE, ignore_errors=True)
    print(f"\ndone in {hours():.2f} h\n" + open(os.path.join(RESULTS, "summary.md")).read(), flush=True)
