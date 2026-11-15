"""
ColorSAR - Kaggle GPU job: train the new Models 3 and 1 and, for a fair comparison,
retrain the previous recipe on the same held-out scenes; evaluate all of them.

Runs as a Kaggle "script" kernel (see kernel-metadata.json) with two inputs attached:
    requiemonk/sentinel12-image-pairs-segregated-by-terrain   the dataset (v_2)
    <user>/colorsar-code                                      built by kaggle/prepare_code.py
Everything it produces lands in /kaggle/working/results (download with
`kaggle kernels output <user>/colorsar-train -p kaggle_out`).

MODE = "smoke" runs every job on a handful of images for one epoch (catches bugs in a
few minutes); MODE = "full" is the real run.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

MODE = "full"
WORK = "/kaggle/working/ColorSAR"
RESULTS = "/kaggle/working/results"
BUDGET_H = 11.0          # Kaggle stops a session at 12 h; leave room for copying results
TRAIN_STOP_H = BUDGET_H - 0.75   # a training job is stopped here, leaving time to evaluate it
T0 = time.time()


def find_dir(predicate, root="/kaggle/input"):
    for d, dirs, files in os.walk(root):
        if predicate(d, dirs, files):
            return d
    return None


def hours():
    return (time.time() - T0) / 3600


def run(name, args, log, cwd=WORK, need_h=0.0, stop_h=BUDGET_H):
    if hours() + need_h > BUDGET_H:
        print(f"\n===== {name}: SKIPPED, {hours():.1f} h used and it needs ~{need_h} h", flush=True)
        return False
    t0 = time.time()
    print(f"\n===== {name}: {' '.join(args)}", flush=True)
    with open(log, "w") as f:
        p = subprocess.Popen([sys.executable, "-u"] + args, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        killed = False
        for line in p.stdout:
            if "Warning" not in line:
                print(line, end="", flush=True)
            f.write(line)
            if hours() > stop_h and not killed:   # never let Kaggle's 12 h limit kill the session
                print(f"===== {name}: STOPPED at {stop_h:.2f} h (budget)", flush=True)
                p.kill()
                killed = True
        p.wait()
    print(f"===== {name}: exit {p.returncode} after {(time.time()-t0)/60:.1f} min "
          f"(total {hours():.2f} h)", flush=True)
    # a stopped training run still saved its best checkpoint, so let its evaluation run
    return p.returncode == 0 or (killed and train_ok(args))


def train_ok(args):
    out = args[args.index("--out") + 1] if "--out" in args else None
    return bool(out) and bool(glob.glob(os.path.join(WORK, "**", out, "model_*.pth"), recursive=True))


def save(base, run_dir):
    """Copy a finished run into the kernel output right away, so it survives a later failure."""
    src = os.path.join(base, run_dir)
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(RESULTS, run_dir), dirs_exist_ok=True)


if __name__ == "__main__":
    import torch
    print("mode", MODE, "| torch", torch.__version__, "| GPUs:",
          [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

    data_root = find_dir(lambda d, dirs, f: "s1" in dirs and "s2" in dirs)
    data_root = os.path.dirname(data_root)
    code = find_dir(lambda d, dirs, files: "train_colorizer.py" in files)
    assert data_root and code, (data_root, code)
    print("data:", data_root, "| code:", code)

    # rebuild the layout from the flat code dataset ("legacy__splits.py" -> legacy/splits.py)
    for src in glob.glob(os.path.join(code, "*.py")):
        dst = os.path.join(WORK, *os.path.basename(src).split("__"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    LEG = os.path.join(WORK, "legacy")
    os.makedirs(RESULTS, exist_ok=True)

    smoke = MODE == "smoke"
    R = ["--root", data_root, "--num_workers", "4"]
    TR = ["--n_images", "64", "--n_val", "32", "--epochs", "1"] if smoke else []
    EV = ["--per_terrain", "--batch_size", "32"] + (["--n_images", "32"] if smoke else []) + R
    L = lambda n: os.path.join(RESULTS, n + ".log")  # noqa: E731
    h = (lambda x: 0.0) if smoke else (lambda x: x)  # noqa: E731

    # 1. new Model 3
    if run("train new model 3", ["Train_model3.py", "--out", "new3"] + TR + R, L("train_new3"),
           need_h=h(3.0), stop_h=TRAIN_STOP_H):
        ck = ["--model", "3", "--checkpoint", "new3/model_3.pth"]
        run("eval new 3", ["evaluate.py"] + ck + EV + ["--out", "new3/metrics_test.json"], L("eval_new3"))
        run("eval new 3 + TTA", ["evaluate.py"] + ck + EV + ["--tta", "--out", "new3/metrics_test_tta.json"],
            L("eval_new3_tta"), need_h=h(0.3))
        run("eval new 3 + TTA @224", ["evaluate.py"] + ck + EV +
            ["--tta", "--eval_size", "224", "--out", "new3/metrics_test_tta_224.json"],
            L("eval_new3_tta_224"), need_h=h(0.3))
        run("eval new 3 + TTA (val)", ["evaluate.py"] + ck + EV +
            ["--tta", "--split", "val", "--out", "new3/metrics_val_tta.json"],
            L("eval_new3_val"), need_h=h(0.3))

    save(WORK, "new3")

    # 2. old Model 3 recipe, same scenes
    if run("train legacy model 3", ["Train_model3.py", "--out", "legacy3"] + TR + R,
           L("train_legacy3"), cwd=LEG, need_h=h(2.5), stop_h=TRAIN_STOP_H):
        run("eval legacy 3", ["evaluate.py", "--model", "3", "--checkpoint", "legacy3/model_3.pth"]
            + EV + ["--out", "legacy3/metrics_test.json"], L("eval_legacy3"), cwd=LEG)

    save(LEG, "legacy3")

    # 3. new Model 1
    if run("train new model 1", ["Train_model1.py", "--out", "new1"] + TR + R, L("train_new1"),
           need_h=h(2.0), stop_h=TRAIN_STOP_H):
        ck = ["--model", "1", "--checkpoint", "new1/model_1.pth"]
        run("eval new 1", ["evaluate.py"] + ck + EV + ["--out", "new1/metrics_test.json"], L("eval_new1"))
        run("eval new 1 + TTA", ["evaluate.py"] + ck + EV + ["--tta", "--out", "new1/metrics_test_tta.json"],
            L("eval_new1_tta"), need_h=h(0.3))
        run("eval new 1 + TTA @224", ["evaluate.py"] + ck + EV +
            ["--tta", "--eval_size", "224", "--out", "new1/metrics_test_tta_224.json"],
            L("eval_new1_tta_224"), need_h=h(0.3))

    save(WORK, "new1")

    # 4. old Model 1b recipe (bins + adapters + ImageNet norm + augmentation), same scenes
    if run("train legacy model 1b", ["Train_model1.py", "--out", "legacy1b", "--loss", "bins",
                                     "--train_adapters", "--imagenet_norm", "--augment"] + TR + R,
           L("train_legacy1b"), cwd=LEG, need_h=h(2.0), stop_h=TRAIN_STOP_H):
        run("eval legacy 1b", ["evaluate.py", "--model", "1", "--checkpoint", "legacy1b/model_1.pth"]
            + EV + ["--out", "legacy1b/metrics_test.json"], L("eval_legacy1b"), cwd=LEG)

    save(LEG, "legacy1b")
    shutil.rmtree(WORK, ignore_errors=True)   # keep the kernel output small
    print(f"\nresults ({hours():.2f} h):", sorted(os.listdir(RESULTS)))
