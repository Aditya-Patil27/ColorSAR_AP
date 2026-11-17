"""
ColorSAR benchmark on Modal - the same jobs as kaggle/bench/run_bench.py, but every job in
its own container (2x T4, like Kaggle), all at the same time.

    python -m modal run scripts/modal_bench.py::fetch                       # dataset -> volume, once
    python -m modal run scripts/modal_bench.py::main --sets gans,ablation --smoke  # minutes, catches bugs
    python -m modal run --detach scripts/modal_bench.py::main --sets gans,ablation # the real runs
    python -m modal volume get colorsar-results full all_results/modal      # download results

Results land in the volume colorsar-results under full/<job>/ (or smoke/<job>/).
"""
import glob
import json
import os
import subprocess
import sys

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = "requiemonk/sentinel12-image-pairs-segregated-by-terrain"

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libgl1", "libglib2.0-0", "unzip")
         .pip_install("torch==2.9.0", "torchvision==0.24.0", "numpy<2.3", "scikit-image==0.26.0",
                      "opencv-python-headless", "matplotlib", "pillow", "scipy", "diffusers",
                      "lpips", "torch-fidelity", "kaggle", "six", "torchmetrics"))
for _f in glob.glob(os.path.join(REPO, "*.py")):
    image = image.add_local_file(_f, "/code/" + os.path.basename(_f))

app = modal.App("colorsar-bench", image=image)
data = modal.Volume.from_name("colorsar-data", create_if_missing=True)
results = modal.Volume.from_name("colorsar-results", create_if_missing=True)


@app.function(volumes={"/data": data}, timeout=3600, cpu=4, memory=8192)
def fetch_data(user, key, token):
    if os.path.isdir("/data/v_2/agri/s1"):
        return "already there"
    env = dict(os.environ, KAGGLE_USERNAME=user, KAGGLE_KEY=key, KAGGLE_API_TOKEN=token)
    r = subprocess.run(["kaggle", "datasets", "download", "-d", DS, "-p", "/tmp/ds", "--unzip"], env=env,
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("kaggle download failed: " + (r.stdout + r.stderr)[-1500:])
    subprocess.run(["cp", "-r", "/tmp/ds/v_2", "/data/v_2"], check=True)
    data.commit()
    return "%d files" % sum(len(f) for _, _, f in os.walk("/data/v_2"))


@app.function(gpu="T4:2", cpu=8, memory=32768, volumes={"/data": data, "/out": results}, timeout=6 * 3600)
def train(name, args, hours, smoke):
    out = "/out/" + ("smoke" if smoke else "full")
    cmd = [sys.executable, "-u", "bench_run.py"] + args + [
        "--root", "/data/v_2", "--cache", "/tmp/cache", "--out", out, "--name", name, "--hours", str(hours)]
    if smoke:
        cmd += ["--smoke", "--pre_hours", "0.01"]
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd="/code")
    results.commit()
    path = os.path.join(out, name, "metrics_test.json")
    return name, p.returncode, json.load(open(path)) if os.path.isfile(path) else None


@app.local_entrypoint()
def fetch():
    k = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
    tok = os.path.expanduser("~/.kaggle/access_token")
    print(fetch_data.remote(k["username"], k["key"], open(tok).read().strip() if os.path.isfile(tok) else ""))


@app.local_entrypoint()
def main(sets: str = "gans,ablation", smoke: bool = False):
    sys.path.insert(0, os.path.join(REPO, "kaggle", "bench"))
    from run_bench import SETS
    jobs = [(name, args, 0.02 if smoke else h, smoke)
            for s in sets.split(",") for name, args, h, _ in SETS[s] if h > 0]
    print("jobs:", ", ".join(j[0] for j in jobs))
    for name, code, m in train.starmap(jobs):
        s = (m or {}).get("sample") or {}
        print(f"{name}: exit {code} | PSNR {s.get('psnr')} SSIM {s.get('ssim')} LPIPS {s.get('lpips')} FID {s.get('fid')}")


@app.local_entrypoint()
def picma(gpu: str = "H100", hours: float = 6.0, pre_hours: float = 0.4, bs: int = 32, smoke: bool = False):
    """The proposed model alone on one big GPU (more steps in the same wall-clock time):
    python -m modal run --detach scripts/modal_bench.py::picma"""
    name = "picma_" + gpu.lower().replace(":", "x")
    args = ["--algo", "picma", "--pre_hours", str(pre_hours), "--bs", str(bs)]
    fn = train.with_options(gpu=gpu, timeout=int((hours + 2.5) * 3600))
    _, code, m = fn.remote(name, args, 0.02 if smoke else hours, smoke)
    print(name, "exit", code, (m or {}).get("sample"))
