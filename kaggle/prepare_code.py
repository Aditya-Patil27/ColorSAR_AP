"""
Build the flat folder that is uploaded as the private Kaggle dataset `<user>/colorsar-code`.

Kaggle datasets are flat here: subfolders are encoded in the file name with "__"
(legacy/splits.py -> legacy__splits.py); run_kaggle.py rebuilds the layout.

Two code trees go up:
    ./        the current code (scene split, fine-tuned ResNet-50, full-resolution head)
    legacy/   the previous recipe, taken from the git tag `v1-legacy`, with its split
              replaced by the scene split - so the old and new models are retrained
              and scored on exactly the same held-out scenes.

    python kaggle/prepare_code.py --user <kaggle username> --out <folder>
    kaggle datasets create -p <folder>            # first time (private by default)
    kaggle datasets version -p <folder> -m "msg"  # updates
"""
import argparse
import json
import os
import shutil
import subprocess

CURRENT = ["color_utils.py", "model_architecture.py", "pipeline.py", "splits.py",
           "bench_common.py", "bench_models.py", "bench_run.py",
           "train_colorizer.py", "evaluate.py", "Train_model1.py", "Train_model3.py"]
LEGACY = ["color_utils.py", "dncnn.py", "model_architecture.py", "pipeline.py", "splits.py",
          "train_colorizer.py", "evaluate.py", "Train_model1.py", "Train_model3.py"]
LEGACY_REF = "v1-legacy"
SCENE_OVERRIDE = """

# ---- added by kaggle/prepare_code.py: the scene split replaces the per-pair hash split
from splits_scene import scene_split as _scene_split  # noqa: E402


def canonical_split(pairs):  # noqa: F811
    return _scene_split(pairs)
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in CURRENT:
        shutil.copy(f, os.path.join(args.out, f))
    for f in LEGACY:
        src = subprocess.run(["git", "show", f"{LEGACY_REF}:{f}"], check=True,
                             capture_output=True, text=True, encoding="utf-8").stdout
        if f == "splits.py":
            src += SCENE_OVERRIDE
        with open(os.path.join(args.out, f"legacy__{f}"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(src)
    shutil.copy("splits.py", os.path.join(args.out, "legacy__splits_scene.py"))
    with open(os.path.join(args.out, "dataset-metadata.json"), "w") as fh:
        json.dump({"title": "colorsar-code", "id": f"{args.user}/colorsar-code",
                   "licenses": [{"name": "CC0-1.0"}]}, fh, indent=2)
    print(f"{len(CURRENT)} current + {len(LEGACY) + 1} legacy files -> {args.out}")
