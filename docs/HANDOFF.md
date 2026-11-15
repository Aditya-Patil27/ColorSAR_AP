# ColorSAR handoff (2026-11-15)

## Goal
The user asked to "check ColorSAR and see if we can improve the result", then approved "go, use Kaggle".
Plan: fix the leaky split, build a stronger model, then train and evaluate it on a Kaggle GPU. The
old recipe is retrained on the same held-out scenes, so the before/after comparison is fair.

## State of the repo (C:\Users\app27\Downloads\ColorSAR_AP)
- Branch `scene-split`. Other refs: `main` holds the old recipe, and tag `v1-legacy` marks it.
- **Nothing is committed yet.** All the changes listed below sit uncommitted in the working tree.
  Commit only if the user asks, and add no Co-Authored-By trailer.
- Removed with `git rm`: dncnn.py, train_dncnn.py, Train_model2.py, _sharpness.py. They are still on `main`.
- The old checkpoints `model*_run/*.pth` are tracked in git and stay in history.

### What changed
- **The split.** `splits.scene_split` holds out whole scenes: 6 val scenes and 7 test scenes.
  It replaces the leaky per-pair hash split. evaluate.py refuses to score any scene that the
  checkpoint lists as a train scene.
- **The model** (model_architecture.py, train_colorizer.py):
  - Encoder: ResNet-50 (ImageNet V2), fine-tuned at 0.1x lr, with its BatchNorm statistics frozen.
  - Decoder: U-Net up to 256 px.
  - Head: Zhang color-bin logits on a grid of 10, decoded with an annealed mean at T=0.38.
    Model 3 also has a tanh L head.
  - Training: AMP, DataParallel, rot90/flip augmentation, AdamW, cosine schedule.
  - Checkpoint format 3.
- **Inputs.** Model 1 takes the optical L. Model 3 takes `sar3`: raw SAR plus Gaussian blurs at
  sigma 1 and 2.5.
- **evaluate.py** scores four rows: the model, a no-color baseline, a mean-color baseline and an
  oracle. It also reports L RMSE. Options: `--tta` (average over 8 rotations/flips),
  `--eval_size 224` (the old scoring resolution), `--per_terrain`, `--split val`.
- **Also rewritten** for the new checkpoints: colorize.py, app.py (Gradio), make_figures.py,
  visual_test.py, run_demo.ps1.
- **The Kaggle tooling:**
  - kaggle/prepare_code.py builds the flat code dataset `aditya32211/colorsar-code`.
    Its file names encode folders: `legacy__x.py` means `legacy/x.py`.
  - The legacy code comes from `git show main:<f>`, with the scene split patched in.
  - kaggle/run_kaggle.py is the kernel script.
  - kaggle/kernel-metadata.json points to `aditya32211/colorsar-train`.
- Torch is not installed locally. The only local check is `py_compile`. Real testing happens on Kaggle.

## Running now
- **Kaggle kernel:** `aditya32211/colorsar-train` version 3, MODE="full", GPU T4 x2.
  Pushed about 10:50 on 2026-11-15. Expect it to take about 11 h.
- **Jobs, in order:**
  1. new3: train, then eval, eval with TTA, eval with TTA at 224 px, and eval with TTA on val.
  2. legacy3: train, then eval.
  3. new1: train, then eval, eval with TTA, and eval with TTA at 224 px.
  4. legacy1b: train, then eval.
- **Safety measures in v3:**
  - Each run folder is copied to the output as soon as it finishes.
  - A training job is stopped at 10.25 h. The best checkpoint saved so far is then evaluated.
  - A job that would not fit in the time left is skipped.
- **v1 (smoke run)** was clean: every job ran, with no tracebacks or NaNs. Its output is in
  `C:\Users\app27\Downloads\kaggle_smoke\results`. Its metrics mean nothing, because each job
  trained for 1 epoch on 64 images.
- v2 was cancelled on purpose.
- The user's account has a limit of 2 batch GPU sessions. The user's other project
  `verikg-smallmodel-dev` may hold one of them. Do not stop that project's sessions.

## Next steps
1. **Check the run.** Run `kaggle kernels status aditya32211/colorsar-train`.
   `kaggle kernels logs ...` returned nothing while the run was in progress.
2. **When it is COMPLETE**, download the output:
   `kaggle kernels output aditya32211/colorsar-train -p ../kaggle_full`
   Read every `*.log` first. Look for "SKIPPED" or "STOPPED", and check that each eval scored the
   full test set (n in the metrics JSON).
3. **Place the results:**
   - Copy `new3/*` into `model3_run/` and `new1/*` into `model1_run/`.
     The checkpoints are `model_3.pth` and `model_1.pth`, about 168 MB each.
   - Copy `legacy3/metrics_test.json` to `model3_run/metrics_legacy_same_scenes.json`.
   - Copy `legacy1b/metrics_test.json` to `model1_run/metrics_legacy_same_scenes.json`.
   - The Results tab in app.py reads these file names.
4. **Rewrite README.md**, which is outdated. Give an honest before/after table: new model against
   legacy on the same test scenes, with TTA and without, scored at 256 px and at 224 px, against
   the no-color and mean-color baselines, plus a per-terrain table. Say clearly that the old
   README numbers came from a leaky split and cannot be compared.
5. **Report to the user, then ask before committing.**

## Gotchas
- **Kaggle CLI 2.2.4** is the installed version (2.2.5 is available). The credentials are in
  `~/.kaggle`. Never print or upload them.
- **If the code changes:**
  1. Run `python kaggle/prepare_code.py --user aditya32211 --out ../colorsar_code_upload`.
  2. Then run `kaggle datasets version -p ../colorsar_code_upload -m "..."`.
  3. Then run `kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4`.
  run_kaggle.py is the kernel itself, so a change to it needs only the push.
- **The CLI cannot cancel a run.** The user stops runs from the Kaggle web UI: the Active Events
  panel has a stop button.
- **The legacy evaluate.py scores at 224 px.** Use the new `--eval_size 224` numbers to compare
  at the same resolution.
- **For Model 1, the "oracle" row is only a Lab round trip** (about 130 dB). It is not a ceiling.

## User preferences
- Keep answers short and plain.
- Ask decision questions as: the question, a short scenario, then a recommendation.
- Publish no Artifacts unless asked.
- The user's device should not need to stay on for long jobs, so use Kaggle.
