# ColorSAR — colorizing SAR and grayscale satellite imagery

Deep-learning colorization of Sentinel-1 SAR and grayscale Sentinel-2 tiles in the CIE
$L^*a^*b^*$ color space.

| | input | predicts | checkpoint |
|---|---|---|---|
| **Model 3** | the SAR (s1) tile | brightness **L** and color **a/b** | `model3_run/model_3.pth` |
| **Model 1** | L of the optical (s2) tile | color **a/b** (L is given) | `model1_run/model_1.pth` |

```
 input (N,3,256,256)
   └─ ResNet-50 encoder (ImageNet, BatchNorm statistics frozen)
        └─ U-Net decoder, skips at strides 2/4/8/16/32, upsampled to the full 256 px
             └─ head: color-bin logits (+ 1 tanh channel for L in Model 3)
                  └─ annealed mean over the bins -> a/b  ->  Lab  ->  RGB
```

- **Inputs.** Model 3 reads the SAR tile itself as three channels: raw, and Gaussian-smoothed
  at σ = 1 and σ = 2.5 (three despeckling levels). Model 1 reads the optical L as gray.
  Both are ImageNet-normalized.
- **Color.** a/b is predicted as a probability over the occupied 10×10 cells of the a/b
  plane (Zhang et al. 2016, *Colorful Image Colorization*), trained with cross-entropy on soft
  targets and decoded with the annealed mean at temperature T. T is picked on the validation
  scenes and stored in the checkpoint: T = 1 is the plain mean (lowest error when the color is
  uncertain), a lower T gives more vivid but riskier colors.
- **Training.** AdamW with a cosine schedule, rot90/flip augmentation, mixed precision,
  multi-GPU via `DataParallel`. The best epoch is picked on the validation scenes by a/b RMSE
  (+ L RMSE for Model 3).

Data: `requiemonk/sentinel12-image-pairs-segregated-by-terrain`, folder `v_2/`. It holds 16,000
SAR/optical pairs at 256×256, with 4,000 per terrain (agri, barrenland, grassland, urban),
cut from 30 Sentinel scenes.

---

## 1. Results

### The split: whole scenes are held out

Patches cut from one scene look alike, so a model that trained on some of them has
effectively seen the rest. [`splits.py`](splits.py) therefore holds out **whole scenes**,
and every terrain has at least one scene in val and in test:

| part | scenes | pairs |
|---|---|---|
| train | 17 | 11,877 |
| val (best epoch, temperature) | 6 | 2,463 |
| test (scored once, never trained on) | 7 | 1,660 |

`evaluate.py` refuses to score a scene that a checkpoint lists as a training scene.

> **The numbers in earlier versions of this README are not comparable.** They came from a
> per-pair hash split, so test patches shared scenes with training patches. For example,
> Model 1b at 30.0 dB and Model 3 at 16.6 dB were inflated. To get a fair comparison, the
> previous recipe (tag `v1-legacy`) was retrained on exactly the scenes above.

### How to read the tables

Every model is shown next to two baselines that use **the same brightness**:
- *no color*: a = b = 0;
- *mean color*: one constant a/b, the average training color.

RGB PSNR/SSIM are dominated by brightness. Color quality is therefore best read from the
**a/b RMSE** (color error in Lab units, lower is better) and its **gap to the mean-color
baseline**. A negative gap means the model beats a constant color.

The previous recipe's `evaluate.py` scores at 224 px and computes the baselines slightly
differently: its mean-color baseline is 8.78, against 9.19 here. So compare each model with
**its own** baseline, not raw a/b RMSE across rows.

### Same test scenes, scored at 224 px

<!-- RESULTS-V5: add the tuned Model 3 / Model 1 rows here when kaggle v5 finishes -->

| model | PSNR (dB) | SSIM | a/b RMSE | gap to mean color | gap to no color | L RMSE |
|---|---|---|---|---|---|---|
| Model 1, previous recipe | 26.64 | 0.916 | 8.74 | −0.04 | −0.78 | 0 (given) |
| Model 1, ResNet-50 U-Net, first recipe (TTA) | 26.71 | 0.910 | 9.09 | −0.10 | −0.76 | 0 (given) |
| Model 3, previous recipe | 13.82 | 0.253 | 12.35 | **+3.57** | +2.83 | 22.50 |
| Model 3, ResNet-50 U-Net, first recipe (TTA) | 13.89 | 0.257 | 14.59 | **+5.40** | +4.74 | 22.32 |

The rows marked TTA average the prediction over the 8 rotations and flips. The previous
recipe has no TTA.

### Per terrain (a/b RMSE at 224 px; this model's mean-color baseline in brackets)

| terrain | Model 1 previous | Model 1 first recipe | Model 3 previous | Model 3 first recipe |
|---|---|---|---|---|
| agri | 10.89 (8.93) | 11.23 (9.19) | 12.61 (8.93) | 17.24 (9.19) |
| barrenland | 7.94 (10.23) | 9.44 (10.67) | 9.24 (10.23) | 9.29 (10.67) |
| grassland | 8.15 (9.53) | 8.11 (9.66) | 16.92 (9.53) | 19.08 (9.66) |
| urban | 7.94 (6.36) | 7.64 (7.22) | 10.00 (6.36) | 12.00 (7.22) |

### What the numbers say

- **On honestly unseen scenes, color is hard.** Model 1 beats a constant color by only about
  0.1 a/b units overall. It does well on barrenland and grassland but loses on agri and urban.
- **Model 3 gets the color wrong.** Both recipes are worse than a constant color. The SAR
  tile carries little color information, and confident colors cost more than safe ones on this
  metric. Model 3's brightness error (≈ 22 Lab units) is also far from the optical image.
- **The first ResNet-50 U-Net recipe did not beat the previous one.** Both models peaked
  early (Model 1 at epoch 1, Model 3 at epoch 6) and then overfit the 17 training scenes, and
  the vivid decoding (T = 0.38, rare colors up-weighted) was penalized. The current defaults
  address both: the encoder is frozen (Model 1) or slowed down (Model 3), fewer epochs are
  trained, rebalancing is off, and T is chosen on val.

Full numbers are in `model*_run/metrics*.json` and in the dashboard's *Results* tab.

---

## 2. Setup

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
# PyTorch first, matching your machine (GPU build shown; use .../whl/cpu for CPU only):
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

All versions are pinned in [`requirements.txt`](requirements.txt). Put the dataset at `v_2/`.

---

## 3. Colorize an image / demo

[`colorize.py`](colorize.py) is the single inference path; the dashboard and the helper scripts
use it. It sends Sentinel-1 tiles to Model 3 and everything else to Model 1.

```powershell
python colorize.py --image v_2/agri/s1/<file>.png           # SAR -> Model 3
python colorize.py --image v_2/agri/s2/<file>.png --tta     # optical -> Model 1, averaged over 8 rotations/flips
python colorize.py --image other_sar_image.png --mode sar   # SAR without an _s1_ name
python colorize.py --image v_2/agri/s1/<file>.png --sharp   # cosmetic unsharp mask (disclosed in the output)
```

- **Mode detection** (`--mode auto`): the input is treated as SAR only if it is a Sentinel-1
  tile by name (`_s1_` in the file name, or inside an `s1/` folder). Everything else, RGB or
  grayscale, is treated as optical.
- **Resolution**: the network runs at 256×256. Model 1 keeps the input's full-resolution L and
  upsamples only the color. Model 3 upsamples its own predicted L.
- **Ground truth**: for a file inside `v_2/`, its pair is shown as a third panel.
- **Out-of-domain inputs**: an everyday photo such as `sample_input.jpeg` gets satellite colors.
- Outputs: `<out>` (side-by-side figure) and `<out stem>_raw.png`.

**Dashboard**: `python app.py` opens http://127.0.0.1:7860 (or `$env:PORT`). *Colorize* runs
either model. *Results* shows the test metrics, the previous recipe on the same scenes, the
baselines and the training curves.

**Scripted demo**: `.\run_demo.ps1`. Use `-Fast` to skip the live evaluation.

---

## 4. Training

```powershell
python Train_model3.py      # SAR -> L + color (GPU recommended)
python Train_model1.py      # optical L -> color
python Train_model3.py --n_images 64 --n_val 32 --epochs 1 --batch_size 8 --out _smoke   # smoke test
```

| | epochs | encoder | decoder lr | other defaults |
|---|---|---|---|---|
| Model 3 | 15 | fine-tuned at 0.05× | 5e-4 | L1 loss on L |
| Model 1 | 10 | frozen | 5e-4 | |

Both models share these defaults: batch 32, AdamW, cosine decay, 10-unit color bins, no class
rebalancing, T picked from {0.38, 0.5, 0.7, 1.0} on val, rot90/flip augmentation, and AMP.

All logic is in [`train_colorizer.py`](train_colorizer.py) and [`pipeline.py`](pipeline.py)
(inputs, bins, loss, decoding, TTA). The options are `--epochs`, `--batch_size`, `--lr`,
`--encoder_lr_mult` (0 = frozen), `--bin_grid`, `--T_grid`, `--rebal_lam`
(0.5 = boost rare colors, 1 = off), `--L_weight`, `--n_images`, `--n_val`, `--[no_]augment` and
`--[no_]amp`.

Each run writes `model_<k>.pth`, `split.json`, `history.json`, `loss_curve.png` and
`sample_predictions.png`.

### Training on Kaggle

[`kaggle/`](kaggle) runs the training and evaluation as a private Kaggle GPU kernel (2× T4):

```powershell
python kaggle/prepare_code.py --user <kaggle-user> --out ../colorsar_code_upload   # flatten the code
kaggle datasets version -p ../colorsar_code_upload -m "update"     # first time: kaggle datasets create
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4           # edit the id in kernel-metadata.json
kaggle kernels output <kaggle-user>/colorsar-train -p ../kaggle_out
```

`run_kaggle.py` has `MODE = "smoke"` for a few-minute bug check and `MODE = "full"` for the
real run. It saves each run folder as soon as that run finishes, and stops training before
Kaggle's 12 h limit. Run `kaggle datasets` commands with a *relative* path, because the CLI
mishandles `C:` paths.

---

## 5. Evaluation

```powershell
python evaluate.py --model 3 --tta --per_terrain --out model3_run/metrics_test_tta.json
python evaluate.py --model 1 --tta --eval_size 224          # the previous recipe's scoring size
python evaluate.py --model 3 --split val                    # validation scenes
python evaluate.py --model 1 --n_images 200                 # quick check
```

`evaluate.py` prints these rows: the model, the no-color baseline, the mean-color baseline and
the oracle. The oracle uses the true a/b with the same L; for Model 1 it is only a Lab round
trip, not a ceiling. It prints a/b RMSE for each row, and L RMSE for Model 3.

---

## 6. Checkpoints

- **Format 3** (written by `train_colorizer.py`) stores the whole network, encoder included,
  plus the metadata: input type, color bins, temperature, training settings and the scenes of
  each split. It does not depend on the torchvision weights of the machine that loads it.
- `model1b_run/` and `model2_run/`, and the format 1/2 checkpoints, belong to the previous
  recipe. Its code (DnCNN, Model 2, adapters) lives under the git tag `v1-legacy`, and the
  current code refuses to load those checkpoints.

---

## 7. Known limitations

- **SAR → color is close to unsolved here.** Model 3 does not beat a constant color on the
  held-out scenes. Its outputs are also smooth, because radar lacks optical texture.
- **a/b RMSE favors safe, desaturated colors.** A vivid colorizer can look better and still
  score worse.
- **Small scene count.** 17 training scenes is little variety, and the large encoder overfits
  quickly.
- **One season.** The data holds one summer scene, so no split can test a season the model has
  not seen.
- **Domain**: the models were trained on 256×256 Sentinel tiles only.

---

## 8. Files

```
├── model1_run/ model3_run/   checkpoint, split.json, history.json, metrics*.json, figures
├── model1b_run/ model2_run/  previous recipe (needs the v1-legacy code)
├── model_architecture.py     ResNet-50 encoder, U-Net decoder, checkpoint save/load (format 3)
├── pipeline.py               inputs, color bins, loss, annealed mean, decoding, TTA
├── color_utils.py            pairing, dataset, Lab conversions
├── splits.py                 scene split, split.json
├── train_colorizer.py        trainer for Models 1 and 3
├── Train_model1.py, Train_model3.py   per-model entry points
├── evaluate.py               held-out PSNR / SSIM / a/b + L RMSE with baselines
├── colorize.py               inference CLI (the one inference path)
├── colorize_image.py         alias for `colorize.py --mode gray`
├── app.py                    Gradio dashboard
├── run_demo.ps1              scripted demo
├── make_figures.py           regenerate a run's figures from its checkpoint
├── kaggle/                   Kaggle GPU kernel (run_kaggle.py, kernel-metadata.json, prepare_code.py)
├── ColorSAR_Kaggle_Training.ipynb   older notebook route (clones upstream; predates this recipe, use kaggle/)
├── visual_test.py, data_pipeline.py, count_pairs.py   sanity checks
├── requirements.txt          pinned dependencies
└── sample_input.jpeg         an everyday photo (out-of-domain example)
```
