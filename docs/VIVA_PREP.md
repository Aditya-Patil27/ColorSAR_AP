# Viva / review prep - likely questions with short answers

## Problem and data
**Q: Why is SAR-to-optical hard?**
SAR measures radar backscatter (roughness, geometry, moisture), not colour. Two fields with
the same texture can be green or brown. So colour has to be *inferred* from context - the
problem has many valid answers.

**Q: Why use SAR at all if optical exists?**
Radar sees through clouds and at night. Optical is often cloudy (monsoon!). Colourised
SAR gives an optical-like image when no optical image is available.

**Q: How did you split the data? Why not random?**
By satellite scene: test scenes never appear in training. Tiles from one scene overlap
and look alike - a random split puts near-copies in train and test and inflates scores.

**Q: Why Lab and not RGB?**
Lab separates brightness (L) from colour (a, b), and Lab distances follow human
perception. SAR carries brightness/structure information but almost no colour, so we
measure the two errors separately (L RMSE, ab RMSE).

## Physics
**Q: What is speckle? Why not a Gaussian blur?**
Interference of radar echoes inside one pixel - multiplicative noise (signal x random
factor), so bright areas are noisier. A blur also destroys edges. The Lee filter smooths
only where the local statistics match pure speckle.

**Q: Explain the Lee filter in one line.**
Compare the local coefficient of variation Ci with the speckle one Cu = 1/sqrt(ENL): if
equal, it is speckle -> replace by the local mean; if much larger, it is a real edge ->
keep the pixel. Weight w = (1 - Cu^2/Ci^2)/(1 + Cu^2).

**Q: What is ENL, where does 4.4 come from?**
Equivalent Number of Looks: how many independent looks were averaged in the product -
higher = less speckle. ~4.4 is the standard value for Sentinel-1 IW GRD high resolution.

**Q: Why is it "physics-informed"?**
The speckle model (multiplicative, known ENL) decides (1) the input channels and (2) a loss:
flat colour where the radar is homogeneous, edges aligned with radar edges where it has
structure. The network is constrained by how the sensor works, not just by data.

**Q: Why |cos| in the edge loss?**
Brightness can flip between sensors (metal roof: bright in SAR, dark in optical). We care
that edges are in the same place and direction, not their sign.

## Model
**Q: Why diffusion and not a U-Net or GAN?**
With many valid answers, a regression network (U-Net) predicts the *average* colour ->
blurry, greyish output. Our ResNet-50 U-Net even lost to the mean-colour baseline. A GAN
is sharper but unstable (mode collapse). Diffusion learns the full distribution and samples
one realistic answer - stable training and sharp output.

**Q: Why latent diffusion?**
32x32x4 latents instead of 256x256x3 pixels: 48x smaller, so training fits on Kaggle T4s
in hours. The pretrained Stable Diffusion VAE already knows how to decode sharp textures.

**Q: What is the cross-modal part?**
Two things: (1) contrastive (InfoNCE) pretraining makes SAR features similar to the
features of the matching optical image; (2) cross-attention: the denoiser (optical
side) queries 320 SAR tokens, so information flows across modalities by attention.

**Q: Cross-attention vs. concatenation - why both?**
Concatenation gives pixel-aligned local guidance. Cross-attention gives global context:
a pixel can look at the whole tile (a river, a town) to decide its colour.

**Q: Where is the frequency-domain part?**
(1) Focal Frequency Loss in PICMA: compares Fourier spectra, weights hard frequencies up,
fights blur. (2) The Wavelet U-Net baseline: Haar wavelet down/up-sampling instead of
pooling.

**Q: What is DDIM? Why 50 steps?**
A deterministic sampler for a model trained as DDPM; it skips steps. 50 instead of 1000
steps gives nearly the same quality ~20x faster.

**Q: What does "mean of K" mean?**
Diffusion gives a different image per random start. We draw 4. One sample = realistic
output (fair for LPIPS/FID). The average of 4 = blurrier, higher PSNR. Shows the
perception-distortion trade-off with our own model.

## Results and metrics
Final numbers (18 Nov, 1660 test tiles, 7 held-out scenes, all on 2x T4). Lower is better for
RMSE/LPIPS/FID.

| algorithm | PSNR | SSIM | ab RMSE | L RMSE | LPIPS | FID | train h |
|---|---|---|---|---|---|---|---|
| PICMA (ours), 1 sample | 7.77 | 0.089 | 32.80 | 40.09 | 0.820 | **244.9** | 1.52 |
| PICMA (ours), mean of 4 | 11.10 | 0.186 | 19.70 | 29.09 | 0.783 | 258.9 | 1.52 |
| histmatch | 10.88 | 0.055 | **9.12** | 30.63 | 0.687 | 293.1 | 0 |
| welsh | **14.15** | 0.210 | 9.22 | **20.78** | **0.665** | 266.1 | 0 |
| unet_r50 (our Model 3) | 13.96 | **0.244** | 11.73 | 22.27 | 0.813 | 245.7 | trained earlier |

Teammates' rows (TransUNet, Wavelet U-Net, DDPM) get added when their results arrive.

**Q: Your PSNR is lower than the U-Net's - so your model is worse?**
On these numbers, yes, on almost every metric. Do not use the perception-distortion
argument here: PICMA's LPIPS (0.820) is also worse than every baseline's. Its only lead is
FID (244.9 vs 245.7 for the U-Net), and that gap is too small to call a win on 1660 images.
Say it straight: the model is undertrained (next question) and we report it as it is.

**Q: Why is PICMA so much worse than the baselines?**
It got 1.5 h of training: 0.4 h contrastive pretraining and 1 h of diffusion, which is only
8 epochs (5239 steps) for a 113M-parameter model. Diffusion models usually need far more
steps than that. The parts we can check did work: contrastive pretraining reached in-batch
top-1 = 1.00, so SAR and optical features do match. The colour head has not converged. A
sign of that: averaging 4 samples cuts the colour error from 32.8 to 19.7, so the samples
still disagree a lot on colour. Fix: the planned 9.3 h run (it was killed - see below).

**Q: What is FID? Any limitation?**
Distance between Inception-feature statistics of all generated vs. all real test images;
lower = more realistic set. With 1660 images it is biased upwards - fine for comparing
methods on the same test set, not comparable with papers using 50k images.

**Q: Is the comparison fair?**
Same split, same test tiles, same scoring code, same GPU (2x T4) for every row. It is not
fair *against* PICMA in time: histmatch and welsh need no training, and the U-Net (our
Model 3) was trained earlier for hours, while PICMA got 1.5 h. Training hours are in the
table.

**Q: What happened to the longer PICMA runs?**
- Planned 9.3 h run on 2x T4 (Kaggle): killed by the platform (exit -9, most likely out of
  memory) at epoch 8, after ~4 h. Nothing was saved.
- Two shorter runs on one H100 (Modal): both cancelled early (the second during contrastive
  pretraining, at 0.34 h). Nothing was saved.
- The backup run on 2x T4 (0.4 h + 1 h) finished. It is the only PICMA row, and it runs on
  the same hardware as the baselines.

**Q: How do you know each component helps?**
We do not yet. Ablations (remove physics, contrastive pretraining, cross-attention one at a
time, same 2 h each) are written and ready (`ablation` set in `kaggle/bench/run_bench.py`)
but were not run: limited GPU time. What we can show: contrastive pretraining converges
(top-1 = 1.00).

## Weak points - know them before they ask
- dB scaling of the 8-bit SAR tiles (-25..0 dB) is an assumption about the dataset's
  export.
- Single polarisation (VV) only; VH would add vegetation information.
- Kaggle T4 budget: all models are trained for hours, not days; absolute numbers are lower
  than in papers with big GPUs.
- FID on 1660 test images (see above).
- Some baselines (classical ones) need no training; the GANs/U-Nets got ~2-4 h each.
- PICMA is undertrained (1.5 h, 8 epochs). It loses to the simple baselines on PSNR, SSIM,
  RMSE and LPIPS, and only ties on FID.
- No ablation results; all longer PICMA runs failed (see "What happened to the longer
  PICMA runs?").
