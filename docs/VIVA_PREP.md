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
**Q: Your PSNR is lower than the U-Net's - so your model is worse?**
No. PSNR rewards the average colour (blurry). The perception-distortion trade-off (Blau &
Michaeli, CVPR 2018) proves no method can be best at both. We compare on LPIPS/FID for
realism and report PSNR/SSIM honestly. (Say this only if the numbers show it.)

**Q: What is FID? Any limitation?**
Distance between Inception-feature statistics of all generated vs. all real test images;
lower = more realistic set. With 1660 images it is biased upwards - fine for comparing
methods on the same test set, not comparable with papers using 50k images.

**Q: Is the comparison fair? PICMA trained longest.**
Same split, same test tiles, same scoring code. Training hours are in the table. The
ablation runs all use the same 2 h budget, so component effects are fair. (If asked:
baselines converge faster; given more time the gap may shrink - honest answer.)

**Q: How do you know each component helps?**
Ablations: remove physics, contrastive pretraining, cross-attention one at a time with
the same budget; the score drop shows each part's contribution. (Only if the ablation
run was done - otherwise: "planned, limited GPU time".)

## Weak points - know them before they ask
- dB scaling of the 8-bit SAR tiles (-25..0 dB) is an assumption about the dataset's
  export.
- Single polarisation (VV) only; VH would add vegetation information.
- Kaggle T4 budget: all models are trained for hours, not days; absolute numbers are lower
  than in papers with big GPUs.
- FID on 1660 test images (see above).
- Some baselines (classical ones) need no training; the GANs/U-Nets got ~2-4 h each.
