# Method section - bullet notes for the writer

Turn each block into 1-2 paragraphs. Numbers are the real settings in `bench_run.py`
(`run_picma`) and `bench_common.py`. Name of the model: **PICMA** - Physics-Informed
Cross-Modal Attention latent diffusion.

## A. Problem setup
- Input: one Sentinel-1 SAR tile (VV, 256x256, 8-bit, dB scale -25..0 dB). Output: the
  matching Sentinel-2 RGB tile.
- Data: SEN1-2 style pairs, 4 terrains (agriculture, barren land, grassland, urban).
- **Scene split**: train / val / test are different satellite scenes, so no test area is
  ever seen in training (tiles of one scene overlap and look alike - a random tile
  split would leak). Test set: 1660 tiles.
- Colour is scored in **CIE Lab**: L = lightness, a/b = colour.

## B. SAR physics (speckle model)
- SAR speckle is **multiplicative**: I = R * n, n has mean 1 and variance 1/ENL.
  ENL (equivalent number of looks) = 4.4 for Sentinel-1 IW GRD.
- Speckle-only areas therefore have a fixed coefficient of variation
  Cu = 1/sqrt(ENL) ~ 0.48.
- **Lee filter** (7x7 window, on linear intensity): local mean m, local variance v,
  Ci = sqrt(v)/m.
  w = (1 - Cu^2/Ci^2) / (1 + Cu^2), clipped to [0, 1];  I_f = m + w (I - m).
  w ~ 0 = homogeneous (pure speckle), w -> 1 = real structure (edges, buildings).
- **Physics input** (3 channels instead of a grey image repeated 3 times):
  raw SAR (dB), Lee-filtered SAR (dB), Ci scaled so the speckle level is a fixed value.
- **Physics loss** on the predicted image (L, ab), two terms:
  1. smoothness: sum over pixels of (1 - w) * |grad ab| - colour must be flat where the
     radar says the surface is homogeneous;
  2. edge alignment: w * (1 - |cos angle(grad L, grad I_f)|) - brightness edges must run
     along the despeckled radar edges. |cos| because radar and optical brightness can
     have opposite sign (bright roof in SAR, dark in optical).
  Weight 0.05.

## C. Cross-modal contrastive pretraining
- Before diffusion training, the SAR encoder (ResNet-50, ImageNet init) is aligned with an
  optical encoder (ResNet-18, ImageNet init) on the paired tiles.
- Each side: global average pooling -> 2-layer MLP projection head -> 128-d, L2 normalised.
- **InfoNCE** (symmetric, temperature 0.1): in a batch of 64 pairs, each SAR tile must pick
  its own optical twin out of 64 (CLIP-style). 0.4 h on 2xT4, lr 1e-4.
- Effect: the SAR features already carry land-cover / colour information before the
  generator sees them. Only the SAR encoder is kept.

## D. Latent diffusion
- Frozen **Stable Diffusion VAE** (`sd-vae-ft-mse`): 256x256x3 optical -> 32x32x4 latent
  (x0.18215). 48x fewer numbers than pixels -> faster training, sharper decoded detail.
- **DDPM** forward process: T = 1000 steps, linear beta 1e-4 .. 0.02; the network predicts the
  added noise epsilon; loss = MSE(eps_pred, eps).
- **Sampling**: DDIM, 50 steps, deterministic given the start noise; predicted x0 clipped to
  the latent range. K = 4 samples per test tile (different start noise).

## E. Denoiser and SAR conditioning (cross-modal attention)
- Denoiser: U-Net on the 32x32 latent, 3 levels (128/256/512 channels), 2 residual blocks
  per level, self-attention + **cross-attention** at 16x16 and 8x8.
- SAR encoder outputs (ResNet-50 on the physics input):
  - **spatial path**: layer2 (32x32x512) -> 1x1 conv -> 64 channels, concatenated to the
    noisy latent - pixel-aligned guidance;
  - **token path**: layer3 (16x16) + layer4 (8x8) = **320 tokens**, projected to 512-d,
    learned position embedding. In cross-attention the latent features are the queries
    and the SAR tokens the keys/values: every latent position can take context from the
    whole tile (river, town, field boundaries).

## F. Auxiliary losses on the decoded image
- Every 4th step, for 4 tiles at low noise (t < 300): predict x0 from eps, decode with the
  VAE, convert to Lab, then add
  - physics loss (B), weight 0.05;
  - **Focal Frequency Loss** (Jiang et al., ICCV 2021), weight 1.0: distance between 2-D
    Fourier spectra of prediction and target, frequencies with large error weighted up -
    fights the blur that pixel losses give (frequency-domain part of the method).
- Low t only: there the decoded x0 is a meaningful image.

## G. Training details
- AdamW, lr 1e-4 (SAR encoder 2e-5), no weight decay; batch 16; gradient clipping 1.0;
  mixed precision; data augmentation = same random 90-degree rotation + flip on SAR and
  optical; EMA of weights (0.999) used for testing.
- Fixed wall-clock budget on 2x NVIDIA T4 (Kaggle); lr decays linearly over the second
  half of the budget. PICMA: 0.4 h contrastive + 9.3 h diffusion.
- Baselines use the same split, the same test tiles and the same scoring code.

## H. Ablations (if the ablation run is done)
Same 2 h budget each: full model vs. without physics (plain SAR input, no physics loss),
without contrastive pretraining (ImageNet encoder only), without cross-attention (spatial
path only).

## Figure for this section
Architecture diagram (Person 5): SAR -> Lee filter -> 3-ch physics input -> ResNet-50 ->
(spatial 32x32 concat | 320 tokens -> cross-attention) -> latent U-Net x 50 DDIM steps ->
VAE decoder -> RGB; side box: contrastive pretraining with the optical ResNet-18;
loss box: eps-MSE + physics + FFL.
