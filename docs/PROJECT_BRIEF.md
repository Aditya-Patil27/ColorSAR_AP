# ColorSAR - complete project brief

One file for three jobs:
1. **Video script** - paste Part A into an AI video tool (scene by scene: narration + visuals).
2. **Revision for the exam** - Part B: every concept we use, in short.
3. **Who did what** - Part C: contribution of all 5 members.

Numbers marked **[PENDING]** come from the benchmark runs (Kaggle + Modal) on the morning of
the exam. Fill them from `results.md` (made by `python scripts/bench_figure.py all_results --out paper_figs`).
Never say a number that is not in that file.

---

## Part A - Video script (about 4 minutes)

Style for the video tool: clean, technical, dark-blue/teal satellite theme, calm narrator,
on-screen text short. 1 scene = 1 shot or slide.

### Scene 1 - The problem (0:00-0:25)
- **Visual:** Earth from orbit; clouds covering India during monsoon; an optical satellite
  image that is fully white with clouds.
- **Text on screen:** "Optical satellites can't see through clouds."
- **Narration:** "Optical satellites like Sentinel-2 take colour pictures of the Earth - but
  they are blind under clouds and at night. During the monsoon, many areas stay hidden for
  weeks."

### Scene 2 - Radar sees everything (0:25-0:50)
- **Visual:** Sentinel-1 radar satellite sending radar waves through clouds; a grey, grainy
  SAR image appears.
- **Text:** "SAR radar: works through clouds, day and night - but grey and noisy."
- **Narration:** "Radar satellites like Sentinel-1 use microwaves. They see through clouds,
  day and night. But the image is grey, grainy and hard to read for a human."

### Scene 3 - Our goal (0:50-1:05)
- **Visual:** arrow from a grey SAR tile to a colour optical tile.
- **Text:** "ColorSAR: turn radar images into optical-like colour images."
- **Narration:** "Our project, ColorSAR, learns to translate a radar image into the colour
  image an optical satellite would have taken - even when the sky is cloudy."

### Scene 4 - The data (1:05-1:25)
- **Visual:** grid of SAR/optical pairs in four groups: agriculture, barren land, grassland,
  urban. Then a map split into colours: train / validation / test scenes.
- **Text:** "16,000 paired tiles - 4 terrains - test scenes never seen in training."
- **Narration:** "We use 16,000 matched Sentinel-1 and Sentinel-2 tiles from four terrains.
  We split by satellite scene, so the test areas are places the model has never seen."

### Scene 5 - Why it is hard (1:25-1:45)
- **Visual:** two fields with the same radar texture - one green, one brown.
- **Text:** "Same radar signal -> many possible colours."
- **Narration:** "Radar measures roughness and shape, not colour. Two fields can look the
  same to radar but have different colours. A normal network just predicts the average -
  a dull, greyish image."

### Scene 6 - Our model: PICMA (1:45-2:35)
- **Visual:** architecture diagram building up step by step:
  SAR -> Lee filter -> 3-channel physics input -> ResNet-50 encoder -> (spatial features +
  320 tokens) -> cross-attention U-Net denoising a 32x32 latent in 50 steps -> VAE decoder ->
  colour image. Side box: contrastive pretraining (SAR encoder <-> optical encoder).
- **Text:** "PICMA - Physics-Informed Cross-Modal Attention latent diffusion"
- **Narration:** "Our proposed model is PICMA. First, physics: radar speckle noise follows a
  known model, so a Lee filter separates real edges from noise, and this guides both the
  input and the loss. Second, cross-modal learning: before training, we teach the radar
  encoder to match the features of the real optical image. Third, a diffusion model creates
  the colour image step by step, and at every step it looks at the radar through
  cross-attention. Finally a frequency-domain loss keeps the fine details sharp."

### Scene 7 - Comparison with 8 other methods (2:35-3:05)
- **Visual:** table of 9 algorithms appearing row by row: Histogram matching, Welsh,
  ResNet-50 U-Net, Pix2Pix, CycleGAN, TransUNet, Wavelet U-Net, DDPM, PICMA.
  Then the comparison figure: rows of test tiles, columns of every method's output.
- **Text:** "9 algorithms - same data, same test tiles, same scoring code."
- **Narration:** "We compared PICMA with eight methods: two classical methods, CNNs, GANs,
  a Transformer, a frequency-domain network and a pixel-space diffusion model. All of them
  use the same split, the same test tiles and the same metrics."

### Scene 8 - Results (3:05-3:30)
- **Visual:** results table with the best value per column highlighted. **[PENDING: insert
  results.md table]**
- **Narration (fill from results.md):** "On the held-out test scenes, PICMA reaches
  [PENDING] LPIPS and [PENDING] FID, against [PENDING] for the best baseline. Methods that
  predict the average colour win on PSNR but look blurry - this is the known
  perception-distortion trade-off."

### Scene 9 - Live demo (3:30-3:50)
- **Visual:** screen recording of the web app: pick a test SAR tile -> click Colorize ->
  input | output | ground truth side by side; then the Results tab.
- **Text:** "Live demo: [DEPLOYMENT URL]"
- **Narration:** "The colorizer is deployed as a web app. Upload a radar tile and get the
  colour image in seconds."

### Scene 10 - Conclusion (3:50-4:05)
- **Visual:** cloudy optical image fading into a colourised SAR image of the same place.
- **Text:** "ColorSAR - see the Earth in colour, even through clouds." + team names.
- **Narration:** "ColorSAR combines radar physics, cross-modal learning and diffusion to
  give an optical view of the Earth when clouds hide it. Thank you."

---

## Part B - Revision for tomorrow

### B1. The project in 5 lines
- Input: Sentinel-1 SAR tile (VV polarisation, 256x256, grey). Output: Sentinel-2 RGB tile.
- Data: SEN1-2 style pairs, 4 terrains (agri, barren land, grassland, urban), ~16,000 pairs.
- **Scene split:** train 17 scenes / 11,877 pairs, val 6 / 2,463, **test 7 scenes / 1,660 pairs**.
- 9 algorithms compared, the 9th is our proposed **PICMA**.
- Scored in Lab colour space with PSNR, SSIM, ab RMSE, L RMSE, LPIPS, FID.

### B2. Lab colour space
- **L** = lightness (0 black ... 100 white). **a** = green <-> red. **b** = blue <-> yellow.
- Why: separates brightness from colour, and distances follow human perception. SAR gives
  structure/brightness hints but almost no colour, so we score L error and ab error separately.

### B3. SAR speckle and the Lee filter (denoising)
- **Speckle:** radar echoes inside one pixel interfere -> grainy noise. It is
  **multiplicative**: I = R x n (true signal x noise with mean 1), so bright areas are noisier.
  A Gaussian blur would also destroy edges, so we use the Lee filter.
- **Pixel:** one value of the image. For each pixel we look at a **7x7 window** around it.
- **mean (m):** average of the 49 values in the window. **std:** how much they vary.
- **ENL** (Equivalent Number of Looks) = **4.4** for Sentinel-1 IW GRD. Higher ENL = less speckle.
- **Cu** = 1 / sqrt(ENL) ~ **0.48** - the variation that pure speckle would produce.
- **Ci** = std / mean of the window - the variation we actually see.
- **w** = (1 - Cu^2/Ci^2) / (1 + Cu^2), clipped to [0, 1].
  - Ci ~ Cu -> only speckle -> w ~ 0 -> use the mean (smooth it).
  - Ci >> Cu -> real edge/building -> w ~ 1 -> keep the pixel.
- Filtered pixel: **I_f = m + w (I - m)**.
- **One-line answer:** "Lee filter compares the local variation with the speckle variation;
  where they match it smooths, where the variation is bigger it keeps the edge."

### B4. Physics-informed parts of PICMA
- **Physics input (3 channels):** raw SAR (dB), Lee-filtered SAR (dB), Ci scaled by
  sqrt(ENL)/4. (The w map is NOT an input; it is used only in the loss.)
- **Physics loss (weight 0.05):**
  1. smoothness: (1 - w) x |gradient of ab| - colour must be flat where radar is homogeneous;
  2. edge alignment: w x (1 - |cos(angle between grad L and grad I_f)|) - brightness edges
     must follow radar edges. |cos| because brightness can flip between sensors (metal roof
     bright in SAR, dark in optical).

### B5. Cross-modal learning
- **Contrastive pretraining (InfoNCE, CLIP-style):** SAR encoder (ResNet-50) and optical
  encoder (ResNet-18), 128-d projections, temperature 0.1, batch 64. Each SAR tile must pick
  its own optical twin out of 64. Result: SAR features already carry land-cover/colour info.
- **Cross-attention:** the denoiser (optical side) asks questions (queries) to **320 SAR
  tokens** (16x16 + 8x8 feature maps) - every position can use context from the whole tile.
- **Spatial path:** ResNet layer2 (32x32) -> 64 channels concatenated to the latent -
  pixel-aligned guidance. Concat = local, cross-attention = global.

### B6. Diffusion
- **DDPM:** add noise to the image in T = 1000 steps; the network learns to predict the noise
  (loss = MSE between predicted and real noise). Generation = remove noise step by step.
- **Latent diffusion:** work on a **32x32x4 latent** from the frozen Stable Diffusion VAE
  instead of 256x256x3 pixels - 48x smaller, so it trains on Kaggle T4s in hours.
- **DDIM:** faster sampler, **50 steps** instead of 1000, nearly same quality.
- **K = 4 samples:** different random start -> different plausible colourings.
  "Sample" row = one realistic output; "mean of K" row = average of 4 (blurrier, higher PSNR).
- **Why diffusion, not U-Net:** a regression U-Net predicts the average colour -> grey.
  Our own ResNet-50 U-Net (Model 3) even lost to the mean-colour baseline
  (ab RMSE 11.73, i.e. 2.54 worse than just painting the mean colour - README). GANs are
  sharper but unstable.

### B7. Frequency domain
- **Focal Frequency Loss (FFL, weight 1.0):** compare 2-D Fourier spectra of prediction and
  target; frequencies with big errors get more weight -> fights blur, keeps detail.
- Applied every 4th training step on 4 tiles at low noise (t < 300), after decoding the
  latent to an image. Also the Wavelet U-Net baseline (Haar wavelets) is frequency-domain.

### B8. Training setup
- AdamW, lr 1e-4 (SAR encoder 2e-5), batch 16, gradient clip 1.0, mixed precision,
  EMA 0.999, augmentation = same random rot90 + flip on SAR and optical.
- PICMA: 0.4 h contrastive + 9.3 h diffusion on 2x NVIDIA T4 (Kaggle).
- Baselines: same split, same test tiles, same scoring code. GANs + ablation run on Modal,
  also 2x T4, so hardware matches.

### B9. The 9 algorithms
| # | Algorithm | Type | One line |
|---|---|---|---|
| 1 | Histogram match + mean colour | classical | match SAR brightness to optical, paint the average colour |
| 2 | Welsh (2002) | classical | copy colour from a reference image by matching local statistics |
| 3 | ResNet-50 U-Net (Model 3) | CNN | encoder-decoder regression of Lab |
| 4 | Pix2Pix | conditional GAN | U-Net generator + PatchGAN discriminator, paired |
| 5 | CycleGAN | unpaired GAN | two generators, cycle-consistency loss |
| 6 | TransUNet | CNN + Transformer | Transformer in the bottleneck for global context |
| 7 | Wavelet U-Net + FFL | frequency domain | Haar wavelet down/up-sampling |
| 8 | DDPM (Palette-style) | pixel diffusion | diffusion on 256x256 pixels, SAR concatenated |
| 9 | **PICMA (ours)** | latent diffusion | physics + contrastive + cross-attention + FFL |

Ablations (same 2 h each): PICMA short, without physics, without cross-attention, without
contrastive pretraining.

### B10. Metrics
| Metric | Meaning | Better |
|---|---|---|
| PSNR | pixel error in dB (log of MSE) | higher |
| SSIM | structure similarity (light, contrast, structure), 0-1 | higher |
| ab RMSE | colour error in Lab a/b | lower |
| L RMSE | lightness error | lower |
| LPIPS | distance of deep (AlexNet) features - looks-alike to humans | lower |
| FID | distance between feature statistics of all outputs vs. all real images - realism of the set | lower |

- **Perception-distortion trade-off** (Blau & Michaeli, CVPR 2018): no method can be best in
  both pixel accuracy (PSNR) and realism (LPIPS/FID). Blurry average = high PSNR, low realism.
- FID on 1,660 images is biased upward - fine for comparing our methods, not with papers.

### B11. Likely viva questions (short answers)
- **Why scene split?** Tiles of one scene look alike; a random split leaks near-copies into test.
- **Why physics-informed?** The known speckle model sets the input channels and a loss.
- **Where is cross-modal?** Contrastive SAR<->optical pretraining + cross-attention to SAR tokens.
- **Where is frequency domain?** Focal Frequency Loss; Wavelet U-Net baseline.
- **Your PSNR is lower than the U-Net?** PSNR rewards blur; see perception-distortion; compare
  LPIPS/FID. (Say only if the numbers show it.)
- **Is the comparison fair?** Same data, tiles, scoring; training hours are in the table.
- **What is deployed?** The trained colorizer web app (Model 3 for SAR, Model 1 for grey
  photos). PICMA needs 50 diffusion steps x 4 samples, too slow for a free CPU server;
  its results are in the paper.
- Full list: `docs/VIVA_PREP.md`.

### B12. Weak points (say them before they ask)
- dB scaling of the 8-bit SAR tiles (-25..0 dB) is an assumption about the dataset export.
- Only VV polarisation; VH would help vegetation.
- Training is hours on T4s, not days on big GPUs - absolute numbers are lower than papers.
- FID on 1,660 images.
- The data holds only one summer scene, so no split can test a season the model has not seen
  (all 7 test scenes are fall scenes).

### B13. Submission checklist (all mandatory, upload only after ma'am approves)
| Item | Status / owner |
|---|---|
| Final PPT (.pptx) | teammates |
| All project files | GitHub repo `Aditya-Patil27/ColorSAR_AP` (branch `benchmark`) |
| Dataset link | https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain |
| Jupyter notebook with outputs (.ipynb) | Kaggle bench notebook output / `ColorSAR_Kaggle_Training.ipynb` |
| CP project report (.docx) | teammates |
| IEEE paper (.docx) | teammates |
| Deployment link | [DEPLOYMENT URL] |
| Similarity report + AI plagiarism report | run on the final report and paper |
| Video demo | Part A of this file + screen recording of the app |

Exam: Room 4008, starts 9:30 AM, formal dress. Batch reporting 10:00 / 11:30 / 12:30.

---

## Part C - Contribution of all 5 members

| Member | Contribution |
|---|---|
| **Person 1 - [Name]** (team lead) | Project idea and code: data pipeline, scene split, Lab scoring, all 9 algorithms and PICMA in `bench_run.py` / `bench_models.py`; Kaggle and Modal training runs (PICMA, GANs, ablations); results table and comparison figure (`bench_figure.py`); deployment of the web app; viva notes |
| **Person 2 - [Name]** | Method section of paper and report: SAR physics (speckle, Lee filter, physics loss), contrastive learning, cross-attention, frequency loss, latent diffusion (from `docs/METHOD_NOTES.md`); ablation analysis |
| **Person 3 - [Name]** | Literature survey: 15-20 papers (classical colorization, GANs, Transformers, frequency methods, diffusion, SAR-to-optical); Related Work section; reference list; baseline descriptions; similarity / AI-plagiarism reports |
| **Person 4 - [Name]** | IEEE paper in Word: template, Introduction, Dataset, Abstract, final formatting; CP project report assembly |
| **Person 5 - [Name]** | Architecture diagram, comparison figures, final PPT, Discussion and Conclusion; demo video |

Kaggle/Modal training runs by member: Person 1 - `picma` set (Kaggle) + `gans` and
`ablation` sets (Modal); teammates - `freq` (TransUNet, Wavelet U-Net) and `ddpm` (Kaggle).
Edit the names and adjust the rows so they match what each person really did.
