# ColorSAR - work split (5 people)

Goal: 9-algorithm comparison + proposed model (physics-informed cross-modal attention
diffusion), IEEE paper, mid-sem review slides.

## Person 1 - Kaggle runs + Results
- Upload the code, run the Kaggle smoke test, then the full benchmark (~11 h)
- Download the results, build the comparison table
- Write the **Results** section

## Person 2 - Ablation runs + Method
- **Today:** Kaggle account with GPU hours (phone verified). Runs the ablation kernel
  (~10 h) from a ready folder - 3 commands, no coding
- Write the **Method** section from the bullet draft: SAR physics (speckle, Lee filter,
  physics loss), cross-modal contrastive learning, cross-attention, frequency loss,
  latent diffusion

## Person 3 - Literature review
- **Related Work**, ~15-20 papers:
  - classical colorization (Welsh 2002)
  - GANs (Pix2Pix, CycleGAN)
  - TransUNet; wavelet networks / Focal Frequency Loss
  - diffusion (DDPM, Palette, Latent Diffusion)
  - SAR-to-optical (SEN1-2 dataset)
- One paragraph per baseline: what it is and why it is in the comparison
- Owns the BibTeX reference file

## Person 4 - IEEE paper + Introduction
- Set up the **IEEE conference template** on Overleaf, share with all 5
- Write **Introduction** and **Dataset** (Sentinel-1/2 pairs, 4 terrains, scene split)
- Write the **Abstract** last
- Final pass: page limit, formatting, every figure/table referenced

## Person 5 - Figures + slides
- **Architecture diagram** of the proposed model (draw.io / PowerPoint)
- **Visual comparison figure**: one row per algorithm (SAR | outputs | real optical) -
  a script makes the grid from the run outputs
- **Mid-sem slides**: problem, 9 algorithms, proposed model, results
- **Discussion + Conclusion** after the results: which method wins, where diffusion
  helps, failure cases

## Timeline
| When | What |
|---|---|
| Today | Smoke test + full run start (P1); Kaggle setup (P2); Overleaf ready (P4); everyone starts writing |
| Tomorrow | Full run done, ablations start (P2); diagram ready (P5) |
| Day 3 | All numbers in, then table, figures, Results, Discussion |
| Before mid-sem | Slides final, paper draft complete |

## The 9 algorithms
| # | Algorithm | Type |
|---|---|---|
| 1 | Histogram match + mean color | classical |
| 2 | Welsh statistics transfer | classical |
| 3 | ResNet-50 U-Net (current Model 3) | CNN |
| 4 | Pix2Pix | conditional GAN |
| 5 | CycleGAN | unpaired GAN |
| 6 | TransUNet | CNN + Transformer |
| 7 | Wavelet U-Net + Focal Frequency Loss | frequency domain |
| 8 | Conditional DDPM (Palette-style) | pixel diffusion |
| 9 | **PICMA (proposed)** | physics-informed cross-modal attention latent diffusion |

Ablations of PICMA: without physics, without contrastive pretraining, without cross-attention.
