"""
ColorSAR - train Model 3: SAR tile -> optical L + a/b.

Defaults: every pair of the training scenes, 15 epochs, batch 32, AdamW 5e-4 (encoder
x0.05) with cosine decay, color-bin loss (no rebalancing) + L1 on L, decoding temperature
picked on the val scenes, rot90/flip augmentation, mixed precision. Meant for a GPU (Kaggle 2x T4: see kaggle/).

    python Train_model3.py
    python Train_model3.py --n_images 64 --n_val 32 --epochs 1 --batch_size 8 --out _smoke
"""
from train_colorizer import main

if __name__ == "__main__":
    main(3)
