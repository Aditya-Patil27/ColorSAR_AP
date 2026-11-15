"""
ColorSAR - train Model 1 (optical L -> a/b).

Defaults: every pair of the training scenes, 20 epochs, batch 32, AdamW 5e-4 (encoder
x0.1) with cosine decay, color-bin loss, rot90/flip augmentation, mixed precision.
Multiple GPUs are used automatically (DataParallel). All logic lives in
train_colorizer.py; see it for every option.

    python Train_model1.py
    python Train_model1.py --n_images 64 --n_val 32 --epochs 1 --batch_size 8 --out _smoke
"""
from train_colorizer import main

if __name__ == "__main__":
    main(1)
