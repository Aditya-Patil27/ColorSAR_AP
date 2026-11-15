"""
ColorSAR - model inputs, output decoding, color bins and losses

One place that knows how a batch becomes network input and how the raw head output
becomes (L, a/b). Training, evaluation and inference all call these functions, so
they cannot disagree.

Inputs (meta["input"])
    "optical_L"   Model 1: L of the optical (s2) image, as gray on 3 channels
    "sar3"        Model 3: the SAR (s1) tile as 3 channels - raw, Gaussian-smoothed
                  (sigma 1) and (sigma 2.5), i.e. three despeckling levels
    Both are ImageNet-normalized, the statistics the ResNet-50 was pretrained on.

Output (head of model_architecture.Colorizer)
    predict_L: the first channel is L (tanh) - Model 3 predicts brightness too
    then Q color-bin logits (Zhang et al. 2016, "Colorful Image Colorization"):
    trained with cross-entropy on soft-encoded targets (optionally class-rebalanced),
    decoded with the annealed mean at temperature T. T is picked on the validation
    scenes and stored in the checkpoint: T=1 is the plain mean (lowest error when the
    color is uncertain), lower T gives more vivid but riskier colors.
"""
import math

import torch
import torch.nn.functional as F

from color_utils import SIZE, ab_to_lab_units, resize_t, rgb_to_L
from model_architecture import IMAGENET_MEAN, IMAGENET_STD

INPUT_OF_MODEL = {1: "optical_L", 3: "sar3"}


# --------------------------------------------------------------------------- inputs
def _gauss(x, sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    return F.conv2d(x, k.view(1, 1, -1, 1))


def _imagenet(x):
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def sar3_input(gray, size=SIZE):
    """SAR gray (N,1,h,w) in [0,1] -> ImageNet-normalized (N,3,size,size)."""
    g = resize_t(gray, size)
    return _imagenet(torch.cat([g, _gauss(g, 1.0), _gauss(g, 2.5)], dim=1))


def optical_input(L):
    """Normalized L (N,1,H,W) in [-1,1] -> ImageNet-normalized gray (N,3,H,W)."""
    return _imagenet(((L + 1.0) * 0.5).repeat(1, 3, 1, 1))


def model_input(meta, gray=None, rgb=None):
    """-> (network input (N,3,H,W), given L (N,1,H,W) or None when L is predicted).

    gray: SAR (N,1,h,w) in [0,1]; rgb: optical (N,3,H,W) in [0,1].
    """
    kind = meta["input"]
    if kind == "optical_L":
        L = rgb_to_L(rgb)
        return optical_input(L), L
    if kind == "sar3":
        return sar3_input(gray), None
    raise ValueError(f"unknown input kind {kind!r}")


# --------------------------------------------------------------------------- bins
def lab_to_norm_ab(ab_lab):
    return (ab_lab + 128) * 2 / 255 - 1


def compute_bins(ab_norm, grid=10.0, min_count=1):
    """Color bins from training a/b. ab_norm: (M,2) normalized samples.

    Returns (centers (Q,2) in Lab units, prior (Q,) probabilities): the cells of a
    `grid`-unit lattice over a/b that actually occur in the data.
    """
    ab = ab_to_lab_units(ab_norm)
    idx = torch.round(ab / grid).long()
    keys, counts = torch.unique(idx, dim=0, return_counts=True)
    keep = counts >= min_count
    centers = keys[keep].float() * grid
    prior = counts[keep].float()
    return centers, prior / prior.sum()


def class_weights(centers, prior, lam=0.5, sigma=5.0):
    """Zhang et al. rebalancing: w ~ 1 / ((1-lam) * smoothed prior + lam / Q), E_p[w] = 1."""
    d2 = torch.cdist(centers, centers).pow(2)
    k = torch.exp(-d2 / (2 * sigma ** 2))
    smooth = (k @ prior)
    smooth = smooth / smooth.sum()
    w = 1.0 / ((1 - lam) * smooth + lam / len(prior))
    return w / (smooth * w).sum()


def bin_loss(logits, ab_norm, centers, weights, stride=4, k=5, sigma=5.0):
    """Class-rebalanced cross-entropy against soft-encoded a/b targets.

    Evaluated on every `stride`-th pixel to keep memory small.
    """
    lg = logits[:, :, ::stride, ::stride].flatten(2).transpose(1, 2).float()   # (N,P,Q)
    ab = ab_to_lab_units(ab_norm[:, :, ::stride, ::stride]).flatten(2).transpose(1, 2)  # (N,P,2)
    d2 = ((ab.unsqueeze(2) - centers.view(1, 1, -1, 2)) ** 2).sum(-1)          # (N,P,Q)
    kk = min(k, centers.shape[0])
    dk, ik = d2.topk(kk, dim=-1, largest=False)
    soft = torch.softmax(-dk / (2 * sigma ** 2), dim=-1)
    target = torch.zeros_like(lg).scatter_(-1, ik, soft)
    ce = -(target * F.log_softmax(lg, dim=-1)).sum(-1)                          # (N,P)
    return (ce * weights[ik[..., 0]]).mean()


def annealed_mean(logits, centers, T=0.38):
    """Bin logits (N,Q,H,W) -> normalized a/b (N,2,H,W). T<1 sharpens toward the mode."""
    p = torch.softmax(logits.float() / T, dim=1)
    ab = torch.einsum("nqhw,qc->nchw", p, centers.to(p.dtype))
    return lab_to_norm_ab(ab)


# --------------------------------------------------------------------------- outputs
def decode_output(raw, meta, centers=None):
    """Raw head output -> (predicted L (N,1,H,W) or None, normalized a/b (N,2,H,W))."""
    raw = raw.float()
    L = torch.tanh(raw[:, :1]) if meta["predict_L"] else None
    logits = raw[:, 1:] if L is not None else raw
    if centers is None:
        centers = torch.tensor(meta["bins"], dtype=raw.dtype, device=raw.device)
    return L, annealed_mean(logits, centers, meta.get("T", 0.38))


# the 8 symmetries of a square: (k quarter turns, horizontal flip)
_D4 = [(k, f) for f in (False, True) for k in range(4)]


def _apply(x, k, flip):
    x = torch.rot90(x, k, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if flip else x


def _undo(x, k, flip):
    x = torch.flip(x, dims=(3,)) if flip else x
    return torch.rot90(x, -k, dims=(2, 3))


@torch.no_grad()
def predict(net, meta, gray=None, rgb=None, tta=False):
    """Full inference on a batch -> (L used for the output (N,1,H,W), a/b normalized).

    tta=True averages L and a/b over the 8 rotations/flips of the input (satellite
    tiles have no "up", and the models were trained with the same augmentation).
    """
    x, L_given = model_input(meta, gray=gray, rgb=rgb)
    if not tta:
        L_pred, ab = decode_output(net(x), meta)
    else:
        L_sum, ab_sum = 0.0, 0.0
        for k, f in _D4:
            L_t, ab_t = decode_output(net(_apply(x, k, f)), meta)
            ab_sum = ab_sum + _undo(ab_t, k, f)
            if L_t is not None:
                L_sum = L_sum + _undo(L_t, k, f)
        ab = ab_sum / len(_D4)
        L_pred = L_sum / len(_D4) if meta["predict_L"] else None
    return (L_pred if L_pred is not None else L_given), ab
