"""
ColorSAR - model: ResNet-50 encoder + U-Net decoder with a full-resolution head

    input (N,3,256,256)
      stem   conv1+bn+relu      64 ch  @128  ─────────────────────────┐
      layer1                   256 ch  @64   ────────────────────┐    │
      layer2                   512 ch  @32   ───────────────┐    │    │
      layer3                  1024 ch  @16   ──────────┐    │    │    │
      layer4                  2048 ch  @8              │    │    │    │
        └─ up ─ cat ───────────────────────────────────┘    │    │    │
             └─ up ─ cat ───────────────────────────────────┘    │    │
                  └─ up ─ cat ───────────────────────────────────┘    │
                       └─ up ─ cat ───────────────────────────────────┘
                            └─ up to 256 ─ cat input ─ conv ─ head (N,k,256,256)

The encoder starts from ImageNet weights (IMAGENET1K_V2, pinned: DEFAULT has changed
between torchvision releases) and is fine-tuned at a lower learning rate than the
decoder; its BatchNorm statistics stay frozen. Every layer is upsampled before the
head, so the output is predicted at the input resolution instead of being upsampled
from half resolution.

A checkpoint stores the whole network (encoder included) and its metadata, so it does
not depend on the torchvision weights on the machine that loads it.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

CHECKPOINT_FORMAT = 3

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        r = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.layers = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])

    def forward(self, x):
        feats = [self.stem(x)]
        y = self.pool(feats[0])
        for layer in self.layers:
            y = layer(y)
            feats.append(y)
        return feats            # strides 2, 4, 8, 16, 32


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.up4 = _block(2048 + 1024, 512)
        self.up3 = _block(512 + 512, 256)
        self.up2 = _block(256 + 256, 128)
        self.up1 = _block(128 + 64, 64)
        self.full = _block(64 + 3, 32)
        self.head = nn.Conv2d(32, out_channels, 1)

    @staticmethod
    def _up_cat(x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x, feats):
        s2, s4, s8, s16, s32 = feats
        y = self.up4(self._up_cat(s32, s16))
        y = self.up3(self._up_cat(y, s8))
        y = self.up2(self._up_cat(y, s4))
        y = self.up1(self._up_cat(y, s2))
        y = self.full(self._up_cat(y, x))
        return self.head(y)


class Colorizer(nn.Module):
    """Encoder input (N,3,H,W) -> raw head output (N,k,H,W); pipeline.decode_output reads it."""

    def __init__(self, out_channels, pretrained=True):
        super().__init__()
        self.encoder = Encoder(pretrained)
        self.decoder = Decoder(out_channels)

    def forward(self, x):
        return self.decoder(x, self.encoder(x))

    def set_mode(self, training, train_encoder=True):
        """Train or eval mode; the encoder's BatchNorm statistics always stay frozen."""
        self.train(training)
        if training:
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d) or not train_encoder:
                    m.eval()


def out_channels(meta):
    """Head size: the a/b bin logits, plus one channel when L is predicted too."""
    return len(meta["bins"]) + (1 if meta["predict_L"] else 0)


def save_checkpoint(path, net, meta):
    """`net` may be wrapped in DataParallel; the bare module is saved."""
    net = getattr(net, "module", net)
    torch.save({"format": CHECKPOINT_FORMAT, "state": net.state_dict(), "meta": meta}, path)


def load_colorizer(path, device):
    """-> (Colorizer in eval mode on `device`, meta). Loads tensors and plain data only."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(obj, dict) or obj.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a format-{CHECKPOINT_FORMAT} ColorSAR checkpoint. "
                         "Checkpoints from before the scene split are kept under the git tag "
                         "v1-legacy and need that version of the code.")
    meta = obj["meta"]
    net = Colorizer(out_channels(meta), pretrained=False)
    net.load_state_dict(obj["state"])
    return net.to(device).eval(), meta
