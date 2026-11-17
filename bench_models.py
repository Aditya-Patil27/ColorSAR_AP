"""
ColorSAR benchmark - network architectures

    UNetGenerator, PatchDiscriminator   Pix2Pix (Isola et al., CVPR 2017)
    ResnetGenerator                     CycleGAN (Zhu et al., ICCV 2017)
    TransUNet                           CNN stem + Transformer bottleneck + U-Net decoder
                                        (Chen et al., 2021)
    WaveletUNet                         Haar-DWT down / IDWT up U-Net (MWCNN, Liu et al.,
                                        CVPR-W 2018), trained with Focal Frequency Loss
    DiffusionUNet                       noise-prediction U-Net with timestep embedding,
                                        self-attention and optional cross-attention to
                                        condition tokens (used by DDPM and the latent model)
    Diffusion                           DDPM training targets (Ho et al., 2020) and the
                                        deterministic DDIM sampler (Song et al., 2021)

All image-to-image networks output 3 channels [L, a, b] in [-1, 1] through tanh.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- Pix2Pix
class _Down(nn.Module):
    def __init__(self, c_in, c_out, norm=True):
        super().__init__()
        layers = [nn.Conv2d(c_in, c_out, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(c_out))
        layers.append(nn.LeakyReLU(0.2, True))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class _Up(nn.Module):
    def __init__(self, c_in, c_out, dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False), nn.BatchNorm2d(c_out)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(True))
        self.f = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.f(x), skip], dim=1)


class UNetGenerator(nn.Module):
    """The 256x256 Pix2Pix U-Net: 8 downsamplings to 1x1, skips at every level."""

    def __init__(self, c_in=1, c_out=3, nf=64):
        super().__init__()
        ch = [nf, nf * 2, nf * 4, nf * 8, nf * 8, nf * 8, nf * 8]
        self.downs = nn.ModuleList([_Down(c_in, ch[0], norm=False)] +
                                   [_Down(ch[i], ch[i + 1]) for i in range(6)])
        self.inner = nn.Sequential(nn.Conv2d(nf * 8, nf * 8, 4, 2, 1), nn.ReLU(True))
        ups, c = [], nf * 8
        for i, skip in enumerate(reversed(ch)):
            ups.append(_Up(c, skip, dropout=i < 3))
            c = skip * 2
        self.ups = nn.ModuleList(ups)
        self.out = nn.Sequential(nn.ConvTranspose2d(c, c_out, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
        x = self.inner(x)
        for u, s in zip(self.ups, reversed(skips)):
            x = u(x, s)
        return self.out(x)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN."""

    def __init__(self, c_in, nf=64, norm=nn.BatchNorm2d):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, nf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False), norm(nf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False), norm(nf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 8, 4, 1, 1, bias=False), norm(nf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, 1, 4, 1, 1))

    def forward(self, x):
        return self.f(x)


# --------------------------------------------------------------------------- CycleGAN
class _ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c), nn.ReLU(True),
                               nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c))

    def forward(self, x):
        return x + self.f(x)


class ResnetGenerator(nn.Module):
    """CycleGAN generator: c7s1-64, d128, d256, 9 residual blocks, u128, u64, c7s1-out."""

    def __init__(self, c_in, c_out, nf=64, n_blocks=9):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(c_in, nf, 7), nn.InstanceNorm2d(nf), nn.ReLU(True)]
        for m in (1, 2):
            layers += [nn.Conv2d(nf * m, nf * m * 2, 3, 2, 1), nn.InstanceNorm2d(nf * m * 2), nn.ReLU(True)]
        layers += [_ResBlock(nf * 4) for _ in range(n_blocks)]
        for m in (4, 2):
            layers += [nn.ConvTranspose2d(nf * m, nf * m // 2, 3, 2, 1, output_padding=1),
                       nn.InstanceNorm2d(nf * m // 2), nn.ReLU(True)]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(nf, c_out, 7), nn.Tanh()]
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


# --------------------------------------------------------------------------- shared blocks
def _conv_block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU())


# --------------------------------------------------------------------------- TransUNet
class TransUNet(nn.Module):
    """CNN encoder to 16x16, a Transformer over the 256 tokens, U-Net decoder with skips."""

    def __init__(self, c_in=1, c_out=3, dim=384, depth=8, heads=6):
        super().__init__()
        ch = [32, 64, 128, 256, dim]                       # @256, 128, 64, 32, 16
        self.enc = nn.ModuleList([_conv_block(c_in, ch[0])] +
                                 [_conv_block(ch[i], ch[i + 1]) for i in range(4)])
        self.pos = nn.Parameter(torch.zeros(1, 256, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth)
        self.dec = nn.ModuleList([_conv_block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(4))])
        self.out = nn.Sequential(nn.Conv2d(ch[0], c_out, 1), nn.Tanh())

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        n, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        pos = self.pos if h * w == 256 else F.interpolate(
            self.pos.transpose(1, 2).view(1, c, 16, 16), size=(h, w), mode="bilinear").flatten(2).transpose(1, 2)
        x = self.transformer(t + pos).transpose(1, 2).view(n, c, h, w)
        for d, s in zip(self.dec, reversed(skips[:-1])):
            x = d(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return self.out(x)


# --------------------------------------------------------------------------- wavelets
def dwt_haar(x):
    """(N,C,H,W) -> (N,4C,H/2,W/2): LL, LH, HL, HH subbands (orthonormal Haar)."""
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return torch.cat([(a + b + c + d), (-a - b + c + d), (-a + b - c + d), (a - b - c + d)], 1) * 0.5


def idwt_haar(y):
    """Inverse of dwt_haar."""
    C = y.shape[1] // 4
    ll, lh, hl, hh = y[:, :C], y[:, C:2 * C], y[:, 2 * C:3 * C], y[:, 3 * C:]
    a = (ll - lh - hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll + lh + hl + hh) * 0.5
    n, _, h, w = ll.shape
    out = ll.new_zeros(n, C, h * 2, w * 2)
    out[..., 0::2, 0::2], out[..., 0::2, 1::2], out[..., 1::2, 0::2], out[..., 1::2, 1::2] = a, b, c, d
    return out


class WaveletUNet(nn.Module):
    """MWCNN-style U-Net: Haar DWT replaces pooling, IDWT replaces upsampling, so speckle
    (high-frequency subbands) and land-cover color (low-frequency) are kept apart losslessly."""

    def __init__(self, c_in=1, c_out=3, nf=32, n_res=4):
        super().__init__()
        ch = [nf, nf * 2, nf * 4, nf * 8, nf * 8]          # @256, 128, 64, 32, 16
        self.inp = _conv_block(c_in, ch[0])
        self.down = nn.ModuleList([_conv_block(4 * ch[i], ch[i + 1]) for i in range(4)])
        self.mid = nn.Sequential(*[_ResBlockGN(ch[4]) for _ in range(n_res)])
        self.to_sub = nn.ModuleList([nn.Conv2d(ch[i + 1], 4 * ch[i], 3, padding=1) for i in range(4)])
        self.up = nn.ModuleList([_conv_block(2 * ch[i], ch[i]) for i in range(4)])
        self.out = nn.Sequential(nn.Conv2d(ch[0], c_out, 1), nn.Tanh())

    def forward(self, x):
        skips = [self.inp(x)]
        for d in self.down:
            skips.append(d(dwt_haar(skips[-1])))
        y = self.mid(skips[-1])
        for i in reversed(range(4)):
            y = self.up[i](torch.cat([idwt_haar(self.to_sub[i](y)), skips[i]], 1))
        return self.out(y)


class _ResBlockGN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.GELU(),
                               nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c))

    def forward(self, x):
        return F.gelu(x + self.f(x))


# --------------------------------------------------------------------------- diffusion U-Net
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _TResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim, dropout=0.0):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t = nn.Linear(t_dim, c_out)
        self.n2, self.c2 = nn.GroupNorm(32, c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(temb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class _Attn(nn.Module):
    """Self-attention over pixels, then (if context_dim) cross-attention to the condition
    tokens: queries from the generated image's features, keys/values from the SAR."""

    def __init__(self, c, heads=4, context_dim=None):
        super().__init__()
        self.n1 = nn.GroupNorm(32, c)
        self.self_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.cross = None
        if context_dim:
            self.n2 = nn.LayerNorm(c)
            self.cross = nn.MultiheadAttention(c, heads, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.n3 = nn.LayerNorm(c)
        self.ff = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))

    def forward(self, x, ctx=None):
        n, c, hh, ww = x.shape
        t = x.flatten(2).transpose(1, 2)
        z = self.n1(x).flatten(2).transpose(1, 2)
        t = t + self.self_attn(z, z, z, need_weights=False)[0]
        if self.cross is not None and ctx is not None:
            q = self.n2(t)
            t = t + self.cross(q, ctx, ctx, need_weights=False)[0]
        t = t + self.ff(self.n3(t))
        return t.transpose(1, 2).reshape(n, c, hh, ww)


class DiffusionUNet(nn.Module):
    """eps-prediction U-Net. in_ch = noisy channels + spatial condition channels.

    attn_levels: indices of resolution levels (0 = full) that get attention blocks;
    context_dim: size of the condition tokens for cross-attention (None = no cross-attn).
    """

    def __init__(self, in_ch, out_ch, ch=64, mult=(1, 2, 2, 4), n_res=2, attn_levels=(3,),
                 context_dim=None, dropout=0.1):
        super().__init__()
        t_dim = ch * 4
        self.ch = ch
        self.temb = nn.Sequential(nn.Linear(ch, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.inp = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.down, skip_ch, c = nn.ModuleList(), [ch], ch
        for lvl, m in enumerate(mult):
            for _ in range(n_res):
                blk = nn.ModuleList([_TResBlock(c, ch * m, t_dim, dropout)])
                c = ch * m
                if lvl in attn_levels:
                    blk.append(_Attn(c, context_dim=context_dim))
                self.down.append(blk)
                skip_ch.append(c)
            if lvl < len(mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)]))
                skip_ch.append(c)
        self.mid = nn.ModuleList([_TResBlock(c, c, t_dim, dropout), _Attn(c, context_dim=context_dim),
                                  _TResBlock(c, c, t_dim, dropout)])
        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(mult))):
            for i in range(n_res + 1):
                blk = nn.ModuleList([_TResBlock(c + skip_ch.pop(), ch * m, t_dim, dropout)])
                c = ch * m
                if lvl in attn_levels:
                    blk.append(_Attn(c, context_dim=context_dim))
                if lvl > 0 and i == n_res:
                    blk.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    blk.append(nn.Conv2d(c, c, 3, padding=1))
                self.up.append(blk)
        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1))

    @staticmethod
    def _run(blk, h, temb, ctx):
        for layer in blk:
            if isinstance(layer, _TResBlock):
                h = layer(h, temb)
            elif isinstance(layer, _Attn):
                h = layer(h, ctx)
            else:
                h = layer(h)
        return h

    def forward(self, x, t, ctx=None):
        temb = self.temb(timestep_embedding(t, self.ch))
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            h = self._run(blk, h, temb, ctx)
            hs.append(h)
        h = self._run(self.mid, h, temb, ctx)
        for blk in self.up:
            h = self._run(blk, torch.cat([h, hs.pop()], 1), temb, ctx)
        return self.out(h)


class Diffusion:
    """Linear-beta DDPM (T=1000) with eps-prediction and a deterministic DDIM sampler."""

    def __init__(self, T=1000, beta1=1e-4, beta2=0.02, device="cpu"):
        self.T = T
        betas = torch.linspace(beta1, beta2, T, dtype=torch.float64)
        self.abar = torch.cumprod(1 - betas, 0).float().to(device)

    def q_sample(self, x0, t, noise):
        a = self.abar[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def x0_from_eps(self, xt, t, eps):
        a = self.abar[t].view(-1, 1, 1, 1)
        return (xt - (1 - a).sqrt() * eps) / a.sqrt()

    @torch.no_grad()
    def ddim(self, eps_fn, shape, steps=50, device="cpu", clip=None, generator=None):
        ts = torch.linspace(self.T - 1, 0, steps, device=device).round().long()
        x = torch.randn(shape, device=device, generator=generator)
        for i, t in enumerate(ts):
            tt = t.repeat(shape[0])
            eps = eps_fn(x, tt).float()
            x0 = self.x0_from_eps(x, tt, eps)
            if clip is not None:
                x0 = x0.clamp(-clip, clip)
                eps = (x - self.abar[t].sqrt() * x0) / (1 - self.abar[t]).sqrt()
            a_next = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
        return x
