"""
ColorSAR benchmark - train one algorithm on the SAR -> optical task, then score it on the
held-out test scenes (bench_common.evaluate). One call = one row of the comparison table.

    python bench_run.py --algo pix2pix --root v_2 --out bench_out --hours 1.0

Algorithms (--algo)
    histmatch   classical: SAR histogram-matched to the optical L of the training scenes,
                constant mean training color
    welsh       classical: Welsh et al. (2002) statistics transfer - every SAR pixel takes
                the mean Lab of training pixels with the same despeckled intensity and
                local standard deviation (a lookup table, no learning)
    unet_r50    the current Model 3 checkpoint (ResNet-50 U-Net, color bins), re-scored
                with the new metrics (--ckpt)
    pix2pix     conditional GAN, U-Net generator + PatchGAN, L1 x 100
    cyclegan    unpaired GAN with cycle consistency (pairs are deliberately shuffled)
    transunet   CNN + Transformer bottleneck U-Net, L1 regression
    wavenet     frequency-aware: Haar-wavelet U-Net + Focal Frequency Loss
    ddpm        conditional pixel-space diffusion (Palette-style) at 128 px, DDIM sampling
    picma       proposed: Physics-Informed Cross-Modal Attention latent diffusion
                  - physics: SAR speckle statistics as input (Lee filter, coefficient of
                    variation) and a speckle-aware structure loss (bench_common.physics_loss)
                  - cross-modal learning: the SAR encoder is first aligned with an optical
                    encoder by contrastive (InfoNCE) pretraining on the paired tiles
                  - cross-modal attention: the denoiser's features attend to SAR tokens
                  - frequency: Focal Frequency Loss on the decoded image
                  - diffusion: in the latent space of the Stable Diffusion VAE (frozen)
                Ablations: --no_physics --no_contrastive --no_xattn --no_freq

The training budget (--hours) is wall time; the learning rate decays linearly over the
second half of it. Every row is scored by the same code on the same test scenes.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench_common import (augment, batches, evaluate, focal_frequency_loss, lab3, lab_to_rgb_t,
                          lee_filter, load_data, physics_input, physics_loss, plain_input,
                          rgb_to_lab_t, split_scene_check)
from bench_models import (Diffusion, DiffusionUNet, PatchDiscriminator, ResnetGenerator, TransUNet,
                          UNetGenerator, WaveletUNet, n_params)

ALGOS = ("histmatch", "welsh", "unet_r50", "pix2pix", "cyclegan", "transunet", "wavenet", "ddpm", "picma")


class Budget:
    def __init__(self, hours, max_epochs):
        self.t0, self.sec, self.max_epochs = time.time(), hours * 3600, max_epochs

    def frac(self):
        return (time.time() - self.t0) / self.sec

    def over(self):
        return self.frac() >= 1.0

    def lr_mult(self):
        """1 for the first half of the budget, then linear decay to 0.05."""
        return max(0.05, min(1.0, 2 * (1 - self.frac())))

    def hours(self):
        return (time.time() - self.t0) / 3600


def set_lr(opt, base, mult):
    for g in opt.param_groups:
        g["lr"] = g.get("base_lr", base) * mult


def wrap(m):
    return nn.DataParallel(m) if torch.cuda.device_count() > 1 else m


def bare(m):
    return getattr(m, "module", m)


def ema_update(ema, model, decay):
    with torch.no_grad():
        for e, p in zip(ema.parameters(), bare(model).parameters()):
            e.mul_(decay).add_(p.detach(), alpha=1 - decay)
        for e, b in zip(ema.buffers(), bare(model).buffers()):
            e.copy_(b)


def save_half(path, model):
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in bare(model).state_dict().items()}, path)


def log(hist, **kw):
    for k, v in kw.items():
        hist.setdefault(k, []).append(float(v))


def lab_split(out):
    return out[:, :1], out[:, 1:]


# =========================================================================== classical
def run_histmatch(a, data, dev):
    tr = data["train"]
    n = min(len(tr), 2000)
    idx = np.random.RandomState(0).permutation(len(tr))[:n]
    L_list, ab_sum = [], torch.zeros(2, device=dev)
    sar_hist = np.zeros(256)
    for b in batches(n, 64, shuffle=False):
        gray, rgb = tr.batch(idx[b], dev)
        L, ab = rgb_to_lab_t(rgb)
        L_list.append(L[:, :, ::4, ::4].flatten().cpu())
        ab_sum += ab.mean(dim=(2, 3)).sum(0)
        sar_hist += np.bincount(tr.sar[idx[b]].ravel(), minlength=256)
    L_all = torch.cat(L_list).numpy()
    cdf = np.cumsum(sar_hist) / sar_hist.sum()
    lut = torch.tensor(np.quantile(L_all, np.clip(cdf - sar_hist / sar_hist.sum() / 2, 0, 1)),
                       dtype=torch.float32, device=dev)
    m_ab = (ab_sum / n).view(1, 2, 1, 1)

    def predict(gray):
        L = lut[(gray[:, 0] * 255).round().long()].unsqueeze(1)
        return [(L, m_ab.expand(len(gray), 2, *gray.shape[-2:]))]
    return predict, {}


def run_welsh(a, data, dev, nb_i=32, nb_s=8):
    tr = data["train"]
    n = min(len(tr), 2000)
    idx = np.random.RandomState(0).permutation(len(tr))[:n]

    def feats(gray):
        g_f, _, _ = lee_filter(gray)
        m = F.avg_pool2d(F.pad(gray, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
        s = (F.avg_pool2d(F.pad(gray * gray, (2, 2, 2, 2), mode="reflect"), 5, stride=1) - m * m).clamp_min(0).sqrt()
        return g_f, s

    s_samples = []
    for b in batches(min(n, 300), 64, shuffle=False):
        gray, _ = tr.batch(idx[b], dev)
        s_samples.append(feats(gray)[1][:, :, ::8, ::8].flatten().cpu())
    s_edges = torch.tensor(np.quantile(torch.cat(s_samples).numpy(), np.linspace(0, 1, nb_s + 1)[1:-1]),
                           dtype=torch.float32, device=dev)

    def cell(gray):
        g_f, s = feats(gray)
        ci = (g_f * (nb_i - 1)).round().long().clamp(0, nb_i - 1)
        cs = torch.bucketize(s.contiguous(), s_edges)
        return (ci * nb_s + cs)[:, 0]

    sums = torch.zeros(nb_i * nb_s, 3, device=dev, dtype=torch.float64)
    cnt = torch.zeros(nb_i * nb_s, device=dev, dtype=torch.float64)
    for b in batches(n, 64, shuffle=False):
        gray, rgb = tr.batch(idx[b], dev)
        c = cell(gray)[:, ::2, ::2].flatten()
        lab = lab3(rgb)[:, :, ::2, ::2].permute(0, 2, 3, 1).reshape(-1, 3).double()
        sums.index_add_(0, c, lab)
        cnt.index_add_(0, c, torch.ones_like(c, dtype=torch.float64))
    glob = sums.sum(0) / cnt.sum()
    table = torch.where(cnt[:, None] > 20, sums / cnt.clamp_min(1)[:, None], glob).float()

    def predict(gray):
        out = table[cell(gray)].permute(0, 3, 1, 2)
        return [lab_split(out)]
    return predict, {"cells_filled": int((cnt > 20).sum())}


def run_unet_r50(a, data, dev):
    from model_architecture import load_colorizer
    from pipeline import predict as p_predict
    net, meta = load_colorizer(a.ckpt, dev)
    from bench_common import scenes_of
    assert sorted(meta["split"]["test"]) == scenes_of(data["test"]) or a.smoke, "checkpoint split differs"

    def predict(gray):
        return [p_predict(net, meta, gray=gray, tta=True)]
    return predict, {"params": n_params(net), "best_epoch": meta.get("best_epoch")}


# =========================================================================== GANs
def run_pix2pix(a, data, dev):
    G, D = wrap(UNetGenerator(1, 3).to(dev)), wrap(PatchDiscriminator(4).to(dev))
    oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    sG, sD = torch.amp.GradScaler(enabled=a.amp), torch.amp.GradScaler(enabled=a.amp)
    bce = nn.BCEWithLogitsLoss()
    tr, rng, bud, hist, ep = data["train"], np.random.RandomState(a.seed), Budget(a.hours, a.epochs), {}, 0
    while ep < bud.max_epochs and not bud.over():
        G.train(); D.train()
        for idx in batches(len(tr), a.bs, True, rng, drop_last=True):
            set_lr(oG, 2e-4, bud.lr_mult()); set_lr(oD, 2e-4, bud.lr_mult())
            gray, rgb = augment(*tr.batch(idx, dev))
            x, y = gray * 2 - 1, lab3(rgb)
            with torch.autocast("cuda", enabled=a.amp):
                fake = G(x)
                dr, df = D(torch.cat([x, y], 1)), D(torch.cat([x, fake.detach()], 1))
                lD = 0.5 * (bce(dr, torch.ones_like(dr)) + bce(df, torch.zeros_like(df)))
            oD.zero_grad(set_to_none=True); sD.scale(lD).backward(); sD.step(oD); sD.update()
            with torch.autocast("cuda", enabled=a.amp):
                df = D(torch.cat([x, fake], 1))
                l1 = (fake.float() - y).abs().mean()
                lG = bce(df, torch.ones_like(df)) + 100 * l1
            oG.zero_grad(set_to_none=True); sG.scale(lG).backward(); sG.step(oG); sG.update()
            if bud.over():
                break
        ep += 1
        log(hist, lD=lD.item(), lG=lG.item(), l1=l1.item(), hours=bud.hours())
        print(f"epoch {ep}  D {lD.item():.3f}  G {lG.item():.3f}  L1 {l1.item():.4f}  ({bud.hours():.2f} h)", flush=True)
    save_half(os.path.join(a.dir, "G.pth"), G)
    g = bare(G).eval()

    def predict(gray):
        return [lab_split(g(gray * 2 - 1).float())]
    return predict, {"params": n_params(g), "epochs": ep, "history": hist}


def run_cyclegan(a, data, dev):
    GA, GB = wrap(ResnetGenerator(1, 3).to(dev)), wrap(ResnetGenerator(3, 1).to(dev))
    DA = wrap(PatchDiscriminator(1, norm=nn.InstanceNorm2d).to(dev))
    DB = wrap(PatchDiscriminator(3, norm=nn.InstanceNorm2d).to(dev))
    oG = torch.optim.Adam(list(GA.parameters()) + list(GB.parameters()), 2e-4, betas=(0.5, 0.999))
    oD = torch.optim.Adam(list(DA.parameters()) + list(DB.parameters()), 2e-4, betas=(0.5, 0.999))
    sG, sD = torch.amp.GradScaler(enabled=a.amp), torch.amp.GradScaler(enabled=a.amp)
    mse = nn.MSELoss()
    tr, rng, bud, hist, ep = data["train"], np.random.RandomState(a.seed), Budget(a.hours, a.epochs), {}, 0
    while ep < bud.max_epochs and not bud.over():
        other = rng.permutation(len(tr))        # unpaired: optical images from a different order
        for k, idx in enumerate(batches(len(tr), a.bs, True, rng, drop_last=True)):
            set_lr(oG, 2e-4, bud.lr_mult()); set_lr(oD, 2e-4, bud.lr_mult())
            gray, _ = tr.batch(idx, dev)
            _, rgb = tr.batch(other[k * a.bs:(k + 1) * a.bs], dev)
            xa, xb = augment(gray * 2 - 1)[0], augment(lab3(rgb))[0]
            with torch.autocast("cuda", enabled=a.amp):
                fb, fa = GA(xa), GB(xb)
                ra, rb = GB(fb), GA(fa)
                pb, pa = DB(fb), DA(fa)
                adv = mse(pb, torch.ones_like(pb)) + mse(pa, torch.ones_like(pa))
                cyc = (ra.float() - xa).abs().mean() + (rb.float() - xb).abs().mean()
                lG = adv + 10 * cyc
            oG.zero_grad(set_to_none=True); sG.scale(lG).backward(); sG.step(oG); sG.update()
            with torch.autocast("cuda", enabled=a.amp):
                lD = 0.0
                for D, real, fake in ((DA, xa, fa), (DB, xb, fb)):
                    pr, pf = D(real), D(fake.detach())
                    lD = lD + 0.5 * (mse(pr, torch.ones_like(pr)) + mse(pf, torch.zeros_like(pf)))
            oD.zero_grad(set_to_none=True); sD.scale(lD).backward(); sD.step(oD); sD.update()
            if bud.over():
                break
        ep += 1
        log(hist, lD=lD.item(), lG=lG.item(), cyc=cyc.item(), hours=bud.hours())
        print(f"epoch {ep}  D {lD.item():.3f}  G {lG.item():.3f}  cyc {cyc.item():.4f}  ({bud.hours():.2f} h)", flush=True)
    save_half(os.path.join(a.dir, "G_sar2opt.pth"), GA)
    g = bare(GA).eval()

    def predict(gray):
        return [lab_split(g(gray * 2 - 1).float())]
    return predict, {"params": n_params(g), "epochs": ep, "history": hist}


# =========================================================================== regression nets
@torch.no_grad()
def val_error(predict, val, dev, n=400):
    se_ab = se_L = cnt = 0.0
    for idx in batches(min(len(val), n), 32, shuffle=False):
        gray, rgb = val.batch(idx, dev)
        L, ab = predict(gray)[0]
        Lt, abt = rgb_to_lab_t(rgb)
        se_ab += ((ab - abt) * 127.5).pow(2).mean(dim=(1, 2, 3)).sum().item()
        se_L += ((L - Lt) * 50).pow(2).mean(dim=(1, 2, 3)).sum().item()
        cnt += len(idx)
    return (se_ab / cnt) ** 0.5, (se_L / cnt) ** 0.5


def run_regression(a, data, dev, net, freq_w=0.0):
    net = wrap(net.to(dev))
    opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=a.amp)
    tr, rng, bud, hist, ep = data["train"], np.random.RandomState(a.seed), Budget(a.hours, a.epochs), {}, 0
    best, best_state = float("inf"), None
    m = bare(net)

    def predict(gray):
        return [lab_split(m(gray * 2 - 1).float())]

    while ep < bud.max_epochs and not bud.over():
        net.train()
        for idx in batches(len(tr), a.bs, True, rng, drop_last=True):
            set_lr(opt, 2e-4, bud.lr_mult())
            gray, rgb = augment(*tr.batch(idx, dev))
            y = lab3(rgb)
            with torch.autocast("cuda", enabled=a.amp):
                out = net(gray * 2 - 1).float()
            loss = (out - y).abs().mean()
            if freq_w:
                loss = loss + freq_w * focal_frequency_loss(out, y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if bud.over():
                break
        ep += 1
        m.eval()
        ab_e, L_e = val_error(predict, data["val"], dev)
        log(hist, loss=loss.item(), val_ab_rmse=ab_e, val_L_rmse=L_e, hours=bud.hours())
        flag = ""
        if ab_e + L_e < best:
            best, best_state, flag = ab_e + L_e, copy.deepcopy(m.state_dict()), "  <- best"
        print(f"epoch {ep}  loss {loss.item():.4f}  val ab {ab_e:.2f}  L {L_e:.2f}  ({bud.hours():.2f} h){flag}", flush=True)
    m.load_state_dict(best_state)
    m.eval()
    save_half(os.path.join(a.dir, "net.pth"), m)
    return predict, {"params": n_params(m), "epochs": ep, "best_val_score": best, "history": hist}


# =========================================================================== pixel diffusion
def run_ddpm(a, data, dev, res=128):
    unet = DiffusionUNet(4, 3, ch=64, mult=(1, 2, 2, 4), attn_levels=(3,)).to(dev)
    net, ema = wrap(unet), copy.deepcopy(unet).eval()
    diff = Diffusion(device=dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=a.amp)
    tr, rng, bud, hist, ep, step = data["train"], np.random.RandomState(a.seed), Budget(a.hours, a.epochs), {}, 0, 0
    down = lambda x: F.interpolate(x, size=(res, res), mode="area")  # noqa: E731
    while ep < bud.max_epochs and not bud.over():
        net.train()
        for idx in batches(len(tr), a.bs, True, rng, drop_last=True):
            set_lr(opt, 2e-4, bud.lr_mult())
            gray, rgb = augment(*tr.batch(idx, dev))
            x0, c = down(lab3(rgb)), down(gray) * 2 - 1
            t = torch.randint(0, diff.T, (len(idx),), device=dev)
            noise = torch.randn_like(x0)
            with torch.autocast("cuda", enabled=a.amp):
                eps = net(torch.cat([diff.q_sample(x0, t, noise), c], 1), t)
            loss = F.mse_loss(eps.float(), noise)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            ema_update(ema, net, 0.999 if step > 1000 else 0.99)
            step += 1
            if bud.over():
                break
        ep += 1
        log(hist, loss=loss.item(), hours=bud.hours())
        print(f"epoch {ep}  loss {loss.item():.4f}  step {step}  ({bud.hours():.2f} h)", flush=True)
    save_half(os.path.join(a.dir, "ema.pth"), ema)

    def predict(gray):
        K, n = a.n_samples, len(gray)
        c = (down(gray) * 2 - 1).repeat(K, 1, 1, 1)
        with torch.autocast("cuda", enabled=a.amp):
            x = diff.ddim(lambda z, t: ema(torch.cat([z, c], 1), t), (K * n, 3, res, res),
                          steps=a.ddim_steps, device=dev, clip=1.0)
        x = F.interpolate(x.float(), size=gray.shape[-2:], mode="bilinear", align_corners=False)
        return [lab_split(x[k * n:(k + 1) * n]) for k in range(K)]
    return predict, {"params": n_params(unet), "epochs": ep, "steps": step, "history": hist}


# =========================================================================== proposed
class SARCond(nn.Module):
    """ResNet-50 over the SAR input -> spatial map at the latent resolution (layer2, 32x32)
    and condition tokens for cross-attention (layer3 16x16 + layer4 8x8 = 320 tokens)."""

    def __init__(self, ctx_dim=512, spatial=64, pretrained=True):
        super().__init__()
        from model_architecture import Encoder
        self.enc = Encoder(pretrained)
        self.spatial = nn.Conv2d(512, spatial, 1)
        self.tok3, self.tok4 = nn.Linear(1024, ctx_dim), nn.Linear(2048, ctx_dim)
        self.pos = nn.Parameter(torch.zeros(1, 320, ctx_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        f = self.enc(x)
        t = torch.cat([self.tok3(f[3].flatten(2).transpose(1, 2)), self.tok4(f[4].flatten(2).transpose(1, 2))], 1)
        pos = self.pos if t.shape[1] == 320 else 0
        return self.spatial(f[2]), t + pos


class PICMA(nn.Module):
    def __init__(self, xattn=True, pretrained=True):
        super().__init__()
        self.cond = SARCond(pretrained=pretrained)
        self.xattn = xattn
        self.unet = DiffusionUNet(4 + 64, 4, ch=128, mult=(1, 2, 4), n_res=2, attn_levels=(1, 2),
                                  context_dim=512 if xattn else None)

    def denoise(self, z, t, spatial, ctx):
        if spatial.shape[-2:] != z.shape[-2:]:
            spatial = F.interpolate(spatial, size=z.shape[-2:], mode="bilinear", align_corners=False)
        return self.unet(torch.cat([z, spatial], 1), t, ctx if self.xattn else None)

    def forward(self, z, t, sar_in):
        spatial, ctx = self.cond(sar_in)
        return self.denoise(z, t, spatial, ctx)


def contrastive_pretrain(a, data, dev, sar_in_fn, hours):
    """InfoNCE between SAR (ResNet-50) and optical (ResNet-18) global embeddings of paired tiles."""
    from torchvision.models import ResNet18_Weights, resnet18
    from model_architecture import Encoder
    sar = Encoder(pretrained=True).to(dev)
    opt_net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    opt_net.fc = nn.Identity()
    opt_net = opt_net.to(dev)
    hs = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(True), nn.Linear(512, 128)).to(dev)
    ho = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 128)).to(dev)

    class Pair(nn.Module):
        def __init__(self):
            super().__init__()
            self.sar, self.opt, self.hs, self.ho = sar, opt_net, hs, ho

        def forward(self, xs, xo):
            zs = self.hs(self.sar(xs)[-1].mean(dim=(2, 3)))
            zo = self.ho(self.opt(xo))
            return F.normalize(zs.float(), dim=1), F.normalize(zo.float(), dim=1)

    model = wrap(Pair())
    opt = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=a.amp)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    tr, rng, bud, hist, ep = data["train"], np.random.RandomState(a.seed + 1), Budget(hours, a.epochs), {}, 0
    bs = max(a.bs * 4, 8)
    while ep < bud.max_epochs and not bud.over():
        model.train()
        for idx in batches(len(tr), bs, True, rng, drop_last=True):
            gray, rgb = augment(*tr.batch(idx, dev))
            with torch.autocast("cuda", enabled=a.amp):
                zs, zo = model(sar_in_fn(gray), (rgb - mean) / std)
            logits = zs @ zo.T / 0.1
            lbl = torch.arange(len(idx), device=dev)
            loss = 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.T, lbl))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if bud.over():
                break
        ep += 1
        acc = (logits.argmax(1) == lbl).float().mean().item()
        log(hist, loss=loss.item(), top1=acc, hours=bud.hours())
        print(f"[contrastive] epoch {ep}  InfoNCE {loss.item():.4f}  in-batch top-1 {acc:.2f}  "
              f"({bud.hours():.2f} h)", flush=True)
    return sar.state_dict(), hist


def load_vae(dev):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


SCALE = 0.18215


@torch.no_grad()
def encode_latents(vae, part, dev, bs=32):
    out = []
    for idx in batches(len(part), bs, shuffle=False):
        _, rgb = part.batch(idx, dev)
        out.append(vae.encode((rgb * 2 - 1).half()).latent_dist.mean.float() * SCALE)
    return torch.cat(out)


def run_picma(a, data, dev):
    phys = not a.no_physics
    sar_in_fn = (lambda g: physics_input(g)[0]) if phys else plain_input
    extra = {"physics": phys, "contrastive": not a.no_contrastive, "xattn": not a.no_xattn,
             "freq": not a.no_freq}
    model = PICMA(xattn=not a.no_xattn).to(dev)
    if not a.no_contrastive:
        state, extra["contrastive_history"] = contrastive_pretrain(a, data, dev, sar_in_fn, a.pre_hours)
        model.cond.enc.load_state_dict(state)
    vae = load_vae(dev)
    t0 = time.time()
    lat = encode_latents(vae, data["train"], dev)
    print(f"encoded {len(lat)} training latents {tuple(lat.shape[1:])} in {time.time() - t0:.0f} s "
          f"(std {lat.std():.3f})", flush=True)

    net, ema = wrap(model), copy.deepcopy(model).eval()
    diff = Diffusion(device=dev)
    enc_params = list(model.cond.enc.parameters())
    enc_ids = {id(p) for p in enc_params}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": rest, "lr": 1e-4, "base_lr": 1e-4},
                             {"params": enc_params, "lr": 2e-5, "base_lr": 2e-5}], weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=a.amp)
    tr, rng, bud, hist, ep, step = data["train"], np.random.RandomState(a.seed), Budget(a.hours, a.epochs), {}, 0, 0
    use_aux = phys or not a.no_freq
    while ep < bud.max_epochs and not bud.over():
        net.train()
        for idx in batches(len(tr), a.bs, True, rng, drop_last=True):
            set_lr(opt, 1e-4, bud.lr_mult())
            gray, rgb, z0 = augment(*tr.batch(idx, dev), lat[torch.as_tensor(idx, device=lat.device)])
            t = torch.randint(0, diff.T, (len(idx),), device=dev)
            noise = torch.randn_like(z0)
            with torch.autocast("cuda", enabled=a.amp):
                eps = net(diff.q_sample(z0, t, noise), t, sar_in_fn(gray))
            loss = F.mse_loss(eps.float(), noise)
            aux = torch.zeros((), device=dev)
            if use_aux and step % a.aux_every == 0:
                # physics / frequency terms on the decoded image of a small sub-batch at low noise
                k = min(4, len(idx))
                tk = torch.randint(0, 300, (k,), device=dev)
                nk = torch.randn_like(z0[:k])
                with torch.autocast("cuda", enabled=a.amp):
                    ek = bare(net)(diff.q_sample(z0[:k], tk, nk), tk, sar_in_fn(gray[:k])).float()
                    x0 = diff.x0_from_eps(diff.q_sample(z0[:k], tk, nk), tk, ek)
                    img = vae.decode((x0 / SCALE).half()).sample.float()
                img = ((img + 1) / 2).clamp(0, 1)
                L, ab = rgb_to_lab_t(img)
                if phys:
                    _, w, g_f = physics_input(gray[:k])
                    aux = aux + a.phys_w * physics_loss(L, ab, g_f, w)
                if not a.no_freq:
                    aux = aux + a.freq_w * focal_frequency_loss(torch.cat([L, ab], 1), lab3(rgb[:k]))
            total = loss + aux
            opt.zero_grad(set_to_none=True); scaler.scale(total).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            ema_update(ema, net, 0.999 if step > 1000 else 0.99)
            step += 1
            if bud.over():
                break
        ep += 1
        log(hist, loss=loss.item(), aux=aux.item(), hours=bud.hours())
        print(f"epoch {ep}  eps-MSE {loss.item():.4f}  aux {aux.item():.4f}  step {step}  "
              f"({bud.hours():.2f} h)", flush=True)
    save_half(os.path.join(a.dir, "ema.pth"), ema)
    del lat

    @torch.no_grad()
    def predict(gray):
        K, n = a.n_samples, len(gray)
        with torch.autocast("cuda", enabled=a.amp):
            spatial, ctx = ema.cond(sar_in_fn(gray))
            spatial, ctx = spatial.repeat(K, 1, 1, 1), ctx.repeat(K, 1, 1)
            z = diff.ddim(lambda zz, tt: ema.denoise(zz, tt, spatial, ctx), (K * n, 4, 32, 32),
                          steps=a.ddim_steps, device=dev)
            img = torch.cat([vae.decode((z[i:i + 32] / SCALE).half()).sample.float()
                             for i in range(0, len(z), 32)])
        L, ab = rgb_to_lab_t(((img + 1) / 2).clamp(0, 1))
        return [(L[k * n:(k + 1) * n], ab[k * n:(k + 1) * n]) for k in range(K)]
    extra.update({"params": n_params(model), "epochs": ep, "steps": step, "history": hist})
    return predict, extra


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=ALGOS, required=True)
    ap.add_argument("--name", default=None, help="output folder name (default: the algo)")
    ap.add_argument("--root", default="v_2")
    ap.add_argument("--out", default="bench_out")
    ap.add_argument("--cache", default=None, help="folder for the uint8 data cache")
    ap.add_argument("--hours", type=float, default=1.0, help="training budget (wall time)")
    ap.add_argument("--pre_hours", type=float, default=0.3, help="picma: contrastive pretraining budget")
    ap.add_argument("--epochs", type=int, default=1000, help="upper bound; the budget usually stops first")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--n_val", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=4, help="diffusion samples per test tile")
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--ckpt", default="model3_run/model_3.pth", help="unet_r50 checkpoint")
    ap.add_argument("--no_physics", action="store_true")
    ap.add_argument("--no_contrastive", action="store_true")
    ap.add_argument("--no_xattn", action="store_true")
    ap.add_argument("--no_freq", action="store_true")
    ap.add_argument("--phys_w", type=float, default=0.05)
    ap.add_argument("--freq_w", type=float, default=1.0)
    ap.add_argument("--aux_every", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="tiny run that only checks the code path")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.smoke:  # tiny sizes: only checks that the code path runs end to end
        a.n_train, a.n_val, a.n_test = a.n_train or 32, a.n_val or 8, a.n_test or 8
        a.epochs, a.bs, a.n_samples, a.ddim_steps = 1, min(a.bs, 4), 2, 3
        a.pre_hours = min(a.pre_hours, 0.05)
    a.name = a.name or a.algo
    a.dir = os.path.join(a.out, a.name)
    os.makedirs(a.dir, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.amp = dev.type == "cuda"
    print(f"[{a.name}] device {dev}  GPUs {torch.cuda.device_count()}  config "
          f"{ {k: v for k, v in vars(a).items() if k not in ('root',)} }", flush=True)

    data, rec = load_data(a.root, a.n_train, a.n_val, a.n_test, a.seed, a.cache)
    split_scene_check(rec, data)
    t0 = time.time()
    runner = {"histmatch": run_histmatch, "welsh": run_welsh, "unet_r50": run_unet_r50,
              "pix2pix": run_pix2pix, "cyclegan": run_cyclegan, "ddpm": run_ddpm, "picma": run_picma,
              "transunet": lambda a, d, dv: run_regression(a, d, dv, TransUNet(1, 3)),
              "wavenet": lambda a, d, dv: run_regression(a, d, dv, WaveletUNet(1, 3), freq_w=1.0)}[a.algo]
    predict, extra = runner(a, data, dev)
    train_h = (time.time() - t0) / 3600
    hist = extra.pop("history", None)
    if hist:
        with open(os.path.join(a.dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)
    t1 = time.time()
    evaluate(predict, data["test"], dev, a.dir, a.name, bs=16 if a.algo in ("ddpm", "picma") else 32,
             extra=dict(extra, train_hours=train_h, config={k: v for k, v in vars(a).items()
                                                           if isinstance(v, (int, float, str, bool))}))
    print(f"[{a.name}] done: train {train_h:.2f} h, eval {(time.time() - t1) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
