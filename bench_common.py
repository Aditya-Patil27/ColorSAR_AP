"""
ColorSAR benchmark - shared data cache, SAR physics, Lab in torch, and scoring

Every benchmarked algorithm (bench_run.py) solves the Model 3 task: the SAR (s1) tile
in, the optical (s2) image out as L and a/b. They all read the same scene split
(splits.scene_split), see the same test scenes and are scored by the same Scorer, so
the rows of the comparison table can be read against each other.

SAR physics (used by the proposed model, see bench_run.py)
    The v_2 tiles are 8-bit Sentinel-1 backscatter. We assume the SEN1-2 scaling: sigma0
    in dB clipped to [DB_MIN, DB_MAX] and mapped linearly to 0..255. Speckle is
    multiplicative (fully developed speckle: Gamma-distributed intensity with ENL looks),
    so it is handled in the linear intensity domain:
      - Lee filter: I_hat = m + w (I - m) with w = (1 - Cu^2/Ci^2) / (1 + Cu^2), where
        Ci is the local coefficient of variation and Cu = 1/sqrt(ENL) the one of pure
        speckle. w ~ 0 in homogeneous areas, ~ 1 on real structure (edges, buildings).
      - The local coefficient of variation itself (a heterogeneity / texture measure).
"""
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from color_utils import (_RGB_TO_XYZ, _WHITE_D65, SIZE, _f_l, _srgb_to_linear, ab_to_lab_units,
                         find_pairs, lab_to_rgb, rgb_to_L)
from splits import scene_split, split_record, subsample

DB_MIN, DB_MAX = -25.0, 0.0   # SEN1-2 dB scaling of the 8-bit SAR tiles (assumption)
ENL = 4.4                     # equivalent number of looks, Sentinel-1 IW GRDH


# --------------------------------------------------------------------------- data
def _read(pair):
    sar_path, opt_path, _ = pair
    g = np.array(Image.open(sar_path).convert("L"))
    rgb = np.array(Image.open(opt_path).convert("RGB"))
    if g.shape != (SIZE, SIZE):
        g = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    if rgb.shape[:2] != (SIZE, SIZE):
        rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return g, rgb


class Part:
    """One split part held in RAM as uint8: sar (N,H,W), rgb (N,H,W,3), terrain (N,)."""

    def __init__(self, sar, rgb, terrain, names):
        self.sar, self.rgb, self.terrain, self.names = sar, rgb, list(terrain), list(names)

    def __len__(self):
        return len(self.sar)

    def batch(self, idx, device):
        """-> SAR gray (B,1,H,W) in [0,1], RGB (B,3,H,W) in [0,1] on device."""
        idx = np.asarray(idx)
        g = torch.from_numpy(self.sar[idx]).to(device).float().unsqueeze(1) / 255.0
        rgb = torch.from_numpy(self.rgb[idx]).to(device).float().permute(0, 3, 1, 2) / 255.0
        return g, rgb


def _load_part(pairs, cache):
    if cache and os.path.isfile(cache + "_sar.npy"):
        with open(cache + ".json") as f:
            meta = json.load(f)
        if meta["names"] == [os.path.basename(p[0]) for p in pairs]:
            return Part(np.load(cache + "_sar.npy"), np.load(cache + "_rgb.npy"),
                        meta["terrain"], meta["names"])
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(_read, pairs))
    sar = np.stack([o[0] for o in out]) if out else np.zeros((0, SIZE, SIZE), np.uint8)
    rgb = np.stack([o[1] for o in out]) if out else np.zeros((0, SIZE, SIZE, 3), np.uint8)
    part = Part(sar, rgb, [p[2] for p in pairs], [os.path.basename(p[0]) for p in pairs])
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.save(cache + "_sar.npy", sar)
        np.save(cache + "_rgb.npy", rgb)
        with open(cache + ".json", "w") as f:
            json.dump({"names": part.names, "terrain": part.terrain}, f)
    return part


def load_data(root, n_train=0, n_val=0, n_test=0, seed=42, cache_dir=None):
    """-> ({"train", "val", "test"}: Part, split record). Test is never subsampled in a full run."""
    pairs = find_pairs(root)
    assert pairs, f"No s1/s2 pairs found under {root}"
    split = scene_split(pairs)
    sizes = {"train": n_train, "val": n_val, "test": n_test}
    data = {}
    for name, n in sizes.items():
        part = subsample(split[name], n, seed)
        cache = os.path.join(cache_dir, f"{name}_{len(part)}_{seed}") if cache_dir else None
        data[name] = _load_part(part, cache)
    rec = split_record(split)
    print(f"{len(pairs)} pairs -> train {len(data['train'])}  val {len(data['val'])}  "
          f"test {len(data['test'])}  | test scenes {rec['test']}", flush=True)
    return data, rec


def batches(n, bs, shuffle, rng=None, drop_last=False):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    stop = n - (n % bs) if drop_last else n
    for i in range(0, stop, bs):
        yield idx[i:i + bs]


def augment(*xs):
    """The same random rot90 + flip for every tensor of the batch (tiles have no 'up')."""
    k = int(torch.randint(4, (1,)))
    flip = bool(torch.randint(2, (1,)))
    out = []
    for x in xs:
        x = torch.rot90(x, k, dims=(-2, -1))
        out.append(torch.flip(x, dims=(-1,)) if flip else x)
    return out


# --------------------------------------------------------------------------- Lab in torch
_XYZ_TO_RGB = np.linalg.inv(np.array(_RGB_TO_XYZ)).tolist()


def rgb_to_lab_t(rgb):
    """RGB (N,3,H,W) in [0,1] -> (L (N,1,H,W), ab (N,2,H,W)), normalized as color_utils.rgb_to_lab."""
    lin = _srgb_to_linear(rgb.clamp(0.0, 1.0))
    m = torch.tensor(_RGB_TO_XYZ, dtype=lin.dtype, device=lin.device)
    xyz = torch.einsum("ij,njhw->nihw", m, lin)
    fx, fy, fz = (_f_l((xyz[:, i] / _WHITE_D65[i]).clamp_min(0.0)) for i in range(3))
    L = 116.0 * fy - 16.0
    a, b = 500.0 * (fx - fy), 200.0 * (fy - fz)
    return ((2 * L / 100 - 1).unsqueeze(1),
            torch.stack([2 * (a + 128) / 255 - 1, 2 * (b + 128) / 255 - 1], dim=1))


def lab_to_rgb_t(L, ab):
    """Normalized L (N,1,H,W) and ab (N,2,H,W) -> RGB (N,3,H,W) in [0,1], differentiable."""
    Lu = (L[:, 0] + 1) * 50.0
    a, b = ((ab[:, i] + 1) * 0.5 * 255 - 128 for i in range(2))
    fy = (Lu + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0
    d = 6.0 / 29.0

    def finv(t):
        return torch.where(t > d, t.pow(3), 3 * d * d * (t - 4.0 / 29.0))

    xyz = torch.stack([finv(fx) * _WHITE_D65[0], finv(fy) * _WHITE_D65[1], finv(fz) * _WHITE_D65[2]], 1)
    m = torch.tensor(_XYZ_TO_RGB, dtype=xyz.dtype, device=xyz.device)
    lin = torch.einsum("ij,njhw->nihw", m, xyz).clamp(0.0, 1.0)
    srgb = torch.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin.clamp_min(1e-8).pow(1 / 2.4) - 0.055)
    return srgb.clamp(0.0, 1.0)


def lab3(rgb):
    """RGB (N,3,H,W) -> Lab as one normalized 3-channel tensor [L, a, b] in [-1, 1]."""
    L, ab = rgb_to_lab_t(rgb)
    return torch.cat([L, ab], dim=1)


# --------------------------------------------------------------------------- SAR physics
def _box(x, k):
    p = k // 2
    return F.avg_pool2d(F.pad(x, (p, p, p, p), mode="reflect"), k, stride=1)


def sar_intensity(gray):
    """8-bit-scaled SAR in [0,1] -> linear intensity sigma0."""
    db = DB_MIN + gray.float() * (DB_MAX - DB_MIN)
    return torch.pow(10.0, db / 10.0)


def lee_filter(gray, k=7, enl=ENL):
    """-> (filtered gray in [0,1], Lee weight w in [0,1], local coefficient of variation Ci)."""
    I = sar_intensity(gray)
    m = _box(I, k)
    v = (_box(I * I, k) - m * m).clamp_min(0.0)
    ci2 = v / (m * m + 1e-12)
    cu2 = 1.0 / enl
    w = ((1.0 - cu2 / ci2.clamp_min(1e-12)) / (1.0 + cu2)).clamp(0.0, 1.0)
    I_f = (m + w * (I - m)).clamp_min(1e-8)
    db_f = 10.0 * torch.log10(I_f)
    g_f = ((db_f - DB_MIN) / (DB_MAX - DB_MIN)).clamp(0.0, 1.0)
    return g_f, w, ci2.sqrt()


def physics_input(gray):
    """SAR gray (N,1,H,W) -> (3 physics channels in [-1,1]: raw dB, Lee-filtered dB, local
    coefficient of variation scaled by the speckle one; Lee weight w (N,1,H,W))."""
    g_f, w, ci = lee_filter(gray)
    cov = (ci * (ENL ** 0.5) / 4.0).clamp(0.0, 1.0)   # 1/4 of the way = speckle-only level
    return torch.cat([gray, g_f, cov], dim=1) * 2 - 1, w, g_f


def plain_input(gray):
    """The SAR tile as it is, repeated on 3 channels, in [-1, 1] (no physics)."""
    return (gray * 2 - 1).repeat(1, 3, 1, 1)


def _grad(x):
    return x[..., :-1, 1:] - x[..., :-1, :-1], x[..., 1:, :-1] - x[..., :-1, :-1]


def physics_loss(L, ab, g_f, w, eps=1e-3):
    """Speckle-aware structure prior.

    Where the speckle statistics say the SAR is homogeneous (Lee weight w ~ 0) the color
    should be flat: penalize a/b gradients there. Where they say there is real structure
    (w ~ 1), predicted brightness edges should run along the despeckled SAR edges
    (|cos| of the gradient angle; optical and radar brightness may have opposite sign).
    """
    w = w[..., :-1, :-1]
    gxa, gya = _grad(ab)
    smooth = ((1 - w) * (gxa.abs() + gya.abs()).sum(1, keepdim=True)).mean()
    gxl, gyl = _grad(L)
    gxs, gys = _grad(g_f)
    dot = (gxl * gxs + gyl * gys).abs()
    norm = torch.sqrt((gxl ** 2 + gyl ** 2) * (gxs ** 2 + gys ** 2) + eps ** 4)
    align = (w * (1 - dot / norm)).sum() / w.sum().clamp_min(1.0)
    return smooth + align


def focal_frequency_loss(pred, target, alpha=1.0):
    """Focal Frequency Loss (Jiang et al., ICCV 2021): spectrum distance, hard frequencies weighted up."""
    fp = torch.fft.fft2(pred.float(), norm="ortho")
    ft = torch.fft.fft2(target.float(), norm="ortho")
    d = (fp - ft).abs() ** 2
    wgt = d.detach().sqrt() ** alpha
    wgt = wgt / wgt.flatten(-2).amax(-1).clamp_min(1e-8)[..., None, None]
    return (wgt * d).mean()


# --------------------------------------------------------------------------- scoring
class Scorer:
    """PSNR / SSIM / a/b RMSE / L RMSE / LPIPS / FID over the test scenes.

    PSNR/SSIM/a/b RMSE/L RMSE use the same definitions as evaluate.py (RGB rebuilt with
    skimage's lab2rgb, scored at 256 px), so these rows compare with the README tables.
    """

    def __init__(self, device, with_fid=True):
        import lpips
        self.device = device
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.fid = None
        if with_fid:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.recs, self.terr = [], []

    @torch.no_grad()
    def add(self, L, ab, rgb_gt, terrain):
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        L, ab = L.float().clamp(-1, 1), ab.float().clamp(-1, 1)
        pred = lab_to_rgb(L, ab)                                     # (N,H,W,3) numpy
        gt = rgb_gt.permute(0, 2, 3, 1).cpu().numpy()
        _, ab_gt = rgb_to_lab_t(rgb_gt)
        ab_err = (ab_to_lab_units(ab) - ab_to_lab_units(ab_gt)).pow(2).mean(dim=(1, 2, 3)).sqrt()
        L_err = ((L - rgb_to_L(rgb_gt)) * 50).pow(2).mean(dim=(1, 2, 3)).sqrt()
        pred_t = torch.from_numpy(pred).permute(0, 3, 1, 2).float().to(self.device)
        lp = self.lpips(pred_t * 2 - 1, rgb_gt * 2 - 1).flatten()
        if self.fid is not None:
            self.fid.update(rgb_gt, real=True)
            self.fid.update(pred_t, real=False)
        for i in range(len(gt)):
            p = peak_signal_noise_ratio(gt[i], pred[i], data_range=1.0)
            s = structural_similarity(gt[i], pred[i], data_range=1.0, channel_axis=2)
            self.recs.append([p, s, ab_err[i].item(), L_err[i].item(), lp[i].item()])
            self.terr.append(terrain[i])

    def result(self):
        keys = ("psnr", "ssim", "ab_rmse", "L_rmse", "lpips")
        a = np.array(self.recs)
        out = {k: float(a[:, i].mean()) for i, k in enumerate(keys)}
        out["n"] = len(a)
        out["fid"] = float(self.fid.compute()) if self.fid is not None and len(a) > 1 else None
        out["per_terrain"] = {}
        for t in sorted(set(self.terr)):
            m = np.array([r for r, tt in zip(self.recs, self.terr) if tt == t])
            out["per_terrain"][t] = {k: float(m[:, i].mean()) for i, k in enumerate(keys)}
            out["per_terrain"][t]["n"] = len(m)
        return out


SHOW = 8   # test tiles kept for the side-by-side comparison figure


@torch.no_grad()
def evaluate(predict, test, device, out_dir, algo, bs=32, extra=None):
    """predict(gray (B,1,H,W) in [0,1]) -> list of K samples [(L, ab), ...] (K=1 if deterministic).

    Writes <out_dir>/metrics_test.json; with K>1 both the first sample ("sample") and the
    mean of the K samples ("mean") are scored. Also keeps SHOW predictions for figures.
    """
    os.makedirs(out_dir, exist_ok=True)
    scorers, shown, k = None, [], 1
    show_idx = np.linspace(0, len(test) - 1, SHOW).astype(int) if len(test) >= SHOW else np.arange(len(test))
    for idx in batches(len(test), bs, shuffle=False):
        gray, rgb = test.batch(idx, device)
        samples = predict(gray)
        k = len(samples)
        if scorers is None:
            scorers = {"sample": Scorer(device)}
            if len(samples) > 1:
                scorers["mean"] = Scorer(device)
        terr = [test.terrain[i] for i in idx]
        L0, ab0 = samples[0]
        scorers["sample"].add(L0, ab0, rgb, terr)
        if "mean" in scorers:
            Lm = torch.stack([s[0] for s in samples]).mean(0)
            abm = torch.stack([s[1] for s in samples]).mean(0)
            scorers["mean"].add(Lm, abm, rgb, terr)
        for j, i in enumerate(idx):
            if i in show_idx:
                shown.append(np.uint8(np.round(lab_to_rgb(L0[j:j + 1].float().clamp(-1, 1),
                                                          ab0[j:j + 1].float().clamp(-1, 1))[0] * 255)))
    res = {"algo": algo, "test_scenes": scenes_of(test), "k_samples": k}
    res.update({name: s.result() for name, s in scorers.items()})
    res.update(extra or {})
    with open(os.path.join(out_dir, "metrics_test.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "shown.npz"), pred=np.stack(shown),
                        idx=show_idx, sar=test.sar[show_idx], gt=test.rgb[show_idx])
    s = res["sample"]
    print(f"[{algo}] test n={s['n']}  PSNR {s['psnr']:.2f}  SSIM {s['ssim']:.4f}  ab RMSE "
          f"{s['ab_rmse']:.2f}  L RMSE {s['L_rmse']:.2f}  LPIPS {s['lpips']:.4f}  FID {s['fid']}", flush=True)
    if "mean" in res:
        m = res["mean"]
        print(f"[{algo}] mean of samples: PSNR {m['psnr']:.2f}  SSIM {m['ssim']:.4f}  "
              f"ab RMSE {m['ab_rmse']:.2f}  L RMSE {m['L_rmse']:.2f}", flush=True)
    return res


def scenes_of(part):
    from splits import scene_of
    return sorted({scene_of((os.path.join("s1", n), "", t)) for n, t in zip(part.names, part.terrain)})


def split_scene_check(rec, data):
    """Refuse to run if a test or val scene is also a training scene."""
    for name in ("val", "test"):
        leak = set(scenes_of(data[name])) & set(scenes_of(data["train"]))
        assert not leak, f"{name} scenes in training: {sorted(leak)}"
