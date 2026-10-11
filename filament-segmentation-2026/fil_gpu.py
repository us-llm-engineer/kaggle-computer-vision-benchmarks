"""GPU pre-processing for the filament images: batched nvjpeg decode and the same fixed transform as fil_common.preprocess, computed with torch on the device.
Same constants as the CPU reference (radial flattening in 120 bins, 20 px local contrast at 1/4 resolution, quantisation to 8 bits); the disk fit uses a threshold,
a 5 x 5 opening and the area and centroid of the mask (the CPU version additionally fills holes and keeps the largest component; the test below bounds the difference).
Inputs and outputs are uint8 tensors, so batches can be written straight into the disk-backed arrays."""
import math

import torch
import torch.nn.functional as TF

S = 2048


def decode_gray(raw_bytes, dev):
    """list of JPEG byte strings -> list of uint8 (S,S) tensors on `dev` (batched nvjpeg on cuda, per-image fallback; raises on a real decode error)."""
    from torchvision.io import ImageReadMode, decode_jpeg
    raws = [torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in raw_bytes]
    try:
        out = decode_jpeg(raws, device=dev, mode=ImageReadMode.GRAY)
    except Exception:
        out = [decode_jpeg(r, mode=ImageReadMode.GRAY).to(dev) for r in raws]
    return [o[0] for o in out]


def disk_params_gpu(img):
    """(cy, cx, r) of the solar disk for a uint8 (S,S) tensor on the device."""
    x = img.float()
    thr = max(8.0, 0.25 * float(torch.quantile(x[::4, ::4].flatten(), 0.99)))
    m = (x > thr).float()[None, None]
    m = -TF.max_pool2d(-m, 5, 1, 2)                       # erosion then dilation: 5 x 5 opening
    m = TF.max_pool2d(m, 5, 1, 2)[0, 0] > 0.5
    area = m.sum().clamp_min(1)
    yy = torch.arange(S, device=img.device, dtype=torch.float32)[:, None]; xx = torch.arange(S, device=img.device, dtype=torch.float32)[None, :]
    cy = float((m * yy).sum() / area); cx = float((m * xx).sum() / area)
    return cy, cx, float(torch.sqrt(area / math.pi))


def _gauss_kernel(sigma, device):
    r = max(1, int(math.ceil(4 * sigma)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=device, dtype=torch.float32) / sigma) ** 2)
    return k / k.sum(), r


def _smooth(a, sigma, f=4):
    """Gaussian smoothing of sigma pixels at 1/f resolution (block mean, sigma / f, bilinear back); a is (S,S) float."""
    lo = TF.avg_pool2d(a[None, None], f)
    k, r = _gauss_kernel(sigma / f, a.device)
    lo = TF.conv2d(TF.pad(lo, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    lo = TF.conv2d(TF.pad(lo, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return TF.interpolate(lo, scale_factor=f, mode="bilinear", align_corners=True)[0, 0]


def preprocess_gpu(img, mode="flat_lcn", return_disk=False):
    """uint8 (S,S) tensor on the device -> uint8 (S,S) tensor, quantised 0 = -4 sigma ... 255 = +4 sigma (mid value off the disk)."""
    x = img.float()
    cy, cx, r = disk_params_gpu(img)
    yy = torch.arange(S, device=img.device, dtype=torch.float32)[:, None]; xx = torch.arange(S, device=img.device, dtype=torch.float32)[None, :]
    rho = torch.hypot(yy - cy, xx - cx) / max(r, 1.0)
    disk = rho < 0.985
    if mode == "flat_lcn":
        b = (rho * 100).long().clamp(0, 119)
        cnt = torch.bincount(b[disk], minlength=120).float()
        sm = torch.bincount(b[disk], weights=x[disk], minlength=120)
        prof = torch.where(cnt > 0, sm / cnt.clamp_min(1), torch.ones_like(cnt))
        k, rr = _gauss_kernel(1.5, img.device)
        prof = TF.conv1d(TF.pad(prof[None, None], (rr, rr), mode="replicate"), k.view(1, 1, -1))[0, 0]
        x = x / prof[b].clamp_min(1.0)
        d = disk.float()
        w = _smooth(d, 20)
        g = _smooth(x * d, 20) / w.clamp_min(1e-3)
        v = _smooth(((x - g) ** 2) * d, 20) / w.clamp_min(1e-3)
        z = (x - g) / torch.sqrt(v + 1e-4)
    elif mode == "raw":
        mu = x[disk].mean(); sd = x[disk].std() + 1e-6
        z = (x - mu) / sd
    else:
        raise ValueError(mode)
    z = torch.where(disk, z, torch.zeros_like(z))
    out = ((z + 4.0) * 31.875).clamp(0, 255).to(torch.uint8)
    return (out, (cy, cx, r)) if return_disk else out
