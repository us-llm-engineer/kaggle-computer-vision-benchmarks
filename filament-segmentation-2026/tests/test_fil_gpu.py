"""GPU pre-processing versus the CPU reference on real images (needs a CUDA device and the downloaded data). Run: python3 helpers/tests/test_fil_gpu.py"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fil_common as F, fil_gpu as G
from PIL import Image
if not torch.cuda.is_available():
    print("no CUDA device: GPU test skipped"); sys.exit(0)
if not os.path.isdir(os.environ.get("FIL_ROOT", "data/MAGFiLO_1.0_Kaggle_2026")):
    print("competition data not found: GPU test skipped"); sys.exit(0)
dev = torch.device("cuda")
root = F.find_root(); d = os.path.join(root, "train", "train_images"); names = sorted(os.listdir(d))[::97][:5]
raw = [open(os.path.join(d, n), "rb").read() for n in names]
dec = G.decode_gray(raw, dev)
for n, t in zip(names, dec):
    ref = np.asarray(Image.open(os.path.join(d, n)).convert("L"))
    assert t.shape == (F.S, F.S) and t.dtype == torch.uint8
    assert np.abs(t.cpu().numpy().astype(int) - ref.astype(int)).mean() < 1.0, "nvjpeg decode differs from PIL decode beyond rounding"
print("ok   nvjpeg batched decode agrees with the PIL decode (mean abs diff < 1 grey level)")
for n, t in zip(names, dec):
    ref = np.asarray(Image.open(os.path.join(d, n)).convert("L"))
    cy0, cx0, r0 = F.disk_params(ref); cy1, cx1, r1 = G.disk_params_gpu(t)
    assert abs(cy0 - cy1) < 4 and abs(cx0 - cx1) < 4 and abs(r0 - r1) < 6, (n, (cy0, cx0, r0), (cy1, cx1, r1))
print("ok   GPU disk fit within 4 px of centre and 6 px of radius of the CPU reference")
for mode in ("flat_lcn", "raw"):
    for n, t in zip(names, dec):
        ref = np.asarray(Image.open(os.path.join(d, n)).convert("L"))
        a = F.preprocess(ref, mode).astype(np.float32); b = G.preprocess_gpu(t, mode).cpu().numpy().astype(np.float32)
        c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        assert c > 0.97 and np.abs(a - b).mean() < 6.0, (mode, n, c, np.abs(a - b).mean())
        assert abs(int(b[0, 0]) - 127) <= 1
print("ok   GPU preprocess matches the CPU reference (correlation > 0.97, mean abs diff < 6 of 255) in both modes")
t = time.time(); [G.preprocess_gpu(x) for x in dec]; torch.cuda.synchronize(); print(f"ok   {(time.time() - t) / len(dec) * 1000:.0f} ms per image on the GPU (CPU reference ~1100 ms)")
