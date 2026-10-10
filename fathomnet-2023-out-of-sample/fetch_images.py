"""Resumable image fetcher for the FathomNet 2023 files. The URLs in the annotation files point at a host that has since moved, so
'fathomnet.org/static' is rewritten to 'database.fathomnet.org/static'. Only the image file is fetched (no metadata API), every image is resized
to 448x256 and stored as JPEG, and the run stops after a time budget so each call stays short; run it again to continue.
usage: python3 fetch_images.py {train|eval} [budget_seconds] [threads]"""
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

split = sys.argv[1]; budget = float(sys.argv[2]) if len(sys.argv) > 2 else 75; threads = int(sys.argv[3]) if len(sys.argv) > 3 else 48
here = os.path.dirname(os.path.abspath(__file__)); out = f"{here}/data/img448_{split}"; os.makedirs(out, exist_ok=True)
imgs = json.load(open(f"{here}/data/object_detection/{'train' if split == 'train' else 'eval'}.json"))["images"]
todo = [i for i in imgs if not os.path.exists(f"{out}/{i['file_name'][:-4]}.jpg")]; t0 = time.time(); sess = requests.Session(); stats = {"ok": 0, "fail": 0}


def get(i):
    if time.time() - t0 > budget: return
    dest = f"{out}/{i['file_name'][:-4]}.jpg"; url = i["coco_url"].replace("://fathomnet.org/", "://database.fathomnet.org/")
    for _ in range(3):
        try:
            r = sess.get(url, timeout=30)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                Image.open(io.BytesIO(r.content)).convert("RGB").resize((448, 256), Image.BILINEAR).save(dest + ".tmp", "JPEG", quality=95); os.replace(dest + ".tmp", dest); stats["ok"] += 1; return
        except Exception: pass
    stats["fail"] += 1


with ThreadPoolExecutor(threads) as ex: list(ex.map(get, todo))
print(f"{split}: {len(imgs) - len(todo) + stats['ok']}/{len(imgs)} stored, this run +{stats['ok']} (failed {stats['fail']}) in {time.time() - t0:.0f}s")
