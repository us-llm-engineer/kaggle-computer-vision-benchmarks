# Filament segmentation 2026: instance segmentation of thin solar structures, four approaches

Instance segmentation of dark, thin, elongated filaments in 2048 x 2048 full-disk H-alpha images, scored by Panoptic Quality. Three
small U-Net variants are trained from scratch and one pre-trained Mask2Former is fine-tuned; all four share one pre-processing stage,
one grouped split, tiled inference and one instance-recovery and scoring code path.
Summary, figures and references are in the [repository README](../README.md); this page is the method and the evidence.

## 1. The task and its metric

One polygon per filament; a submission row per predicted instance as a run-length-encoded 2048 x 2048 mask. Predictions are matched to
ground truth at IoU > 0.5 and scored by Panoptic Quality,

`PQ = sum IoU(matched pairs) / (|TP| + 0.5 |FP| + 0.5 |FN|) = SQ x RQ`,

with SQ the mean IoU of matched pairs and RQ = |TP| / (|TP| + 0.5 |FP| + 0.5 |FN|). The competition's Evaluation and Leaderboard pages
name PQ as the ranking metric; its Submission File page still says the scoreboard shows a mean Dice score. Which one the scoreboard
displays could not be established, so PQ is reported together with SQ, RQ and pixel Dice. **No platform score was obtained**
(no submission was made).

## 2. What the data look like (`investigate_data.py`, [`results/data_investigation.json`](results/data_investigation.json))

| Quantity | Value |
|---|---|
| Training images / test images | 707 / 180 JPEG files, 2048 x 2048, 8-bit grayscale |
| Annotation records | 1,154 (one, two or three annotator copies for 411 / 145 / 151 images), 8,199 single-piece polygons |
| Instances per record | median 7, up to 26; area median 1,228 px; long side of the box median 96 px |
| Structure width (distance transform) | median 10 px; per-record 10th percentile median 5.9 px |
| Touching instances | 1.6% of sampled records |
| Train / test shift | none visible: same 2011 to 2022 span, same six instrument codes in the file names |
| **Agreement of two annotators of one image** | **median PQ 0.359** (10th / 90th percentile 0.133 / 0.575, maximum 0.748; 400 of 598 pairs) |

The last row is the ceiling that matters: against one annotator's labels a score of 0.55 sits above what two annotators usually reach
on each other (median SQ 0.631, median RQ 0.571).

## 3. Pre-processing, split and inference (`prep_fil.py`, `fil_gpu.py`, `fil_infer.py`)

- **Pre-processing, once, on the GPU:** batched nvjpeg decode, fit of the solar disk, radial flattening of limb darkening (120 bins),
  local contrast normalisation at 20 px (computed at 1/4 resolution), 8-bit quantisation. The GPU path is tested against a CPU reference
  (`tests/test_fil_gpu.py`: correlation above 0.97).
- **Split by file** (annotator copies of one image always share a role), stratified by instrument code, seed 0: FIT 496 files / 821
  records, CHOOSE 105 / 169, REPORT 106 / 164. CHOOSE drives early stopping and checkpoint choice; REPORT is meant to be read once.
- **Training crops** of 512 x 512 at full resolution (70% centred near an annotated vertex), flips and 90-degree rotations.
- **Inference:** overlapping 512 px tiles (stride 384) blended with a Hann window; threshold 0.5, 8-connected components, components
  below 30 px removed (`instances_from_prob`); a grid over threshold, minimum area and closing radius is implemented in `fil_infer.py`
  but was **not run**, so the reported numbers use these defaults.

## 4. The four approaches

| Variant | Model | Loss | Parameters |
|---|---|---|---|
| V0 control | 4-stage U-Net, base 16, from scratch | BCE + soft Dice | 1.94 M |
| V1 | V0 | (1 - 0.3)(BCE + soft Dice) + 0.3 (1 - soft-clDice), 15 skeleton iterations | 1.94 M |
| V2 | V0 + two edge-guided self-attention blocks at the 1/16 bottleneck (edge map = sigmoid(conv3x3(input)) added to queries and keys) | BCE + soft Dice | 2.47 M |
| V3 | Mask2Former, Swin-tiny, COCO-instance checkpoint, one class, 100 queries | the model's mask-classification loss | 47.4 M |

V1 follows the soft-skeleton loss of Shit et al. (2021) with the base term changed from soft Dice to BCE + Dice; V2 follows the
edge-guided attention of Solomon et al. (2025) at a smaller width; V3 turns the queries into one probability map,
`p(x) = sum_q P(filament | q) sigmoid(mask_q(x))`, so that touching instances are separated only by the shared component step.
**V3 is the one pre-trained model in this repository**; the others use no pre-trained weights.

Optimisation: U-Nets use Adam, 3e-4, batch 8, a cosine schedule to 5% of the rate; V0 to V2 seed 0 ran a first 30-epoch cycle and then
two warm restarts at 1e-4 (60 epochs for V0, 52 for V1, 58 for V2); seeds 1 and 2 of V0 and V2 are single 60-epoch cycles. V3 uses
AdamW 1e-4 (backbone 0.1x), weight decay 0.05, batch 6, gradient clipping 1.0, and stopped at a 26-minute wall-clock budget after
9 epochs. Each run's recorded epoch time stays under 30 minutes.

## 5. Results (best checkpoint, CHOOSE monitoring subset)

[`results/run_summary.csv`](results/run_summary.csv); the monitoring subset is a fixed 16 CHOOSE records (12 for V3, a subset of the
16). **These are not REPORT scores**: REPORT was never read, so there is no held-out number here.

| Variant | Runs | Best smoothed PQ per seed | Mean PQ / SQ / RQ at the best checkpoint | Extra predictions per true instance | Missed per true instance |
|---|---|---|---|---|---|
| V0 U-Net | 3 seeds | 0.334 / 0.359 / 0.345 | 0.351 / 0.665 / 0.516 | 1.02 | 0.32 |
| V1 + soft-clDice | 1 seed | 0.312 | 0.321 / 0.651 / 0.492 | 1.09 | 0.36 |
| V2 + edge attention | 3 seeds | 0.346 / 0.342 / 0.342 | 0.350 / 0.681 / 0.512 | 1.00 | 0.34 |
| V3 Mask2Former | 1 seed, 9 epochs | 0.319 | 0.329 / 0.686 / 0.479 | 1.36 | 0.24 |

Pixel Dice on the subset is 0.63 to 0.66 for the U-Nets and 0.59 for V3. How to read it:

- **V0 and V2 are indistinguishable** (seed ranges 0.334 to 0.359 and 0.342 to 0.346). The edge-attention blocks and the extra
  parameters did not move PQ; V1's soft-clDice term scored lowest in its single seed and was never given more.
- **RQ, not SQ, limits the score.** Matched instances overlap well (SQ 0.65 to 0.69) but only about two thirds of true instances are
  matched, while each true instance comes with about one extra predicted instance (Post 2, Post 4).
- **Fragmentation is the dominant error** for the U-Nets: about 0.14 true instances per true instance are split, 0.03 merged
  (merging is rare because touching instances are rare). Recall falls to 0.50 to 0.58 for instances of 3,000 px and more, which are
  the ones broken into pieces (Post 5).
- **The pre-trained Mask2Former reached PQ 0.33 after 9 epochs** (about 1,200 steps), with higher recall on large instances (0.72) and
  fewer splits (0.09), but more extra predictions (1.36). It is competitive per unit of training, not better; with one seed on 12
  images this is a lead, not a result.
- **Against the human ceiling:** the mean PQ of 0.35 is close to the human-vs-human median of 0.36 and far below its 90th percentile
  (0.58); per-image PQ of the best checkpoints spans 0.13 to 0.56 (Post 1, Post 3).
- The probability maps are confident (mean 0.70 inside a filament, 0.005 outside) and PQ is insensitive to the threshold between 0.3
  and 0.7 (Fig. 5), so the errors are in instance structure, not in thresholding.

### 5.1 First pass: 30 epochs

The first cosine cycle (seed 0, [`results/first_pass_30_epochs.csv`](results/first_pass_30_epochs.csv)) ended at best smoothed PQ 0.295
(V0), 0.200 (V1) and 0.305 (V2); V3 stopped at 0.319 after 9 epochs. At that point training Dice was only about 0.70 and the loss was
still falling, which is why the U-Nets were trained further. A statistics snapshot taken during the continuation is analysed in
[`results/first_pass_analysis.txt`](results/first_pass_analysis.txt) (trajectories, size recall, gradient and update ratios by layer
group). The continuation (30 more epochs, 22 for V1) moved V0 and V2 by about 0.04 and V1 by about 0.11.

## 6. Figures

Six in-training figures ([`make_figures_fil.py`](make_figures_fil.py)) and five post-training figures
([`make_posthoc_figures_fil.py`](make_posthoc_figures_fil.py)), all drawn from the files in `results/`; they are embedded with captions in
the [repository README](../README.md#filament-segmentation-2026).

## 7. Compute

One NVIDIA L4 (22 GiB), several jobs sharing it. Pre-processing of the 887 images takes about 40 s. A U-Net epoch of 102 steps takes a
median of 23 to 33 s (25 to 35 images per second) while sharing the GPU; a Mask2Former epoch of 136 steps takes about 90 s. The recorded
epoch time of every run is 24 to 30 minutes (26.5 for V3); the per-run table in `results/run_summary.csv` holds the epochs and scores.

## 8. Tests (`tests/`)

| File | Checks |
|---|---|
| `test_fil_common.py` | PQ on known cases including split and merge, relations, instance recovery, RLE round-trip at 2048 x 2048, grouped split (a per-record split is caught), preprocessing |
| `test_fil_models.py` | shapes, gradients through the edge branch, soft skeleton, soft-clDice (a swapped definition is caught), loss weights, tiling |
| `test_fil_m2f.py` | parameter groups, probability combination, loss decreases on a toy batch, empty targets (needs `transformers` and the public checkpoint) |
| `test_fil_gpu.py` | nvjpeg decode and GPU pre-processing against the CPU reference (needs a CUDA device and the data) |

## 9. Limitations

- **No held-out score.** Selection and every number above come from a 12 to 16 image monitoring subset of CHOOSE; REPORT (106 images)
  was not scored, and instance parameters were not tuned. A smoothed criterion over about 20 evaluations on 16 images is noisy:
  consecutive evaluations differ by up to 0.14 early in a run and by about 0.02 on average near the end.
- **One seed for V1 and V3**, three for V0 and V2; no seed ensemble was evaluated.
- Seeds 1 and 2 were trained with a different schedule from seed 0, and V1 stopped at 52 epochs; the runs are not equal-budget.
- The metric shown on the platform scoreboard (PQ or Dice) is unresolved, and the local PQ scores against one annotator at a time.
- V3 used a pre-trained checkpoint and got far fewer steps; its comparison with the U-Nets is not like for like.

## 10. Files

| Path | Role |
|---|---|
| `fil_common.py` | records, grouped split, rasterisation, pre-processing, PQ and relations, instance recovery, RLE |
| `fil_models.py`, `fil_m2f.py` | U-Net, edge-guided attention, losses; Mask2Former wrapper |
| `fil_gpu.py`, `prep_fil.py` | GPU pre-processing and the pre-processing stage |
| `train_fil.py`, `fil_infer.py`, `make_submission_fil.py` | training, tiled inference and scoring, submission writer |
| `investigate_data.py`, `analyze_round1.py`, `summarize_runs.py` | data investigation, statistics analysis, result tables |
| `make_figures_fil.py`, `make_posthoc_figures_fil.py` | the eleven figures |
| `results/` | run table, first-pass table, per-run training statistics and example crops, data investigation |

## 11. Reproduce

```bash
pip install numpy scipy pillow torch torchvision matplotlib pycocotools transformers kaggle
python3 tests/test_fil_common.py && python3 tests/test_fil_models.py       # seconds, no data needed
kaggle competitions download -c filament-segmentation-2026 && unzip -q filament-segmentation-2026.zip -d data
python3 investigate_data.py --root data/MAGFiLO_1.0_Kaggle_2026
python3 prep_fil.py --root data/MAGFiLO_1.0_Kaggle_2026 --out prep          # GPU decode and transform, then labels
python3 train_fil.py --variant v0 --seed 0 --prep prep --out runs --crop 512 --bs 8 --base 16 --epochs 30 --cosine-epochs 30 --min-epochs 30
python3 train_fil.py --variant v3 --seed 0 --prep prep --out runs --crop 512 --bs 6 --lr 1e-4 --epochs 1000 --time-budget-min 26
python3 fil_infer.py --run runs/v0_s0 --prep prep                          # CHOOSE scoring and instance-parameter grid
python3 summarize_runs.py && python3 make_figures_fil.py && python3 make_posthoc_figures_fil.py
```

The tables and figures in this folder are regenerated from `results/runs/` by the last line without any data.
