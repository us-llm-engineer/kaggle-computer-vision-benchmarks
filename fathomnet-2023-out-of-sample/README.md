# FathomNet 2023: out-of-sample detection, three from-scratch methodologies

Multi-label recognition of 290 deep-sea categories plus detection of out-of-sample images. Everything is trained from scratch; the
competition rules allow pre-trained weights, this project deliberately does not use them.
Summary, figures and references are in the [repository README](../README.md); this page is the method and the evidence.

## 1. The task and its metric

For every test image a submission lists up to 20 categories ordered by confidence and an `osd` score for "drawn from a different
distribution". The score is `0.5 * (sAUC + MAP@20)` with `sAUC = 2 AUC - 1`. Training images come from 0-800 m, evaluation images
from 0-1300 m; categories live in 20 supercategories. The evaluation set has no public labels, so local numbers come from a
validation proxy described in section 3.

## 2. What the data look like

All values from [`results/preprocess_meta.json`](results/preprocess_meta.json).

| Quantity | Value |
|---|---|
| Labelled images / evaluation images | 5,950 / 10,744 |
| Categories with at least one labelled image | 133 of 290 (157 never appear) |
| Categories with five or fewer images | 63 |
| Labels per image | mean 1.51, maximum 10, 70.7% have one |
| Most common category | 2,643 images |
| Image sizes | 645x366 up to 1920x1080 |
| Channel mean / std (training split only) | (0.296, 0.385, 0.354) / (0.155, 0.148, 0.134) |

Two consequences drive every design choice: a very long tail (a model can only ever rank the 133 seen categories), and animals that
are small relative to the frame, so images are kept at 448x256 instead of the usual 224.

## 3. Pre-processing and the validation proxy (`preprocess.py`)

Run once, before any training, and every methodology reads its outputs:

1. decode all 16,694 JPEGs in batches on the GPU (nvjpeg) with a per-image CPU fallback; no decode errors occurred;
2. resize with the torchvision transforms v2 pipeline (float conversion, bilinear antialiased resize, back to uint8) to **448x256**
   and **224x128**; 63.8 s for all images on one GPU;
3. align the multi-hot label matrix to image order and compute per-channel mean and standard deviation on the **training split only**;
4. build the split once with a seed (`fn_common.make_split`): six categories with 40-150 images are chosen at random and every image
   that shows one of them is removed from training and becomes a **novel** validation image (550 images); the remaining images are
   split 85/15 into training (4,590) and in-distribution validation (810), stratified on each image's rarest label;
5. write the dataset statistics (above) to `prep/meta.json`.

The proxy measures whether a model can tell images with *unseen categories* from images with seen ones. It does not reproduce the
depth shift of the real evaluation set, and six held-out categories make it a noisy estimate (the novelty score moves by about 0.1
between consecutive epochs, Fig. 1c).

## 4. The three methodologies

| | Input | Model | Parameters | Training |
|---|---|---|---|---|
| 1 | 224x128 | 312 hand-crafted features (4x4x4 colour histogram, colour mean/std and gradient-orientation histograms on a 2x4 grid, Laplacian energy, 16x8 thumbnail) and a closed-form multi-output ridge regression | 312 x 290 weights | alpha grid {1, 10, 100, 1e3, 1e4, 3e4}, alpha = 100 chosen on validation |
| 2 | 224x128 | residual CNN: stem + 3 stages of 2 residual blocks (64/128/256 channels), average + max pooling, dropout 0.3, one logit per category | 2.91 M | 40 epochs, AdamW (lr 2e-3, weight decay 1e-2, one-cycle), batch 128, mixed precision, gradient clipping 5 |
| 3 | 448x256 | tile-MIL: the image is cut into 2x4 tiles of 128x112; a shared residual encoder (48/96/160 channels) scores every tile; image logit = (1/r) log mean exp(r * tile logit), r = 3 | 1.31 M | 30 epochs, batch 64, same optimiser |

Methodologies 2 and 3 train with a class-weighted binary cross-entropy (weight `sqrt((N - n_pos) / n_pos)` clipped to [1, 30])
and GPU augmentation: flips, scale 0.75-1 with shift and +-10 degrees rotation, brightness and contrast jitter, small noise. The
model with the best `0.5 * (MAP@20 + sAUC)` on validation is kept, so the reported validation numbers are slightly optimistic
(selection and reporting share the set). One seed per methodology.

Out-of-sample scores computed from the logits (higher = more novel): `1 - max probability`, negative energy `logsumexp`, and the
entropy of the normalised probabilities.

## 5. Results

[`results/final_metrics.csv`](results/final_metrics.csv):

| Methodology | Best epoch | MAP@20 | sAUC max-prob | sAUC energy | sAUC entropy | Train-subset MAP@20 |
|---|---|---|---|---|---|---|
| 1 Classical + ridge | - | 0.6167 | 0.2486 | 0.1348 | 0.2868 | 0.6683 |
| 2 Residual CNN | 33 / 40 | 0.6113 | 0.1773 | 0.0640 | 0.0932 | 0.6447 |
| 3 Tile-MIL CNN | 3 / 30 | 0.5631 | 0.2439 | 0.1517 | -0.0796 | 0.5690 |

What the in-training statistics add (figures in the repository README):

- Train and validation losses of both networks stay close (Fig. 1a) and the MAP@20 generalisation gap is 0.03 (CNN) and 0.01 (MIL)
  at the last epoch: these models under-fit the long tail rather than over-fit it.
- Mean per-category AP is 0.10 (classical), 0.11 (CNN) and 0.05 (MIL); the mean AP of rare supercategories rises slowly and some never
  leave zero (Figs. 2, 7).
- Gradient norms fall by an order of magnitude within a few epochs and the relative weight update shrinks in every layer group
  as the schedule decays (Figs. 1d, 8).
- Max-probability separates novel from seen images only weakly: the last-epoch mean max-probability is 0.705 for seen and 0.663 for
  novel images (CNN), 0.569 and 0.548 (MIL) (Fig. 1e-f, 3).
- Grad-CAM attention is broad and central rather than on the animals (Fig. 5).

## 6. Per-epoch statistics collected during training (`train_cnn.py`)

Rewritten atomically each epoch to `<name>_history.json` ([`results/history/`](results/history/)): train and validation loss,
learning rate, running top-1, train-subset and validation MAP@20, sAUC for three scores, gradient-norm mean and maximum (overall and
per layer group), relative weight update per layer group, weight norm, feature mean/std/zero share, logit statistics,
max-probability histograms for seen and novel images, validation AP for all 290 categories, reliability-bin counts, throughput,
epoch seconds, peak accelerator memory and loss scale. Checkpoints are written every epoch (resumable) and on improvement.

## 7. Compute

One NVIDIA L4 (23 GB) in a hosted notebook runtime; the two networks trained in parallel as separate processes sharing the device.
Median epoch 3.5 s (CNN, about 1,600 images/s, 1.9 GB peak) and 5.6 s (MIL, about 970 images/s, 5.0 GB peak); classical features
and ridge fit 95 s on CPU. The training run reported a mean device utilisation of 94% (median 99%) from the run's printed summary;
the raw samples were not kept.

## 8. Tests (`tests/test_fn_common.py`)

18 checks, all passing: the three worked examples of the competition text (all score 1.0), a plain-list reference implementation
against the vectorised MAP@20 on random data, perfect/worst rankings, sAUC for perfect/constant/inverted scores, ordering of the
three novelty scores, split disjointness and coverage, held-out categories never present in training, determinism, and a mutant split
that leaks novel images into training being detected.

## 9. Files

| File | Role |
|---|---|
| `fn_common.py` | label parsing, MAP@20, sAUC, novelty scores, validation split, optional shard reader |
| `fn_models.py` | residual CNN and tile-MIL network definitions |
| `fetch_images.py` | resumable image download to 448x256 JPEG (the annotation URLs moved hosts; the host is rewritten) |
| `preprocess.py` | the pre-processing stage of section 3 |
| `train_classical.py`, `train_cnn.py` | methodologies 1, 2 and 3 |
| `make_figures.py` | validation table and the nine figures |
| `tests/test_fn_common.py` | unit tests |
| `results/` | final metrics, dataset statistics, per-epoch histories |
| `figures/` | the nine figures |

## 10. Limitations

- No leaderboard result: Kaggle's scorer for this archived competition failed on every submission, including the unchanged
  sample file, so only the validation proxy exists.
- Single seed, no hyper-parameter search beyond the ridge strength, model selection on the reporting set.
- The proxy holds out six categories; the real shift is depth, and 157 categories are never labelled, so no model trained on the
  provided labels can rank them.
- Model checkpoints and the predictions on the evaluation set are not included.
