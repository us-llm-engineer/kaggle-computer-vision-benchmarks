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

**Correction.** The `proxy_score` column (0.4326 / 0.3943 / 0.4035) uses MAP@20 on seen images only. With MAP@20 over all validation images
the competition-style score is 0.3347 / 0.2967 / 0.3141 (columns `val_MAP@20_all_images` and `score_all_images` in
`results/final_metrics.csv`; derivation in section 11.1).

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
| `figures/` | the nine figures of the first attempt |
| `fn_oos_common.py`, `fn_oos_models.py`, `rank_tail.py`, `prep_oos.py`, `train_oos.py`, `combiner_oos.py`, `eval_oos.py`, `make_figures_oos.py` | the out-of-sample round (section 11) |
| `tests/test_fn_oos_common.py` | 34 checks of the three-set split, the scores and the samplers |
| `results/oos/`, `figures_oos/` | results, per-epoch histories and the nine figures of the out-of-sample round |

## 10. Limitations

- No leaderboard result: Kaggle's scorer for this archived competition failed on every submission, including the unchanged
  sample file, so only the validation proxy exists.
- Single seed, no hyper-parameter search beyond the ridge strength, model selection on the reporting set.
- The proxy holds out six categories; the real shift is depth, and 157 categories are never labelled, so no model trained on the
  provided labels can rank them.
- Model checkpoints and the predictions on the evaluation set are not included.

## 11. Out-of-sample case: four variants on a three-set split

This round targets the half of the competition score that the first attempt handled worst (the out-of-sample score, sAUC 0.18 to 0.25), and
it repairs three flaws of the first attempt's validation proxy. Everything is trained from scratch; code is in `fn_oos_common.py`,
`fn_oos_models.py`, `rank_tail.py`, `prep_oos.py`, `train_oos.py`, `combiner_oos.py`, `eval_oos.py`, `make_figures_oos.py`, results in
`results/oos/`, figures in `figures_oos/`.

### 11.1 Flaws of the first proxy, and the corrected measure

1. **MAP@20 was computed on seen images only.** The competition averages MAP@20 over every test image. Recomputed from the saved
   logits, seen-image MAP@20 is 0.616 / 0.611 / 0.563 for the three methodologies but only 0.133 / 0.129 / 0.121 on the 550 novel
   images, so all-image MAP@20 is 0.421 / 0.416 / 0.384 and the competition-style score `0.5 (sAUC + MAP@20)` is **0.335 / 0.297 /
   0.314**, not the 0.43 / 0.39 / 0.40 obtained with seen-image MAP.
2. **Selection leaked into the reported set.** The checkpoint and the score were chosen on the same 550 novel images that were reported.
3. **54% of the novel images are mixed:** 296 of the 550 also show a seen category (254 show only held-out categories).

### 11.2 Data methodology: three disjoint held-back sets

Fourteen categories are held out of training (`fn_oos_common.make_oos_split`, seed 0). Every image that shows one of them leaves the
in-distribution pool; an image showing categories of several sets goes to the highest-priority set (report > select > pool). A **novel
image is any image that shows a held-out category**; sAUC is also reported separately for pure-novel and mixed images.

| Set | Categories | Images | Used for |
|---|---|---|---|
| train | 106 seen categories | 4,279 | fitting |
| ID validation | same 106 | 756 | checkpoint choice, negatives of every sAUC |
| REPORT | 6 (the first attempt's: 16, 120, 133, 203, 242, 260) | 550 | read **once** per frozen configuration, by `eval_oos.py` only |
| SELECT | 4 (105, 202, 228, 259) | 204 | choosing the ranking-tail strength; never the reported number |
| POOL | 4 (61, 69, 100, 224) | 161 | only the learned combiner (V4) |

Shares of mixed images: REPORT 51%, SELECT 67%, POOL 70%. Pre-processing is the torchvision-v2 stage of section 3, reused unchanged;
only the split and the channel statistics (new training set only) are regenerated (`prep_oos.py`, which asserts disjointness and that
no held-out category reaches training or ID validation). `tests/test_fn_oos_common.py` (34 checks, three leak mutants) guards this.

### 11.3 The four variants and the mathematics (all multi-label adaptations are the author's own, untested in the cited papers)

All variants share the residual trunk of section 4 (2.91 M parameters, 224x128 input, GPU augmentation, class-weighted binary
cross-entropy with positive weights clipped to [1, 30]) plus a 20-way supercategory head. With sigmoid logits `f_i(x)`, i = 1..290:

**V0 control.** Novelty score `1 - max_i sigmoid(f_i)`, the competition baseline.

**V1 hypersphere outliers** (idea of VMF-GOS, Peng et al. 2026). The L2-normalised penultimate feature `z` lives on `S^(d-1)`. Per seen
category `c` an exponential moving average keeps a mean direction `mu_c` (over images where `c` is positive, weight `1/#labels`) and a
concentration `kappa_c` from the mean resultant length `r` (`kappa ~ r (d - r^2) / (1 - r^2)`). Virtual outliers are drawn at cosine
`t = 1 - xi / (2 kappa_c)` from `mu_c`, with `xi` uniform on `[mu_chi + 2 s, mu_chi + 3 s]`, `mu_chi = d - 1`, `s = sqrt(2 (d - 1))`
(the 2 to 3 sigma tail of `chi^2_(d-1)`), as `z_out = t mu_c + sqrt(1 - t^2) v`, with `v` Gram-Schmidt-orthogonal to `mu_c`. The head is
trained to give them a negative label on every seen category. The paper does not say how `mu` and `kappa` are obtained or give most
hyper-parameters; the estimators above are this project's choice. Score: negative joint energy.

**V2 radius head** (idea of BootOOD, Wang et al. 2025). Features of image pairs with **disjoint label sets** are mixed,
`h = a h_i + (1 - a) h_j`, `a ~ Beta(1, 1)`, and a head is trained to place them in one of `K = 4` inner shells of radii
`rho_k in [0.25 r_ref, (1 - gamma) r_ref]` around the moving-average ID centre (`gamma = 0.2`): shell classification + squared radius
regression + an angular term `mean |cos(h, w_c)|`, weight 0.1 with linear warm-up after epoch 10. Score: distance to the ID centre.

**V3 evidential sum** (Aguilar et al. 2025 with the JointEnergy aggregation of Wang et al. 2021). Two heads give positive and negative
evidence `alpha = ELU(f_p) + 2`, `beta = ELU(f_n) + 2`; the prediction is `alpha / (alpha + beta)` and the loss is the Bayes risk of the
binary cross-entropy under `Beta(alpha, beta)`:
`y (psi(alpha + beta) - psi(alpha)) + (1 - y) (psi(alpha + beta) - psi(beta))` (checked against numerical integration in the tests). Scores
`U_p = L / sum alpha`, `U_n = 1 - (1/L) sum 1/beta`, blend `0.5 U_p + 0.5 U_n` (the paper's default weight).

**V4 learned combiner** (idea of COOD, Hogeweg et al. 2024, with the outlier expert of MARVEL, Anudeep and Sundaresan 2026). On the
frozen features of V0, 15 measures (temperature-scaled and plain max probability, joint energy `sum log(1 + e^f)`, entropy, kNN class
probabilities with k = 10, linear/kNN agreement, a supercategory distance between the linear and the kNN prediction, nearest-neighbour
distances, entropy-weighted distances, feature entropy, global and per-class PCA reconstruction error) plus a logistic outlier expert are
combined by a random forest trained on ID validation images (negatives) against POOL and SELECT images (positives), with five-fold
cross-fitting; REPORT images are scored by the mean of the fold models and never enter a fit.

**Ranking tail** (all variants). The competition scores categories no model can rank: 157 categories never occur in training. The tail
adds, behind a novelty gate `g(x)` (empirical CDF of the score against ID validation), the unseen categories of the predicted
supercategory with score `tau g(x) P(supercategory) / n_unseen`, leaving the order of seen categories unchanged; `tau` is chosen on SELECT.

**Fixed scores and selection.** Each variant's novelty score is fixed in advance from its paper's design (V0 max probability, V1 energy,
V2 radius, V3 evidential blend, V4 combiner); checkpoints and early stopping use MAP@20 on ID validation only (3-epoch smoothing, cap 120
epochs, learning rate cut by 0.3 after 6 epochs without improvement, stop after 18). Three seeds per variant; the table reports the
ensemble of the three seeds and the seed half-range.

### 11.4 Results (REPORT, 550 novel + 756 ID-validation images)

| Variant | Score `0.5 (sAUC + MAP@20)` | Seed scores | sAUC (all / pure / mixed) | MAP@20 seen / novel / all |
|---|---|---|---|---|
| **V0 control** | **0.402** (95% CI 0.368 to 0.432) | 0.351, 0.416, 0.394 | 0.327 / 0.213 / 0.435 | 0.722 / 0.140 / 0.477 |
| V4 learned combiner | 0.394 (0.359 to 0.423) | 0.354, 0.399, 0.372 | 0.310 / 0.063 / 0.545 | 0.722 / 0.140 / 0.477 |
| V1 hypersphere outliers | 0.303 (0.270 to 0.337) | 0.250, 0.306, 0.315 | 0.111 / 0.319 / -0.086 | 0.750 / 0.145 / 0.495 |
| V2 radius head | 0.231 (0.199 to 0.263) | 0.217, 0.208, 0.261 | 0.024 / -0.201 / 0.238 | 0.665 / 0.126 / 0.439 |
| V3 evidential sum | 0.171 (0.142 to 0.199) | 0.150, 0.184, 0.183 | -0.184 / 0.113 / -0.466 | 0.797 / 0.154 / 0.526 |

Source: [`results/oos/oos_results.json`](results/oos/oos_results.json); per-epoch histories in `results/oos/history/`. The chosen
tail strength is `tau = 0` for every variant, i.e. the tail never helped.

**Read this honestly.**
- **The target of a score of at least 0.55 was not reached.** The best variant is the plain control at 0.402; the gap is about 0.15
  (sAUC 0.33 against about 0.62 needed at the observed MAP).
- **None of the four novelty methods beats the control** on the competition-style score. V1 to V3 are clearly worse on sAUC; V4 ties V0
  within the seed band. Longer, correctly selected training raised all-image MAP@20 from 0.38-0.42 to 0.44-0.53; that, not the
  novelty methods, is the real gain of this round.
- **V3 has the best MAP and the worst sAUC**; it was still improving at the 120-epoch cap in all three seeds, so it is under-trained.
  Five of the twelve runs reached the cap.
- **MAP@20 on novel images stays at 0.13 to 0.15**, which caps the MAP half near 0.48 whatever the novelty score does.
- **Mixed images behave differently from pure-novel ones** (V0 pure 0.21 vs mixed 0.44; V4 0.06 vs 0.55); the cited papers never test
  that case.
- **Protocol history.** A first run selected, per epoch, the best of about ten novelty scores on the 204 SELECT images; that maximum was
  optimistic (SELECT sAUC about 0.3, REPORT negative), stopped training at epochs 4 to 11 and made one variant look like a winner
  (0.29). The numbers above come from the corrected protocol (fixed scores, selection on ID-validation MAP). The correction was made
  after seeing REPORT, so these numbers are not a clean first read. The per-score REPORT sAUCs are stored in the results file as
  description only; some other scores look better than the fixed ones (for example the temperature-scaled maximum, 0.34 to 0.39 on V0,
  V1 and V3) and were deliberately not used.
- **Cross-fitting caveat for V4.** Its out-of-fold SELECT sAUC is 0.71, far above its REPORT value (0.31), because folds mix images of
  the same category; a category-grouped split would be the honest cross-fit. REPORT categories are disjoint from every fit, so the
  REPORT figure is the valid one.
- Only a proxy: six held-out categories from the same depth range stand in for a real 800 to 1300 m shift, and the Kaggle scorer for
  this archived competition returned errors for every file, so there is no leaderboard figure for this round.

### 11.5 Per-epoch statistics (the three in-training figures)

Each epoch logs, beside losses and gradient norms per layer group, the ID-validation MAP, novelty sAUC on SELECT split into pure and
mixed images, on POOL and for train against validation (a memorisation gap), behaviour of the 184 outputs that never receive a
positive in the seen set (mean logit, share of arg-max predictions), seen-label probability on mixed images, distance to seen-category
prototypes, the weights of seen versus unseen output units, for V1 the joint-energy novelty of fresh virtual outliers against real
novel images, for V2 radii of ID, novel and mixed features, for V3 mean positive and negative evidence. REPORT is never read per epoch.

### 11.6 Compute

One NVIDIA L4 (22 GiB, 12 vCPUs), twelve training jobs (four variants, three seeds) as up to ten concurrent processes plus three CPU
combiner jobs that start as soon as their control run ends. Wall time of the corrected run is about half an hour; mean GPU utilisation 98%
(median 100%, at least 85% in 97% of 373 samples), peak memory 16.7 GiB ([`results/oos/gpu_utilisation_summary.json`](results/oos/gpu_utilisation_summary.json)).
Per-step host synchronisation was removed from the training loop so the processes keep the GPU busy.

### 11.7 Reproduce

```bash
python3 tests/test_fn_common.py && python3 tests/test_fn_oos_common.py          # needs the labels from the competition download
python3 prep_oos.py                                                            # three-set split from the pre-processing output of section 3
for s in 0 1 2; do for v in bce vmf radius evid; do python3 train_oos.py $v --seed $s --name ${v}_s$s; done; done
for s in 0 1 2; do python3 combiner_oos.py --pred out/bce_s${s}_pred.npz --out out/comb_s${s}_pred.npz --seed $s; done
python3 eval_oos.py --variants bce vmf radius evid comb --seeds 0 1 2 --out results/oos_results.json
python3 make_figures_oos.py --results results/oos_results.json
```

References for this section: Peng et al., *VMF-GOS: Geometry-guided virtual Outlier Synthesis for Long-Tailed OOD Detection*, 2026;
Wang, Qin, Valle, Abrahao, *BootOOD: Self-Supervised Out-of-Distribution Detection via Synthetic Sample Exposure under Neural Collapse*, 2025;
Wang, Liu, Bocchieri, Li, *Can multi-label classification networks know what they don't know?*, NeurIPS 2021; Aguilar, Raducanu, Radeva,
*Multi-label out-of-distribution detection via evidential learning*, 2025; Hogeweg et al., *COOD: Combined out-of-distribution detection
using multiple measures for anomaly and novel class detection in large-scale hierarchical classification*, 2024; Anudeep, Sundaresan,
*MARVEL: Margin-Aware Robust von Mises-Fischer Expert Learning for Long-Tailed Out-of-Distribution Detection*, 2026. Four of these papers use
ImageNet-pretrained backbones; only their mechanisms are used here, never their reported numbers.
