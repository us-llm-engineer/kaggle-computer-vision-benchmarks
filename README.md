# Kaggle computer-vision benchmarks

Worked, reproducible solutions to Kaggle computer-vision competitions, trained **from scratch** (no pre-trained weights) and
evaluated with the competition's own metric where it can be computed offline. Each folder is self-contained: code, unit tests,
result tables, figures and a README with the method, the numbers and the caveats.

| Folder | Competition | Task | Metric |
|---|---|---|---|
| [`fathomnet-2023-out-of-sample/`](fathomnet-2023-out-of-sample/) | [FathomNet 2023 (FGVC10)](https://www.kaggle.com/competitions/fathomnet-out-of-sample-detection) | Multi-label recognition of 290 deep-sea animal categories and detection of images from a different depth range | mean of sAUC (out-of-sample detection) and MAP@20 |

## FathomNet 2023 in one page

**Problem.** 5,950 labelled images from 0-800 m depth; the evaluation images (10,744) come from 0-1300 m. A model must rank the
categories present in each image (MAP@20) and score how likely the image is out-of-sample (sAUC = 2 AUC - 1).

**Baseline.** 0.664 at the current Leaderboard

**What is in the data** (all from [`results/preprocess_meta.json`](fathomnet-2023-out-of-sample/results/preprocess_meta.json)):
only 133 of the 290 categories have any labelled image, 157 have none; 63 of the 133 have at most five images; images carry 1.51 labels
on average (70.7% have one); the most common category appears in 2,643 images.

**Pipeline.** One pre-processing stage (batched GPU JPEG decode, torchvision transforms v2 resizing to 448x256 and 224x128, channel
statistics from the training split only, a fixed split with six categories held out as "novel"), then three from-scratch methodologies
trained on identical inputs:

1. hand-crafted colour/gradient/texture features with a closed-form multi-output ridge model;
2. a residual CNN on 224x128 inputs, trained with GPU augmentation and a class-weighted binary cross-entropy;
3. a tile-MIL CNN that scores a 2x4 grid of tiles of the 448x256 image and pools tile logits with log-sum-exp.

**Validation results** ([`results/final_metrics.csv`](fathomnet-2023-out-of-sample/results/final_metrics.csv); 810 in-distribution and 550
novel validation images; sAUC uses 1 - max probability):

| Methodology | MAP@20 | sAUC (max-prob) | sAUC (energy) | sAUC (entropy) | Mean per-category AP | Parameters |
|---|---|---|---|---|---|---|
| 1 Classical features + ridge | **0.6167** | **0.2486** | 0.1348 | 0.2868 | 0.1044 | closed form |
| 2 Residual CNN, 224x128 | 0.6113 | 0.1773 | 0.0640 | 0.0932 | 0.1061 | 2.91 M |
| 3 Tile-MIL CNN, 448x256 | 0.5631 | 0.2439 | 0.1517 | -0.0796 | 0.0473 | 1.31 M |

**Read this honestly.**
- The simplest model is competitive: on 4,590 training images with a very long tail, learned features add little to hand-crafted ones.
- Out-of-sample detection is weak for every method (best sAUC 0.29), and per-category average precision is about 0.10: most of the 133
  learnable categories are barely learned.
- The tile-MIL model reaches its best validation score at epoch 3 of 30 and does not improve afterwards.
- **No leaderboard score is reported.** The three prediction files were submitted, but Kaggle's scorer for this archived
  competition returned "Scoring session had non-zero exit code" for every file, including the competition's own unchanged
  `sample_submission.csv`, so the validation numbers above are the only measurements. They are a *proxy*: the novelty proxy
  (held-out categories) is not the depth shift of the real evaluation set.

## Visualizations

Six result figures and three that read the training statistics collected per epoch. Each caption names the file it comes from.

<table><tr><td colspan="2" align="center"><img src="fathomnet-2023-out-of-sample/figures/fig1_training_dynamics.png" alt="Figure 1: training dynamics"><br><sub><b>Fig. 1</b> Loss, MAP@20 with the generalisation gap, novelty-detection sAUC, learning rate and gradient norm, and how the max-probability histograms of in-distribution and novel images separate. Source: <code>results/history/cnn_history.json</code>, <code>mil_history.json</code>.</sub></td></tr>
<tr><td colspan="2" align="center"><img src="fathomnet-2023-out-of-sample/figures/fig2_long_tail.png" alt="Figure 2: long tail"><br><sub><b>Fig. 2</b> Per-category average precision against the number of training images (log scale). Source: per-category AP in <code>results/history/*.json</code> and <code>results/final_metrics.csv</code>.</sub></td></tr>
<tr><td width="50%"><img src="fathomnet-2023-out-of-sample/figures/fig3_out_of_sample.png" alt="Figure 3: out-of-sample scores"><br><sub><b>Fig. 3</b> Distribution of the 1 - max-probability score for in-distribution and novel images, and ROC curves of the max-probability and energy scores.</sub></td>
<td width="50%"><img src="fathomnet-2023-out-of-sample/figures/fig6_supercategory_heatmap.png" alt="Figure 6: supercategory heatmap"><br><sub><b>Fig. 6</b> Mean average precision per supercategory and methodology, rows clustered.</sub></td></tr>
<tr><td colspan="2" align="center"><img src="fathomnet-2023-out-of-sample/figures/fig4_embedding.png" alt="Figure 4: t-SNE"><br><sub><b>Fig. 4</b> t-SNE of the validation features of both networks, coloured by the supercategory of each image's rarest label; novel images are crosses.</sub></td></tr>
<tr><td colspan="2" align="center"><img src="fathomnet-2023-out-of-sample/figures/fig5_gradcam.png" alt="Figure 5: Grad-CAM"><br><sub><b>Fig. 5</b> Grad-CAM of the residual CNN on the four best and four worst validation images with the ground-truth boxes in green: attention is broad and central rather than on the small animals, which the model identifies from global scene cues.</sub></td></tr>
<tr><td colspan="2" align="center"><img src="fathomnet-2023-out-of-sample/figures/fig7_supercategory_trajectories.png" alt="Figure 7: supercategory trajectories"><br><sub><b>Fig. 7</b> (in training) Mean AP of each supercategory per epoch. Source: <code>results/history/*.json</code>.</sub></td></tr>
<tr><td colspan="2" align="center"><img src="fathomnet-2023-out-of-sample/figures/fig8_layerwise_learning.png" alt="Figure 8: layer-wise learning"><br><sub><b>Fig. 8</b> (in training) Relative weight update and mean gradient norm per layer group and epoch.</sub></td></tr>
<tr><td colspan="2" align="center"><img src="fathomnet-2023-out-of-sample/figures/fig9_calibration_and_learning_time.png" alt="Figure 9: calibration and learning time"><br><sub><b>Fig. 9</b> (in training) Reliability curves by epoch, calibration error, and the epoch at which each category first reaches AP 0.3 against its training frequency.</sub></td></tr></table>

## Reproduce

Needs Python 3.10+, PyTorch with torchvision, scikit-learn, seaborn, a Kaggle account that accepted the competition rules and the
Kaggle CLI. A GPU is used when present; the unit tests and a small pre-processing run work on CPU.

```bash
cd fathomnet-2023-out-of-sample
python3 tests/test_fn_common.py                      # 18 checks of the metrics and the split, ~seconds
kaggle competitions download fathomnet-out-of-sample-detection -p data && (cd data && unzip -oq *.zip)
python3 fetch_images.py train && python3 fetch_images.py eval     # resumable; re-run until complete
python3 preprocess.py                                # arrays, split and statistics -> prep/
python3 train_classical.py && python3 train_cnn.py cnn --epochs 40 --batch 128 && python3 train_cnn.py mil --epochs 30 --batch 64
python3 make_figures.py                              # tables and the nine figures
```

Details, design decisions and the file map are in [`fathomnet-2023-out-of-sample/README.md`](fathomnet-2023-out-of-sample/README.md).

## Data and licence

The images are **not** redistributed here: they are licensed CC BY-NC-ND 4.0 by FathomNet/MBARI and the competition rules forbid
passing the data to non-participants. `fetch_images.py` downloads them from the URLs in the competition annotation files. Please
acknowledge [FathomNet](https://fathomnet.org) and the [Monterey Bay Aquarium Research Institute](https://www.mbari.org).

## Candidate next experiments (not run)

Each follows from a measured weakness above and is **untested** here:

- class-balanced sigmoid loss for the 63 categories with five or fewer images (Cui et al., 2019);
- choosing learning rate and weight decay from the early parameter norm and dropping momentum (Brigato and Mougiakakou, 2023), noting that its precondition of full training accuracy is not met by these models;
- attention pooling over tiles in place of log-sum-exp (Ilse et al., 2018);
- coarse-to-fine heads over the 20 supercategories (Zhu and Bain, 2017);
- a normalised-logit loss or energy score for novelty detection (Wei et al., 2022; Liu et al., 2020), both defined for softmax outputs, so an extension to independent sigmoid outputs would be required.

## References

- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., Belongie, S. *Class-Balanced Loss Based on Effective Number of Samples.* CVPR 2019.
- Brigato, L., Mougiakakou, S. *No Data Augmentation? Alternative Regularizations for Effective Training on Small Datasets.* ICCV Workshops 2023.
- Ilse, M., Tomczak, J., Welling, M. *Attention-based Deep Multiple Instance Learning.* ICML 2018.
- Zhu, X., Bain, M. *B-CNN: Branch Convolutional Neural Network for Hierarchical Classification.* arXiv:1709.09890, 2017.
- Wei, H., Xie, R., Cheng, H., Feng, L., An, B., Li, Y. *Mitigating Neural Network Overconfidence with Logit Normalization.* ICML 2022.
- Liu, W., Wang, X., Owens, J. D., Li, Y. *Energy-based Out-of-distribution Detection.* NeurIPS 2020.
- FGVC10 workshop, CVPR 2023: FathomNet 2023 competition, MBARI.
