# Multi-Branch Ensemble for Cross-Avatar Teleoperator Verification

Code for our first-place system in the **WBA 2026 challenge (Who is Behind the Avatar?)**, from the paper *Multi-Branch Ensemble for Cross-Avatar Teleoperator Verification: Correlation Descriptors, Graph Networks, and Video Encoders* (AUC CV Lab, The American University in Cairo).

| | Same generator | Different generator | Overall |
|---|---|---|---|
| Official test AUC (CodaBench submission 980274, rank 1) | 0.9275 | 0.6852 | 0.8063 |

## Checkpoints and final submission

The trained checkpoints and the submitted file are too large for GitHub and are on Google Drive:

**https://drive.google.com/drive/folders/1CZO8eeVMWaDIB_-7IITbfyvTksNEFCDb?usp=drive_link**

The Drive folder contains:
- `checkpoints/`: the 11 trained models used in the submission, one folder per model type;
- `best_submission/`: the exact file uploaded to CodaBench, `wba2026-al23-gen2robust3nv.zip`.

Download both folders and place them in the root of this repository, next to `code/`:

```text
wba-auc-cv-lab/
  README.md
  code/
    requirements.txt
    models/            model code, one folder per model type
    scripts/           setup, export, fusion and submission scripts
  checkpoints/         from Google Drive
  best_submission/     from Google Drive
```

## Models

The system fuses 22 branches. The same-generator file (G1) uses 15 of them and the different-generator file (G2) uses 13. The eleven closed-form correlation branches have no checkpoint; their LDA + WCCN back-ends are refitted deterministically from the training data.

| Name in the paper | Run ID | Code folder | Checkpoint | Used in |
|---|---|---|---|---|
| Corr-Pearson | `c_classical` | `code/models/correlation_descriptors` | (closed form) | G1, G2 |
| Corr-Spearman | `c_rank` | `code/models/correlation_descriptors` | (closed form) | G1, G2 |
| Corr-Velocity | `c_delta` | `code/models/correlation_descriptors` | (closed form) | G1, G2 |
| Corr-Acceleration | `c_accel` | `code/models/correlation_descriptors` | (closed form) | G1, G2 |
| Corr-Partial | `c_partial` | `code/models/correlation_descriptors` | (closed form) | G1, G2 |
| Corr-Lagged | `c_lagged` | `code/models/correlation_descriptors` | (closed form) | G2 |
| Corr-BandLow / -Mid / -High | `c_band_low`, `c_band_mid`, `c_band_high` | `code/models/correlation_descriptors` | (closed form) | G2 |
| Corr-Coherence / -RankCoherence | `c_coherence`, `c_rank_coherence` | `code/models/correlation_descriptors` | (closed form) | G2 |
| Corr-LearnedHead | `c_d8_hand` | `code/models/correlation_descriptors` | `correlation_descriptors/c_d8_hand` | G1, G2 |
| Video-Std-s1 / -s2 | `b3`, `b5` | `code/models/motion_encoder_video` | `motion_encoder_video/b3`, `b5` | G1 |
| Video-Aligned-s0 / -s1 / -s2 | `al1`, `al2_aligned_s1`, `al3_aligned_s2` | `code/models/motion_encoder_video` | `motion_encoder_video/al1`, `al2_aligned_s1`, `al3_aligned_s2` | G1 |
| LM-STGCN | `stgcn` | `code/models/stgcn` | `stgcn/stgcn` | G1 |
| LM-ECAPA-192 / -384 | `ktdnn_1`, `ktdnn_2` | `code/models/ecapa_tdnn` | `ecapa_tdnn/ktdnn_1`, `ktdnn_2` | G1 |
| LM-PDTCNN (τ = 0.5) | `nvidia` | `code/models/pairwise_distance_tcnn` | `pairwise_distance_tcnn/nvidia` | G1 |
| LM-PDTCNN (τ = 0.1) | `nv_t01_s1` | `code/models/pairwise_distance_tcnn` | `pairwise_distance_tcnn/nv_t01_s1` | G2 |

### Training hyperparameters

All networks are trained from scratch with AdamW and a cosine schedule with warm-up. Checkpoints are selected on validation AUC (20 held-out teleoperators, 12 held-out avatar templates).

| Branch | Iterations | LR / WD | Batch (teleop. × templ. × clips) | Window | Loss | Selected at |
|---|---|---|---|---|---|---|
| Video (all seeds) | 3,000 | 1e-3 / 1e-4 | 8 × 8 × 2 | 64 frames | SupCon τ = 0.07 + uniformity 0.1 | best (2,000–3,000) |
| LM-STGCN | 30,000 | 3e-4 / 0.01 | 4 × 2 × 2 | 75 frames | InfoNCE τ = 0.5 | 14,000 |
| LM-ECAPA-192 / -384 | 10,000 | 5e-4 / 0.05 | 32 × 4 × 1 | 48–160 frames | AAM + 0.5 SupCon + GRL | 500 / 1,000 |
| LM-PDTCNN τ = 0.5 | 30,000 | 3e-4 / 0.01 | 8 × 4 × 1 | 71 frames | three-term contrastive loss | 3,500 |
| LM-PDTCNN τ = 0.1 | 6,000 | 3e-4 / 0.01 | 8 × 4 × 1 | 71 frames | three-term contrastive loss | best |
| Corr-LearnedHead | 6,000 | 5e-4 / 1e-3 | 32 × 8 clips | 80–100 frames | AAM + 0.5 SupCon + 0.1 GRL | best |

At inference the video encoders embed the whole clip with five test-time views (identity, 0.93 zoom, 0.97 zoom shifted 3% left / right / down). ST-GCN and PD-TCNN average 100-frame windows with stride 36. ECAPA-TDNN and the correlation descriptors use the whole clip.

### Fusion weights

Each branch embedding is L2-normalised and scaled by sqrt(w / W), and the results are concatenated, so the cosine of two fused embeddings equals the weighted mean of the per-branch cosines. The weights are defined in code: `C51B5AL23` in `code/scripts/build_avatar_centered.py` (G1) and `gen2_weights()` in `code/scripts/build_final_submission.py` (G2). Normalised to sum to 1:

| Same generator (G1) | Weight | Different generator (G2) | Weight |
|---|---|---|---|
| Video-Aligned-s1 | 0.1547 | 11 correlation branches (all except BandHigh) | 0.0824 each |
| Corr-Pearson | 0.1275 | Corr-BandHigh | 0.0412 |
| Video-Std-s1, Video-Aligned-s0 | 0.0774 each | LM-PDTCNN (τ = 0.1) | 0.0528 |
| Video-Std-s2, Video-Aligned-s2 | 0.0773 each | | |
| LM-STGCN | 0.0645 | | |
| Corr-Acceleration | 0.0512 | | |
| Corr-LearnedHead, Corr-Velocity | 0.0490 each | | |
| Corr-Spearman, Corr-Partial | 0.0392 each | | |
| LM-ECAPA-192, LM-ECAPA-384, LM-PDTCNN (τ = 0.5) | 0.0387 each | | |

### Test-time normalisation

Both files are normalised with statistics of the unlabelled embeddings of their own test set; no labels or identity information are used.
- **Avatar centering:** subtract the mean of the 402 nearest clips in a static face-shape space. In G2 the correlation branches use one global mean instead.
- **Speaking/listening mean removal:** G1 only, with the mode predicted by a classifier trained on training clips.
- **Neighbourhood averaging:** k = 8, applied twice.

See `code/scripts/build_avatar_centered.py` and `code/scripts/build_final_submission.py`.

## Setup

Tested on Windows 11, Python 3.12, NVIDIA RTX 3050 Laptop GPU (4 GB).

```bash
pip install -r code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

The challenge data (`meta/`, `landmark_2d_106/`, `mp4/`, `train_val_splits/`) is distributed by the WBA 2026 organisers and is not included here.

The code expects the folder layout used during the competition. One command creates it and copies the checkpoints into place:

```bash
python code/scripts/setup_workspace.py --out D:/wba_workspace --data D:/WBA2026_data
cd D:/wba_workspace/wba
```

All following commands run inside the new working directory.

## Reproducing the submission

**1. Preprocessing**

```bash
python models/landmarks/dcs/build_cache.py --split all
python models/video/mdnet/preprocess.py --split train --size 96 --workers 6
python models/video/mdnet/preprocess_aligned.py --split train --size 96 --workers 6
# repeat both crop steps with --split test-gen1 and --split test-gen2
```

**2. Training** (optional: all checkpoints are on Google Drive)

```bash
# video encoder; per model: b3 --cache-suffix y --seed 1 | b5 y 2 | al1 ya 0 | al2 ya 1 | al3 ya 2
python models/video/mdnet/trainer.py --variant pixdiff --size 96 --cache-suffix ya --input-size 64 \
  --frames 64 --batch-drivers 8 --batch-sources 8 --cross-samples 2 --iterations 3000 --validation-interval 500 \
  --lr 1e-3 --weight-decay 1e-4 --temperature 0.07 --warmup-steps 500 --ema-decay 0.999 \
  --mixstyle 0.5 --embed-bn --uniformity 0.1 --no-register --seed 1 --output-dir output/video/study/al2_aligned_s1

# PD-TCNN at temperature 0.1
python scripts/train_nvidia_variant.py --tag nv_t01_s1 --temperature 0.1 --seed 1 --iterations 6000
```

ST-GCN, PD-TCNN (τ = 0.5), ECAPA-TDNN and Corr-LearnedHead are trained with the trainers in their model folders, using the settings in the table above.

**3. Per-model test embeddings** (for `test-gen1` and `test-gen2`)

```bash
# closed-form correlation branches; --signals: base | delta | accel | partial | lagged |
#   band_low | band_mid | band_high | coherence; Spearman variants add --rank
python models/landmarks/dcs/infer_classical.py --split test-gen1 --signals delta --shrinkage 0.05 \
  --out output/landmarks/dcs/classical_delta/gen1.bl2
python models/landmarks/dcs/infer.py --split test-gen1 \
  --checkpoint output/landmarks/dcs/d8_hand_seed1/best_model.pt --out output/landmarks/dcs/d8_hand_seed1/gen1.bl2
python models/video/mdnet/infer.py --geni gen1 --checkpoint output/video/study/al2_aligned_s1/best_model.pt \
  --frames 100 --tta 5 --output output/video/submission_build/al2_aligned_s1-gen1-f100.bl2
python scripts/export_fusion_landmarks.py --model stgcn --split gen1 --out-dir output/fusion/parity_fixed/stgcn
python scripts/export_fusion_landmarks.py --model nvidia --split gen1 \
  --checkpoint output/landmark_2d_106/variants/nv_t01_s1/best_model.pt --out-dir output/fusion/variants/nv_t01_s1
```

The paths read by the final build are listed in `scripts/build_share.py` and `scripts/build_avatar_centered.py`.

**4. Inputs of the test-time normalisation**

```bash
python scripts/avatar_shape_features.py --split train
python scripts/avatar_shape_features.py --split test-gen1
python scripts/avatar_shape_features.py --split test-gen2
python scripts/mode_classifier.py
```

**5. Build and verify the submission**

```bash
python scripts/build_final_submission.py --out submission/final_rebuild.zip \
  --verify <repository>/best_submission/wba2026-al23-gen2robust3nv.zip
```

A correct rebuild reports a row cosine of 1.000000 for all 40,200 rows of both files.

## Citation

```bibtex
@inproceedings{ragab2026multibranch,
  title     = {Multi-Branch Ensemble for Cross-Avatar Teleoperator Verification: Correlation Descriptors, Graph Networks, and Video Encoders},
  author    = {Yehia Ragab and Mohamed A. Ayoub and Aly Mohamed Mahmoud},
  booktitle = {Asian Conference on Computer Vision (ACCV) Workshops},
  year      = {2026}
}
```
