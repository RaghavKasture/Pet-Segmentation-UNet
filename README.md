# Pet Image Segmentation with UNet

Binary semantic segmentation on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 37 cat and dog breeds with pixel-level trimap annotations — comparing two approaches:

1. **Vanilla UNet** — built from scratch in PyTorch
2. **Transfer Learning UNet** — ResNet-34 encoder pretrained on ImageNet, fine-tuned with a warm-up + fine-tuning regime

---

## Notebook Structure

Both models were originally developed in a single notebook, but the combined runtime came out to around **8.5 hours**. To make it practical to run, compare, and iterate on each approach independently, they were split into two self-contained notebooks — each covering EDA, data cleaning, preprocessing, model definition, training, and evaluation end-to-end.

```
├── unet-for-segmentation.ipynb          # Vanilla UNet — built from scratch
├── transfer-unet-for-segmentation.ipynb # Transfer Learning UNet — ResNet-34 encoder
├── requirements.txt
└── README.md
```

---

## Dataset

**Oxford-IIIT Pet Dataset** — ~7,400 images across 37 breeds, with pixel-level trimap masks.

Masks are binarized: foreground (pet) = 1, background = 0.

### Preprocessing & Augmentation

- Samples with foreground coverage <10% or >90% are removed (noisy extremes)
- 80/20 train/val split (seed=10)
- Training augmentations: resize to 496×496, horizontal flip, shift-scale-rotate, hue-saturation, brightness-contrast, ImageNet normalization
- Validation: resize + normalize only
- Training set is 2× augmented

---

## Model 1 — Vanilla UNet (`unet-for-segmentation.ipynb`)

A clean PyTorch implementation of the classic [UNet architecture](https://arxiv.org/abs/1505.04597).

**Architecture:**
- 4-level encoder: 64 → 128 → 256 → 512 channels via `DoubleConv` blocks (Conv → ReLU → Conv → ReLU)
- MaxPool2d downsampling
- 1024-channel bottleneck
- Transposed convolution upsampling with centre-cropped skip connections
- 1×1 conv output head (sigmoid applied at inference)

**Training config:**

| Setting | Value |
|---|---|
| Optimizer | Adam, lr=1e-3 |
| Loss | BCE + Dice |
| Epochs | 20 |
| Batch size | 8 |
| Input size | 496×496 |
| Checkpoint | Best val IoU |

---

## Model 2 — Transfer Learning UNet (`transfer-unet-for-segmentation.ipynb`)

Uses `segmentation-models-pytorch` with a **ResNet-34** encoder pretrained on ImageNet.

**Two-phase training:**

| Phase | Encoder | LR | Epochs |
|---|---|---|---|
| Warm-up | Frozen | 1e-3 | 5 |
| Fine-tuning | Unfrozen | 1e-4 | 10 |

Freezing the encoder during warm-up lets the decoder adapt to the pretrained features before end-to-end fine-tuning at a lower learning rate.

---

## Loss & Metrics

**Loss:** BCE-with-Logits + Dice Loss (combined)

**Metrics (threshold = 0.5):**
- **IoU** — intersection over union between predicted and ground truth masks
- **Dice Coefficient** — harmonic mean of precision and recall on mask pixels

Both notebooks also include failure case analysis: the 5 worst-performing validation samples by IoU are visualized for error inspection.

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Dataset

Both notebooks use `kagglehub` to download the dataset automatically at runtime:

```python
import kagglehub
_ = kagglehub.dataset_download("julinmaloof/the-oxfordiiit-pet-dataset")
```

You'll need a Kaggle account and your API credentials configured (`~/.kaggle/kaggle.json`). See the [kagglehub docs](https://github.com/Kaggle/kagglehub) for setup instructions.

---

## References

- Ronneberger et al., [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), MICCAI 2015
- Parkhi et al., [Cats and Dogs](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf), CVPR 2012
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
