# Nuclei Segmentation with U-Net and Watershed Post-processing

This repository provides a Keras-based implementation of U-Net for instance segmentation of nuclei in histopathology images, enhanced with watershed post-processing and evaluated using five-fold cross-validation on the **NuInsSeg** dataset.



## Table of Contents

* [Features](#features)
* [Dataset](#dataset)
* [Installation](#installation)
* [Usage](#usage)
* [Configuration](#configuration)
* [Model Architectures](#model-architectures)
* [Data Augmentation](#data-augmentation)
* [Training & Evaluation](#training--evaluation)
* [Metrics](#metrics)
* [Results](#results)
* [Visualization](#visualization)


## Features

* **Shallow and Deep U-Net** implementations
* **Albumentations**-based data augmentation (CLAHE, flips, noise, rotations, etc.)
* **K‑fold cross-validation** (default: 5 folds)
* **Watershed-based instance separation** using distance maps & peak local maxima
* **Post-processing** to remove small objects and vague regions
* **Evaluation metrics**: Dice, Aggregated Jaccard Index (AJI), Panoptic Quality (PQ)
* **Logging** of training history and per-fold metrics


## Dataset

We use the [NuInsSeg] dataset, organized by organ type:

```
nuinsseg/
├── human melanoma/
│   ├── tissue images/            # RGB PNG images
│   ├── mask binary/              # Binary masks (cell vs. background)
│   ├── distance maps/            # Distance transforms for watershed
│   ├── label masks modify/       # Ground truth instance labels (TIFF)
│   └── vague areas/              # Regions marked as “vague” (PNG)
├── human liver/
│   └── …
└── mouse spleen/
    └── …
```


## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/nuclei-unet-watershed.git
   cd nuclei-unet-watershed
   ```

2. **Create and activate a Python environment**

   ```bash
   conda create -n nuclei python=3.8
   conda activate nuclei
   ```

3. **Install dependencies**

   ```bash
   pip install \
     numpy pandas opencv-python scikit-image \
     tensorflow keras albumentations \
     scipy tqdm matplotlib
   ```


## Usage

1. **Set dataset path** in the script (default `base_path = '../input/nuinsseg/'`).
2. **Configure options** at the top of the script (`opts` dictionary).
3. **Run training & evaluation**:

   ```bash
   python train_unet_watershed.py
   ```

   This will:

   * Build directories for saving models and predictions
   * Perform 5‑fold cross-validation
   * Train a Deep U-Net on each fold
   * Save the best weights per fold (`output_model/unet_#.h5`)
   * Generate instance predictions (with and without watershed/vague filtering)
   * Compute and log Dice, AJI, PQ per fold
   * Export CSV summaries (`dice.csv`, `aji.csv`, `pq.csv`)



## Configuration

All hyperparameters and paths are defined in the `opts` dictionary:

```python
opts = {
    'number_of_channel':   3,
    'threshold':           0.5,
    'epoch_num':           100,
    'batch_size':          16,
    'k_fold':              5,
    'init_LR':             1e-3,
    'LR_decay_factor':     0.5,
    'LR_drop_after_nth_epoch': 20,
    'crop_size':           512,
    'save_val_results':    True,
    'model_save_path':     './output_model/',
    'result_save_path':    './prediction_image/'
}
```

* **`threshold`**: binarization cutoff for U‑Net output
* **`crop_size`**: random crop size for augmentation
* **`LR_decay_factor`**: factor by which learning rate is multiplied every `LR_drop_after_nth_epoch`


## Model Architectures

### Deep U-Net

```python
def deep_unet(IMG_CHANNELS, learn_rate):
    inputs = Input((None, None, IMG_CHANNELS))
    # Encoder: Conv–Dropout–Pool blocks (16 → 32 → 64 → 128 → 256 filters)
    # Bridge: two 512-filter conv layers
    # Decoder: Conv2DTranspose + skip connections back to 256 → 128 → 64 → 32 → 16
    outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learn_rate),
        loss=bce_dice_loss,
        metrics=[dice_coef]
    )
    return model
```

* **Loss**: `0.5*BCE(y_true,y_pred) − Dice(y_true,y_pred)`
* **Metric**: Dice coefficient



## Data Augmentation

Using **Albumentations** for on‑the‑fly augmentation:

```python
Compose([
    RandomCrop(512,512),
    CLAHE(p=0.5),
    RandomBrightnessContrast(p=0.4),
    HueSaturationValue(p=0.1),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    ShiftScaleRotate(p=0.1),
])
```



## Training & Evaluation

* **Cross-validation**: `KFold(n_splits=5, shuffle=True, random_state=19)`

* **Per-fold**:

  1. Train U‑Net with early stopping on best validation Dice.
  2. Predict on held‑out fold: raw binary mask and watershed-labeled mask.
  3. Remove “vague” regions, relabel, and save PNGs under:

     ```
     prediction_image/
     ├── validation/unet/<image_id>.png
     └── validation/watershed_unet/<image_id>.png
     ```
  4. Compute **Dice**, **AJI**, **PQ** with and without vague regions.

* **Logging**:

  * `output_model/unet_<fold>.log` (CSVLogger)
  * Summary CSVs: `dice.csv`, `aji.csv`, `pq.csv`



## Metrics

* **Dice Coefficient**: overlap measure
* **Aggregated Jaccard Index (AJI)**: instance‑aware IoU
* **Panoptic Quality (PQ)**: combines detection & segmentation quality

Custom implementations:

* `get_fast_aji(true, pred)`
* `get_fast_pq(true, pred)`



## Results

Example five‑fold averages:

|   Fold   | Dice (%) |  AJI (%) |  PQ (%)  |
| :------: | :------: | :------: | :------: |
|     1    |   80.6   |   41.8   |   42.2   |
|     2    |   79.2   |   42.0   |   42.7   |
|     3    |   79.6   |   45.7   |   44.9   |
|     4    |   77.9   |   41.2   |   41.7   |
|     5    |   81.0   |   45.1   |   44.3   |
| **Mean** | **79.7** | **43.2** | **43.2** |

Watershed post-processing and vague-region filtering further improve AJI and PQ.



## Visualization

After training, random validation examples are displayed showing:

1. **Original image**
2. **Ground-truth labels**
3. **Vague-region mask**
4. **Watershed-labeled prediction**
5. **Post-processed (without vague)**

Use the included plotting code at the end of the script to inspect results.
